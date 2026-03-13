import argparse
import json
import re
from pathlib import Path
from typing import List, Optional


SPLIT_PATTERNS = [
    r"\?",
    r"\band\b",
    r"\bthen\b",
    r"\bafter that\b",
    r"，",
    r"。",
    r"；",
]

WH_STARTERS = ("what", "which", "who", "whom", "whose", "where", "when", "why", "how")
AUX_STARTERS = (
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "can",
    "could",
    "will",
    "would",
    "should",
)

MULTIHOP_CUES = [
    r"\band\b",
    r"\bthen\b",
    r"\bafter that\b",
    r"\bbefore\b",
    r"之后",
    r"然后",
    r"并且",
    r"且",
]


def _to_full_question(text: str) -> str:
    """Normalize a fragment into a complete-looking interrogative sentence."""
    q = (text or "").strip(" ,.;，。；")
    if not q:
        return q

    q = re.sub(r"\s+", " ", q)
    lower = q.lower()

    if lower.startswith(WH_STARTERS) or lower.startswith(AUX_STARTERS):
        return q if q.endswith("?") else f"{q}?"

    if re.match(r"^(has|have|had|contains?|include[sd]?|inspire[sd]?|located|born|founded)\b", lower):
        return f"Which entities {q}?"

    if any(x in q for x in ["哪个", "哪一个", "哪位", "是谁", "哪里", "哪儿", "什么"]):
        return q if q.endswith("？") else f"{q}？"

    return f"What is {q}?"


def _semantic_decompose(question: str) -> Optional[List[str]]:
    """Template-based semantic decomposition with dependency placeholders.

    Returns None when no semantic template matches.
    """
    q = (question or "").strip()
    if not q:
        return None

    m = re.match(r"^(.+?)的(.+?)出生的城市是哪个国家[？?]?$", q)
    if m:
        entity = m.group(1).strip()
        role = m.group(2).strip()
        return [
            f"谁是{entity}的{role}？",
            "[B]出生在哪个城市？",
            "[C]位于哪个国家？",
        ]

    m = re.match(r"^(.+?)的(.+?)是哪个国家[？?]?$", q)
    if m:
        left = m.group(1).strip()
        right = m.group(2).strip()
        return [
            f"{left}的{right}是什么？",
            "[B]位于哪个国家？",
        ]

    return None


def _is_multihop_question(question: str) -> bool:
    """Heuristic hop detection: only decompose likely multi-hop questions."""
    q = (question or "").strip()
    if not q:
        return False

    # Strong signal: semantic templates explicitly matched.
    if _semantic_decompose(q):
        return True

    # Coordination/temporal connectors.
    for cue in MULTIHOP_CUES:
        if re.search(cue, q, flags=re.IGNORECASE):
            return True

    # Chinese nested possession/attributes often indicate multi-hop chains.
    if "哪个国家" in q and q.count("的") >= 2:
        return True

    # Placeholder/coref marker in raw text (if present in input).
    if re.search(r"\[[A-Z]\]", q):
        return True

    return False

def split_question(question: str):
    text = question.strip()
    if not text:
        return []
    
    # 单跳问题不分解，直接返回原题。
    if not _is_multihop_question(text):
        return [text]

    # Prefer semantic decomposition when possible.
    semantic_parts = _semantic_decompose(text)
    if semantic_parts:
        return semantic_parts

    # Fallback: regex split + normalization.
    merged = text
    for pattern in SPLIT_PATTERNS:
        merged = re.sub(pattern, " [SPLIT] ", merged, flags=re.IGNORECASE)

    parts = [p.strip(" ,.;，。；") for p in merged.split("[SPLIT]")]
    parts = [p for p in parts if len(p) > 3]

    # 保底：如果没有拆出来，返回原问题
    if not parts:
        parts = [text]

    # 去重并保序
    uniq = []
    seen = set()
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(_to_full_question(p))
    return uniq


def _infer_dep_type(subq: str, sub_id: int) -> str:
    if sub_id == 0:
        return "none"
    q = (subq or "").lower()
    if any(x in q for x in ["[b]", "[c]", " it ", " its ", " they ", " them ", " that one ", " this one "]):
        return "coref"
    if any(x in q for x in [" which ", " what ", " where ", " when ", " who ", " whose ", "哪个", "什么", "谁"]):
        return "filter"
    return "bridge"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, type=str)
    parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument("--max_subquestions", type=int, default=3)
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    out_rows = 0

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            item = json.loads(line)
            q = item.get("question", "")
            subquestions = split_question(q)[: args.max_subquestions]

            if len(subquestions) == 1 and subquestions[0].strip("?").strip("？").lower() == q.strip("?").strip("？").lower():
                row = dict(item)
                if "id" in row:
                    row["id"] = str(row["id"])
                row["parent_id"] = str(row.get("id", total - 1))
                row["sub_id"] = 0
                row["dep_prev_sub_id"] = None
                row["dep_type"] = "none"
                row["needs_prev_answer"] = False
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_rows += 1
                continue

            parent_id = str(item.get("id", total - 1))
            for i, sq in enumerate(subquestions):
                row = dict(item)
                row["parent_id"] = parent_id
                row["sub_id"] = i
                row["id"] = f"{parent_id}_{i}"
                row["question"] = sq

                # 依赖链字段（供 cascade 推理 / bridge 注入使用）
                row["dep_prev_sub_id"] = i - 1 if i > 0 else None
                row["dep_type"] = _infer_dep_type(sq, i)
                row["needs_prev_answer"] = i > 0

                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_rows += 1

    print(f"Input samples: {total}")
    print(f"Output sub-question samples: {out_rows}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
