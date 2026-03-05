import argparse
import json
import re
from pathlib import Path


SPLIT_PATTERNS = [
    r"\?",
    r"\band\b",
    r"\bthen\b",
    r"\bafter that\b",
    r"\bwhich\b",
    r"\bwhose\b",
    r"，",
    r"。",
    r"；",
]


def split_question(question: str):
    text = question.strip()
    if not text:
        return []

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
        if not p.endswith("?"):
            p = p + "?"
        uniq.append(p)
    return uniq


def _infer_dep_type(subq: str, sub_id: int) -> str:
    if sub_id == 0:
        return "none"
    q = (subq or "").lower()
    if any(x in q for x in [" it ", " its ", " they ", " them ", " that one ", " this one "]):
        return "coref"
    if any(x in q for x in [" which ", " what ", " where ", " when ", " who ", " whose "]):
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

            # 如果拆不出来，就把原样本输出一次（同时补齐依赖字段）
            if len(subquestions) == 1 and subquestions[0].strip("?").lower() == q.strip("?").lower():
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
                row["needs_prev_answer"] = (i > 0)

                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_rows += 1

    print(f"Input samples: {total}")
    print(f"Output sub-question samples: {out_rows}")
    print(f"Saved to: {output_path}")
if __name__ == "__main__":
    main()
