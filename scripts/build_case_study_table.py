#!/usr/bin/env python3
"""Build paper-ready case-study tables from poisoned sub-question data.

Outputs markdown with:
- A/B/C categories (5 high-quality examples each by default)
- one short statistical conclusion paragraph
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


GENERIC_TARGETS = {
    "person", "actor", "university", "city/town/village", "published", "disease or medical condition",
    "country", "organization", "unknown", "none", "entity",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="poisoned jsonl file")
    p.add_argument("--output", default="results/case_study_table.md", help="markdown output path")
    p.add_argument("--top_k", type=int, default=5, help="examples per category")
    return p.parse_args()


def _safe_int(x: Any, default: int = 10**9) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _parent_id(item: Dict[str, Any]) -> str:
    pid = str(item.get("parent_id", "")).strip()
    if pid:
        return pid
    qid = str(item.get("id", "")).strip()
    if "_" in qid:
        return qid.split("_")[0]
    return qid


def _target(item: Dict[str, Any]) -> str:
    return str(item.get("poison_target") or item.get("poison_target_entity") or "").strip()


def _quality_score(question: str, target: str) -> float:
    q = (question or "").strip()
    t = (target or "").strip()
    if not q or not t:
        return -1e9

    score = 0.0
    score += min(len(q), 120) / 40.0
    score += min(len(t), 60) / 30.0
    if "?" in q:
        score += 0.8
    if t.lower() not in GENERIC_TARGETS:
        score += 1.2
    if 2 <= len(t.split()) <= 6:
        score += 0.6
    if re.search(r"[A-Za-z]", t):
        score += 0.4
    if re.search(r"[^\x00-\x7F]", q):  # likely encoding issue
        score -= 0.8
    if q.lower().count("?") > 1:
        score -= 0.4
    return score


def load_groups(path: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            pid = _parent_id(item)
            if not pid:
                continue
            grouped[pid].append(item)

    for pid in grouped:
        grouped[pid].sort(key=lambda r: (_safe_int(r.get("sub_id")), str(r.get("id", ""))))
    return grouped


def summarize(pid: str, arr: List[Dict[str, Any]]) -> Dict[str, Any]:
    first = None
    first_target = ""
    rows = []
    edges: List[Tuple[Any, Any]] = []
    inherited_count = 0

    for r in arr:
        sid = _safe_int(r.get("sub_id"))
        tgt = _target(r)
        dep_inj = bool(r.get("is_dependency_injected", False))
        frm = r.get("inherited_from_sub_id", None)

        if first is None and bool(r.get("is_poisoned", False)) and tgt:
            first = sid
            first_target = tgt

        if dep_inj:
            inherited_count += 1
            edges.append((frm, sid))

        rows.append({
            "sub_id": sid,
            "question": str(r.get("question", "")).strip(),
            "target": tgt,
            "dep_inj": dep_inj,
            "from": frm,
        })

    chain = " -> ".join([str(first)] + [str(x[1]) for x in edges]) if first is not None and edges else (str(first) if first is not None else "-")
    base_q = rows[0]["question"] if rows else ""
    score = _quality_score(base_q, first_target)

    return {
        "parent_id": pid,
        "first_poison_sub_id": first,
        "first_target": first_target,
        "inherit_edges": edges,
        "chain": chain,
        "inherited_count": inherited_count,
        "rows": rows,
        "score": score,
    }


def to_row_md(case: Dict[str, Any]) -> str:
    q_preview = " / ".join([f"s{r['sub_id']}: {r['question']}" for r in case["rows"][:3]])
    q_preview = q_preview.replace("|", "\\|")
    target = case["first_target"].replace("|", "\\|")
    edges = ", ".join([f"{a}->{b}" for a, b in case["inherit_edges"]]) or "-"
    return f"| {case['parent_id']} | {case['first_poison_sub_id']} | {case['chain']} | {edges} | {target} | {q_preview} |"


def main() -> None:
    args = parse_args()
    groups = load_groups(args.input)
    cases = [summarize(pid, arr) for pid, arr in groups.items()]
    cases = [c for c in cases if c["first_poison_sub_id"] is not None]

    A = sorted([c for c in cases if c["inherited_count"] == 0], key=lambda x: x["score"], reverse=True)
    B = sorted([c for c in cases if c["inherited_count"] >= 1], key=lambda x: x["score"], reverse=True)
    C = sorted([c for c in cases if c["inherited_count"] >= 2], key=lambda x: x["score"], reverse=True)

    topA, topB, topC = A[: args.top_k], B[: args.top_k], C[: args.top_k]

    total = len(cases)
    a_n, b_n, c_n = len(A), len(B), len(C)
    b_ratio = (b_n / total * 100) if total else 0.0
    c_ratio = (c_n / total * 100) if total else 0.0

    out = []
    out.append("# Case Study Tables (A/B/C)\n")
    out.append(f"Total poisoned parent groups: **{total}**. A-only: **{a_n}**, B-propagation: **{b_n}**, C-multi-step: **{c_n}**.\n")

    out.append("## A) First-poison only (no dependency propagation)\n")
    out.append("| parent_id | first_poison_sub_id | chain | inherit_edges | poison_target | sub-questions (preview) |")
    out.append("|---|---:|---|---|---|---|")
    out.extend(to_row_md(c) for c in topA)

    out.append("\n## B) Has dependency propagation (>=1 inherited node)\n")
    out.append("| parent_id | first_poison_sub_id | chain | inherit_edges | poison_target | sub-questions (preview) |")
    out.append("|---|---:|---|---|---|---|")
    out.extend(to_row_md(c) for c in topB)

    out.append("\n## C) Multi-step propagation (>=2 inherited nodes)\n")
    out.append("| parent_id | first_poison_sub_id | chain | inherit_edges | poison_target | sub-questions (preview) |")
    out.append("|---|---:|---|---|---|---|")
    out.extend(to_row_md(c) for c in topC)

    out.append(
        "\n## Statistical conclusion\n"
        f"Across {total} poisoned parent groups, dependency propagation appears in {b_n} groups ({b_ratio:.2f}%), "
        f"while multi-step propagation appears in {c_n} groups ({c_ratio:.2f}%). "
        "This indicates poisoning most often lands on the first sub-question, with a smaller but non-trivial fraction "
        "showing cross-step cascading effects."
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out), encoding="utf-8")
    print(f"Saved markdown: {out_path}")


if __name__ == "__main__":
    main()

# python scripts/build_case_study_table.py \
#   --input datasets/poisoned_cwq_dynamic_full.jsonl \
#   --output results/case_study_table.md \
#   --top_k 5