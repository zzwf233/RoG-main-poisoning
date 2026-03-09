#!/usr/bin/env python3
"""Trace first-poison and dependency propagation paths per parent_id.

Usage:
  python scripts/trace_poison_paths.py \
      --input datasets/poisoned_subquestions.jsonl

Optional:
  --output_json results/poison_trace.json
  --only_poisoned
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="poisoned_subquestions.jsonl path")
    parser.add_argument("--output_json", default="", help="Optional JSON output path")
    parser.add_argument(
        "--only_poisoned",
        action="store_true",
        help="Only print parent groups with at least one poisoned sub-question",
    )
    return parser.parse_args()


def _safe_int(x: Any, default: int = 10**9) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _resolve_parent_id(item: Dict[str, Any]) -> str:
    parent_id = str(item.get("parent_id", "")).strip()
    if parent_id:
        return parent_id
    qid = str(item.get("id", "")).strip()
    if "_" in qid:
        return qid.split("_")[0]
    return qid


def _resolve_input_path(path: str) -> Path:
    raw = Path(path)
    candidates = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        cwd = Path.cwd()
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent
        candidates.extend([cwd / raw, repo_root / raw, script_dir / raw])

    unique_candidates = []
    seen = set()
    for c in candidates:
        key = str(c.resolve()) if c.exists() else str(c)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(c)

    for c in unique_candidates:
        if c.exists() and c.is_file():
            return c

    # Fallback: try basename under common data folders.
    basename = raw.name
    common_roots = [Path.cwd(), Path(__file__).resolve().parent.parent]
    common_dirs = ["datasets", "data", "results", "outputs"]
    for root in common_roots:
        for d in common_dirs:
            c = root / d / basename
            if c.exists() and c.is_file():
                return c

    hint_list = "\n".join(f"  - {c}" for c in unique_candidates) if unique_candidates else "  (none)"
    raise FileNotFoundError(
        f"Input file not found: {path}\n"
        f"Current working directory: {Path.cwd()}\n"
        f"Tried these paths:\n{hint_list}\n"
        "Tip: pass an absolute path, or run from project root containing your datasets/."
    )


def load_groups(path: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    resolved_path = _resolve_input_path(path)
    with open(resolved_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            parent_id = _resolve_parent_id(item)
            if not parent_id:
                continue
            grouped[parent_id].append(item)

    for parent_id in grouped:
        grouped[parent_id].sort(
            key=lambda x: (
                _safe_int(x.get("sub_id"), 10**9),
                str(x.get("id", "")),
            )
        )
    return grouped


def summarize_parent(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    first_poison_sub_id = None
    first_poison_qid = ""
    first_poison_target = ""

    nodes: List[Dict[str, Any]] = []
    for it in items:
        sid = _safe_int(it.get("sub_id"), 10**9)
        is_poisoned = bool(it.get("is_poisoned", False))
        target = str(it.get("poison_target") or it.get("poison_target_entity") or "").strip()
        is_dep_injected = bool(it.get("is_dependency_injected", False))
        inherited_from = it.get("inherited_from_sub_id", None)

        if first_poison_sub_id is None and is_poisoned and target:
            first_poison_sub_id = sid
            first_poison_qid = str(it.get("id", ""))
            first_poison_target = target

        nodes.append(
            {
                "sub_id": sid,
                "qid": str(it.get("id", "")),
                "is_poisoned": is_poisoned,
                "target": target,
                "is_dependency_injected": is_dep_injected,
                "inherited_from_sub_id": inherited_from,
            }
        )

    inherit_edges: List[Tuple[Any, int]] = []
    inherited_sub_ids: List[int] = []
    for n in nodes:
        if n["is_dependency_injected"]:
            inherited_sub_ids.append(n["sub_id"])
            inherit_edges.append((n["inherited_from_sub_id"], n["sub_id"]))

    return {
        "subquestion_count": len(nodes),
        "first_poison_sub_id": first_poison_sub_id,
        "first_poison_qid": first_poison_qid,
        "first_poison_target": first_poison_target,
        "inherited_sub_ids": inherited_sub_ids,
        "inherit_edges": inherit_edges,
        "nodes": nodes,
    }


def main() -> None:
    args = parse_args()
    try:
        groups = load_groups(args.input)
    except FileNotFoundError as e:
        print(str(e))
        return

    parent_ids = sorted(groups.keys(), key=lambda x: (_safe_int(x, 10**9), x))
    report: Dict[str, Any] = {}

    print("parent_id\tfirst_poison_sub_id\tpropagation_path")
    for pid in parent_ids:
        summary = summarize_parent(groups[pid])
        if args.only_poisoned and summary["first_poison_sub_id"] is None:
            continue

        first = summary["first_poison_sub_id"]
        inherited = summary["inherited_sub_ids"]
        if first is None:
            path = "(no poison target found)"
        elif not inherited:
            path = f"{first}"
        else:
            chain = " -> ".join(str(x) for x in [first] + inherited)
            edges = ", ".join(f"{a}->{b}" for a, b in summary["inherit_edges"])
            path = f"{chain} | inherit_edges: {edges}"

        print(f"{pid}\t{first}\t{path}")
        report[pid] = summary

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
