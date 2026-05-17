#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List


GENERIC_KEYWORDS = {
    "country",
    "movie",
    "movies",
    "language",
    "currency",
    "city",
    "location",
    "place",
    "person",
    "team",
    "college",
    "university",
}


def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"\s+", text.strip().lower()) if t]


def _spread_to_percent(value) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    if v <= 1.0:
        return max(0.0, min(100.0, v * 100.0))
    return max(0.0, min(100.0, v))


def _label_question(question: str) -> List[str]:
    q = question.strip().lower()
    toks = _tokenize(q)
    labels = []

    if re.search(r"\b[bcd]\b", q):
        labels.append("placeholder")

    if re.search(r"\b(is|are|was|were)\s+(is|are|was|were)\b", q):
        labels.append("grammar_degraded")
    if "what is was" in q:
        labels.append("grammar_degraded")

    starts_with_wh = toks[:1] and toks[0] in {"what", "which", "where", "who", "when", "how"}
    generic_hits = sum(1 for t in toks if t in GENERIC_KEYWORDS)
    very_short = len(toks) <= 3
    if (starts_with_wh and very_short) or generic_hits >= 1 and len(toks) <= 5:
        labels.append("generic_template")

    if any(name in q for name in ["scarlett johansson", "tom hanks", "barack obama"]):
        labels.append("entity_hotspot")

    if not labels:
        labels.append("other")

    return sorted(set(labels))


def _priority_score(parents: int, spread_pct: float, labels: List[str]) -> float:
    # Base risk: high spread + broad cross-parent reuse.
    parent_component = min(100.0, math.log1p(max(0, parents)) * 18.0)
    score = 0.65 * spread_pct + 0.35 * parent_component

    boosts = {
        "grammar_degraded": 12,
        "generic_template": 10,
        "placeholder": 8,
        "entity_hotspot": 6,
    }
    for lb in labels:
        score += boosts.get(lb, 0)
    return min(100.0, round(score, 2))


def _tier(score: float) -> str:
    if score >= 85:
        return "P0"
    if score >= 70:
        return "P1"
    if score >= 55:
        return "P2"
    return "P3"


def _recommendation(labels: List[str]) -> str:
    recs = []
    if "placeholder" in labels:
        recs.append("Normalize [B]/[C] placeholders to <PREV_ANSWER> before spread grouping")
    if "generic_template" in labels:
        recs.append("Add min-information filter or enrich template with relation/entity constraints")
    if "grammar_degraded" in labels:
        recs.append("Patch decomposition rewrite rules to avoid malformed WH questions")
    if "entity_hotspot" in labels:
        recs.append("Track hotspot entities separately to avoid over-attributing global spread")
    if not recs:
        recs.append("Manual review")
    return "; ".join(recs)


def load_shared(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            q = str(item.get("question", "")).strip().lower()
            if not q:
                continue
            parents = int(item.get("parents", 0))
            hit_parents = int(item.get("hit_parents", 0))
            spread_pct = _spread_to_percent(item.get("spread_rate", 0.0))
            labels = _label_question(q)
            score = _priority_score(parents, spread_pct, labels)
            rows.append(
                {
                    "question": q,
                    "parents": parents,
                    "hit_parents": hit_parents,
                    "spread_rate": round(spread_pct, 2),
                    "labels": labels,
                    "priority_score": score,
                    "priority_tier": _tier(score),
                    "recommendation": _recommendation(labels),
                }
            )
    rows.sort(key=lambda x: (-x["priority_score"], -x["spread_rate"], -x["parents"], x["question"]))
    return rows


def write_jsonl(rows: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_summary(rows: List[Dict], path: Path):
    label_counts: Dict[str, int] = {}
    tier_counts: Dict[str, int] = {}
    for r in rows:
        tier_counts[r["priority_tier"]] = tier_counts.get(r["priority_tier"], 0) + 1
        for lb in r["labels"]:
            label_counts[lb] = label_counts.get(lb, 0) + 1

    summary = {
        "total": len(rows),
        "tier_counts": dict(sorted(tier_counts.items())),
        "label_counts": dict(sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))),
        "top10": rows[:10],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Label shared subquestions and create a prioritized fix list.")
    parser.add_argument("--input_file", required=True, help="Input shared subquestions JSONL.")
    parser.add_argument(
        "--output_file",
        default="results/evaluation/shared_subquestions_priority.jsonl",
        help="Output prioritized JSONL.",
    )
    parser.add_argument(
        "--summary_file",
        default="results/evaluation/shared_subquestions_priority_summary.json",
        help="Output summary JSON.",
    )
    args = parser.parse_args()

    rows = load_shared(Path(args.input_file))
    write_jsonl(rows, Path(args.output_file))
    write_summary(rows, Path(args.summary_file))

    print(f"Loaded shared subquestions: {len(rows)}")
    print(f"Saved prioritized list to: {args.output_file}")
    print(f"Saved summary to: {args.summary_file}")


if __name__ == "__main__":
    main()
