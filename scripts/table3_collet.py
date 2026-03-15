#!/usr/bin/env python
import argparse
import csv
import json
import re
import string
from pathlib import Path
from typing import Dict, List


def normalize(s: str) -> str:
    s = (s or "").lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    return " ".join(s.split())


def match(pred: str, ans: str) -> bool:
    return normalize(ans) in normalize(pred)


def parse_ranked_answers(prediction) -> List[str]:
    if isinstance(prediction, list):
        return [str(x).strip() for x in prediction if str(x).strip()]

    txt = str(prediction or "").strip()
    if not txt:
        return []

    if txt.startswith("[") and txt.endswith("]"):
        content = txt[1:-1]
        items = [x.strip().strip("'\"") for x in content.split(",")]
        return [x for x in items if x]

    lines = [x.strip() for x in txt.split("\n") if x.strip()]
    if len(lines) > 1:
        cleaned = [re.sub(r"^\d+\.\s*", "", x).strip() for x in lines]
        return [x for x in cleaned if x]

    if "," in txt:
        return [x.strip() for x in txt.split(",") if x.strip()]

    return [txt]


def eval_f1(pred_list: List[str], answers: List[str]):
    if not pred_list:
        return 0.0, 0.0, 0.0
    pred_str = " ".join(pred_list)
    matched = sum(1 for a in answers if match(pred_str, a))
    precision = matched / len(pred_list)
    recall = matched / len(answers) if answers else 0.0
    if precision + recall == 0:
        return 0.0, precision, recall
    return 2 * precision * recall / (precision + recall), precision, recall


def eval_file(path: str) -> Dict[str, float]:
    total = 0
    hit_sum = 0.0
    f1_sum = 0.0
    p_sum = 0.0
    r_sum = 0.0
    h1_sum = 0.0
    em_sum = 0.0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            answers = item.get("ground_truth", [])
            if isinstance(answers, str):
                answers = [answers]
            answers = [str(a).strip() for a in answers if str(a).strip()]
            if not answers:
                continue

            preds = parse_ranked_answers(item.get("prediction", ""))
            if not preds:
                continue

            total += 1
            pred_join = " ".join(preds)
            hit = 1.0 if any(match(pred_join, a) for a in answers) else 0.0
            f1, p, r = eval_f1(preds, answers)

            top1 = preds[0]
            h1 = 1.0 if any(match(top1, a) for a in answers) else 0.0
            top1_norm = normalize(top1)
            em = 1.0 if any(top1_norm == normalize(a) for a in answers) else 0.0

            hit_sum += hit
            f1_sum += f1
            p_sum += p
            r_sum += r
            h1_sum += h1
            em_sum += em

    if total == 0:
        return {"total": 0, "hit": 0, "f1": 0, "precision": 0, "recall": 0, "hits1": 0, "em": 0}

    return {
        "total": total,
        "hit": hit_sum * 100 / total,
        "f1": f1_sum * 100 / total,
        "precision": p_sum * 100 / total,
        "recall": r_sum * 100 / total,
        "hits1": h1_sum * 100 / total,
        "em": em_sum * 100 / total,
    }


def main():
    ap = argparse.ArgumentParser(description="Collect Clean/Rand/Ours metrics into a Table-3 style CSV")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", default="RoG")
    ap.add_argument("--clean_pred", required=True)
    ap.add_argument("--rand_pred", required=True)
    ap.add_argument("--ours_pred", required=True)
    ap.add_argument("--output_csv", default="results/evaluation/table3_rows.csv")
    args = ap.parse_args()

    rows = []
    for attacker, path in [("Clean", args.clean_pred), ("Rand", args.rand_pred), ("Ours", args.ours_pred)]:
        m = eval_file(path)
        rows.append({
            "Dataset": args.dataset,
            "KG-RAG": args.method,
            "Attacker": attacker,
            "#Eval": m["total"],
            "Hit": round(m["hit"], 2),
            "F1": round(m["f1"], 2),
            "Precision": round(m["precision"], 2),
            "Recall": round(m["recall"], 2),
            "Hits@1": round(m["hits1"], 2),
            "EM": round(m["em"], 2),
        })

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    with open(out_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Saved rows to: {out_path}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()

