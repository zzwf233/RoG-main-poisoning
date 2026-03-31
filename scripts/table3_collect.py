#!/usr/bin/env python
import argparse
import csv
import os
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

def eval_acc(pred: str, answers: List[str]) -> float:
    if not answers:
        return 0.0
    matched = sum(1 for a in answers if match(pred, a))
    return matched / len(answers)

def parse_ranked_answers(prediction) -> List[str]:
    if isinstance(prediction, list):
        # Keep consistent with evaluate_results.py: deduplicate by frequency.
        freq = {}
        for p in prediction:
            p = str(p).strip()
            if not p:
                continue
            freq[p] = freq.get(p, 0) + 1
        ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [x[0] for x in ranked]

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
    acc_sum = 0.0
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
            acc = eval_acc(pred_join, answers)
            # Unify with paper-oriented comparison: use top-1 hit as the main "Hit".
            hit = 1.0 if any(match(preds[0], a) for a in answers) else 0.0
            f1, p, r = eval_f1(preds, answers)

            top1 = preds[0]
            h1 = 1.0 if any(match(top1, a) for a in answers) else 0.0
            top1_norm = normalize(top1)
            em = 1.0 if any(top1_norm == normalize(a) for a in answers) else 0.0

            acc_sum += acc
            hit_sum += hit
            f1_sum += f1
            p_sum += p
            r_sum += r
            h1_sum += h1
            em_sum += em

    if total == 0:
        return {"total": 0, "accuracy": 0, "hit": 0, "f1": 0, "precision": 0, "recall": 0, "hits1": 0, "em": 0}

    return {
        "total": total,
        "accuracy": acc_sum * 100 / total,
        "hit": hit_sum * 100 / total,
        "f1": f1_sum * 100 / total,
        "precision": p_sum * 100 / total,
        "recall": r_sum * 100 / total,
        "hits1": h1_sum * 100 / total,
        "em": em_sum * 100 / total,
    }

def eval_from_detailed_file(path: str) -> Dict[str, float]:
    """
    Reuse metrics already computed by qa_prediction/evaluate_results.py
    for Hits@1/F1/Precision/Recall, then combine with Hits@1/EM from raw predictions.
    """
    detailed_path = path.replace("predictions.jsonl", "detailed_eval_result.jsonl")
    if not os.path.exists(detailed_path):
        return {}

    total = 0
    acc_sum = 0.0
    hit_sum = 0.0
    f1_sum = 0.0
    p_sum = 0.0
    r_sum = 0.0

    with open(detailed_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            total += 1
            acc_sum += float(item.get("acc", 0.0))
            hit_sum += float(item.get("hits1", item.get("hit", 0.0)))
            f1_sum += float(item.get("f1", 0.0))
            p_sum += float(item.get("precission", item.get("precision", 0.0)))
            r_sum += float(item.get("recall", 0.0))

    if total == 0:
        return {}

    # Keep Hits@1 and EM from current parser on raw prediction file.
    base = eval_file(path)
    return {
        "total": total,
        "accuracy": acc_sum * 100 / total,
        "hit": hit_sum * 100 / total,
        "f1": f1_sum * 100 / total,
        "precision": p_sum * 100 / total,
        "recall": r_sum * 100 / total,
        "hits1": base["hits1"],
        "em": base["em"],
    }


def main():
    ap = argparse.ArgumentParser(description="Collect Clean/Rand/Ours metrics into a Table-3 style CSV")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", default="RoG")
    ap.add_argument("--clean_pred", required=True)
    ap.add_argument("--rand_pred", required=True)
    ap.add_argument("--ours_pred", required=True)
    ap.add_argument("--output_csv", default="results/evaluation/table3_rows.csv")
    ap.add_argument("--prefer_detailed_eval", action="store_true",
                    help="If detailed_eval_result.jsonl exists, reuse its Hit/F1/Precision/Recall.")
    args = ap.parse_args()

    rows = []
    for attacker, path in [("Clean", args.clean_pred), ("Rand", args.rand_pred), ("Ours", args.ours_pred)]:
        m = eval_from_detailed_file(path) if args.prefer_detailed_eval else {}
        if not m:
            m = eval_file(path)
        rows.append({
            "Dataset": args.dataset,
            "KG-RAG": args.method,
            "Attacker": attacker,
            "#Eval": m["total"],
            "Accuracy": round(m["accuracy"], 2),
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
