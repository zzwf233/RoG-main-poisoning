#!/usr/bin/env python
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Tuple


PAPER_CLEAN = {
    "cwq": {"f1": 54.86, "hit": 64.94},
    "webqsp": {"f1": 70.34, "hit": 85.87},
}


def parse_eval_txt(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    txt = path.read_text(encoding="utf-8", errors="ignore")
    out = {}
    for key in ["Accuracy", "Hit", "F1", "Precision", "Recall"]:
        m = re.search(rf"{key}:\s*([0-9.]+)", txt)
        if m:
            out[key.lower()] = float(m.group(1))
    return out


def rule_sanity(path: Path) -> Tuple[int, int, float]:
    if not path.exists():
        return 0, 0, 0.0
    total = non_empty = rule_cnt = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                item = json.loads(line)
            except Exception:
                continue
            rules = item.get("rules", item.get("prediction", []))
            if not isinstance(rules, list):
                rules = [rules]
            rules = [r for r in rules if str(r).strip()]
            if rules:
                non_empty += 1
                rule_cnt += len(rules)
    avg = (rule_cnt / non_empty) if non_empty else 0.0
    return total, non_empty, avg


def pred_diag(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    total = empty_pred = multiline = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                item = json.loads(line)
            except Exception:
                continue
            p = str(item.get("prediction", "")).strip()
            if not p:
                empty_pred += 1
            if "\n" in p:
                multiline += 1
    return {
        "total": total,
        "empty_pred": empty_pred,
        "empty_ratio": (100 * empty_pred / total) if total else 0.0,
        "multiline": multiline,
    }


def model_sanity(model_path: str) -> Dict[str, bool]:
    p = Path(model_path)
    if not p.exists() or not p.is_dir():
        return {"exists": False, "full_weights": False, "adapter_only": False}
    has_full = any((p / n).exists() for n in [
        "pytorch_model.bin", "model.safetensors", "pytorch_model.bin.index.json", "model.safetensors.index.json"
    ])
    has_adapter = any((p / n).exists() for n in [
        "adapter_config.json", "adapter_model.bin", "adapter_model.safetensors"
    ])
    return {"exists": True, "full_weights": has_full, "adapter_only": (has_adapter and not has_full)}


def main():
    ap = argparse.ArgumentParser(description="Post-hoc diagnosis for Table-3 artifacts (no inference rerun).")
    ap.add_argument("--datasets", default="cwq,webqsp")
    ap.add_argument("--model_name", default="RoG")
    ap.add_argument("--rule_root", default="results/gen_rule_path")
    ap.add_argument("--pred_root", default="results/KGQA")
    ap.add_argument("--cwq_clean", default="datasets/clean_cwq.jsonl")
    ap.add_argument("--webqsp_clean", default="datasets/clean_webqsp.jsonl")
    args = ap.parse_args()

    datasets = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]
    clean_map = {"cwq": args.cwq_clean, "webqsp": args.webqsp_clean}

    for d in datasets:
        clean_file = Path(clean_map[d])
        clean_tag = clean_file.stem
        rule_file = Path(args.rule_root) / clean_tag / args.model_name / "test" / "predictions_3_False.jsonl"
        rule_postfix = str(rule_file).replace("/", "_").replace(".", "_")

        print(f"\n=== [{d}] artifact diagnosis ===")
        print(f"[path] clean_file={clean_file}")
        print(f"[path] rule_file={rule_file}")

        rt, rn, ravg = rule_sanity(rule_file)
        rcov = (100 * rn / rt) if rt else 0.0
        print(f"[rule] total={rt}, non_empty={rn} ({rcov:.2f}%), avg_rules={ravg:.2f}")

        for split in ["clean", "rand", "ours"]:
            pred = Path(args.pred_root) / f"{d}-{split}-paper" / args.model_name / "test" / rule_postfix / "predictions.jsonl"
            args_txt = pred.parent / "args.txt"
            eval_txt = pred.parent / "eval_result.txt"
            diag = pred_diag(pred)
            ev = parse_eval_txt(eval_txt)
            print(f"[{split}] pred={pred}")
            if diag:
                print(f"[{split}] total={diag['total']} empty={diag['empty_pred']} ({diag['empty_ratio']:.2f}%) multiline={diag['multiline']}")
            if ev:
                print(f"[{split}] Hit={ev.get('hit', -1):.2f} F1={ev.get('f1', -1):.2f} Precision={ev.get('precision', -1):.2f} Recall={ev.get('recall', -1):.2f}")
                if split == "clean" and d in PAPER_CLEAN:
                    print(f"[{split}] gap_vs_paper: ΔHit={ev.get('hit', 0)-PAPER_CLEAN[d]['hit']:.2f}, ΔF1={ev.get('f1', 0)-PAPER_CLEAN[d]['f1']:.2f}")
            if args_txt.exists():
                cfg = json.loads(args_txt.read_text(encoding="utf-8"))
                mp = cfg.get("model_path", "")
                ms = model_sanity(mp)
                print(f"[{split}] model_path={mp}")
                print(f"[{split}] model_exists={ms['exists']} full_weights={ms['full_weights']} adapter_only={ms['adapter_only']}")


if __name__ == "__main__":
    main()

