#!/usr/bin/env python3
"""Plot Figure-6 style poisoning-scale results.

Expected CSV columns:
  method,scale,metric,value

Example rows:
  RoG,clean,F1,70.3
  RoG,k=2,F1,43.1
  RoG,k=2,A-Precision,47.5

Values are percentages. Metrics F1/Hits@1/EM are drawn in the KGQA panel;
A-Precision/A-H@1/A-MRR are drawn in the adversarial panel.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


KGQA_METRICS = ["F1", "Hits@1", "EM"]
ATTACK_METRICS = ["A-Precision", "A-H@1", "A-MRR"]
DEFAULT_COLORS = {
    "clean": "#f6d6d6",
    "k=2": "#e7b1bd",
    "k=4": "#c996c5",
    "k=6": "#a985c5",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Input CSV with method,scale,metric,value columns.")
    p.add_argument("--out", default="results/figures/poison_scale.png", help="Output image path.")
    p.add_argument("--methods", default="", help="Comma-separated method order, e.g. RoG,GCR.")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def load_rows(path):
    data = defaultdict(dict)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"method", "scale", "metric", "value"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            method = row["method"].strip()
            scale = row["scale"].strip()
            metric = row["metric"].strip()
            if not method or not scale or not metric:
                continue
            data[method][(scale, metric)] = float(row["value"])
    return data


def sorted_scales(method_data, metrics):
    scales = []
    for scale, metric in method_data:
        if metric in metrics and scale not in scales:
            scales.append(scale)

    def key(s):
        if s.lower() == "clean":
            return -1
        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else 10**9

    return sorted(scales, key=key)


def draw_panel(ax, method_data, metrics, scales, title, colors):
    import numpy as np

    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(scales))

    for i, scale in enumerate(scales):
        vals = [method_data.get((scale, m), np.nan) for m in metrics]
        offset = (i - (len(scales) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=scale,
            color=colors.get(scale, None),
            edgecolor="#777777",
            linewidth=0.35,
        )
        for bar, val in zip(bars, vals):
            if np.isnan(val):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.6,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_title(title, fontsize=11, y=-0.28)
    ax.set_xlabel("Metrics", fontsize=9)
    ax.set_ylabel("Value(%)", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    ax.legend(fontsize=7, frameon=True, loc="upper right")


def main():
    args = parse_args()
    data = load_rows(args.csv)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] or list(data.keys())

    import matplotlib.pyplot as plt

    ncols = len(methods) * 2
    fig, axes = plt.subplots(1, ncols, figsize=(3.1 * ncols, 2.8), squeeze=False)
    panel_letters = "abcdefghijklmnopqrstuvwxyz"

    for mi, method in enumerate(methods):
        if method not in data:
            raise ValueError(f"Method {method!r} not found in CSV.")
        method_data = data[method]
        kgqa_scales = sorted_scales(method_data, KGQA_METRICS)
        attack_scales = [s for s in sorted_scales(method_data, ATTACK_METRICS) if s.lower() != "clean"]

        draw_panel(
            axes[0][mi * 2],
            method_data,
            KGQA_METRICS,
            kgqa_scales,
            f"({panel_letters[mi * 2]}) KGQA performances of {method}",
            DEFAULT_COLORS,
        )
        draw_panel(
            axes[0][mi * 2 + 1],
            method_data,
            ATTACK_METRICS,
            attack_scales,
            f"({panel_letters[mi * 2 + 1]}) Adversarial Results of {method}",
            DEFAULT_COLORS,
        )

    fig.suptitle("Attack Effectiveness across increasing poisoning scales.", fontsize=12, y=0.02)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure to: {out}")


if __name__ == "__main__":
    main()
