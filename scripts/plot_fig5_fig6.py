#!/usr/bin/env python3
"""Draw paper-style Figure 5 and Figure 6 from fixed result tables."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("results/figures")


FIG5 = {
    "RoG": {
        "WebQSP": {"A-RR": 100.00, "A-GR": 79.87, "A-Precision\u2020": 98.54},
        "CWQ": {"A-RR": 100.00, "A-GR": 69.24, "A-Precision\u2020": 93.56},
    },
    "SubgraphRAG": {
        "WebQSP": {"A-RR": 100.00, "A-GR": 43.27, "A-Precision\u2020": 78.91},
        "CWQ": {"A-RR": 99.80, "A-GR": 35.49, "A-Precision\u2020": 91.99},
    },
}


FIG6 = {
    "RoG": {
        "kgqa": {
            "clean": {"F1": 70.45, "Hits@1": 75.79, "EM": 52.38},
            "k=2": {"F1": 34.25, "Hits@1": 48.10, "EM": 13.27},
            "k=4": {"F1": 16.94, "Hits@1": 22.31, "EM": 8.13},
            "k=6": {"F1": 15.74, "Hits@1": 29.24, "EM": 5.47},
        },
        "attack": {
            "k=2": {"A-Precision": 56.92, "A-H@1": 57.56, "A-MRR": 67.11},
            "k=4": {"A-Precision": 69.91, "A-H@1": 73.80, "A-MRR": 77.54},
            "k=6": {"A-Precision": 75.90, "A-H@1": 79.55, "A-MRR": 82.04},
        },
    },
    "SubgraphRAG": {
        "kgqa": {
            "clean": {"F1": 72.21, "Hits@1": 82.98, "EM": 46.93},
            "k=2": {"F1": 46.7995, "Hits@1": 59.8033, "EM": 24.2163},
            "k=4": {"F1": 47.1462, "Hits@1": 60.1106, "EM": 24.3393},
            "k=6": {"F1": 47.1380, "Hits@1": 60.1106, "EM": 24.3393},
        },
        "attack": {
            "k=2": {"A-Precision": 34.7154, "A-H@1": 37.3239, "A-MRR": 44.6567},
            "k=4": {"A-Precision": 36.0827, "A-H@1": 38.0282, "A-MRR": 45.8392},
            "k=6": {"A-Precision": 36.7752, "A-H@1": 39.0845, "A-MRR": 46.9689},
        },
    },
}


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#555555",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_figure5(out_dir: Path):
    metrics = ["A-Precision\u2020", "A-GR", "A-RR"]
    datasets = ["CWQ", "WebQSP"]
    y = np.arange(len(metrics))
    offsets = {"CWQ": 0.18, "WebQSP": -0.18}
    height = 0.30

    left_colors = {"CWQ": "#d9c7ec", "WebQSP": "#9f86c0"}
    right_colors = {"CWQ": "#cfe8d6", "WebQSP": "#6fb3a8"}

    fig, ax = plt.subplots(figsize=(6.4, 3.2))

    for dataset in datasets:
        ys = y + offsets[dataset]
        rog_vals = [FIG5["RoG"][dataset][m] for m in metrics]
        sub_vals = [FIG5["SubgraphRAG"][dataset][m] for m in metrics]

        ax.barh(
            ys,
            [-v for v in rog_vals],
            height=height,
            color=left_colors[dataset],
            edgecolor="#7d7190",
            linewidth=0.45,
            alpha=0.55,
            label=dataset if dataset == "CWQ" else None,
        )
        ax.barh(
            ys,
            sub_vals,
            height=height,
            color=right_colors[dataset],
            edgecolor="#5c8d87",
            linewidth=0.45,
            alpha=0.65,
            label=dataset if dataset == "WebQSP" else None,
        )

        for yy, value in zip(ys, rog_vals):
            ax.text(
                -value - 2.0,
                yy,
                f"{value:.2f}",
                ha="right",
                va="center",
                fontsize=7,
                clip_on=False,
            )
        for yy, value in zip(ys, sub_vals):
            ax.text(
                value + 2.0,
                yy,
                f"{value:.2f}",
                ha="left",
                va="center",
                fontsize=7,
                clip_on=False,
            )

    ax.axvline(0, color="#777777", linewidth=0.7)
    ax.set_xlim(-118, 118)
    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=8)
    ax.invert_yaxis()
    ticks = np.arange(-100, 101, 20)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(abs(t)) for t in ticks], fontsize=7)
    ax.grid(axis="x", linestyle=":", linewidth=0.45, alpha=0.45)
    ax.set_axisbelow(True)

    ax.text(0.25, -0.11, "RoG (%)", transform=ax.transAxes, ha="center", va="top", fontsize=8)
    ax.text(0.75, -0.11, "SubgraphRAG (%)", transform=ax.transAxes, ha="center", va="top", fontsize=8)

    left_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=left_colors[d], edgecolor="#7d7190", alpha=0.55)
        for d in datasets
    ]
    right_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=right_colors[d], edgecolor="#5c8d87", alpha=0.65)
        for d in datasets
    ]
    leg1 = ax.legend(
        left_handles,
        datasets,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        frameon=False,
        fontsize=7,
        borderaxespad=0.0,
    )
    ax.add_artist(leg1)
    ax.legend(
        right_handles,
        datasets,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        frameon=False,
        fontsize=7,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.16, right=0.94, top=0.84, bottom=0.18)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"figure5_attack_stages.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_grouped_bars(ax, rows, metrics, scales, colors, ylim, legend_loc):
    x = np.arange(len(metrics))
    width = 0.68 / len(scales)

    for i, scale in enumerate(scales):
        vals = [rows[scale][m] for m in metrics]
        offset = (i - (len(scales) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=scale,
            color=colors[scale],
            edgecolor="#888888",
            linewidth=0.35,
            alpha=0.88,
        )
        for bar, val in zip(bars, vals):
            label_offset = (ylim[1] - ylim[0]) * 0.012
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + label_offset,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=5.0,
                clip_on=False,
            )

    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylabel("Value(%)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.15, linewidth=0.45)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.0, loc=legend_loc, frameon=True, edgecolor="#dddddd")


def plot_figure6(out_dir: Path):
    kgqa_metrics = ["F1", "Hits@1", "EM"]
    attack_metrics = ["A-Precision", "A-H@1", "A-MRR"]
    scale_colors = {
        "clean": "#d9d9d9",
        "k=2": "#6baed6",
        "k=4": "#f4a261",
        "k=6": "#7a6f9b",
    }

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.05))
    panels = [
        (
            "RoG",
            "kgqa",
            kgqa_metrics,
            ["clean", "k=2", "k=4", "k=6"],
            scale_colors,
            (0, 96),
            "(a) KGQA performances of RoG",
            "upper right",
        ),
        (
            "RoG",
            "attack",
            attack_metrics,
            ["k=2", "k=4", "k=6"],
            scale_colors,
            (45, 91),
            "(b) Adversarial Results of RoG",
            "upper left",
        ),
        (
            "SubgraphRAG",
            "kgqa",
            kgqa_metrics,
            ["clean", "k=2", "k=4", "k=6"],
            scale_colors,
            (20, 98),
            "(c) KGQA performances of GCR",
            "upper right",
        ),
        (
            "SubgraphRAG",
            "attack",
            attack_metrics,
            ["k=2", "k=4", "k=6"],
            scale_colors,
            (30, 53),
            "(d) Adversarial Results of GCR",
            "upper left",
        ),
    ]

    for ax, (method, section, metrics, scales, colors, ylim, title, legend_loc) in zip(axes, panels):
        draw_grouped_bars(ax, FIG6[method][section], metrics, scales, colors, ylim, legend_loc)

    fig.subplots_adjust(left=0.055, right=0.995, top=0.83, bottom=0.30, wspace=0.28)
    for ax, (*_, title, legend_loc) in zip(axes, panels):
        pos = ax.get_position()
        fig.text((pos.x0 + pos.x1) / 2, 0.11, title, ha="center", fontsize=8.2)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"figure6_poison_scale.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_figure5(OUT_DIR)
    plot_figure6(OUT_DIR)
    print(f"Saved figures to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
