"""
Erzeugt aus results/master_results.csv Heatmaps (Modell x Perturbation)
für die drei zentralen Kennzahlen - direkt verwendbar für Kap. 6
(Ergebnisse und Diskussion).

Aufruf:
    python analyze_results.py --dataset kinetics400_subset
    python analyze_results.py --dataset ssv2_subset
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config


METRICS_TO_PLOT = {
    "class_mAP": "Cross-Condition Retrieval (mAP)",
    "identity_recall@1": "Identity-Retrieval (Recall@1)",
    "shift_mean_cosine_shift": "Embedding-Shift (mean Cosine-Distanz)",
}


def plot_heatmaps(dataset_name: str):
    df = pd.read_csv(config.RESULTS_DIR / "master_results.csv")
    df = df[(df["dataset"] == dataset_name) & (df["perturbation"] != "clean")]

    fig, axes = plt.subplots(1, len(METRICS_TO_PLOT), figsize=(6 * len(METRICS_TO_PLOT), 5))

    for ax, (col, title) in zip(axes, METRICS_TO_PLOT.items()):
        if col not in df.columns:
            ax.set_title(f"{title}\n(Spalte fehlt: {col})")
            continue
        pivot = df.pivot(index="model", columns="perturbation", values=col)
        # Perturbationen in fixer, sinnvoller Reihenfolge (schwach -> stark)
        order = [p for p in config.PERTURBATION_NAMES if p in pivot.columns]
        pivot = pivot[order]

        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar=True)
        ax.set_title(title)
        ax.set_xlabel("Perturbation")
        ax.set_ylabel("Modell")

    fig.suptitle(f"Dataset: {dataset_name}")
    fig.tight_layout()
    out_path = config.RESULTS_DIR / f"heatmap_{dataset_name}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Gespeichert: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(config.DATASETS.keys()))
    args = parser.parse_args()
    plot_heatmaps(args.dataset)
