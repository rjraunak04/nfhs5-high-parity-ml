"""Generate exact README charts from the versioned manuscript metric file."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    metrics = json.loads(Path("research/manuscript_metrics.json").read_text(encoding="utf-8"))
    names = ["Nested CV\nIndia", "India\nholdout", "Harmonised\nIndia", "Nepal\nvalidation"]
    auc_values = [
        metrics["india_primary"]["nested_cv_auc_mean"],
        metrics["india_primary"]["holdout_auc"],
        metrics["transport"]["india_holdout_auc"],
        metrics["transport"]["nepal_auc"],
    ]
    colors = ["#25557D", "#2E75B6", "#2F8F5B", "#ED7D1A"]
    figure, axis = plt.subplots(figsize=(8.8, 4.8))
    bars = axis.bar(names, auc_values, color=colors, width=0.68)
    axis.set_ylim(0.75, 0.91)
    axis.set_ylabel("AUC ROC")
    axis.set_title("Leakage-aware internal and geographic validation")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, auc_values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.4f}", ha="center")
    figure.text(
        0.5,
        0.01,
        "Manuscript-reported results; public demo uses a separate synthetic model.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    output = Path("docs/assets/performance_summary.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
