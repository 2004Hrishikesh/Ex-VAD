"""Render visual summaries of demo_cmaf.py results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_path: str = "outputs/demo_results.json") -> Dict:
    """Load JSON results produced by demo_cmaf.py."""
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found at {path.resolve()}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_confusion(results: Iterable[Dict]) -> Tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) counts from demo outputs."""
    tp = sum(1 for r in results if r["predicted_label"] == 1 and r["expected_label"] == 1)
    fp = sum(1 for r in results if r["predicted_label"] == 1 and r["expected_label"] == 0)
    fn = sum(1 for r in results if r["predicted_label"] == 0 and r["expected_label"] == 1)
    tn = sum(1 for r in results if r["predicted_label"] == 0 and r["expected_label"] == 0)
    return tp, fp, fn, tn


def accuracy_by_group(results: Iterable[Dict], key: str) -> Dict[str, float]:
    """Compute accuracy percentages grouped by a given key (category, expected_type, etc.)."""
    groups: Dict[str, list] = {}
    for record in results:
        groups.setdefault(record[key], []).append(record["is_correct"])
    return {
        group: np.mean(flags) * 100
        for group, flags in groups.items()
        if flags
    }


def render_figures(results: Iterable[Dict], threshold: float,
                   output_dir: str = "outputs/visualizations") -> None:
    """Render performance plots to a summary figure."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tp, fp, fn, tn = compute_confusion(results)
    conf_matrix = np.array([[tn, fp], [fn, tp]])

    per_category = accuracy_by_group(results, "category")
    per_type = accuracy_by_group(results, "expected_type")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Category accuracy bar chart
    axes[0, 0].barh(list(per_category.keys()), list(per_category.values()), color="steelblue")
    axes[0, 0].set_title("Accuracy by Category")
    axes[0, 0].set_xlabel("Accuracy (%)")
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(axis="x", alpha=0.3)

    # Normal vs anomaly accuracy
    types = list(per_type.keys())
    values = list(per_type.values())
    bars = axes[0, 1].bar(types, values, color=["seagreen", "coral"])
    axes[0, 1].set_title("Normal vs Anomaly Accuracy")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, val + 2,
                        f"{val:.1f}%", ha="center", va="bottom")

    # Confusion matrix heatmap
    im = axes[1, 0].imshow(conf_matrix, cmap="Blues")
    axes[1, 0].set_title("Confusion Matrix")
    axes[1, 0].set_xticks([0, 1], labels=["Normal", "Anomaly"])
    axes[1, 0].set_yticks([0, 1], labels=["Normal", "Anomaly"])
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, int(conf_matrix[i, j]), ha="center", va="center",
                           fontsize=16, fontweight="bold")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Score distribution scatter
    scores = [r["anomaly_score"] for r in results]
    expected = [r["expected_label"] for r in results]
    colors = ["crimson" if label == 1 else "forestgreen" for label in expected]
    axes[1, 1].scatter(range(len(scores)), scores, c=colors, alpha=0.75)
    axes[1, 1].axhline(y=threshold, color="navy", linestyle="--",
                      label=f"Threshold = {threshold}")
    axes[1, 1].set_title("Anomaly Scores per Sample")
    axes[1, 1].set_xlabel("Sample Index")
    axes[1, 1].set_ylabel("Anomaly Score")
    axes[1, 1].legend(loc="lower right")

    plt.tight_layout()
    summary_path = out_dir / "performance_summary.png"
    fig.savefig(summary_path, dpi=300)
    plt.close(fig)

    print(f"Saved summary figure to {summary_path}")


if __name__ == "__main__":
    payload = load_results("outputs/demo_results.json")
    render_figures(payload["results"], threshold=payload["threshold"])
