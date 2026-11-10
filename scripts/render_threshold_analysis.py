"""Generate threshold analysis visualizations from demo results."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_path: str = "outputs/demo_results.json") -> Dict:
    """Load JSON results produced by demo_cmaf.py."""
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found at {path.resolve()}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_threshold(results: Iterable[Dict], threshold: float) -> Dict[str, float]:
    """Compute metrics for a single decision threshold."""
    tp = fp = fn = tn = 0
    for record in results:
        score = float(record["anomaly_score"])
        expected = int(record["expected_label"])
        predicted = 1 if score > threshold else 0

        if predicted == 1 and expected == 1:
            tp += 1
        elif predicted == 1 and expected == 0:
            fp += 1
        elif predicted == 0 and expected == 1:
            fn += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    normal_total = tn + fp
    anomaly_total = tp + fn
    normal_accuracy = tn / normal_total if normal_total > 0 else 0.0
    anomaly_accuracy = tp / anomaly_total if anomaly_total > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "normal_accuracy": normal_accuracy,
        "anomaly_accuracy": anomaly_accuracy,
        "false_positive_rate": false_positive_rate,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def generate_threshold_grid(start: float = 0.0, end: float = 1.0, step: float = 0.02) -> np.ndarray:
    """Return an array of thresholds spanning the decision range."""
    thresholds = np.arange(start, end + step, step)
    return np.clip(thresholds, 0.0, 1.0)


def save_metrics_csv(metrics: List[Dict[str, float]], output_path: Path) -> None:
    """Persist threshold metrics to CSV for reporting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not metrics:
        raise ValueError("No metrics computed; ensure results are available.")

    fieldnames = list(metrics[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def plot_threshold_curves(metrics: List[Dict[str, float]], output_path: Path) -> None:
    """Render threshold performance curves to an image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    thresholds = [m["threshold"] for m in metrics]
    accuracy = [m["accuracy"] * 100 for m in metrics]
    precision = [m["precision"] * 100 for m in metrics]
    recall = [m["recall"] * 100 for m in metrics]
    f1 = [m["f1"] * 100 for m in metrics]
    normal_acc = [m["normal_accuracy"] * 100 for m in metrics]
    anomaly_acc = [m["anomaly_accuracy"] * 100 for m in metrics]
    fpr = [m["false_positive_rate"] * 100 for m in metrics]

    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(thresholds, accuracy, label="Overall Accuracy", color="navy", linewidth=2)
    axes[0].plot(thresholds, precision, label="Precision", color="seagreen", linestyle="--", linewidth=2)
    axes[0].plot(thresholds, recall, label="Recall", color="orangered", linestyle="-.", linewidth=2)
    axes[0].plot(thresholds, f1, label="F1 Score", color="purple", linestyle=":", linewidth=2)
    axes[0].set_ylabel("Metric (%)")
    axes[0].set_title("Global Metrics vs. Decision Threshold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(thresholds, anomaly_acc, label="Anomaly Accuracy", color="crimson", linewidth=2)
    axes[1].plot(thresholds, normal_acc, label="Normal Accuracy", color="forestgreen", linestyle="--", linewidth=2)
    axes[1].plot(thresholds, fpr, label="False Positive Rate", color="goldenrod", linestyle=":", linewidth=2)
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Metric (%)")
    axes[1].set_title("Class-Specific Metrics vs. Decision Threshold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    payload = load_results("outputs/demo_results.json")
    results = payload.get("results", [])
    if not results:
        raise ValueError("No per-sample results found in outputs/demo_results.json.")

    thresholds = generate_threshold_grid(0.0, 1.0, 0.01)
    metrics = [evaluate_threshold(results, t) for t in thresholds]

    vis_dir = Path("outputs/visualizations")
    csv_path = Path("outputs/threshold_analysis.csv")

    save_metrics_csv(metrics, csv_path)
    plot_threshold_curves(metrics, vis_dir / "threshold_analysis.png")

    print(f"Saved threshold metrics to {csv_path}")
    print(f"Saved threshold plot to {vis_dir / 'threshold_analysis.png'}")


if __name__ == "__main__":
    main()
