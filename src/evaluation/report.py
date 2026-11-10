"""Evaluation report from saved per-video scores.

Recomputes every metric in the README from the score files in results/, so the
numbers can be checked without the dataset or a GPU.

Usage:
    python -m src.evaluation.report results/cmaf_per_video_scores_test290.json --out results/report
    python -m src.evaluation.report temporal_extension/results/per_video_scores_test290.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Display names for the score keys used in the result files
MODEL_NAMES = {
    "exvad": "ExVAD baseline",
    "cmaf": "ExVAD + CMAF",
    "v5": "ExVAD + CMAF + Temporal",
    "v5_shuffled": "Temporal, frames shuffled",
}


def load_scores(path):
    """Read a per-video score file and return labels, categories and {model: (scores, threshold)}."""
    data = json.loads(Path(path).read_text())
    samples = data["samples"]
    labels = np.array([s["label"] for s in samples])
    categories = [s["category"] for s in samples]

    if "thresholds" in data:  # results/cmaf_per_video_scores_test290.json
        thresholds = data["thresholds"]
        keys = {"exvad": "score_baseline", "cmaf": "score_cmaf"}
        models = {m: (np.array([s[k] for s in samples]), thresholds[m]) for m, k in keys.items()}
    else:  # temporal_extension/results/per_video_scores_test290.json
        thresholds = data["config"]["thresholds"]
        models = {
            m: (np.array([s["scores"][m] for s in samples]), thresholds.get(m, thresholds.get("v5")))
            for m in samples[0]["scores"]
        }
    return labels, categories, models


def compute_metrics(labels, scores, threshold):
    """Threshold-based and ranking metrics for one model."""
    preds = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": round(float((tp + tn) / len(labels)), 4),
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, preds, zero_division=0)), 4),
        "auc_roc": round(float(roc_auc_score(labels, scores)), 4),
        "auc_pr": round(float(average_precision_score(labels, scores)), 4),
        "false_alarm_rate": round(float(fp / (fp + tn)), 4),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def per_category_correct(labels, categories, scores, threshold):
    """Number of correctly classified videos per category."""
    preds = (scores > threshold).astype(int)
    totals, correct = defaultdict(int), defaultdict(int)
    for label, cat, pred in zip(labels, categories, preds):
        totals[cat] += 1
        correct[cat] += int(label == pred)
    return {cat: (correct[cat], totals[cat]) for cat in sorted(totals)}


def to_markdown(results):
    cols = ["accuracy", "precision", "recall", "f1", "auc_roc", "fp", "fn"]
    lines = ["| Model | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for model, m in results.items():
        lines.append(f"| {MODEL_NAMES.get(model, model)} | " + " | ".join(str(m[c]) for c in cols) + " |")
    return "\n".join(lines)


def plot_roc(labels, models, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    for model, (scores, _) in models.items():
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr, tpr, label=f"{MODEL_NAMES.get(model, model)} (AUC {roc_auc_score(labels, scores):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC on UCF-Crime test split")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_report(score_file, out_dir=None):
    labels, categories, models = load_scores(score_file)
    results = {m: compute_metrics(labels, s, t) for m, (s, t) in models.items()}
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(results, indent=2))
        (out / "metrics.md").write_text(to_markdown(results) + "\n")
        plot_roc(labels, models, out / "roc.png")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("score_file")
    parser.add_argument("--out", default=None, help="Folder for metrics.json, metrics.md and roc.png")
    args = parser.parse_args()
    print(to_markdown(build_report(args.score_file, args.out)))


if __name__ == "__main__":
    main()
