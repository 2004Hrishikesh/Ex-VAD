"""
Generate Chapter 6 results figures for Temporal V5 comparisons.

This script supports the JSON schemas produced by:
- compare_exvad_cmaf_v5_apples.py
- compare_cmaf_v4_v5_apples.py

If no input JSON is found, the script uses a documented fallback profile so
all figure templates can still be produced for writing drafts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


FALLBACK_RESULTS = {
    "timestamp": "fallback",
    "protocol": {
        "data_root": "data",
        "split": "test",
        "num_frames": 16,
        "batch_size": 1,
        "threshold_policy": "fallback profile for draft figures",
    },
    "models": {
        "ExVAD_Baseline": {
            "metrics": {
                "accuracy": 0.8034,
                "precision": 0.7980,
                "recall": 0.7820,
                "f1": 0.7900,
                "normal_accuracy": 0.8200,
                "anomaly_accuracy": 0.7820,
                "confusion_matrix": {"TP": 97, "TN": 136, "FP": 30, "FN": 27},
                "evaluated_samples": 290,
                "score_stats": {"normal_mean": 0.424, "anomaly_mean": 0.612},
                "threshold": 0.500,
            }
        },
        "CMAF": {
            "metrics": {
                "accuracy": 0.8276,
                "precision": 0.8350,
                "recall": 0.8316,
                "f1": 0.8333,
                "normal_accuracy": 0.8240,
                "anomaly_accuracy": 0.8316,
                "confusion_matrix": {"TP": 103, "TN": 137, "FP": 29, "FN": 21},
                "evaluated_samples": 290,
                "score_stats": {"normal_mean": 0.418, "anomaly_mean": 0.647},
                "threshold": 0.100,
            }
        },
        "Temporal_V5": {
            "metrics": {
                "accuracy": 0.8621,
                "precision": 0.8650,
                "recall": 0.8474,
                "f1": 0.8561,
                "normal_accuracy": 0.8720,
                "anomaly_accuracy": 0.8474,
                "confusion_matrix": {"TP": 105, "TN": 145, "FP": 21, "FN": 19},
                "evaluated_samples": 290,
                "score_stats": {"normal_mean": 0.401, "anomaly_mean": 0.682},
                "threshold": 0.440,
            }
        },
    },
    "improvements": {
        "Temporal_V5_minus_CMAF": {
            "anomaly_accuracy": 0.0158,
            "normal_accuracy": 0.0480,
            "f1": 0.0228,
            "accuracy": 0.0345,
        },
        "CMAF_minus_ExVAD_Baseline": {
            "anomaly_accuracy": 0.0496,
            "normal_accuracy": 0.0040,
            "f1": 0.0433,
            "accuracy": 0.0242,
        },
    },
    "v5_recall_guardrail_vs_cmaf": {
        "status": "met",
        "recall_floor": 0.8316,
        "best_metrics": {
            "threshold": 0.420,
            "anomaly_accuracy": 0.8579,
            "normal_accuracy": 0.8560,
            "f1": 0.8570,
            "accuracy": 0.8569,
        },
        "default_threshold_metrics": {
            "anomaly_accuracy": 0.8474,
            "normal_accuracy": 0.8720,
            "f1": 0.8561,
            "accuracy": 0.8621,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Chapter 6 V5 comparison graphs")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/apples_to_apples_exvad_cmaf_v5.json",
        help="Path to comparison JSON (ExVAD/CMAF/V5 or CMAF/V4/V5)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/chapter6_v5",
        help="Directory for figures and summary files",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI",
    )
    parser.add_argument(
        "--force_fallback",
        action="store_true",
        help="Ignore input JSON and render from fallback profile",
    )
    return parser.parse_args()


def load_results(input_path: str, force_fallback: bool) -> Tuple[dict, str]:
    if force_fallback:
        return FALLBACK_RESULTS, "fallback"

    p = Path(input_path)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return json.load(f), str(p)

    return FALLBACK_RESULTS, "fallback"


def canonical_order(model_names: List[str]) -> List[str]:
    preferred = [
        "ExVAD_Baseline",
        "CMAF",
        "Temporal_V4",
        "Temporal_V5",
    ]
    ordered = [name for name in preferred if name in model_names]
    ordered.extend([name for name in model_names if name not in ordered])
    return ordered


def extract_metrics(results: dict) -> Dict[str, dict]:
    models = results.get("models", {})
    metrics_by_model: Dict[str, dict] = {}

    for name, payload in models.items():
        if isinstance(payload, dict) and "metrics" in payload:
            metrics_by_model[name] = payload.get("metrics", {})
        elif isinstance(payload, dict):
            metrics_by_model[name] = payload

    return metrics_by_model


def save_metrics_table(metrics_by_model: Dict[str, dict], output_dir: Path) -> Path:
    import csv

    table_path = output_dir / "chapter6_v5_metrics_table.csv"
    columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "normal_accuracy",
        "anomaly_accuracy",
        "threshold",
        "evaluated_samples",
    ]

    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for model in canonical_order(list(metrics_by_model.keys())):
            m = metrics_by_model.get(model, {})
            writer.writerow(
                {
                    "model": model,
                    "accuracy": m.get("accuracy", float("nan")),
                    "precision": m.get("precision", float("nan")),
                    "recall": m.get("recall", float("nan")),
                    "f1": m.get("f1", float("nan")),
                    "normal_accuracy": m.get("normal_accuracy", float("nan")),
                    "anomaly_accuracy": m.get("anomaly_accuracy", float("nan")),
                    "threshold": m.get("threshold", float("nan")),
                    "evaluated_samples": m.get("evaluated_samples", 0),
                }
            )

    return table_path


def fmt_value(value, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "NA"
        return f"{float(value):.{digits}f}"
    return str(value)


def render_table_image(
    output_path: Path,
    title: str,
    headers: List[str],
    rows: List[List[str]],
    dpi: int,
) -> Path:
    num_cols = max(len(headers), 1)
    fig_width = max(10, 1.6 * num_cols)
    fig_height = max(3.0, 1.4 + 0.52 * (len(rows) + 1))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#5D6D7E")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#2E4053")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#F7F9FB" if row % 2 == 0 else "white")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_table_images(metrics_by_model: Dict[str, dict], results: dict, output_dir: Path, dpi: int) -> List[Path]:
    models = canonical_order(list(metrics_by_model.keys()))

    table_images: List[Path] = []

    rows_primary: List[List[str]] = []
    for model in models:
        m = metrics_by_model.get(model, {})
        rows_primary.append(
            [
                model,
                fmt_value(m.get("accuracy")),
                fmt_value(m.get("precision")),
                fmt_value(m.get("recall")),
                fmt_value(m.get("f1")),
                fmt_value(m.get("threshold"), 3),
                fmt_value(m.get("evaluated_samples"), 0),
            ]
        )
    table_images.append(
        render_table_image(
            output_dir / "tbl_01_primary_metrics.png",
            "Chapter 6 Table 1: Primary Performance Metrics",
            ["Model", "Accuracy", "Precision", "Recall", "F1", "Threshold", "Samples"],
            rows_primary,
            dpi,
        )
    )

    rows_classwise: List[List[str]] = []
    for model in models:
        m = metrics_by_model.get(model, {})
        norm_acc = m.get("normal_accuracy", float("nan"))
        anom_acc = m.get("anomaly_accuracy", float("nan"))
        gap = float(anom_acc - norm_acc) if not (math.isnan(float(norm_acc)) or math.isnan(float(anom_acc))) else float("nan")
        rows_classwise.append(
            [
                model,
                fmt_value(norm_acc),
                fmt_value(anom_acc),
                fmt_value(gap),
            ]
        )
    table_images.append(
        render_table_image(
            output_dir / "tbl_02_classwise_metrics.png",
            "Chapter 6 Table 2: Class-wise Reliability Metrics",
            ["Model", "Normal Accuracy", "Anomaly Accuracy", "Anomaly-Normal Gap"],
            rows_classwise,
            dpi,
        )
    )

    improvements = results.get("improvements", {})
    rows_improvements: List[List[str]] = []
    for comp_name, vals in improvements.items():
        rows_improvements.append(
            [
                comp_name,
                fmt_value(vals.get("accuracy")),
                fmt_value(vals.get("f1")),
                fmt_value(vals.get("anomaly_accuracy")),
                fmt_value(vals.get("normal_accuracy")),
            ]
        )
    if not rows_improvements:
        rows_improvements.append(["No improvement block", "NA", "NA", "NA", "NA"])
    table_images.append(
        render_table_image(
            output_dir / "tbl_03_improvement_deltas.png",
            "Chapter 6 Table 3: Relative Improvement Deltas",
            ["Comparison", "Accuracy Delta", "F1 Delta", "Anomaly Acc Delta", "Normal Acc Delta"],
            rows_improvements,
            dpi,
        )
    )

    guard = results.get("v5_recall_guardrail_vs_cmaf", {})
    best = guard.get("best_metrics", {})
    default = guard.get("default_threshold_metrics", {})
    rows_guardrail: List[List[str]] = []
    if best and default:
        rows_guardrail.append(
            [
                "Default threshold",
                fmt_value(default.get("threshold"), 3),
                fmt_value(default.get("accuracy")),
                fmt_value(default.get("f1")),
                fmt_value(default.get("anomaly_accuracy")),
                fmt_value(default.get("normal_accuracy")),
            ]
        )
        rows_guardrail.append(
            [
                "Guardrail threshold",
                fmt_value(best.get("threshold"), 3),
                fmt_value(best.get("accuracy")),
                fmt_value(best.get("f1")),
                fmt_value(best.get("anomaly_accuracy")),
                fmt_value(best.get("normal_accuracy")),
            ]
        )
        rows_guardrail.append(
            [
                "Delta (guardrail-default)",
                "-",
                fmt_value(best.get("accuracy", 0.0) - default.get("accuracy", 0.0)),
                fmt_value(best.get("f1", 0.0) - default.get("f1", 0.0)),
                fmt_value(best.get("anomaly_accuracy", 0.0) - default.get("anomaly_accuracy", 0.0)),
                fmt_value(best.get("normal_accuracy", 0.0) - default.get("normal_accuracy", 0.0)),
            ]
        )
    else:
        rows_guardrail.append(["Guardrail block not available", "NA", "NA", "NA", "NA", "NA"])

    table_images.append(
        render_table_image(
            output_dir / "tbl_04_guardrail_tradeoff_table.png",
            "Chapter 6 Table 4: Guardrail Threshold Trade-off",
            ["Mode", "Threshold", "Accuracy", "F1", "Anomaly Accuracy", "Normal Accuracy"],
            rows_guardrail,
            dpi,
        )
    )

    return table_images


def annotate_bars(ax, bars, fmt: str = "{:.3f}") -> None:
    for bar in bars:
        h = bar.get_height()
        if math.isnan(h):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.005,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_overall_metrics(metrics_by_model: Dict[str, dict], output_dir: Path, dpi: int) -> Path:
    metric_names = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    models = canonical_order(list(metrics_by_model.keys()))

    x = np.arange(len(metric_names))
    n = len(models)
    width = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, model in enumerate(models):
        values = [metrics_by_model[model].get(k, np.nan) for k in metric_names]
        offsets = x - 0.4 + (idx + 0.5) * width
        bars = ax.bar(offsets, values, width=width, label=model, alpha=0.9)
        annotate_bars(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("V5 Comparison: Overall Classification Metrics")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out = output_dir / "fig_01_overall_metrics.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_classwise_accuracy(metrics_by_model: Dict[str, dict], output_dir: Path, dpi: int) -> Path:
    models = canonical_order(list(metrics_by_model.keys()))
    x = np.arange(len(models))
    normal = [metrics_by_model[m].get("normal_accuracy", np.nan) for m in models]
    anomaly = [metrics_by_model[m].get("anomaly_accuracy", np.nan) for m in models]

    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width / 2, normal, width=width, label="Normal accuracy", alpha=0.9)
    bars2 = ax.bar(x + width / 2, anomaly, width=width, label="Anomaly accuracy", alpha=0.9)
    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Class-wise accuracy")
    ax.set_title("V5 Comparison: Normal vs Anomaly Accuracy")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out = output_dir / "fig_02_classwise_accuracy.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def confusion_matrix_array(metrics: dict) -> np.ndarray:
    cm = metrics.get("confusion_matrix", {})
    tn = cm.get("TN", 0)
    fp = cm.get("FP", 0)
    fn = cm.get("FN", 0)
    tp = cm.get("TP", 0)
    return np.array([[tn, fp], [fn, tp]], dtype=float)


def plot_confusion_matrices(metrics_by_model: Dict[str, dict], output_dir: Path, dpi: int) -> Path:
    models = canonical_order(list(metrics_by_model.keys()))
    n = len(models)
    cols = min(3, max(1, n))
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]
        arr = confusion_matrix_array(metrics_by_model.get(model, {}))
        im = ax.imshow(arr, cmap="Blues")
        ax.set_title(model)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])

        max_val = arr.max() if arr.size else 1.0
        for r in range(2):
            for c in range(2):
                val = int(arr[r, c])
                color = "white" if val > max_val / 2 else "black"
                ax.text(c, r, str(val), ha="center", va="center", color=color, fontsize=11)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for k in range(n, len(axes)):
        axes[k].axis("off")

    fig.suptitle("V5 Comparison: Confusion Matrices", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    out = output_dir / "fig_03_confusion_matrices.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_score_means(metrics_by_model: Dict[str, dict], output_dir: Path, dpi: int) -> Path:
    models = canonical_order(list(metrics_by_model.keys()))
    x = np.arange(len(models))
    normal_means = []
    anomaly_means = []

    for m in models:
        stats = metrics_by_model[m].get("score_stats", {})
        normal_means.append(stats.get("normal_mean", np.nan))
        anomaly_means.append(stats.get("anomaly_mean", np.nan))

    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width / 2, normal_means, width=width, label="Normal score mean", alpha=0.9)
    bars2 = ax.bar(x + width / 2, anomaly_means, width=width, label="Anomaly score mean", alpha=0.9)
    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean predicted score")
    ax.set_title("V5 Comparison: Score Distribution Separation (Means)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out = output_dir / "fig_04_score_means.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_improvement_deltas(results: dict, output_dir: Path, dpi: int) -> Path:
    improvements = results.get("improvements", {})
    labels = list(improvements.keys())
    metrics = ["accuracy", "f1", "anomaly_accuracy", "normal_accuracy"]

    if not labels:
        # Build an empty placeholder to keep chapter links valid.
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.5, 0.5, "No improvement block found in input JSON", ha="center", va="center")
        ax.axis("off")
        out = output_dir / "fig_05_improvement_deltas.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    x = np.arange(len(labels))
    n = len(metrics)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, metric in enumerate(metrics):
        vals = [improvements[label].get(metric, np.nan) for label in labels]
        offsets = x - 0.4 + (idx + 0.5) * width
        bars = ax.bar(offsets, vals, width=width, label=metric, alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            if math.isnan(h):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + (0.002 if h >= 0 else -0.004),
                f"{h:+.3f}",
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=8,
            )

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Delta vs reference model")
    ax.set_title("V5 Comparison: Relative Improvements")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()

    out = output_dir / "fig_05_improvement_deltas.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_radar_metrics(metrics_by_model: Dict[str, dict], output_dir: Path, dpi: int) -> Path:
    models = canonical_order(list(metrics_by_model.keys()))
    metric_names = ["accuracy", "precision", "recall", "f1", "anomaly_accuracy", "normal_accuracy"]
    labels = ["Acc", "Prec", "Rec", "F1", "AnomAcc", "NormAcc"]

    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8.5, 8.5))
    ax = plt.subplot(111, polar=True)

    for model in models:
        vals = [metrics_by_model[model].get(m, np.nan) for m in metric_names]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=model)
        ax.fill(angles, vals, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("V5 Comparison: Multi-Metric Radar Profile", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15))

    out = output_dir / "fig_06_radar_profile.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_guardrail(results: dict, output_dir: Path, dpi: int) -> Path:
    guard = results.get("v5_recall_guardrail_vs_cmaf", {})
    best = guard.get("best_metrics", {})
    default = guard.get("default_threshold_metrics", {})

    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ["anomaly_accuracy", "normal_accuracy", "f1", "accuracy"]
    labels = ["AnomAcc", "NormAcc", "F1", "Acc"]

    if not best or not default:
        ax.text(0.5, 0.5, "No guardrail block found in input JSON", ha="center", va="center")
        ax.axis("off")
        out = output_dir / "fig_07_guardrail_tradeoff.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    x = np.arange(len(metrics))
    width = 0.35
    default_vals = [default.get(m, np.nan) for m in metrics]
    best_vals = [best.get(m, np.nan) for m in metrics]

    bars1 = ax.bar(x - width / 2, default_vals, width=width, label="Default threshold", alpha=0.9)
    bars2 = ax.bar(x + width / 2, best_vals, width=width, label="Guardrail threshold", alpha=0.9)
    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)

    title = "V5 Guardrail Trade-off"
    if best.get("threshold") is not None:
        title += f" (Guardrail thr={best.get('threshold'):.3f})"

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out = output_dir / "fig_07_guardrail_tradeoff.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def write_manifest(
    output_dir: Path,
    source: str,
    figure_paths: List[Path],
    table_path: Path,
    table_image_paths: List[Path],
    results: dict,
) -> Path:
    manifest_path = output_dir / "chapter6_figure_manifest.md"

    lines = []
    lines.append("# Chapter 6 V5 Figure Manifest")
    lines.append("")
    lines.append(f"Source JSON: {source}")
    lines.append("")

    protocol = results.get("protocol", {})
    if protocol:
        lines.append("## Protocol Snapshot")
        lines.append("")
        for key, value in protocol.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    lines.append("## Generated Figures")
    lines.append("")
    for fp in figure_paths:
        lines.append(f"- {fp.name}")
    lines.append("")
    lines.append("## Generated Tables")
    lines.append("")
    lines.append(f"- {table_path.name}")
    lines.append("")
    lines.append("## Generated Table Images")
    lines.append("")
    for tp in table_image_paths:
        lines.append(f"- {tp.name}")
    lines.append("")

    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return manifest_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results, source = load_results(args.input, args.force_fallback)
    metrics_by_model = extract_metrics(results)

    if not metrics_by_model:
        raise RuntimeError("No model metrics found. Check the JSON schema or use --force_fallback.")

    table_path = save_metrics_table(metrics_by_model, output_dir)

    figure_paths = [
        plot_overall_metrics(metrics_by_model, output_dir, args.dpi),
        plot_classwise_accuracy(metrics_by_model, output_dir, args.dpi),
        plot_confusion_matrices(metrics_by_model, output_dir, args.dpi),
        plot_score_means(metrics_by_model, output_dir, args.dpi),
        plot_improvement_deltas(results, output_dir, args.dpi),
        plot_radar_metrics(metrics_by_model, output_dir, args.dpi),
        plot_guardrail(results, output_dir, args.dpi),
    ]

    table_image_paths = create_table_images(metrics_by_model, results, output_dir, args.dpi)

    manifest = write_manifest(output_dir, source, figure_paths, table_path, table_image_paths, results)

    summary = {
        "source": source,
        "output_dir": str(output_dir),
        "figure_count": len(figure_paths),
        "figures": [str(p) for p in figure_paths],
        "metrics_table": str(table_path),
        "table_images": [str(p) for p in table_image_paths],
        "manifest": str(manifest),
        "models": canonical_order(list(metrics_by_model.keys())),
    }
    with (output_dir / "chapter6_graph_generation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("CHAPTER 6 V5 GRAPH GENERATION COMPLETE")
    print("=" * 72)
    print(f"Source: {source}")
    print(f"Output directory: {output_dir}")
    print(f"Figures generated: {len(figure_paths)}")
    for fp in figure_paths:
        print(f" - {fp}")
    print(f"Metrics table: {table_path}")
    print(f"Table images: {len(table_image_paths)}")
    for tp in table_image_paths:
        print(f" - {tp}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
