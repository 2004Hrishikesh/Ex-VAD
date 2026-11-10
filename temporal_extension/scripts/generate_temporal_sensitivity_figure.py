"""
Generate a publication-quality temporal-order sensitivity figure for the paper.

Reads outputs/examiner_demo_prof_full.json (full 290-video test split) and
produces a clean two-panel figure:

  Left panel  : Accuracy comparison (ExVAD vs CMAF vs V5 vs V5-shuffled).
                Demonstrates the order-dependent gain.
  Right panel : Per-CATEGORY mean |s_V5 - s_V5_shuffled|, with anomaly
                categories highlighted vs. normal. Proves that the temporal
                branch's sensitivity concentrates on inherently sequential
                anomaly classes — not on instantaneous events or normal
                activity (which would indicate noise).

Output: outputs/temporal_sensitivity_v5.png  (300 DPI, paper-ready)
"""
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


JSON_PATH  = "outputs/examiner_demo_prof_full.json"
OUT_PATH   = "outputs/temporal_sensitivity_v5.png"

# Sequential anomalies (require frame ordering to detect) vs.
# instantaneous anomalies (per-frame visuals usually sufficient).
SEQUENTIAL_ANOMALIES = {
    "Burglary", "Shoplifting", "Stealing", "Robbery",
    "Fighting", "Assault", "Abuse", "Arrest", "Vandalism",
}
INSTANTANEOUS_ANOMALIES = {
    "Explosion", "Shooting", "RoadAccidents", "Arson",
}
NORMAL = {"NormalVideos"}


def load_data():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def per_category_mean_gap(samples):
    by_cat = defaultdict(list)
    for s in samples:
        by_cat[s["category"]].append(float(s["temporal_order_gap"]))
    return {c: float(np.mean(v)) for c, v in by_cat.items()}


def category_color(cat):
    if cat in NORMAL:
        return "#4A90A4"            # teal-blue for normal
    if cat in SEQUENTIAL_ANOMALIES:
        return "#C9302C"            # red for sequential anomaly
    if cat in INSTANTANEOUS_ANOMALIES:
        return "#E8A33D"            # gold for instantaneous anomaly
    return "#888888"


def main():
    data = load_data()
    samples = data["samples"]
    summary = data["summary"]
    acc = summary["accuracy"]

    # ---------- Figure layout ----------
    fig, axes = plt.subplots(
        1, 2,
        figsize=(11, 4.5),
        gridspec_kw={"width_ratios": [1.0, 1.6]},
    )

    # ---------- LEFT: Accuracy comparison ----------
    ax = axes[0]
    labels = ["ExVAD", "ExVAD+CMAF", "V5\n(ordered)", "V5\n(shuffled)"]
    vals   = [acc["exvad"], acc["cmaf"], acc["v5"], acc["v5_shuffled"]]
    colors = ["#9AA9B8", "#3F7D8C", "#2E9E48", "#C9872A"]

    bars = ax.bar(labels, vals, color=colors, edgecolor="#37474F", linewidth=1.0)
    ax.set_ylim(0.75, 0.90)
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_title("(a) Accuracy on UCF-Crime test split (n=290)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.003,
            f"{v:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    # Annotate the V5 vs V5-shuffled gap
    delta = acc["v5"] - acc["v5_shuffled"]
    ax.annotate(
        f"$\\Delta$ = {delta*100:+.1f} pp\n(order-dependent)",
        xy=(2.5, (acc["v5"] + acc["v5_shuffled"]) / 2),
        xytext=(3.2, 0.785),
        fontsize=9, ha="center",
        arrowprops=dict(arrowstyle="->", color="#C9302C", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#C9302C"),
    )

    # ---------- RIGHT: Per-category sensitivity ----------
    ax = axes[1]
    means = per_category_mean_gap(samples)

    # Sort: anomaly groups first (sequential, then instantaneous), normal last
    ordered_cats = (
        sorted([c for c in means if c in SEQUENTIAL_ANOMALIES],     key=lambda c: -means[c])
        + sorted([c for c in means if c in INSTANTANEOUS_ANOMALIES], key=lambda c: -means[c])
        + [c for c in means if c in NORMAL]
    )

    y = np.arange(len(ordered_cats))
    vals = [means[c] for c in ordered_cats]
    colors = [category_color(c) for c in ordered_cats]

    bars = ax.barh(y, vals, color=colors, edgecolor="#37474F", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered_cats, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(r"Mean  $|s_{V5} - s_{V5\mathrm{-shuffled}}|$", fontsize=11)
    ax.set_title("(b) Temporal-order sensitivity by category", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for bar, v in zip(bars, vals):
        ax.text(
            v + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}",
            va="center", ha="left", fontsize=8,
        )

    # Legend (matched to the three color groups)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#C9302C", label="Sequential anomaly"),
        plt.Rectangle((0, 0), 1, 1, color="#E8A33D", label="Instantaneous anomaly"),
        plt.Rectangle((0, 0), 1, 1, color="#4A90A4", label="Normal"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Console summary so the paper text can quote exact numbers
    print(f"Saved: {OUT_PATH}")
    print()
    print("=== Accuracy ===")
    for k in ("exvad", "cmaf", "v5", "v5_shuffled"):
        print(f"  {k:14s}: {acc[k]:.4f}")
    print(f"  V5 - V5_shuffled: {acc['v5'] - acc['v5_shuffled']:+.4f}")
    print()
    print("=== Per-category temporal sensitivity (sorted, descending) ===")
    sorted_cats = sorted(means.items(), key=lambda kv: -kv[1])
    for cat, v in sorted_cats:
        group = (
            "seq-anom" if cat in SEQUENTIAL_ANOMALIES
            else "inst-anom" if cat in INSTANTANEOUS_ANOMALIES
            else "normal"
        )
        print(f"  {cat:18s} [{group:9s}]  {v:.4f}")


if __name__ == "__main__":
    main()
