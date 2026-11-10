"""
Render the V5 system architecture in the same horizontal layout style as
generate_v5_architecture.py, with minor refinements for paper readability.

Layout:
                       Input Video
                            |
                       Frame Sampling (16 frames)
                       /         |          \\
                      v          v           v
              CLIP Visual    CMAF (center)   BLIP + DistilBERT
              Encoder       (cross-modal)    Text Semantics
                  |             |
                  v          Base Score
              Temporal          |
              Module            |
              (Trans+BiLSTM     |
              +Motion)          |
                  |             |
                  v             |
              Temporal -------> Confidence-Gated Fusion
              Score             |
                                v
                          Final Anomaly Score
                                |
                                v
                            Decision -----> Anomaly | Normal

Output:
  outputs/system_architecture_v5_clean.png
  outputs/system_architecture_v5_clean.pdf
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def draw_box(ax, x, y, w, h, text, color, fontsize=9, bold=False):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.15",
        facecolor=color, edgecolor="#37474F", linewidth=1.4,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color="#212121",
    )


def draw_path_arrow(ax, points, lw=1.6):
    if len(points) < 2:
        return
    for i in range(len(points) - 2):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        ax.plot([x0, x1], [y0, y1], color="#455A64", lw=lw)
    arrow = FancyArrowPatch(
        points[-2], points[-1],
        arrowstyle="->", mutation_scale=10,
        lw=lw, color="#455A64",
    )
    ax.add_patch(arrow)


def box_anchor(box, side):
    x, y, w, h = box
    if side == "top":    return (x + w / 2, y + h)
    if side == "bottom": return (x + w / 2, y)
    if side == "left":   return (x, y + h / 2)
    if side == "right":  return (x + w, y + h / 2)
    return (x + w / 2, y + h / 2)


def main():
    colors = {
        "input":       "#E3F2FD",
        "feature":     "#BBDEFB",
        "temporal":    "#FFE0B2",
        "fusion":      "#F8BBD9",
        "output":      "#D1C4E9",
        "result_anom": "#FFCDD2",
        "result_norm": "#C8E6C9",
    }

    fig, ax = plt.subplots(figsize=(22, 10))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 10.2)
    ax.axis("off")

    ax.text(
        11, 9.7,
        "Temporal Transformer-LSTM Fusion with Confidence-Gated Boosting\n"
        "for Vision-Language Video Anomaly Detection (V5)",
        ha="center", va="center",
        fontsize=22, fontweight="bold", color="#1A237E",
    )

    boxes = {
        "input":    (8.0, 8.4, 6.0, 0.9),
        "sampling": (8.0, 7.2, 6.0, 0.9),

        "clip": (1.2, 5.8, 5.3, 0.95),
        "blip": (15.5, 5.8, 5.3, 0.95),

        "cmaf": (8.0, 5.8, 6.0, 0.95),
        "base": (8.0, 4.5, 6.0, 0.85),

        "temporal": (1.2, 4.3, 5.3, 1.0),
        "tscore":   (1.2, 2.95, 5.3, 0.85),

        "gate":  (8.0, 3.0, 6.0, 1.0),
        "final": (8.0, 1.7, 6.0, 0.85),

        "decision": (15.5, 1.7, 5.3, 0.85),
        "anom":     (15.5, 0.4, 2.5, 0.7),
        "normal":   (18.3, 0.4, 2.5, 0.7),
    }

    draw_box(ax, *boxes["input"],    "Input Video",                    colors["input"],   fontsize=15, bold=True)
    draw_box(ax, *boxes["sampling"], "Frame Sampling (16 frames)",     colors["input"],   fontsize=13)

    draw_box(ax, *boxes["clip"], "CLIP Visual Encoder",                colors["feature"], fontsize=14, bold=True)
    draw_box(ax, *boxes["blip"], "BLIP + DistilBERT\nText Semantics",  colors["feature"], fontsize=14, bold=True)

    draw_box(ax, *boxes["cmaf"], "Cross-Modal Attention Fusion (CMAF)", colors["fusion"],  fontsize=14, bold=True)
    draw_box(ax, *boxes["base"], "Base Score",                          colors["fusion"],  fontsize=13)

    draw_box(ax, *boxes["temporal"], "Temporal Module\n(Transformer + BiLSTM + Motion)",
             colors["temporal"], fontsize=13, bold=True)
    draw_box(ax, *boxes["tscore"],   "Temporal Score",                  colors["temporal"], fontsize=13)

    draw_box(ax, *boxes["gate"],
             "Confidence-Gated Fusion\n(base_low=0.15 | temp_high=0.70)",
             colors["fusion"], fontsize=13, bold=True)
    draw_box(ax, *boxes["final"], "Final Anomaly Score\n(Threshold ~0.44)",
             colors["output"], fontsize=13, bold=True)

    draw_box(ax, *boxes["decision"], "Decision",                        colors["output"],      fontsize=13, bold=True)
    draw_box(ax, *boxes["anom"],     "Anomaly",                         colors["result_anom"], fontsize=12, bold=True)
    draw_box(ax, *boxes["normal"],   "Normal",                          colors["result_norm"], fontsize=12, bold=True)

    # Arrows
    draw_path_arrow(ax, [box_anchor(boxes["input"], "bottom"),    box_anchor(boxes["sampling"], "top")])

    sampling_left  = (boxes["sampling"][0] + 1.1, boxes["sampling"][1])
    sampling_right = (boxes["sampling"][0] + boxes["sampling"][2] - 1.1, boxes["sampling"][1])
    draw_path_arrow(ax, [sampling_left,  box_anchor(boxes["clip"], "top")])
    draw_path_arrow(ax, [sampling_right, box_anchor(boxes["blip"], "top")])

    draw_path_arrow(ax, [box_anchor(boxes["clip"], "right"), box_anchor(boxes["cmaf"], "left")])
    draw_path_arrow(ax, [box_anchor(boxes["blip"], "left"),  box_anchor(boxes["cmaf"], "right")])

    draw_path_arrow(ax, [box_anchor(boxes["clip"], "bottom"),     box_anchor(boxes["temporal"], "top")])

    draw_path_arrow(ax, [box_anchor(boxes["cmaf"], "bottom"),     box_anchor(boxes["base"], "top")])
    draw_path_arrow(ax, [box_anchor(boxes["temporal"], "bottom"), box_anchor(boxes["tscore"], "top")])

    draw_path_arrow(ax, [box_anchor(boxes["base"], "bottom"),     box_anchor(boxes["gate"], "top")])
    draw_path_arrow(ax, [box_anchor(boxes["tscore"], "right"),    box_anchor(boxes["gate"], "left")])

    draw_path_arrow(ax, [box_anchor(boxes["gate"], "bottom"),     box_anchor(boxes["final"], "top")])
    draw_path_arrow(ax, [box_anchor(boxes["final"], "right"),     box_anchor(boxes["decision"], "left")])

    decision_x, decision_y, decision_w, _ = boxes["decision"]
    decision_anom_start = (decision_x + decision_w * 0.32, decision_y)
    decision_norm_start = (decision_x + decision_w * 0.68, decision_y)
    anom_top   = box_anchor(boxes["anom"],   "top")
    normal_top = box_anchor(boxes["normal"], "top")
    split_y = 1.35
    draw_path_arrow(ax, [decision_anom_start, (decision_anom_start[0], split_y), (anom_top[0], split_y),   anom_top])
    draw_path_arrow(ax, [decision_norm_start, (decision_norm_start[0], split_y), (normal_top[0], split_y), normal_top])

    os.makedirs("outputs", exist_ok=True)
    png_path = "outputs/system_architecture_v5_clean.png"
    pdf_path = "outputs/system_architecture_v5_clean.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path,         bbox_inches="tight", facecolor="white")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
