"""
Ablation experiments for Tables 4 and 5 in the paper.

Table 4 — Temporal encoder ablation:
  Inference-time feature masking on the jointly-trained full checkpoint.
  Encoder contributions are zeroed in the concatenated feature vector before
  the shared fusion layer. This is a standard post-hoc knockout approach.

  Fusion vector layout (512 dims total):
    [0:256]   Transformer (hidden_dim=256)
    [256:384] BiLSTM      (hidden_dim//2=128)
    [384:512] Motion      (hidden_dim//2=128)

Table 5 — Gating mechanism ablation:
  Same checkpoint; different fusion strategies override the standard gating.

Usage:
    python scripts/run_paper_ablation.py \
        --checkpoint models_saved/temporal_v3_adaptive_full_cont.pth \
        --data_root data

Outputs:
    outputs/ablation_table4.md
    outputs/ablation_table5.md
"""
import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from temporal_extension.models.temporal_enhanced_exvad_v3 import TemporalEnhancedExVADv3
from src.utils.data_loader_new import create_dataloaders


# ---------------------------------------------------------------------------
# Encoder mask definitions for Table 4
# Layout: [0:256] Transformer | [256:384] BiLSTM | [384:512] Motion
# ---------------------------------------------------------------------------
ENCODER_COMBOS = [
    ("Transformer only",     [True,  False, False]),
    ("BiLSTM only",          [False, True,  False]),
    ("Motion Encoder only",  [False, False, True ]),
    ("Transformer + BiLSTM", [True,  True,  False]),
    ("Transformer + Motion", [True,  False, True ]),
    ("BiLSTM + Motion",      [False, True,  True ]),
    ("All three (Ours)",     [True,  True,  True ]),
]

TRANS_SLICE  = slice(0,   256)
LSTM_SLICE   = slice(256, 384)
MOTION_SLICE = slice(384, 512)


def make_encoder_mask(use_tr, use_lstm, use_mot, device):
    mask = torch.zeros(512, device=device)
    if use_tr:
        mask[TRANS_SLICE] = 1.0
    if use_lstm:
        mask[LSTM_SLICE] = 1.0
    if use_mot:
        mask[MOTION_SLICE] = 1.0
    return mask  # [512]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics_at_threshold(scores, labels, thr):
    preds = [1 if s > thr else 0 for s in scores]
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    total = len(labels)
    acc  = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    nacc = tn / (tn + fp) if (tn + fp) else 0.0
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, nacc=nacc,
                tp=tp, tn=tn, fp=fp, fn=fn, threshold=float(thr))


def best_metrics(scores, labels, thresholds):
    all_m = [metrics_at_threshold(scores, labels, t) for t in thresholds]
    return max(all_m, key=lambda m: (m["acc"], m["f1"]))


# ---------------------------------------------------------------------------
# Cache building — run CLIP + CMAF once for all ablation variants
# ---------------------------------------------------------------------------
def build_cache(model, loader, device):
    """
    Returns list of dicts:
      frame_features : (1, T, 512) float32 on CPU
      video_features : (1, 512)    float32 on CPU
      text_inputs    : list[str]
      label          : int
    """
    cache = []
    model.eval()
    with torch.no_grad():
        for idx, (frames_batch, y, _, _) in enumerate(loader):
            if not frames_batch or len(frames_batch[0]) == 0:
                continue
            frames = frames_batch[0]
            label  = int(y[0].item())

            ff = model.extract_per_frame_features(frames).float().cpu()  # (1,T,512)
            vf = ff.mean(dim=1).cpu()                                     # (1,512)
            explanation, captions = model.base_model.aegm(frames)
            text_inputs = model._prepare_text_inputs(explanation, captions)

            cache.append(dict(frame_features=ff, video_features=vf,
                              text_inputs=text_inputs, label=label))
            if (idx + 1) % 30 == 0:
                print(f"  cached {idx+1} samples...", flush=True)
    return cache


# ---------------------------------------------------------------------------
# Encoder-ablation forward (Table 4)
# ---------------------------------------------------------------------------
def score_with_encoder_mask(model, item, mask, device):
    """
    Run temporal detector with masked encoder features, then apply
    standard confidence-gated fusion.
    """
    ff = item["frame_features"].to(device)   # (1,T,512)
    vf = item["video_features"].to(device)   # (1,512)
    text = item["text_inputs"]

    with torch.no_grad():
        # Base score from frozen CMAF
        base_score = model.base_model.madm.predict_from_visual_features(
            vf, explanations=text
        )
        if base_score.dim() == 0:
            base_score = base_score.view(1, 1)
        elif base_score.dim() == 1:
            base_score = base_score.unsqueeze(1)

        # Temporal features — partial forward through the encoder
        det = model.temporal_detector
        x = det.input_norm(ff)

        features_list = []
        if det.use_transformer:
            tf, _ = det.temporal_transformer(x)
            tf = F.normalize(tf, p=2, dim=-1)
            features_list.append(tf)

        if det.use_lstm:
            lf, _ = det.temporal_lstm(x)
            lf = F.normalize(lf, p=2, dim=-1)
            features_list.append(lf)

        if det.use_motion:
            mf = det.motion_encoder(x)
            mf = F.normalize(mf, p=2, dim=-1)
            features_list.append(mf)

        fused_raw = torch.cat(features_list, dim=-1)  # (1, 512)

        # Apply encoder mask (zero out unwanted encoder slots)
        fused_masked = fused_raw * mask.unsqueeze(0)  # (1, 512)

        temporal_features = det.fusion(fused_masked)
        temporal_logits   = det.temporal_anomaly_head(temporal_features)
        temporal_score    = torch.sigmoid(temporal_logits)

        # Standard confidence-gated fusion
        return _gated_fusion(model, base_score, temporal_score)


def _gated_fusion(model, base_score, temporal_score):
    fw = torch.sigmoid(model.temporal_weight) * 0.4 + 0.1
    bl  = (base_score < model.base_low_threshold).float()
    bh  = (base_score > model.base_high_threshold).float()
    tac = (temporal_score > model.temporal_high_conf).float()
    tnc = (temporal_score < model.temporal_low_conf).float()

    allow_boost    = (1.0 - bl) + (bl * tac)
    allow_suppress = bh * tnc

    up_corr   = torch.relu(temporal_score - base_score) * allow_boost
    down_corr = torch.relu(base_score - temporal_score) * allow_suppress

    s = base_score + fw * up_corr - (fw * model.down_correction_scale) * down_corr
    s = torch.clamp(s, 0.0, 1.0)
    return float(s.view(-1)[0].item())


def run_encoder_ablation(model, cache, thresholds, device):
    rows = []
    for name, (use_tr, use_lstm, use_mot) in ENCODER_COMBOS:
        mask = make_encoder_mask(use_tr, use_lstm, use_mot, device)
        scores, labels = [], []
        for item in cache:
            s = score_with_encoder_mask(model, item, mask, device)
            scores.append(s)
            labels.append(item["label"])
        m = best_metrics(scores, labels, thresholds)
        m["name"] = name
        rows.append(m)
        print(f"  {name:<28s}  acc={m['acc']:.4f}  f1={m['f1']:.4f}  fp={m['fp']:3d}  nacc={m['nacc']:.4f}")
    return rows


# ---------------------------------------------------------------------------
# Gating-ablation forward (Table 5)
# ---------------------------------------------------------------------------
def score_standard(model, item, device):
    ff = item["frame_features"].to(device)
    vf = item["video_features"].to(device)
    text = item["text_inputs"]
    with torch.no_grad():
        base_score = model.base_model.madm.predict_from_visual_features(vf, explanations=text)
        if base_score.dim() == 0:
            base_score = base_score.view(1, 1)
        elif base_score.dim() == 1:
            base_score = base_score.unsqueeze(1)
        logits, _ = model.temporal_detector(ff)
        temporal_score = torch.sigmoid(logits)
        return base_score, temporal_score


def gating_fixed_blend(base, temporal, alpha=0.3):
    s = base + alpha * (temporal - base)
    return float(torch.clamp(s, 0.0, 1.0).view(-1)[0].item())


def gating_boost_only(model, base, temporal):
    fw = torch.sigmoid(model.temporal_weight) * 0.4 + 0.1
    bl  = (base < model.base_low_threshold).float()
    tac = (temporal > model.temporal_high_conf).float()
    allow_boost = (1.0 - bl) + (bl * tac)
    up_corr = torch.relu(temporal - base) * allow_boost
    s = base + fw * up_corr
    return float(torch.clamp(s, 0.0, 1.0).view(-1)[0].item())


def gating_suppress_only(model, base, temporal):
    fw = torch.sigmoid(model.temporal_weight) * 0.4 + 0.1
    bh  = (base > model.base_high_threshold).float()
    tnc = (temporal < model.temporal_low_conf).float()
    allow_suppress = bh * tnc
    down_corr = torch.relu(base - temporal) * allow_suppress
    s = base - (fw * model.down_correction_scale) * down_corr
    return float(torch.clamp(s, 0.0, 1.0).view(-1)[0].item())


def run_gating_ablation(model, cache, thresholds, device):
    strategies = [
        ("Fixed blend (α=0.3, no gate)", "fixed"),
        ("Boost only (no suppress gate)", "boost"),
        ("Suppress only (no boost gate)", "suppress"),
        ("Bidirectional gate (Ours)",     "bidirectional"),
    ]
    rows = []
    for name, mode in strategies:
        scores, labels = [], []
        for item in cache:
            base, temporal = score_standard(model, item, device)
            if mode == "fixed":
                s = gating_fixed_blend(base, temporal)
            elif mode == "boost":
                s = gating_boost_only(model, base, temporal)
            elif mode == "suppress":
                s = gating_suppress_only(model, base, temporal)
            else:
                s = _gated_fusion(model, base, temporal)
            scores.append(s)
            labels.append(item["label"])
        m = best_metrics(scores, labels, thresholds)
        m["name"] = name
        rows.append(m)
        print(f"  {name:<38s}  acc={m['acc']:.4f}  f1={m['f1']:.4f}  fp={m['fp']:3d}  nacc={m['nacc']:.4f}")
    return rows


# ---------------------------------------------------------------------------
# Markdown render
# ---------------------------------------------------------------------------
def render_table4(rows):
    lines = [
        "# Table 4: Temporal Encoder Ablation",
        "",
        "All variants use frozen ExVAD+CMAF base + confidence-gated fusion.",
        "Encoder ablation is post-hoc: unwanted encoder outputs are zeroed",
        "in the concatenated feature vector of the jointly-trained checkpoint.",
        "",
        "| Temporal Encoders | Acc. | F1 | FP | N-Acc |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        bold = "**" if r["name"] == "All three (Ours)" else ""
        lines.append(
            f"| {bold}{r['name']}{bold} | {bold}{r['acc']:.4f}{bold} | "
            f"{bold}{r['f1']:.4f}{bold} | {bold}{r['fp']}{bold} | "
            f"{bold}{r['nacc']:.4f}{bold} |"
        )
    return "\n".join(lines)


def render_table5(rows):
    lines = [
        "# Table 5: Gating Mechanism Ablation",
        "",
        "Same checkpoint; different fusion strategies applied at inference time.",
        "",
        "| Fusion Strategy | Acc. | F1 | FP | N-Acc |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        bold = "**" if "Ours" in r["name"] else ""
        lines.append(
            f"| {bold}{r['name']}{bold} | {bold}{r['acc']:.4f}{bold} | "
            f"{bold}{r['f1']:.4f}{bold} | {bold}{r['fp']}{bold} | "
            f"{bold}{r['nacc']:.4f}{bold} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models_saved/temporal_v5_high_eff.pth")
    p.add_argument("--data_root",  default="data")
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--thr_min",    type=float, default=0.0)
    p.add_argument("--thr_max",    type=float, default=0.55)
    p.add_argument("--thr_steps",  type=int, default=111)
    p.add_argument("--output_dir", default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=1,
        num_frames=args.num_frames,
        device=str(device),
        num_workers=0,
    )

    print("Loading model...", flush=True)
    model = TemporalEnhancedExVADv3(device=device).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    print("Building inference cache...", flush=True)
    cache = build_cache(model, loader, device)
    print(f"Cache ready: {len(cache)} samples", flush=True)

    thresholds = np.round(np.linspace(args.thr_min, args.thr_max, args.thr_steps), 3)

    print("\n=== Table 4: Encoder Ablation ===", flush=True)
    t4_rows = run_encoder_ablation(model, cache, thresholds, device)

    print("\n=== Table 5: Gating Ablation ===", flush=True)
    t5_rows = run_gating_ablation(model, cache, thresholds, device)

    # Save markdown
    t4_path = os.path.join(args.output_dir, "ablation_table4.md")
    t5_path = os.path.join(args.output_dir, "ablation_table5.md")
    with open(t4_path, "w") as f:
        f.write(render_table4(t4_rows))
    with open(t5_path, "w") as f:
        f.write(render_table5(t5_rows))

    # Save JSON for easy re-use
    json_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(json_path, "w") as f:
        json.dump({"table4": t4_rows, "table5": t5_rows}, f, indent=2)

    print(f"\nSaved:\n  {t4_path}\n  {t5_path}\n  {json_path}", flush=True)

    print("\n--- Table 4 summary ---")
    print(render_table4(t4_rows))
    print("\n--- Table 5 summary ---")
    print(render_table5(t5_rows))


if __name__ == "__main__":
    main()
