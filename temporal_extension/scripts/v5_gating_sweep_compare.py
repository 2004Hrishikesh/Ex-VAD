import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from temporal_extension.models.temporal_enhanced_exvad_v3 import TemporalEnhancedExVADv3
from src.utils.data_loader_new import create_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Compare checkpoints with default and tuned gating")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="List of checkpoint paths",
    )
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--threshold_min", type=float, default=0.0)
    parser.add_argument("--threshold_max", type=float, default=0.55)
    parser.add_argument("--num_steps", type=int, default=111)
    parser.add_argument(
        "--output_json",
        type=str,
        default="outputs/v5_gating_sweep_results.json",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default="outputs/v5_before_after_comparison.md",
    )
    return parser.parse_args()


def metrics_at_threshold(scores, labels, thr):
    preds = [1 if s > thr else 0 for s in scores]
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)

    total = len(labels)
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    norm_acc = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "threshold": float(thr),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "normal_accuracy": float(norm_acc),
        "anomaly_accuracy": float(rec),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def best_threshold_metrics(scores, labels, thresholds):
    all_metrics = [metrics_at_threshold(scores, labels, float(t)) for t in thresholds]
    best_acc = max(all_metrics, key=lambda m: (m["accuracy"], m["f1"]))
    best_f1 = max(all_metrics, key=lambda m: (m["f1"], m["accuracy"]))
    return best_acc, best_f1


def set_gating_params(model, params):
    model.base_low_threshold = float(params["base_low_threshold"])
    model.base_high_threshold = float(params["base_high_threshold"])
    model.temporal_high_conf = float(params["temporal_high_conf"])
    model.temporal_low_conf = float(params["temporal_low_conf"])
    model.down_correction_scale = float(params["down_correction_scale"])


def score_from_cached(model, frame_features, video_features, text_inputs):
    with torch.no_grad():
        base_score = model.base_model.madm.predict_from_visual_features(
            video_features,
            explanations=text_inputs,
        )
        if base_score.dim() == 0:
            base_score = base_score.view(1, 1)
        elif base_score.dim() == 1:
            base_score = base_score.unsqueeze(1)

        temporal_logits, _, _ = model.temporal_detector(frame_features, return_components=True)
        temporal_score = torch.sigmoid(temporal_logits)

        fusion_weight = torch.sigmoid(model.temporal_weight) * 0.4 + 0.1
        base_is_low = (base_score < model.base_low_threshold).float()
        base_is_high = (base_score > model.base_high_threshold).float()
        temporal_is_anom_confident = (temporal_score > model.temporal_high_conf).float()
        temporal_is_normal_confident = (temporal_score < model.temporal_low_conf).float()

        allow_boost = (1.0 - base_is_low) + (base_is_low * temporal_is_anom_confident)
        allow_suppress = base_is_high * temporal_is_normal_confident

        up_correction = torch.relu(temporal_score - base_score) * allow_boost
        down_correction = torch.relu(base_score - temporal_score) * allow_suppress

        combined_score = (
            base_score
            + fusion_weight * up_correction
            - (fusion_weight * model.down_correction_scale) * down_correction
        )
        combined_score = torch.clamp(combined_score, 0.0, 1.0)
        return float(combined_score.view(-1)[0].item())


def build_cache(model, data_loader, device):
    cache = []
    with torch.no_grad():
        for idx, (frames_batch, y, _, _) in enumerate(data_loader):
            if not frames_batch or len(frames_batch[0]) == 0:
                continue

            frames = frames_batch[0]
            label = int(y[0].item())

            frame_features = model.extract_per_frame_features(frames).float().cpu()
            video_features = frame_features.mean(dim=1).cpu()
            explanation, captions = model.base_model.aegm(frames)
            text_inputs = model._prepare_text_inputs(explanation, captions)

            cache.append(
                {
                    "frame_features": frame_features,
                    "video_features": video_features,
                    "text_inputs": text_inputs,
                    "label": label,
                }
            )

            if (idx + 1) % 25 == 0:
                print(f"  cached {idx + 1} samples...", flush=True)

    return cache


def evaluate_checkpoint(model, checkpoint_path, cache, device, thresholds, sweep_grid):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state, strict=False)
    model.eval()

    default_params = {
        "base_low_threshold": model.base_low_threshold,
        "base_high_threshold": model.base_high_threshold,
        "temporal_high_conf": model.temporal_high_conf,
        "temporal_low_conf": model.temporal_low_conf,
        "down_correction_scale": model.down_correction_scale,
    }

    def score_all_with_params(params):
        set_gating_params(model, params)
        scores = []
        labels = []
        for i, item in enumerate(cache):
            frame_features = item["frame_features"].to(device)
            video_features = item["video_features"].to(device)
            text_inputs = item["text_inputs"]
            label = item["label"]
            score = score_from_cached(model, frame_features, video_features, text_inputs)
            scores.append(score)
            labels.append(label)
        return scores, labels

    baseline_scores, labels = score_all_with_params(default_params)
    baseline_best_acc, baseline_best_f1 = best_threshold_metrics(baseline_scores, labels, thresholds)

    best_sweep = {
        "params": dict(default_params),
        "best_accuracy_metrics": dict(baseline_best_acc),
        "best_f1_metrics": dict(baseline_best_f1),
    }

    for idx, params in enumerate(sweep_grid):
        scores, labels = score_all_with_params(params)
        best_acc_metrics, best_f1_metrics = best_threshold_metrics(scores, labels, thresholds)

        curr = best_sweep["best_accuracy_metrics"]
        if (
            best_acc_metrics["accuracy"] > curr["accuracy"]
            or (
                best_acc_metrics["accuracy"] == curr["accuracy"]
                and best_acc_metrics["f1"] > curr["f1"]
            )
        ):
            best_sweep = {
                "params": dict(params),
                "best_accuracy_metrics": dict(best_acc_metrics),
                "best_f1_metrics": dict(best_f1_metrics),
            }

        if (idx + 1) % 8 == 0:
            print(f"  sweep configs checked: {idx + 1}/{len(sweep_grid)}", flush=True)

    return {
        "checkpoint": checkpoint_path,
        "epoch": int(checkpoint.get("epoch")) if isinstance(checkpoint, dict) and checkpoint.get("epoch") is not None else None,
        "samples": len(labels),
        "positives": int(sum(labels)),
        "negatives": int(len(labels) - sum(labels)),
        "default": {
            "gating_params": default_params,
            "best_accuracy_metrics": baseline_best_acc,
            "best_f1_metrics": baseline_best_f1,
        },
        "tuned": best_sweep,
    }


def render_markdown(results):
    lines = []
    lines.append("# V5 Before And After Comparison")
    lines.append("")

    for r in results:
        name = os.path.basename(r["checkpoint"])
        lines.append(f"## {name} (Epoch {r['epoch']})")
        lines.append("")
        lines.append("| Variant | Threshold | Accuracy | Precision | Recall | F1 | Normal Acc | Anomaly Acc | TP | TN | FP | FN |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

        d = r["default"]["best_accuracy_metrics"]
        t = r["tuned"]["best_accuracy_metrics"]

        lines.append(
            "| Default gating | "
            f"{d['threshold']:.3f} | {d['accuracy']:.4f} | {d['precision']:.4f} | {d['recall']:.4f} | {d['f1']:.4f} | "
            f"{d['normal_accuracy']:.4f} | {d['anomaly_accuracy']:.4f} | {d['tp']} | {d['tn']} | {d['fp']} | {d['fn']} |"
        )
        lines.append(
            "| Tuned gating | "
            f"{t['threshold']:.3f} | {t['accuracy']:.4f} | {t['precision']:.4f} | {t['recall']:.4f} | {t['f1']:.4f} | "
            f"{t['normal_accuracy']:.4f} | {t['anomaly_accuracy']:.4f} | {t['tp']} | {t['tn']} | {t['fp']} | {t['fn']} |"
        )

        lines.append("")
        lines.append("Tuned gating parameters:")
        p = r["tuned"]["params"]
        lines.append(
            f"- base_low_threshold={p['base_low_threshold']}, "
            f"base_high_threshold={p['base_high_threshold']}, "
            f"temporal_high_conf={p['temporal_high_conf']}, "
            f"temporal_low_conf={p['temporal_low_conf']}, "
            f"down_correction_scale={p['down_correction_scale']}"
        )
        lines.append("")

    return "\n".join(lines)


def main():
    args = parse_args()
    os.makedirs("outputs", exist_ok=True)

    for ckpt in args.checkpoints:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    _, data_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=1,
        num_frames=args.num_frames,
        device=str(device),
        num_workers=0,
    )

    model = TemporalEnhancedExVADv3(device=device).to(device)
    model.eval()

    print("Building shared cache...", flush=True)
    cache_start = time.time()
    cache = build_cache(model, data_loader, device)
    print(f"Cache complete: {len(cache)} samples in {time.time() - cache_start:.1f}s", flush=True)

    thresholds = np.round(np.linspace(args.threshold_min, args.threshold_max, args.num_steps), 3)

    sweep_grid = []
    for values in itertools.product(
        [0.14, 0.16],
        [0.48, 0.52],
        [0.68, 0.72],
        [0.28, 0.32],
        [0.7, 0.9],
    ):
        sweep_grid.append(
            {
                "base_low_threshold": values[0],
                "base_high_threshold": values[1],
                "temporal_high_conf": values[2],
                "temporal_low_conf": values[3],
                "down_correction_scale": values[4],
            }
        )

    all_results = []
    for ckpt_path in args.checkpoints:
        print(f"Evaluating {ckpt_path}", flush=True)
        start = time.time()
        result = evaluate_checkpoint(
            model=model,
            checkpoint_path=ckpt_path,
            cache=cache,
            device=device,
            thresholds=thresholds,
            sweep_grid=sweep_grid,
        )
        result["elapsed_seconds"] = float(time.time() - start)
        all_results.append(result)
        print(f"Completed {ckpt_path} in {result['elapsed_seconds']:.1f}s", flush=True)

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "num_frames": args.num_frames,
        "samples": len(cache),
        "search_space_size": len(sweep_grid),
        "results": all_results,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    md_text = render_markdown(all_results)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md_text)

    print(f"SAVED: {args.output_json}", flush=True)
    print(f"SAVED: {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
