"""Terminal-first evaluation for Temporal Enhanced ExVAD V3.

This script prints per-video and per-frame anomaly decisions in the terminal,
including captions and inner-layer diagnostics.
"""

import argparse
import json
import os
import sys
from collections import defaultdict


import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from temporal_extension.models.temporal_enhanced_exvad_v3 import TemporalEnhancedExVADv3
from src.utils.data_loader_new import create_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Verbose Temporal V3 evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models_saved/temporal_v3_adaptive_full.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--threshold-file",
        type=str,
        default="models_saved/temporal_v3_adaptive_full_threshold.json",
        help="Path to threshold JSON",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold override",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Dataset root",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of frames per sample",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Limit videos to evaluate (0 means all)",
    )
    parser.add_argument(
        "--no-frame-details",
        action="store_true",
        help="Disable per-frame terminal output",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/temporal_v3_terminal_verbose_results.json",
        help="Path to save JSON results",
    )
    return parser.parse_args()


def load_threshold(threshold_file: str, override_threshold: float):
    if override_threshold is not None:
        return float(override_threshold)

    if not os.path.exists(threshold_file):
        return 0.5

    with open(threshold_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        if "best_threshold" in payload:
            return float(payload["best_threshold"])
        if "optimal_threshold" in payload:
            return float(payload["optimal_threshold"])
        if "threshold" in payload:
            return float(payload["threshold"])

    return 0.5


def as_float(value, default=0.0):
    if value is None:
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    if torch.is_tensor(value):
        return float(value.view(-1)[0].item())
    return float(default)


def normalize_frames(frames_list):
    if isinstance(frames_list, list) and len(frames_list) > 0:
        return frames_list[0] if isinstance(frames_list[0], list) else frames_list
    return frames_list


def derive_category(class_names, video_paths):
    if class_names and class_names[0]:
        return class_names[0]

    if video_paths and video_paths[0]:
        path_parts = video_paths[0].replace("\\", "/").split("/")
        for i, part in enumerate(path_parts):
            if part.lower() == "test" and i + 1 < len(path_parts):
                return path_parts[i + 1]

    return "Unknown"


def derive_video_name(video_paths, fallback_idx):
    if video_paths and video_paths[0]:
        raw_path = video_paths[0]
        base = os.path.basename(raw_path)
        # For grouped frame-sequence entries, basename is the logical video id.
        return base if base else f"video_{fallback_idx:04d}"
    return f"video_{fallback_idx:04d}"


def madm_score_with_cached_text(madm, visual_features: torch.Tensor, text_proj: torch.Tensor) -> torch.Tensor:
    """Compute base anomaly score with precomputed text projection."""
    cmaf = madm.cmaf

    cmaf_dtype = next(cmaf.visual_projection[0].parameters()).dtype
    visual_features = visual_features.to(dtype=cmaf_dtype)
    text_proj = text_proj.to(dtype=cmaf_dtype)

    if text_proj.shape[0] != visual_features.shape[0]:
        text_proj = text_proj.expand(visual_features.shape[0], -1)

    visual_proj = cmaf.visual_projection(visual_features)
    visual_seq = visual_proj.unsqueeze(1)
    text_seq = text_proj.unsqueeze(1)

    visual_attended, _ = cmaf.visual_to_text_attention(query=visual_seq, key=text_seq, value=text_seq)
    visual_attended = visual_attended.squeeze(1)

    text_attended, _ = cmaf.text_to_visual_attention(query=text_seq, key=visual_seq, value=visual_seq)
    text_attended = text_attended.squeeze(1)

    combined_features = torch.cat([visual_attended, text_attended], dim=-1)
    gate = cmaf.fusion_gate(combined_features)
    fused_hidden = gate * visual_attended + (1.0 - gate) * text_attended
    fused_features = cmaf.final_projection(fused_hidden)

    enhanced_features = visual_features + fused_features
    target_dtype = next(madm.anomaly_head.parameters()).dtype
    enhanced_features = enhanced_features.to(dtype=target_dtype)
    return madm.anomaly_head(enhanced_features)


def apply_bidirectional_fusion(model, base_score: torch.Tensor, temporal_score: torch.Tensor, fusion_weight: torch.Tensor):
    base_is_low = (base_score < model.base_low_threshold).float()
    base_is_high = (base_score > model.base_high_threshold).float()

    temporal_is_anom_confident = (temporal_score > model.temporal_high_conf).float()
    temporal_is_normal_confident = (temporal_score < model.temporal_low_conf).float()

    allow_boost = (1.0 - base_is_low) + (base_is_low * temporal_is_anom_confident)
    allow_suppress = base_is_high * temporal_is_normal_confident

    up_correction = torch.relu(temporal_score - base_score) * allow_boost
    down_correction = torch.relu(base_score - temporal_score) * allow_suppress

    combined = (
        base_score
        + fusion_weight * up_correction
        - (fusion_weight * model.down_correction_scale) * down_correction
    )
    combined = torch.clamp(combined, 0.0, 1.0)
    return combined, allow_boost, allow_suppress, up_correction, down_correction


def print_video_header(sample_idx, total_samples, category, video_name):
    print("\n" + "-" * 90)
    print(f"VIDEO {sample_idx:04d}/{total_samples:04d} | category={category} | id={video_name}")
    print("-" * 90)


def print_video_prediction(label, pred, score, threshold):
    gt_name = "Anomaly" if label == 1 else "Normal"
    pred_name = "Anomaly" if pred == 1 else "Normal"
    is_correct = pred == label
    print(
        "Prediction: "
        f"GT={gt_name} | Pred={pred_name} | Score={score:.4f} | Threshold={threshold:.4f} | "
        f"Correct={is_correct}"
    )


def print_inner_details(details):
    base_score = as_float(details.get("base_score"))
    temporal_score = as_float(details.get("temporal_score"))
    temporal_logits = as_float(details.get("temporal_logits"))
    fusion_weight = as_float(details.get("fusion_weight"))
    allow_boost = as_float(details.get("allow_boost"))
    allow_suppress = as_float(details.get("allow_suppress"))
    up_correction = as_float(details.get("up_correction"))
    down_correction = as_float(details.get("down_correction"))

    temporal_features = details.get("temporal_features")
    temporal_feat_mean_abs = float(temporal_features.abs().mean().item()) if torch.is_tensor(temporal_features) else 0.0
    temporal_feat_std = float(temporal_features.std().item()) if torch.is_tensor(temporal_features) else 0.0

    print(
        "Inner: "
        f"base={base_score:.4f} | temporal={temporal_score:.4f} | temporal_logits={temporal_logits:.4f} | "
        f"fusion_w={fusion_weight:.4f} | "
        f"allow_boost={allow_boost:.4f} | allow_suppress={allow_suppress:.4f} | "
        f"up_corr={up_correction:.4f} | down_corr={down_correction:.4f}"
    )
    print(
        "Temporal features: "
        f"mean_abs={temporal_feat_mean_abs:.6f} | std={temporal_feat_std:.6f}"
    )

    components = details.get("components", {})
    component_stats = {}
    for name, tensor in components.items():
        if torch.is_tensor(tensor):
            component_stats[name] = {
                "mean_abs": float(tensor.abs().mean().item()),
                "mean": float(tensor.mean().item()),
                "std": float(tensor.std().item()),
                "min": float(tensor.min().item()),
                "max": float(tensor.max().item()),
                "shape": list(tensor.shape),
            }

    if component_stats:
        print("Component stats:")
        for name in sorted(component_stats.keys()):
            stats = component_stats[name]
            shape_str = "x".join(str(x) for x in stats["shape"])
            print(
                f"  {name:<12} shape={shape_str:<12} mean_abs={stats['mean_abs']:.6f} "
                f"mean={stats['mean']:.6f} std={stats['std']:.6f} "
                f"min={stats['min']:.6f} max={stats['max']:.6f}"
            )


def extract_component_stats(details):
    components = details.get("components", {})
    stats = {}
    for name, tensor in components.items():
        if torch.is_tensor(tensor):
            stats[name] = {
                "mean_abs": float(tensor.abs().mean().item()),
                "mean": float(tensor.mean().item()),
                "std": float(tensor.std().item()),
                "min": float(tensor.min().item()),
                "max": float(tensor.max().item()),
                "shape": list(tensor.shape),
            }
    return stats


def test_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "=" * 60)
    print("Loading Temporal Enhanced ExVAD V3 Model")
    print("=" * 60)

    model = TemporalEnhancedExVADv3(device=device).to(device)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state, strict=False)
    model.eval()

    if isinstance(checkpoint, dict):
        print(f"Loaded checkpoint: {args.checkpoint}")
        if "epoch" in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch']}")
        if "val_f1" in checkpoint:
            print(f"Checkpoint val_f1: {checkpoint['val_f1']:.4f}")
        if "val_acc" in checkpoint:
            print(f"Checkpoint val_acc: {checkpoint['val_acc']:.4f}")

    threshold = load_threshold(args.threshold_file, args.threshold)
    print(f"Using threshold: {threshold:.4f}")

    print("\n" + "=" * 60)
    print("Loading Test Dataset")
    print("=" * 60)

    _, test_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=1,
        num_frames=args.num_frames,
        num_workers=0,
    )

    dataset_size = len(test_loader.dataset)
    print(f"Test samples available: {dataset_size}")

    max_samples = args.max_videos if args.max_videos and args.max_videos > 0 else dataset_size
    max_samples = min(max_samples, dataset_size)
    print(f"Videos to evaluate: {max_samples}")
    print(f"Per-frame details: {not args.no_frame_details}")

    print("\n" + "=" * 60)
    print("Evaluating on Test Set (terminal verbose)")
    print("=" * 60)

    all_preds = []
    all_labels = []
    all_scores = []
    category_results = defaultdict(lambda: {"preds": [], "labels": [], "scores": []})
    skipped = 0
    per_video_details = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", total=max_samples)
        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= max_samples:
                break

            frames_list, labels, class_names, video_paths = batch
            category = derive_category(class_names, video_paths)
            video_name = derive_video_name(video_paths, batch_idx + 1)

            frames = normalize_frames(frames_list)
            if len(frames) == 0:
                skipped += 1
                continue

            label = labels[0].item() if hasattr(labels[0], "item") else int(labels[0])

            try:
                output = model(frames, return_details=True)
                combined_score = output[0]
                explanation = output[1] if len(output) > 1 else ""
                captions = output[2] if len(output) > 2 else []
                details = output[3] if len(output) > 3 else {}

                score = as_float(combined_score)
                pred = 1 if score > threshold else 0

                all_scores.append(score)
                all_preds.append(pred)
                all_labels.append(label)

                category_results[category]["preds"].append(pred)
                category_results[category]["labels"].append(label)
                category_results[category]["scores"].append(score)

                print_video_header(batch_idx + 1, max_samples, category, video_name)
                print_video_prediction(label, pred, score, threshold)

                caption0 = ""
                if isinstance(captions, (list, tuple)) and len(captions) > 0:
                    caption0 = str(captions[0])
                elif isinstance(captions, str):
                    caption0 = captions

                print(f"Caption[0]: {caption0 if caption0 else 'N/A'}")
                print(f"Explanation: {str(explanation)}")
                print_inner_details(details)

                frame_entries = []
                if not args.no_frame_details:
                    text_inputs = model._prepare_text_inputs(explanation, captions)
                    cmaf = model.base_model.madm.cmaf
                    text_embeddings = cmaf.encode_text(text_inputs, model.device).float()
                    text_proj = cmaf.text_projection(text_embeddings)

                    frame_features = model.extract_per_frame_features(frames).float()
                    temporal_score_tensor = details.get("temporal_score")
                    if temporal_score_tensor is None:
                        temporal_score_tensor = torch.tensor([[as_float(details.get("temporal_score"))]], device=device)
                    fusion_weight_tensor = torch.tensor([[as_float(details.get("fusion_weight"))]], device=device)

                    print("Frame-level decisions:")
                    for frame_idx in range(frame_features.shape[1]):
                        visual_i = frame_features[:, frame_idx, :]
                        base_i = madm_score_with_cached_text(model.base_model.madm, visual_i, text_proj)
                        combined_i, allow_boost_i, allow_suppress_i, up_i, down_i = apply_bidirectional_fusion(
                            model,
                            base_i,
                            temporal_score_tensor,
                            fusion_weight_tensor,
                        )

                        frame_score = as_float(combined_i)
                        frame_pred = 1 if frame_score > threshold else 0
                        frame_correct = frame_pred == label

                        base_i_float = as_float(base_i)
                        allow_boost_float = as_float(allow_boost_i)
                        allow_suppress_float = as_float(allow_suppress_i)
                        up_i_float = as_float(up_i)
                        down_i_float = as_float(down_i)

                        frame_entries.append(
                            {
                                "frame_index": int(frame_idx),
                                "base_score": base_i_float,
                                "combined_score": frame_score,
                                "pred": int(frame_pred),
                                "gt": int(label),
                                "correct": bool(frame_correct),
                                "allow_boost": allow_boost_float,
                                "allow_suppress": allow_suppress_float,
                                "up_correction": up_i_float,
                                "down_correction": down_i_float,
                            }
                        )

                        pred_name = "Anomaly" if frame_pred == 1 else "Normal"
                        gt_name = "Anomaly" if label == 1 else "Normal"
                        print(
                            f"  frame={frame_idx:02d} | pred={pred_name:<7} | gt={gt_name:<7} | "
                            f"base={base_i_float:.4f} | score={frame_score:.4f} | "
                            f"allow_boost={allow_boost_float:.1f} | allow_suppress={allow_suppress_float:.1f} | "
                            f"up={up_i_float:.4f} | down={down_i_float:.4f} | correct={frame_correct}"
                        )

                component_stats = extract_component_stats(details)
                temporal_features = details.get("temporal_features")
                temporal_feature_stats = {
                    "mean_abs": float(temporal_features.abs().mean().item()) if torch.is_tensor(temporal_features) else 0.0,
                    "mean": float(temporal_features.mean().item()) if torch.is_tensor(temporal_features) else 0.0,
                    "std": float(temporal_features.std().item()) if torch.is_tensor(temporal_features) else 0.0,
                    "min": float(temporal_features.min().item()) if torch.is_tensor(temporal_features) else 0.0,
                    "max": float(temporal_features.max().item()) if torch.is_tensor(temporal_features) else 0.0,
                    "shape": list(temporal_features.shape) if torch.is_tensor(temporal_features) else [],
                }

                per_video_details.append(
                    {
                        "index": int(batch_idx + 1),
                        "category": category,
                        "video_name": video_name,
                        "label": int(label),
                        "pred": int(pred),
                        "score": float(score),
                        "correct": bool(pred == label),
                        "explanation": str(explanation),
                        "captions": captions if isinstance(captions, (list, tuple)) else [str(captions)],
                        "inner": {
                            "base_score": as_float(details.get("base_score")),
                            "temporal_logits": as_float(details.get("temporal_logits")),
                            "temporal_score": as_float(details.get("temporal_score")),
                            "fusion_weight": as_float(details.get("fusion_weight")),
                            "allow_boost": as_float(details.get("allow_boost")),
                            "allow_suppress": as_float(details.get("allow_suppress")),
                            "up_correction": as_float(details.get("up_correction")),
                            "down_correction": as_float(details.get("down_correction")),
                            "temporal_feature_stats": temporal_feature_stats,
                            "component_stats": component_stats,
                        },
                        "frames": frame_entries,
                    }
                )

            except Exception as exc:
                print(f"\nError processing sample {batch_idx + 1}: {exc}")
                skipped += 1
                continue

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    if len(all_labels) == 0:
        print("No valid samples were evaluated.")
        return

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    all_scores_np = np.array(all_scores)

    accuracy = accuracy_score(all_labels_np, all_preds_np)
    precision = precision_score(all_labels_np, all_preds_np, zero_division=0)
    recall = recall_score(all_labels_np, all_preds_np, zero_division=0)
    f1 = f1_score(all_labels_np, all_preds_np, zero_division=0)
    cm = confusion_matrix(all_labels_np, all_preds_np, labels=[0, 1])

    normal_mask = all_labels_np == 0
    anomaly_mask = all_labels_np == 1
    normal_acc = (all_preds_np[normal_mask] == 0).sum() / normal_mask.sum() if normal_mask.sum() > 0 else 0.0
    anomaly_acc = (all_preds_np[anomaly_mask] == 1).sum() / anomaly_mask.sum() if anomaly_mask.sum() > 0 else 0.0

    print(f"Total samples tested: {len(all_labels_np)}")
    print(f"Skipped samples: {skipped}")
    print(f"Normal samples: {int(normal_mask.sum())}")
    print(f"Anomaly samples: {int(anomaly_mask.sum())}")

    print("\nMetric          Value")
    print("--------------------------")
    print(f"Accuracy       {accuracy:10.4f}")
    print(f"Precision      {precision:10.4f}")
    print(f"Recall         {recall:10.4f}")
    print(f"F1 Score       {f1:10.4f}")
    print(f"Normal Acc     {normal_acc:10.4f}")
    print(f"Anomaly Acc    {anomaly_acc:10.4f}")

    print("\nConfusion Matrix (rows=actual [Normal, Anomaly], cols=pred [Normal, Anomaly]):")
    print(cm)

    print("\nPER-CATEGORY RESULTS")
    print("Category              Samples      Acc  AvgScore")
    print("-------------------------------------------------")
    for category in sorted(category_results.keys()):
        data = category_results[category]
        if len(data["labels"]) == 0:
            continue
        labels_cat = np.array(data["labels"])
        preds_cat = np.array(data["preds"])
        scores_cat = np.array(data["scores"])
        cat_acc = accuracy_score(labels_cat, preds_cat)
        print(f"{category:<20} {len(labels_cat):>7} {cat_acc:>8.4f} {scores_cat.mean():>9.4f}")

    normal_scores = all_scores_np[all_labels_np == 0]
    anomaly_scores = all_scores_np[all_labels_np == 1]
    score_gap = float(anomaly_scores.mean() - normal_scores.mean()) if len(normal_scores) > 0 and len(anomaly_scores) > 0 else 0.0

    results = {
        "checkpoint": args.checkpoint,
        "threshold_file": args.threshold_file,
        "threshold": float(threshold),
        "device_used": str(device),
        "evaluated_samples": int(len(all_labels_np)),
        "skipped_samples": int(skipped),
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "normal_accuracy": float(normal_acc),
            "anomaly_accuracy": float(anomaly_acc),
            "confusion_matrix": cm.tolist(),
        },
        "score_stats": {
            "normal_mean": float(normal_scores.mean()) if len(normal_scores) > 0 else None,
            "normal_std": float(normal_scores.std()) if len(normal_scores) > 0 else None,
            "anomaly_mean": float(anomaly_scores.mean()) if len(anomaly_scores) > 0 else None,
            "anomaly_std": float(anomaly_scores.std()) if len(anomaly_scores) > 0 else None,
            "score_gap": score_gap,
        },
        "per_category": {
            cat: {
                "samples": len(data["labels"]),
                "accuracy": float(accuracy_score(np.array(data["labels"]), np.array(data["preds"]))),
                "avg_score": float(np.mean(np.array(data["scores"]))) if len(data["scores"]) > 0 else 0.0,
            }
            for cat, data in category_results.items()
        },
        "detailed_terminal_records": per_video_details,
    }

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    cli_args = parse_args()
    test_model(cli_args)
