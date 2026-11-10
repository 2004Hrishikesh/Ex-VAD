"""Terminal demo for Temporal V5 over the full test split.

This script is intended for internal review demos:
- Loads the Temporal V5 checkpoint and threshold
- Iterates over test videos one-by-one
- Prints per-video prediction details in terminal
- Reports overall and per-category metrics
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from temporal_extension.models.temporal_enhanced_exvad_v3 import TemporalEnhancedExVADv3
from src.utils.data_loader_new import VideoAnomalyDataset, custom_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Temporal V5 demo on each video in the test split"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models_saved/temporal_v5_high_eff.pth",
        help="Path to Temporal V5 checkpoint",
    )
    parser.add_argument(
        "--threshold-file",
        type=str,
        default="models_saved/temporal_v5_high_eff_threshold.json",
        help="Path to threshold JSON",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Manual threshold override",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Dataset root path",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Frames sampled per video",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Evaluate only first N test videos (0 means all)",
    )
    parser.add_argument(
        "--show-explanation",
        action="store_true",
        help="Print model explanation text for each sample",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/demo_v5_testsplit_results.json",
        help="Path to save JSON report",
    )
    return parser.parse_args()


def load_threshold(threshold_file, threshold_override):
    if threshold_override is not None:
        return float(threshold_override)

    if not os.path.exists(threshold_file):
        return 0.44

    with open(threshold_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        if "best_threshold" in payload:
            return float(payload["best_threshold"])
        if "optimal_threshold" in payload:
            return float(payload["optimal_threshold"])
        if "threshold" in payload:
            return float(payload["threshold"])

    return 0.44


def extract_score_and_text(output):
    if isinstance(output, tuple):
        score = output[0]
        explanation = output[1] if len(output) > 1 else ""
    else:
        score = output
        explanation = ""

    if torch.is_tensor(score):
        if score.dim() == 0:
            score_value = float(score.item())
        else:
            score_value = float(score.squeeze().item())
    else:
        score_value = float(score)

    return score_value, str(explanation)


def label_name(label):
    return "anomaly" if int(label) == 1 else "normal"


def derive_category(class_names, video_paths):
    if class_names and class_names[0]:
        return str(class_names[0])

    if video_paths and video_paths[0]:
        path_parts = str(video_paths[0]).replace("\\", "/").split("/")
        for idx, part in enumerate(path_parts):
            if part.lower() == "test" and idx + 1 < len(path_parts):
                return path_parts[idx + 1]

    return "Unknown"


def derive_video_id(video_paths, fallback_index):
    if video_paths and video_paths[0]:
        raw = str(video_paths[0])
        base = os.path.basename(raw)
        return base if base else f"video_{fallback_index:04d}"
    return f"video_{fallback_index:04d}"


def build_summary(y_true, y_pred, y_score):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    total = len(y_true)
    normal_total = sum(1 for t in y_true if t == 0)
    anomaly_total = sum(1 for t in y_true if t == 1)

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    normal_acc = tn / normal_total if normal_total > 0 else 0.0
    anomaly_acc = tp / anomaly_total if anomaly_total > 0 else 0.0

    normal_scores = [s for s, t in zip(y_score, y_true) if t == 0]
    anomaly_scores = [s for s, t in zip(y_score, y_true) if t == 1]

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "normal_accuracy": float(normal_acc),
        "anomaly_accuracy": float(anomaly_acc),
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "evaluated_samples": int(total),
        "normal_samples": int(normal_total),
        "anomaly_samples": int(anomaly_total),
        "score_stats": {
            "normal_mean": float(sum(normal_scores) / len(normal_scores)) if normal_scores else 0.0,
            "anomaly_mean": float(sum(anomaly_scores) / len(anomaly_scores)) if anomaly_scores else 0.0,
        },
    }


def load_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = TemporalEnhancedExVADv3(device=device).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    metadata = {}
    if isinstance(checkpoint, dict):
        for key in ["epoch", "val_acc", "val_f1", "val_precision", "val_recall"]:
            if key in checkpoint:
                value = checkpoint[key]
                if isinstance(value, (float, int)):
                    metadata[key] = float(value)
                else:
                    metadata[key] = value

    return model, metadata


def create_test_loader(data_root, num_frames, device, num_workers=0):
    dataset = VideoAnomalyDataset(
        data_root=data_root,
        split="test",
        num_frames=num_frames,
        device=str(device),
    )

    loader_args = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": str(device) == "cuda",
        "persistent_workers": num_workers > 0,
        "collate_fn": custom_collate_fn,
    }

    return DataLoader(dataset, **loader_args)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 84)
    print("TEMPORAL V5 TERMINAL DEMO (FULL TEST SPLIT)")
    print("=" * 84)
    print(f"timestamp         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"device            : {device}")
    print(f"checkpoint        : {args.checkpoint}")

    threshold = load_threshold(args.threshold_file, args.threshold)
    print(f"threshold         : {threshold:.6f}")
    print(f"data_root         : {args.data_root}")
    print(f"num_frames        : {args.num_frames}")

    print("\nLoading model...")
    model, metadata = load_model(args.checkpoint, device)
    if metadata:
        for key in ["epoch", "val_acc", "val_f1", "val_precision", "val_recall"]:
            if key in metadata:
                value = metadata[key]
                if isinstance(value, float):
                    print(f"checkpoint_{key:<8}: {value:.6f}")
                else:
                    print(f"checkpoint_{key:<8}: {value}")

    print("\nLoading data...")
    test_loader = create_test_loader(
        data_root=args.data_root,
        num_frames=args.num_frames,
        device=device,
        num_workers=0,
    )

    total_available = len(test_loader.dataset)
    limit = args.max_videos if args.max_videos and args.max_videos > 0 else total_available
    limit = min(limit, total_available)

    print(f"test_videos_found : {total_available}")
    print(f"test_videos_run   : {limit}")

    y_true = []
    y_pred = []
    y_score = []
    records = []
    category_stats = defaultdict(lambda: {"total": 0, "correct": 0, "normal": 0, "anomaly": 0})
    skipped = 0

    print("\n" + "-" * 84)
    print("Per-video results")
    print("-" * 84)

    with torch.no_grad():
        for idx, batch in enumerate(test_loader, start=1):
            if idx > limit:
                break

            frames_batch, labels, class_names, video_paths = batch
            if not frames_batch or len(frames_batch[0]) == 0:
                skipped += 1
                continue

            frames = frames_batch[0]
            label = int(labels[0].item())
            category = derive_category(class_names, video_paths)
            video_id = derive_video_id(video_paths, idx)

            try:
                output = model(frames)
                score, explanation = extract_score_and_text(output)
                pred = 1 if score > threshold else 0
                correct = pred == label

                y_true.append(label)
                y_pred.append(pred)
                y_score.append(score)

                category_stats[category]["total"] += 1
                category_stats[category]["correct"] += int(correct)
                category_stats[category]["normal"] += int(label == 0)
                category_stats[category]["anomaly"] += int(label == 1)

                status = "OK" if correct else "ERR"
                gt_name = label_name(label)
                pred_name = label_name(pred)
                print(
                    f"[{idx:04d}/{limit:04d}] {status:<3} "
                    f"category={category:<15} "
                    f"gt={gt_name:<7} pred={pred_name:<7} "
                    f"score={score:0.4f} "
                    f"video={video_id}"
                )

                if args.show_explanation:
                    explanation_preview = explanation.replace("\n", " ").strip()
                    if len(explanation_preview) > 220:
                        explanation_preview = explanation_preview[:220] + "..."
                    print(f"             explanation={explanation_preview}")

                records.append(
                    {
                        "index": int(idx),
                        "category": category,
                        "video": video_id,
                        "path": str(video_paths[0]) if video_paths else "",
                        "label": int(label),
                        "prediction": int(pred),
                        "score": float(score),
                        "correct": bool(correct),
                        "explanation": explanation if args.show_explanation else "",
                    }
                )
            except Exception as exc:
                skipped += 1
                print(f"[{idx:04d}/{limit:04d}] ERR failed_to_process video={video_id} error={exc}")

    summary = build_summary(y_true, y_pred, y_score)

    per_category = {}
    for category in sorted(category_stats.keys()):
        total = category_stats[category]["total"]
        correct = category_stats[category]["correct"]
        per_category[category] = {
            "samples": int(total),
            "correct": int(correct),
            "accuracy": float((correct / total) if total > 0 else 0.0),
            "normal_samples": int(category_stats[category]["normal"]),
            "anomaly_samples": int(category_stats[category]["anomaly"]),
        }

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "checkpoint": args.checkpoint,
            "threshold_file": args.threshold_file,
            "threshold": float(threshold),
            "data_root": args.data_root,
            "num_frames": int(args.num_frames),
            "max_videos": int(args.max_videos),
            "device": str(device),
            "skipped": int(skipped),
        },
        "checkpoint_metadata": metadata,
        "summary": summary,
        "per_category": per_category,
        "per_video": records,
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 84)
    print("FINAL SUMMARY")
    print("=" * 84)
    print(f"evaluated_samples : {summary['evaluated_samples']}")
    print(f"skipped_samples   : {skipped}")
    print(f"normal_samples    : {summary['normal_samples']}")
    print(f"anomaly_samples   : {summary['anomaly_samples']}")
    print(f"accuracy          : {summary['accuracy']:.4f}")
    print(f"precision         : {summary['precision']:.4f}")
    print(f"recall            : {summary['recall']:.4f}")
    print(f"f1                : {summary['f1']:.4f}")
    print(f"normal_accuracy   : {summary['normal_accuracy']:.4f}")
    print(f"anomaly_accuracy  : {summary['anomaly_accuracy']:.4f}")
    print(f"confusion_matrix  : {summary['confusion_matrix']}")

    print("\nPer-category accuracy")
    print("-" * 84)
    print(f"{'category':<22} {'samples':>7} {'accuracy':>10} {'normal':>8} {'anomaly':>8}")
    for category in sorted(per_category.keys()):
        stats = per_category[category]
        print(
            f"{category:<22} {stats['samples']:>7} {stats['accuracy']:>10.4f} "
            f"{stats['normal_samples']:>8} {stats['anomaly_samples']:>8}"
        )

    print(f"\nSaved JSON report: {args.output}")


if __name__ == "__main__":
    main()