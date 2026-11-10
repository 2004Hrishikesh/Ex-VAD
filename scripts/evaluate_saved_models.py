import json
import os
from collections import defaultdict

import torch

from src.models.enhanced_exvad_model import EnhancedExVADModel
from src.models.exvad_model import ExVADModel
from src.utils.data_loader_new import create_dataloaders


def load_enhanced_model(path: str, device: torch.device) -> torch.nn.Module:
    model = EnhancedExVADModel(device).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    return model


def load_baseline_model(path: str, device: torch.device) -> torch.nn.Module:
    model = ExVADModel(device).to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    return model


def evaluate(model: torch.nn.Module, dataloader, thresholds, device: torch.device):
    stats = {thr: defaultdict(int) for thr in thresholds}
    model.eval()

    with torch.no_grad():
        for frames_batch, labels, _, _ in dataloader:
            frames = frames_batch[0]
            label = int(labels[0].item())

            anomaly_score, _, _ = model(frames)
            if isinstance(anomaly_score, torch.Tensor):
                score_val = anomaly_score.squeeze().detach().float().cpu().item()
            else:
                score_val = float(anomaly_score)

            prob = score_val if 0.0 <= score_val <= 1.0 else torch.sigmoid(torch.tensor(score_val)).item()

            for thr in thresholds:
                pred = 1 if prob > thr else 0

                stats[thr]["total"] += 1
                stats[thr]["correct"] += int(pred == label)

                if label == 0:
                    stats[thr]["normal_total"] += 1
                    stats[thr]["normal_correct"] += int(pred == label)
                else:
                    stats[thr]["anomaly_total"] += 1
                    stats[thr]["anomaly_correct"] += int(pred == label)

                if pred == 1 and label == 1:
                    stats[thr]["tp"] += 1
                elif pred == 1 and label == 0:
                    stats[thr]["fp"] += 1
                elif pred == 0 and label == 1:
                    stats[thr]["fn"] += 1
                else:
                    stats[thr]["tn"] += 1

    summary = {}
    for thr, values in stats.items():
        total = values["total"]
        normal_total = values.get("normal_total", 0)
        anomaly_total = values.get("anomaly_total", 0)
        tp = values.get("tp", 0)
        fp = values.get("fp", 0)
        fn = values.get("fn", 0)

        overall_acc = values["correct"] / total if total else 0.0
        normal_acc = values["normal_correct"] / normal_total if normal_total else 0.0
        anomaly_acc = values["anomaly_correct"] / anomaly_total if anomaly_total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        summary[thr] = {
            "overall_accuracy": overall_acc,
            "normal_accuracy": normal_acc,
            "anomaly_accuracy": anomaly_acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "total_samples": total,
            "normal_samples": normal_total,
            "anomaly_samples": anomaly_total,
        }

    return summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader = create_dataloaders(
        data_root="data",
        batch_size=1,
        num_frames=16,
        device=device,
        num_workers=0,
    )

    thresholds = [0.1, 0.25, 0.5]

    candidates = {
        "enhanced_best_model": ("enhanced", "models_saved/best_model.pth"),
        "enhanced_backup": ("enhanced", "models_saved/best_model_bak.pth"),
        "enhanced_old_arch": ("enhanced", "models_saved/best_model_old_architecture.pth"),
        "baseline_best": ("baseline", "models_saved/baseline_best_model.pth"),
    }

    results = {}

    for name, (model_type, path) in candidates.items():
        if not os.path.exists(path):
            continue

        if model_type == "enhanced":
            model = load_enhanced_model(path, device)
        else:
            model = load_baseline_model(path, device)

        summary = evaluate(model, test_loader, thresholds, device)
        results[name] = summary

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
