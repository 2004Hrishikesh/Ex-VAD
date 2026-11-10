"""
Appendix Sample Code: Temporal V5 Architecture

This file contains concise, labeled sample code blocks for the key parts of the
Temporal V5 pipeline used in this project.

Architecture labels used in this appendix:
- A1: Input Pipeline (dataset loading + fixed frame sampling)
- A2: Semantic Branch (AEGM + CMAF baseline scoring)
- A3: Shared Visual Feature Extraction (per-frame CLIP features)
- A4: Temporal Branch (Transformer/LSTM/Motion temporal evidence)
- A5: Bidirectional Confidence-Gated Fusion (core V5 calibration logic)
- A6: Decision Layer (balanced and guardrail operating thresholds)
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.enhanced_exvad_model import EnhancedExVADModel
from temporal_extension.models.temporal_enhanced_exvad_v3 import TemporalEnhancedExVADv3
from src.utils.data_loader_new import create_dataloaders


def architecture_map() -> Dict[str, str]:
    """A0: Human-readable map of architecture parts used in this appendix."""
    return {
        "A1": "Input Pipeline",
        "A2": "Semantic Branch (AEGM + CMAF)",
        "A3": "Shared Visual Feature Extraction",
        "A4": "Temporal Branch",
        "A5": "Bidirectional Confidence-Gated Fusion",
        "A6": "Decision Layer + Threshold Policy",
    }


# -----------------------------------------------------------------------------
# A1: INPUT PIPELINE
# -----------------------------------------------------------------------------
def load_one_test_sample(data_root: str, num_frames: int, device: torch.device):
    """
    A1: Load one sample from the test split with fixed frame count.

    Returns:
        frames: list of PIL frames
        label: int (0 normal, 1 anomaly)
        class_name: str
        video_path: str
    """
    _, test_loader = create_dataloaders(
        data_root=data_root,
        batch_size=1,
        num_frames=num_frames,
        device=str(device),
        num_workers=0,
    )

    batch = next(iter(test_loader))
    frames_batch, labels, class_names, video_paths = batch

    frames = frames_batch[0]
    label = int(labels[0].item())
    class_name = class_names[0] if class_names else "Unknown"
    video_path = video_paths[0] if video_paths else "Unknown"
    return frames, label, class_name, video_path


# -----------------------------------------------------------------------------
# A2: SEMANTIC BRANCH (AEGM + CMAF)
# -----------------------------------------------------------------------------
def semantic_branch_score(frames: List, device: torch.device) -> Tuple[float, str, List[str]]:
    """
    A2: Run semantic baseline branch (AEGM explanation + CMAF score).
    """
    model = EnhancedExVADModel(device=str(device), use_cross_modal_attention=True).to(device)
    model.eval()

    with torch.no_grad():
        anomaly_score, explanation, captions = model(frames)

    score = float(anomaly_score.view(-1)[0].item()) if torch.is_tensor(anomaly_score) else float(anomaly_score)
    return score, str(explanation), captions if isinstance(captions, list) else [str(captions)]


# -----------------------------------------------------------------------------
# A3 + A4 + A5 + A6: V5 INFERENCE WITH DETAILS
# -----------------------------------------------------------------------------
def load_temporal_v5_model(device: torch.device, checkpoint_path: str) -> TemporalEnhancedExVADv3:
    """Load Temporal V5 model and optional checkpoint."""
    model = TemporalEnhancedExVADv3(device=str(device)).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state, strict=False)
    else:
        print(f"[Warning] Checkpoint not found: {checkpoint_path}. Running with current model weights.")

    model.eval()
    return model


def bidirectional_confidence_gated_fusion(
    base_score: torch.Tensor,
    temporal_score: torch.Tensor,
    fusion_weight: torch.Tensor,
    base_low_threshold: float,
    base_high_threshold: float,
    temporal_high_conf: float,
    temporal_low_conf: float,
    down_correction_scale: float,
) -> Dict[str, torch.Tensor]:
    """
    A5: Core V5 fusion equation (standalone version).

    This mirrors the fusion logic in TemporalEnhancedExVADv3.forward.
    """
    base_is_low = (base_score < base_low_threshold).float()
    base_is_high = (base_score > base_high_threshold).float()
    temporal_is_anom_confident = (temporal_score > temporal_high_conf).float()
    temporal_is_normal_confident = (temporal_score < temporal_low_conf).float()

    allow_boost = (1.0 - base_is_low) + (base_is_low * temporal_is_anom_confident)
    allow_suppress = base_is_high * temporal_is_normal_confident

    up_correction = torch.relu(temporal_score - base_score) * allow_boost
    down_correction = torch.relu(base_score - temporal_score) * allow_suppress

    combined_score = (
        base_score
        + fusion_weight * up_correction
        - (fusion_weight * down_correction_scale) * down_correction
    )
    combined_score = torch.clamp(combined_score, 0.0, 1.0)

    return {
        "combined_score": combined_score,
        "allow_boost": allow_boost,
        "allow_suppress": allow_suppress,
        "up_correction": up_correction,
        "down_correction": down_correction,
    }


def decision_threshold(mode: str) -> float:
    """
    A6: Decision policy used in the report.

    - balanced: default V5 threshold
    - guardrail: high-recall operating threshold
    """
    if mode == "guardrail":
        return 0.025
    return 0.44


def run_v5_sample_demo(data_root: str, num_frames: int, checkpoint_path: str, mode: str) -> None:
    """
    Run one end-to-end sample with labeled architecture outputs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A1
    frames, gt_label, class_name, video_path = load_one_test_sample(data_root, num_frames, device)

    # A2 (optional semantic baseline view)
    semantic_score, explanation, captions = semantic_branch_score(frames, device)

    # A3/A4/A5/A6 via full V5 forward
    v5_model = load_temporal_v5_model(device, checkpoint_path)

    with torch.no_grad():
        combined_score, enhanced_explanation, _, details = v5_model(frames, return_details=True)

    final_score = float(combined_score.view(-1)[0].item())
    threshold = decision_threshold(mode)
    pred = 1 if final_score > threshold else 0

    # Optional explicit standalone fusion recomputation (A5)
    base_score = details["base_score"]
    temporal_score = details["temporal_score"]
    fusion_weight = torch.tensor([[details["fusion_weight"]]], device=base_score.device)
    fusion_debug = bidirectional_confidence_gated_fusion(
        base_score=base_score,
        temporal_score=temporal_score,
        fusion_weight=fusion_weight,
        base_low_threshold=v5_model.base_low_threshold,
        base_high_threshold=v5_model.base_high_threshold,
        temporal_high_conf=v5_model.temporal_high_conf,
        temporal_low_conf=v5_model.temporal_low_conf,
        down_correction_scale=v5_model.down_correction_scale,
    )

    print("=" * 80)
    print("TEMPORAL V5 APPENDIX SAMPLE RUN")
    print("=" * 80)
    print("Architecture map:")
    for key, name in architecture_map().items():
        print(f"  {key}: {name}")

    print("\n[A1] Input Pipeline")
    print(f"  video_path: {video_path}")
    print(f"  class_name: {class_name}")
    print(f"  gt_label: {gt_label} ({'Anomaly' if gt_label == 1 else 'Normal'})")
    print(f"  num_frames_loaded: {len(frames)}")

    print("\n[A2] Semantic Branch")
    print(f"  semantic_score: {semantic_score:.4f}")
    print(f"  explanation_preview: {explanation[:140]}")
    if captions:
        print(f"  caption_preview: {str(captions[0])[:140]}")

    print("\n[A3] Shared Visual Features")
    frame_features = v5_model.extract_per_frame_features(frames)
    print(f"  frame_features_shape: {tuple(frame_features.shape)}")

    print("\n[A4] Temporal Branch")
    print(f"  temporal_score: {float(details['temporal_score'].view(-1)[0].item()):.4f}")

    print("\n[A5] Bidirectional Confidence-Gated Fusion")
    print(f"  base_score: {float(base_score.view(-1)[0].item()):.4f}")
    print(f"  fusion_weight: {float(details['fusion_weight']):.4f}")
    print(f"  allow_boost: {float(fusion_debug['allow_boost'].view(-1)[0].item()):.4f}")
    print(f"  allow_suppress: {float(fusion_debug['allow_suppress'].view(-1)[0].item()):.4f}")
    print(f"  up_correction: {float(fusion_debug['up_correction'].view(-1)[0].item()):.4f}")
    print(f"  down_correction: {float(fusion_debug['down_correction'].view(-1)[0].item()):.4f}")

    print("\n[A6] Decision Layer")
    print(f"  operating_mode: {mode}")
    print(f"  decision_threshold: {threshold:.3f}")
    print(f"  final_score: {final_score:.4f}")
    print(f"  prediction: {pred} ({'Anomaly' if pred == 1 else 'Normal'})")

    print("\nEnhanced explanation preview:")
    print(f"  {str(enhanced_explanation)[:220]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Appendix sample code for Temporal V5 architecture")
    parser.add_argument("--data_root", type=str, default="data", help="Dataset root")
    parser.add_argument("--num_frames", type=int, default=16, help="Frames per sample")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models_saved/temporal_v5_high_eff.pth",
        help="Path to Temporal V5 checkpoint",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="balanced",
        choices=["balanced", "guardrail"],
        help="Decision operating mode",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_v5_sample_demo(
        data_root=cli_args.data_root,
        num_frames=cli_args.num_frames,
        checkpoint_path=cli_args.checkpoint,
        mode=cli_args.mode,
    )
