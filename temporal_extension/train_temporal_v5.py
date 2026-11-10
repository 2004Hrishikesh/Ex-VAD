"""
Temporal V5 High-Efficiency Training Launcher.

This script uses the adaptive Temporal V3 trainer with V5-style defaults:
- Longer training (default 50 epochs)
- Larger effective batch size via accumulation (default 8)
- More frames per sample (default 16)
- Extended patience (default 15)
- Continuation from latest strong checkpoint
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from temporal_extension.train_temporal_v3_improved import ImprovedTemporalTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Temporal V5 High-Efficiency model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--base_unfreeze_epoch", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default="models_saved/temporal_v3_adaptive_full_cont.pth",
    )
    parser.add_argument("--hard_scan_limit", type=int, default=400)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--save_prefix", type=str, default="temporal_v5_high_eff")
    parser.add_argument("--save_every_epochs", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.init_checkpoint and not os.path.exists(args.init_checkpoint):
        raise FileNotFoundError(
            f"Initialization checkpoint not found: {args.init_checkpoint}"
        )

    print("=" * 72)
    print("TEMPORAL V5 HIGH-EFFICIENCY RUN")
    print("=" * 72)
    print(f"epochs={args.epochs}")
    print(f"lr={args.lr}")
    print(f"accumulation_steps={args.accumulation_steps}")
    print(f"num_frames={args.num_frames}")
    print(f"patience={args.patience}")
    print(f"base_unfreeze_epoch={args.base_unfreeze_epoch}")
    print(f"init_checkpoint={args.init_checkpoint}")
    print(f"save_prefix={args.save_prefix}")
    print(f"save_every_epochs={args.save_every_epochs}")

    trainer = ImprovedTemporalTrainer(
        num_epochs=args.epochs,
        lr=args.lr,
        accumulation_steps=args.accumulation_steps,
        warmup_epochs=args.warmup_epochs,
        base_unfreeze_epoch=args.base_unfreeze_epoch,
        num_frames=args.num_frames,
        patience=args.patience,
        init_checkpoint=args.init_checkpoint,
        hard_scan_limit=args.hard_scan_limit,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        save_prefix=args.save_prefix,
        save_every_epochs=args.save_every_epochs,
    )
    trainer.train()


if __name__ == "__main__":
    main()
