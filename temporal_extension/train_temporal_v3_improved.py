"""
Temporal V3 Adaptive Training Script

Implements the full improvement plan:
1. Bidirectional confidence-gated fusion in model forward.
2. Loss optimized on final fused score (plus temporal auxiliary loss).
3. Per-epoch threshold recalibration on validation data.
4. Selective base-model fine-tuning after warmup.
"""
import argparse
import json
import os
import sys
from collections import deque
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from temporal_extension.models.temporal_enhanced_exvad_v3 import TemporalEnhancedExVADv3
from src.utils.data_loader_new import create_dataloaders


class ImprovedTemporalTrainer:
    def __init__(self,
                 num_epochs=20,
                 lr=2e-4,
                 accumulation_steps=6,
                 warmup_epochs=4,
                 base_unfreeze_epoch=6,
                 num_frames=8,
                 patience=8,
                 init_checkpoint=None,
                 hard_scan_limit=None,
                 max_train_batches=None,
                 max_val_batches=None,
                 save_prefix='temporal_v3_adaptive',
                 save_every_epochs=0,
                 device='cuda'):

        self.num_epochs = num_epochs
        self.lr = lr
        self.accumulation_steps = accumulation_steps
        self.warmup_epochs = warmup_epochs
        self.base_unfreeze_epoch = base_unfreeze_epoch
        self.num_frames = num_frames
        self.patience = patience
        self.init_checkpoint = init_checkpoint
        self.hard_scan_limit = hard_scan_limit
        self.max_train_batches = max_train_batches
        self.max_val_batches = max_val_batches
        self.save_every_epochs = max(0, int(save_every_epochs))
        self.save_prefix = save_prefix
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.model_path = f'models_saved/{save_prefix}.pth'
        self.threshold_path = f'models_saved/{save_prefix}_threshold.json'
        self.history_path = f'models_saved/{save_prefix}_history.json'

        self.normal_embeddings = deque(maxlen=32)
        self.anomaly_embeddings = deque(maxlen=32)
        self.hard_examples = set()

        self.history = []
        self.current_threshold = 0.5
        self.base_finetune_enabled = False

    def _trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self._trainable_parameters())

    def _build_optimizer(self):
        self.optimizer = AdamW(self._trainable_parameters(), lr=self.lr, weight_decay=0.01)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=4, T_mult=2)

    def _add_new_trainable_params(self, lr_scale=0.2):
        existing_ids = {
            id(param)
            for group in self.optimizer.param_groups
            for param in group['params']
        }
        new_params = [
            p for p in self._trainable_parameters()
            if id(p) not in existing_ids
        ]
        if new_params:
            self.optimizer.add_param_group({
                'params': new_params,
                'lr': self.optimizer.param_groups[0]['lr'] * lr_scale,
                'weight_decay': 0.01,
            })
            print(f"   Added {sum(p.numel() for p in new_params):,} parameters for stage-2 fine-tuning")

    def setup(self):
        print("=" * 72)
        print("TEMPORAL V3 ADAPTIVE TRAINING")
        print("=" * 72)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.num_epochs}")
        print(f"LR: {self.lr}")
        print(f"Gradient accumulation: {self.accumulation_steps}")
        print(f"Warmup epochs: {self.warmup_epochs}")
        print(f"Selective base unfreeze epoch: {self.base_unfreeze_epoch}")
        print(f"Frames per sample: {self.num_frames}")
        if self.save_every_epochs > 0:
            print(f"Periodic checkpoint interval: every {self.save_every_epochs} epoch(s)")
        else:
            print("Periodic checkpoint interval: disabled")

        print("\n[1] Loading model...")
        self.model = TemporalEnhancedExVADv3(device=self.device, freeze_base=True).to(self.device)

        if self.init_checkpoint and os.path.exists(self.init_checkpoint):
            print(f"   Loading initialization checkpoint: {self.init_checkpoint}")
            checkpoint = torch.load(self.init_checkpoint, map_location='cpu', weights_only=False)
            state = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
            load_result = self.model.load_state_dict(state, strict=False)
            missing = len(getattr(load_result, 'missing_keys', []))
            unexpected = len(getattr(load_result, 'unexpected_keys', []))
            print(f"   Checkpoint loaded (missing={missing}, unexpected={unexpected})")

        self._build_optimizer()
        self.bce_logits = nn.BCEWithLogitsLoss()
        print(f"   Initial trainable params: {self._trainable_parameter_count():,}")

        print("\n[2] Loading data...")
        self.train_loader, self.val_loader = create_dataloaders(
            data_root='data',
            batch_size=1,
            num_frames=self.num_frames,
            num_workers=0,
        )
        print(f"   Train samples: {len(self.train_loader.dataset)}")
        print(f"   Val samples: {len(self.val_loader.dataset)}")

        print("\n[3] Finding hard examples...")
        self.find_hard_examples()
        print(f"   Hard examples found: {len(self.hard_examples)}")

    def maybe_enable_stage2_finetuning(self, epoch_num: int):
        if self.base_finetune_enabled:
            return
        if epoch_num < self.base_unfreeze_epoch:
            return

        print("\n[Stage-2] Enabling selective base fine-tuning...")
        self.model.unfreeze_base_adaptation_layers()
        self._add_new_trainable_params(lr_scale=0.2)
        self.base_finetune_enabled = True
        print(f"   Total trainable params after unfreeze: {self._trainable_parameter_count():,}")

    def find_hard_examples(self):
        self.model.eval()
        threshold = 0.149

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc='Hard-example scan')):
                if self.hard_scan_limit is not None and batch_idx >= self.hard_scan_limit:
                    break

                frames_list, labels, _, _ = batch
                frames = frames_list[0] if isinstance(frames_list[0], list) else frames_list
                label = int(labels[0].item())

                if len(frames) == 0:
                    continue

                try:
                    result = self.model(frames, return_details=True)
                    base_score = float(result[3]['base_score'].view(-1)[0].item())
                    base_pred = int(base_score > threshold)
                    if base_pred != label:
                        self.hard_examples.add(batch_idx)
                except Exception:
                    continue

        self.model.train()

    def compute_contrastive_loss(self, embedding: torch.Tensor, label: int) -> torch.Tensor:
        embedding = embedding.view(1, -1)
        embedding = F.normalize(embedding, dim=-1).squeeze(0)

        if label == 0:
            self.normal_embeddings.append(embedding.detach())
        else:
            self.anomaly_embeddings.append(embedding.detach())

        if len(self.normal_embeddings) < 4 or len(self.anomaly_embeddings) < 4:
            return torch.tensor(0.0, device=self.device)

        if label == 0:
            positives = torch.stack(list(self.normal_embeddings)[-4:])
            negatives = torch.stack(list(self.anomaly_embeddings)[-4:])
        else:
            positives = torch.stack(list(self.anomaly_embeddings)[-4:])
            negatives = torch.stack(list(self.normal_embeddings)[-4:])

        pos_proto = F.normalize(positives.mean(dim=0, keepdim=True), dim=-1)
        neg_proto = F.normalize(negatives.mean(dim=0, keepdim=True), dim=-1)

        emb = embedding.unsqueeze(0)
        pos_sim = F.cosine_similarity(emb, pos_proto, dim=-1)
        neg_sim = F.cosine_similarity(emb, neg_proto, dim=-1)

        margin = 0.35
        return F.relu(neg_sim - pos_sim + margin).mean()

    def compute_losses(self, result, label: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        combined_score = torch.clamp(result[0].view(-1), 1e-4, 1.0 - 1e-4)
        details = result[3]
        temporal_logits = details['temporal_logits'].view(-1)
        temporal_features = details['temporal_features'].view(1, -1)

        label_val = float(label.item())
        smooth_label = label_val * 0.95 + 0.025
        target_probs = torch.full_like(combined_score, smooth_label)

        weight = 2.0 if batch_idx in self.hard_examples else 1.0

        combined_bce = weight * F.binary_cross_entropy(combined_score, target_probs)
        temporal_bce = weight * self.bce_logits(temporal_logits, target_probs)
        contrastive = self.compute_contrastive_loss(temporal_features, int(round(label_val)))

        total = 0.60 * combined_bce + 0.30 * temporal_bce + 0.10 * contrastive
        return {
            'total': total,
            'combined_bce': combined_bce,
            'temporal_bce': temporal_bce,
            'contrastive': contrastive,
            'combined_score': combined_score.detach(),
        }

    def train_epoch(self, epoch: int):
        self.model.train()

        total_loss = 0.0
        total_combined_bce = 0.0
        total_temporal_bce = 0.0
        total_contrastive = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad()
        accumulated = 0.0
        error_count = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs}')

        for batch_idx, batch in enumerate(pbar):
            if self.max_train_batches is not None and batch_idx >= self.max_train_batches:
                break

            frames_list, labels, _, _ = batch
            frames = frames_list[0] if isinstance(frames_list[0], list) else frames_list
            label = labels[0].float().to(self.device)

            if len(frames) == 0:
                continue

            try:
                result = self.model(frames, return_details=True)
                losses = self.compute_losses(result, label, batch_idx)

                step_loss = losses['total'] / self.accumulation_steps
                step_loss.backward()
                accumulated += step_loss.item()

                total_combined_bce += float(losses['combined_bce'].item())
                total_temporal_bce += float(losses['temporal_bce'].item())
                total_contrastive += float(losses['contrastive'].item())

                pred = int(losses['combined_score'].item() > self.current_threshold)
                correct += int(pred == int(label.item()))
                total += 1

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    total_loss += accumulated
                    accumulated = 0.0

                pbar.set_postfix({
                    'loss': f"{total_loss:.4f}",
                    'cmb_bce': f"{total_combined_bce / max(total, 1):.4f}",
                    'tmp_bce': f"{total_temporal_bce / max(total, 1):.4f}",
                    'ctr': f"{total_contrastive / max(total, 1):.4f}",
                    'acc': f"{correct / max(total, 1):.4f}",
                })
            except Exception as exc:
                error_count += 1
                if error_count <= 3:
                    print(f"\nTrain batch {batch_idx} error: {exc}")
                continue

        if accumulated > 0:
            torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += accumulated

        if total == 0:
            raise RuntimeError('No successful training batches. Check printed batch errors.')

        self.scheduler.step()

        return {
            'train_loss': total_loss,
            'train_combined_bce': total_combined_bce / max(total, 1),
            'train_temporal_bce': total_temporal_bce / max(total, 1),
            'train_contrastive': total_contrastive / max(total, 1),
            'train_acc': correct / max(total, 1),
        }

    @staticmethod
    def _f1_metrics(preds: np.ndarray, labels: np.ndarray):
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, f1

    def validate(self):
        self.model.eval()

        all_scores = []
        all_base_scores = []
        all_temporal_scores = []
        all_labels = []
        error_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validating')):
                if self.max_val_batches is not None and batch_idx >= self.max_val_batches:
                    break

                frames_list, labels, _, _ = batch
                frames = frames_list[0] if isinstance(frames_list[0], list) else frames_list
                label = int(labels[0].item())

                if len(frames) == 0:
                    continue

                try:
                    result = self.model(frames, return_details=True)
                    score = float(result[0].view(-1)[0].item())
                    base_score = float(result[3]['base_score'].view(-1)[0].item())
                    temporal_score = float(result[3]['temporal_score'].view(-1)[0].item())

                    all_scores.append(score)
                    all_base_scores.append(base_score)
                    all_temporal_scores.append(temporal_score)
                    all_labels.append(label)
                except Exception as exc:
                    error_count += 1
                    if error_count <= 3:
                        print(f"\nValidation batch {batch_idx} error: {exc}")
                    continue

        if not all_scores:
            return {
                'val_acc': 0.0,
                'val_precision': 0.0,
                'val_recall': 0.0,
                'val_f1': 0.0,
                'best_threshold': self.current_threshold,
                'normal_mean': 0.0,
                'anomaly_mean': 0.0,
                'base_mean': 0.0,
                'temporal_mean': 0.0,
                'score_gap': 0.0,
            }

        all_scores = np.array(all_scores)
        all_base_scores = np.array(all_base_scores)
        all_temporal_scores = np.array(all_temporal_scores)
        all_labels = np.array(all_labels)

        best = {
            'threshold': self.current_threshold,
            'f1': -1.0,
            'precision': 0.0,
            'recall': 0.0,
            'acc': 0.0,
        }

        for thresh in np.arange(0.05, 0.96, 0.01):
            preds = (all_scores > thresh).astype(int)
            precision, recall, f1 = self._f1_metrics(preds, all_labels)
            acc = float((preds == all_labels).mean())
            if f1 > best['f1']:
                best = {
                    'threshold': float(thresh),
                    'f1': float(f1),
                    'precision': float(precision),
                    'recall': float(recall),
                    'acc': float(acc),
                }

        normal_scores = all_scores[all_labels == 0]
        anomaly_scores = all_scores[all_labels == 1]

        normal_mean = float(normal_scores.mean()) if len(normal_scores) > 0 else 0.0
        anomaly_mean = float(anomaly_scores.mean()) if len(anomaly_scores) > 0 else 0.0

        return {
            'val_acc': best['acc'],
            'val_precision': best['precision'],
            'val_recall': best['recall'],
            'val_f1': best['f1'],
            'best_threshold': best['threshold'],
            'normal_mean': normal_mean,
            'anomaly_mean': anomaly_mean,
            'base_mean': float(all_base_scores.mean()),
            'temporal_mean': float(all_temporal_scores.mean()),
            'score_gap': anomaly_mean - normal_mean,
        }

    def train(self):
        self.setup()

        best_f1 = -1.0
        no_improve = 0

        print("\n" + "=" * 72)
        print("TRAINING")
        print("=" * 72)

        for epoch in range(self.num_epochs):
            epoch_num = epoch + 1
            self.maybe_enable_stage2_finetuning(epoch_num)

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            self.current_threshold = val_metrics['best_threshold']

            metrics = {
                'epoch': epoch_num,
                'lr': float(self.scheduler.get_last_lr()[0]),
                'stage2_enabled': self.base_finetune_enabled,
                **train_metrics,
                **val_metrics,
            }
            self.history.append(metrics)

            checkpoint_payload = {
                'epoch': epoch_num,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_f1': val_metrics['val_f1'],
                'val_acc': val_metrics['val_acc'],
                'best_threshold': val_metrics['best_threshold'],
                'score_gap': val_metrics['score_gap'],
                'architecture': 'TemporalEnhancedExVADv3Adaptive',
                'stage2_enabled': self.base_finetune_enabled,
            }

            print(f"\nEpoch {epoch_num}/{self.num_epochs}")
            print(
                f"  Train: loss={train_metrics['train_loss']:.4f}, "
                f"acc={train_metrics['train_acc']:.4f}, "
                f"cmb_bce={train_metrics['train_combined_bce']:.4f}, "
                f"tmp_bce={train_metrics['train_temporal_bce']:.4f}"
            )
            print(
                f"  Val: acc={val_metrics['val_acc']:.4f}, "
                f"prec={val_metrics['val_precision']:.4f}, "
                f"rec={val_metrics['val_recall']:.4f}, "
                f"f1={val_metrics['val_f1']:.4f}, "
                f"thr={val_metrics['best_threshold']:.3f}"
            )
            print(
                f"  Means: normal={val_metrics['normal_mean']:.3f}, "
                f"anomaly={val_metrics['anomaly_mean']:.3f}, "
                f"base={val_metrics['base_mean']:.3f}, "
                f"temporal={val_metrics['temporal_mean']:.3f}, "
                f"gap={val_metrics['score_gap']:.3f}"
            )

            if val_metrics['val_f1'] > best_f1:
                best_f1 = val_metrics['val_f1']
                no_improve = 0

                torch.save(checkpoint_payload, self.model_path)

                with open(self.threshold_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'best_threshold': val_metrics['best_threshold'],
                        'val_f1': val_metrics['val_f1'],
                        'val_acc': val_metrics['val_acc'],
                        'val_precision': val_metrics['val_precision'],
                        'val_recall': val_metrics['val_recall'],
                    }, f, indent=2)

                print(f"  New best model saved with F1={best_f1:.4f}")
            else:
                no_improve += 1
                print(f"  No improvement for {no_improve} epoch(s)")

            if self.save_every_epochs > 0 and epoch_num % self.save_every_epochs == 0:
                periodic_path = f"models_saved/{self.save_prefix}_epoch_{epoch_num:03d}.pth"
                torch.save(checkpoint_payload, periodic_path)
                print(f"  Periodic checkpoint saved: {periodic_path}")

            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2)

            if no_improve >= self.patience:
                print(f"\nEarly stopping triggered at epoch {epoch_num}")
                break

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        with open(self.history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)

        print("\n" + "=" * 72)
        print("TRAINING COMPLETE")
        print("=" * 72)
        print(f"Best F1: {best_f1:.4f}")
        print(f"Checkpoint: {self.model_path}")
        print(f"Threshold file: {self.threshold_path}")

        return best_f1


def parse_args():
    parser = argparse.ArgumentParser(description='Train Temporal V3 Adaptive model')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--accumulation_steps', type=int, default=6)
    parser.add_argument('--warmup_epochs', type=int, default=4)
    parser.add_argument('--base_unfreeze_epoch', type=int, default=6)
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--init_checkpoint', type=str, default='models_saved/temporal_v3_improved.pth')
    parser.add_argument('--hard_scan_limit', type=int, default=300)
    parser.add_argument('--max_train_batches', type=int, default=None)
    parser.add_argument('--max_val_batches', type=int, default=None)
    parser.add_argument('--save_prefix', type=str, default='temporal_v3_adaptive')
    parser.add_argument('--save_every_epochs', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
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


if __name__ == '__main__':
    main()
