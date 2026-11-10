# Temporal extension (follow-up work, not in the published paper)

CMAF scores a video from mean-pooled frame features, which throws away the order of events. Shoplifting (approach, conceal, leave) and fights (contact, escalation) are sequences. This module adds a temporal branch next to the CMAF model and corrects its score only when the temporal evidence is decisive.

## Design

- **Shared input:** the same per-frame CLIP features CMAF already computes (16 x 512), so there is no extra feature-extraction cost.
- **Three encoders in parallel** ([`models/temporal_modeling.py`](models/temporal_modeling.py)):
  - Temporal Transformer (2 layers, 4 heads, sinusoidal positions, attention pooling) for long-range structure
  - BiLSTM (2 layers, 128 per direction) for direction of events
  - Motion encoder: frame-to-frame feature differences -> MLP -> BiLSTM, for short bursts
- **Confidence-gated fusion** ([`models/temporal_enhanced_exvad_v3.py`](models/temporal_enhanced_exvad_v3.py)):
  - boost the CMAF score when it is low but the temporal score is > 0.70
  - suppress it when it is high but the temporal score is < 0.30
  - otherwise leave it unchanged
  - fusion strength `alpha = 0.1 + 0.4 * sigmoid(theta)` is learned
- **Loss:** BCE with label smoothing, plus a feature-variance regularizer and a class-mean contrastive margin.
- **Training:** stage 1 trains the temporal branch with CMAF frozen. Stage 2 (from epoch 8) also fine-tunes a few CMAF adaptation layers at a lower learning rate.

## Results (UCF-Crime test split, 290 videos)

Recomputed from saved per-video scores in [`results/per_video_scores_test290.json`](results/per_video_scores_test290.json).

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | False alarms |
|---|---:|---:|---:|---:|---:|---:|
| ExVAD + CMAF | 0.828 | 0.781 | 0.893 | 0.833 | 0.909 | 35 |
| **+ Temporal (V5)** | **0.862** | **0.862** | 0.850 | **0.856** | **0.914** | **19** |
| + Temporal, frames shuffled | 0.852 | - | - | - | 0.908 | 21 |

- False alarms drop by 45.7% (35 -> 19). Most of that comes from the suppress gate.
- Shuffling frame order costs about 1 accuracy point, so the branch does use ordering.
- Encoder and gating ablations are in [`results/tables/`](results/tables/).
- **Limitation:** there is no separate validation split yet. The threshold (0.44) and the checkpoint epoch were chosen on the test split, so these numbers are optimistic.

## Run (from the repository root)

```bash
python temporal_extension/train_temporal_v5.py --init_checkpoint models_saved/<checkpoint>.pth
python temporal_extension/scripts/evaluate_temporal.py --checkpoint models_saved/temporal_v5_high_eff.pth \
    --threshold-file models_saved/temporal_v5_high_eff_threshold.json
python temporal_extension/scripts/run_paper_ablation.py --checkpoint models_saved/temporal_v5_high_eff.pth
```
