# Temporal extension

CMAF averages the frame features of a video, so it cannot tell in what order things happened. Many UCF-Crime classes are sequences, like shoplifting (approach, conceal, leave) or a fight that escalates. This branch models the frame order and corrects the CMAF score when it has strong evidence.

## Model

- **Input:** the same 16 x 512 per-frame CLIP features CMAF already computes.
- **Encoders** (`models/temporal_modeling.py`):
  - Transformer: 2 layers, 4 heads, sinusoidal positions, attention pooling
  - BiLSTM: 2 layers, 128 units per direction
  - Motion encoder: differences between consecutive frame features, an MLP, then a BiLSTM
- **Fusion** (`models/temporal_enhanced_exvad_v3.py`):
  - the encoder outputs are normalized, concatenated and scored
  - if the CMAF score is low and the temporal score is above 0.70, the score is raised
  - if the CMAF score is high and the temporal score is below 0.30, it is lowered
  - otherwise it is left unchanged
  - the correction strength is learned and stays between 0.1 and 0.5
- **Loss:** BCE with label smoothing, a feature-variance term and a contrastive margin between class means.
- **Training:** the temporal branch is trained first with CMAF frozen. A few CMAF projection layers are unfrozen from epoch 8 at a lower learning rate.

## Results

UCF-Crime test split, 290 videos. Per-video scores are in `results/per_video_scores_test290.json`.

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | False alarms |
|---|---:|---:|---:|---:|---:|---:|
| ExVAD + CMAF | 0.828 | 0.781 | 0.893 | 0.833 | 0.909 | 35 |
| + Temporal | 0.862 | 0.862 | 0.850 | 0.856 | 0.914 | 19 |
| + Temporal, frames shuffled | 0.852 | | | | 0.908 | 21 |

- **False alarms:** they drop from 35 to 19, mostly because of the lowering rule.
- **Frame order:** shuffling the frames costs 1 point of accuracy.
- **Low threshold:** at a threshold of 0.025 the same checkpoint reaches 95% recall.
- **Selection:** the 0.44 threshold and the checkpoint epoch were picked on this split.
- **Ablations:** encoder and gating ablations are in `results/tables/`.

## Usage

Run from the repository root:

```bash
python temporal_extension/train_temporal_v5.py --init_checkpoint models_saved/<checkpoint>.pth
python temporal_extension/scripts/evaluate_temporal.py --checkpoint models_saved/temporal_v5_high_eff.pth --threshold-file models_saved/temporal_v5_high_eff_threshold.json
python temporal_extension/scripts/run_paper_ablation.py --checkpoint models_saved/temporal_v5_high_eff.pth
```
