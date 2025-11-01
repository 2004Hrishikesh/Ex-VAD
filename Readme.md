# Ex-VAD — Explainable Video Anomaly Detection (CMAF)

This repository implements Ex-VAD and an enhanced Cross-Modal Attention Fusion (CMAF) variant that fuses CLIP
visual features with BLIP-generated textual explanations. The repo includes training, demo and evaluation scripts
for fast experimentation and comparison between the baseline ExVAD and the CMAF-enhanced model.

Quick start

1. Create and activate a Python virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Train (examples):

```powershell
python -m src.train --epochs 10
python -m src.train_enhanced --epochs 4 --use_cross_modal_attention 1
```

3. Quick demo (uses `models_saved/best_model.pth` by default):

```powershell
python demo_cmaf.py
```

4. Compare saved checkpoints / run full eval:

```powershell
python scripts/evaluate_saved_models.py
python -m src.evaluate_cmaf_comprehensive
```

Files of interest
- `src/models/cross_modal_attention.py` — CMAF implementation
- `src/models/enhanced_exvad_model.py` — enhanced CMAF wrapper
- `src/models/aegm.py` — BLIP caption generator
- `src/models/madm.py` — CLIP-based motion anomaly module
- `src/utils/data_loader_new.py` — supports both video files and PNG frame sequences
- `src/utils/video_utils.py` — `sample_frames` / `frames_to_tensor`
- `scripts/evaluate_saved_models.py` — evaluation harness for saved checkpoints

Results (selected)
- Demo (small sample set): Overall  87.8% (36/41) — precision/recall/F1  0.94/0.92/0.93
- Full test (290 samples, threshold=0.1): Enhanced CMAF `best_model.pth`  77.24% overall (F1  0.768)

Notes
- Baseline outputs logits — apply `torch.sigmoid` before interpreting as probabilities.
- Recommended checkpoint: `models_saved/best_model.pth` (enhanced CMAF).
- Evaluation outputs and plots are saved under `outputs/`.

Want next? I can: (a) create a short README_quick.md, (b) add a minimal Dockerfile for reproducible runs, or (c) add per-category threshold tuning scripts — tell me which and I'll add it.
