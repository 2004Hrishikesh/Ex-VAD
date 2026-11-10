# Ex-VAD AI Agent Guide

## Architecture
- Ex-VAD pairs BLIP captions (`src/models/aegm.py`) with CLIP anomaly scoring (`src/models/madm.py`), wrapped by `src/models/exvad_model.py`.
- The enhanced stack (`src/models/enhanced_exvad_model.py`) adds cross-modal attention via `src/models/cross_modal_attention.py`, projecting DistilBERT text embeddings into CLIP's visual space.
- `EnhancedMADM.anomaly_head` already applies `Sigmoid`; baseline `SimpleMADM` outputs logits. Preserve this split unless you also update training losses.
- Captions/explanations feed back into anomaly scoring only in the enhanced path; baseline ignores text beyond logging.

## Data Handling
- `src/utils/data_loader_new.py` expects `data/videos/{train,test}` or legacy `data/{Train,Test}` folders containing per-class subdirs of videos or PNG frame sequences.
- Each dataset item returns a Python list of PIL frames; the custom collate keeps that structure, so training loops access `frames_batch[0]` and iterate over frames directly.
- Frame sampling relies on `src/utils/video_utils.sample_frames` (OpenCV → PIL) and keeps frame count fixed via padding/repeat.
- When you need tensors, call `src/utils/video_utils.frames_to_tensor`; most inference paths still accept raw frame lists.

## Training Workflows
- Baseline training: `python -m src.train --epochs N` from the repo root; it will auto-create `models_saved/best_model.pth` and `threshold_info.json`.
- Enhanced CMAF training: `python -m src.train_enhanced --epochs N --use_cross_modal_attention 1`; this script lowers `num_frames` to 8 and saves richer checkpoints.
- Both scripts assume GPU availability, aggressively clear CUDA cache, and set `batch_size=1`; keep those optimizations when extending the loops.
- Threshold tuning (`src/utils/threshold_tuning.py`) runs after training; downstream demos rely on the saved optimal threshold.

## Evaluation & Demos
- Quick sanity runs live in the root (`test_cmaf_model.py`, `test_enhanced_model.py`, `test_frame_fix.py`) and expect paths relative to the repo.
- Paper-quality comparisons use `python -m src.evaluate_cmaf_comprehensive`, which emits CSVs/plots into `outputs/` and compares CMAF vs baseline side by side.
- Demos (`src/demo.py`, `demo_cmaf.py`, `demo_extended.py`) read inference thresholds, process a handful of category samples, and persist summaries to `outputs/*.json`.
- `visualize_cmaf.py` (root) surfaces attention diagnostics by calling `EnhancedExVADModel.forward_with_attention_weights`.

## Conventions & Pitfalls
- Metric helpers in `src/utils/evaluation_metrics.py` expect probabilistic inputs; baseline flows wrap logits with `torch.sigmoid` just before metrics.
- Hugging Face downloads (BLIP, DistilBERT) and OpenAI CLIP weights are loaded at module init; cache them or stub in tests to avoid repeated network hits.
- Keep datasets balanced by honoring the class map in `VideoAnomalyDataset`; unknown folders default to anomaly (label 1).
- Many scripts mix relative imports (`from .models ...`) with CLI execution; prefer `python -m` invocations or ensure `PYTHONPATH=src` when running ad hoc utilities.
- GPU checks (`src/test_gpu.py`) gate long trainings; run them if you touch device handling logic.

## Environment
- Core dependencies are pinned in `requirements.txt` with CLIP installed from Git; update both the file and cached wheels if you bump versions.
- Models and history live under `models_saved/`; never rename `best_model.pth` without updating the demo/eval scripts that assume that filename.
- Outputs (plots, CSVs, JSON summaries) belong in `outputs/`; helper scripts expect that directory to exist and will create it if missing.

## When Unsure
- Review the markdown reports in the repo (e.g., `TRAINING_GUIDE.md`, `DEMO_RESULTS_ANALYSIS.md`) for expected behaviors before altering evaluation criteria.
- If a change touches both text and visual branches, cross-check with `EnhancedExVADModel.get_cross_modal_similarity` to maintain CMAF diagnostics.
- Ask for clarification whenever data layout or model checkpoints differ from the defaults above; downstream scripts are brittle to path changes.
