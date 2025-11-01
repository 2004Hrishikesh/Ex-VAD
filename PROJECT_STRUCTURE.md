# Ex-VAD CMAF Project Structure

## Overview
This document provides a complete structural overview of the Ex-VAD with Cross-Modal Attention Fusion (CMAF) project for comprehensive report preparation.

## Core Implementation Files

### Model Architecture (`src/models/`)
- **`cross_modal_attention.py`** - Core CMAF implementation
  - `CrossModalAttentionFusion` class
  - Bidirectional attention (Visual→Text, Text→Visual)
  - Learnable gating mechanism for fusion
  - Text projection from 768D (DistilBERT) to 512D (CLIP visual space)

- **`enhanced_exvad_model.py`** - Enhanced CMAF-integrated model
  - Combines AEGM (BLIP) + MADM (CLIP) + CMAF
  - Enhanced anomaly detection head with sigmoid activation
  - Forward methods with attention weight extraction
  - Cross-modal similarity computation

- **`aegm.py`** - Anomaly Explanation Generation Module
  - BLIP-based image caption generation
  - Generates natural language explanations for detected anomalies
  
- **`madm.py`** - Motion Anomaly Detection Module  
  - CLIP ViT-B/32 visual feature extraction (512D embeddings)
  - Baseline anomaly detection head (logits output)

- **`exvad_model.py`** - Baseline ExVAD model
  - Original model without CMAF
  - BLIP captions + CLIP features
  - For comparison with CMAF-enhanced version

- **`temporal_cross_modal_attention.py`** - Temporal extension (experimental)
  - Temporal pooling across frame sequences
  - Extended CMAF with temporal dynamics

### Data Handling (`src/utils/`)
- **`data_loader_new.py`** - Dataset loading utilities
  - Supports both video files (.mp4, .avi, etc.) and PNG frame sequences
  - UCF-Crime dataset handling
  - Returns list of PIL frames for processing
  
- **`video_utils.py`** - Video/frame processing
  - `sample_frames()` - Uniform temporal sampling from video/sequence
  - `frames_to_tensor()` - Convert PIL frames to PyTorch tensors
  - Frame preprocessing and normalization

- **`evaluation_metrics.py`** - Comprehensive metrics
  - ROC-AUC, PR-AUC calculation
  - Precision, Recall, F1 Score
  - Confusion matrix generation
  - Per-category performance analysis
  - Visualization plotting

- **`threshold_tuning.py`** - Optimal threshold search
  - Grid search across threshold values
  - Per-category threshold optimization
  - Balance normal/anomaly tradeoffs

### Training Scripts (`src/`)
- **`train.py`** - Baseline ExVAD training
  - Standard training loop
  - No cross-modal attention
  - Outputs logits (requires sigmoid for probabilities)

- **`train_enhanced.py`** - CMAF-enhanced training
  - Integrates cross-modal attention fusion
  - Enhanced anomaly head
  - Probability outputs (sigmoid in model)
  - Recommended for new training runs

### Demo & Evaluation Scripts (root)
- **`demo_cmaf.py`** (if exists) - Quick demo script
  - Tests saved CMAF checkpoint
  - Supports video files and PNG sequences
  - Quick evaluation on sample data

- **`test_enhanced_model.py`** - Model testing utilities
  - Load and test enhanced CMAF models
  - Evaluation harness

- **`visualize_cmaf.py`** - Visualization tools
  - Attention weight visualization
  - Cross-modal similarity heatmaps
  - Result plotting

## Documentation Files

### Technical Documentation
- **`CMAF_EXPLANATION.md`** - Detailed CMAF architecture explanation
  - Mathematical formulation
  - Architecture diagrams
  - Implementation details
  - Design rationale

- **`IMPLEMENTATION_ROADMAP.md`** - Development roadmap
  - Feature implementation timeline
  - Milestones achieved
  - Future enhancements

- **`ADVANCED_RESEARCH_DIRECTIONS.md`** - Research extensions
  - Potential improvements
  - Novel research directions
  - Integration with other techniques

- **`CMAF_Architecture.png`** - Architecture diagram
  - Visual representation of CMAF model
  - Data flow illustration

### Project Management
- **`Readme.md`** - Project README (cleaned, concise)
  - Quick start guide
  - Installation instructions
  - Usage examples
  - Key results summary

- **`requirements.txt`** - Python dependencies
  - PyTorch 2.0+
  - Transformers 4.35.2
  - CLIP (from GitHub)
  - Other dependencies

- **`.gitignore`** - Git exclusions
  - Excludes: data/, models_saved/, outputs/
  - Virtual environments
  - Temporary files

- **`.github/copilot-instructions.md`** - AI agent guidance
  - Development guidelines
  - Code standards
  - Common operations

## Data Structure (Excluded from repo)

```
data/
├── Train/
│   ├── Abuse/
│   ├── Arrest/
│   ├── Arson/
│   ├── Assault/
│   ├── Burglary/
│   ├── Explosion/
│   ├── Fighting/
│   ├── NormalVideos/
│   ├── RoadAccidents/
│   ├── Robbery/
│   ├── Shooting/
│   ├── Shoplifting/
│   ├── Stealing/
│   └── Vandalism/
└── Test/
    └── [same category structure]
```

Note: Contains video files (.mp4) or PNG frame sequences grouped by category.

## Model Checkpoints (Excluded from repo)

```
models_saved/
├── best_model.pth              # Best CMAF checkpoint (~77.24% accuracy)
├── baseline_best_model.pth     # Best baseline checkpoint (~51.38% accuracy)
├── best_model_bak.pth          # Backup CMAF checkpoint
└── best_model_old_architecture.pth  # Legacy checkpoint
```

Note: Use `best_model.pth` for CMAF demos and evaluations.

## Output Artifacts (Excluded from repo)

```
outputs/
├── demo_results.json           # Demo run metrics
├── demo_visualization.png      # Visual results
├── model_comparison_results.json  # Checkpoint comparison
├── overall_comparison.csv      # CMAF vs Baseline comparison
├── category_comparison.csv     # Per-category performance
└── [various evaluation plots]
```

## Key Architecture Components

### CMAF Pipeline
```
Video Frames
    ↓
CLIP (ViT-B/32) → Visual Features (512D)
                      ↓
                Cross-Modal Attention Fusion
                      ↑
BLIP Captions → DistilBERT → Text Embeddings (768D) → Projection (512D)
                      ↓
            Enhanced Features → Anomaly Head → Probability Score
```

### Model Variants
1. **Baseline ExVAD**: BLIP captions + CLIP features → Simple anomaly head (logits)
2. **Enhanced CMAF**: BLIP + CLIP + Cross-Modal Fusion → Enhanced head (probabilities)

### Key Differences
| Feature | Baseline | Enhanced CMAF |
|---------|----------|---------------|
| Text Embedding | Not fused | Projected & fused |
| Attention | None | Bidirectional + gating |
| Output | Logits | Probabilities |
| Anomaly Head | Single layer | Multi-layer + LayerNorm |
| Performance | ~51% overall | ~77% overall |

## Usage Examples

### Training
```powershell
# Baseline
python -m src.train --epochs 10

# Enhanced CMAF
python -m src.train_enhanced --epochs 4 --use_cross_modal_attention 1
```

### Demo/Evaluation
```powershell
# Quick demo
python demo_cmaf.py

# Full evaluation
python scripts/evaluate_saved_models.py
python -m src.evaluate_cmaf_comprehensive
```

### Threshold Tuning
```powershell
python -m src.utils.threshold_tuning --checkpoint models_saved/best_model.pth
```

## Performance Summary

### Demo Results (41 samples)
- Overall Accuracy: **87.80%** (36/41)
- Anomaly Detection: **92.11%** (35/38)
- Normal Detection: 33.33% (1/3)
- Precision: 0.9459 | Recall: 0.9211 | F1: 0.9333

### Full Test Set (290 samples, threshold=0.1)
- Overall: **77.24%** | Normal: **76.67%** | Anomaly: **77.86%**
- Precision: **75.69%** | Recall: **77.86%** | F1: **76.76%**
- Best threshold: 0.1 (recommended)

### Comparison vs Baseline
| Metric | Baseline | Enhanced CMAF | Δ Improvement |
|--------|----------|---------------|---------------|
| Overall Acc | 51.38% | 77.24% | +25.86% |
| F1 Score | 62.80% | 76.76% | +13.96% |
| Normal Acc | 20.00% | 76.67% | +56.67% |
| Anomaly Acc | 85.00% | 77.86% | -7.14% |

Note: CMAF significantly improves normal video detection while maintaining strong anomaly detection.

## Repository Branches
- **`main`** - Initial baseline implementation
- **`cmaf-research`** - CMAF development and experiments
- **`readme-update`** - Documentation updates (current)

## For Report Preparation

### Essential Files to Review
1. **Architecture**: `src/models/cross_modal_attention.py`, `src/models/enhanced_exvad_model.py`
2. **Training**: `src/train_enhanced.py`
3. **Evaluation**: `src/utils/evaluation_metrics.py`
4. **Documentation**: `CMAF_EXPLANATION.md`, `CMAF_Architecture.png`
5. **Results**: `outputs/` artifacts (saved locally, not in repo)

### Key Metrics to Highlight
- Overall accuracy: 77.24% (26% improvement over baseline)
- Balanced performance: 76.67% normal, 77.86% anomaly
- F1 score: 76.76%
- Threshold recommendation: 0.1

### Novel Contributions
1. Cross-Modal Attention Fusion between CLIP visual and BLIP text
2. Learnable gating for adaptive fusion
3. Bidirectional attention (Visual→Text, Text→Visual)
4. Enhanced anomaly head with layer normalization
5. Support for both video files and frame sequences

## Contact & Repository
- **Repository**: [2004Hrishikesh/Ex-VAD](https://github.com/2004Hrishikesh/Ex-VAD)
- **Branch**: `readme-update` (latest documentation)
- **License**: MIT

---

*Last Updated: November 2025*
*Generated for CMAF comprehensive report preparation*
