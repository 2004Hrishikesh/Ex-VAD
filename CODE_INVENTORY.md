# Ex-VAD CMAF - Complete Code Inventory

## Purpose
This document catalogs all code files in the repository to facilitate comprehensive report preparation. All files listed here are tracked in git (except dataset and model checkpoints which are excluded).

---

## Source Code Files

### Core Model Implementations (`src/models/`)

#### 1. `src/models/cross_modal_attention.py`
**Purpose**: Core CMAF (Cross-Modal Attention Fusion) implementation  
**Key Classes/Functions**:
- `CrossModalAttentionFusion` class
  - `__init__()` - Initialize text projection, attention layers, gating
  - `encode_text()` - Project DistilBERT embeddings to CLIP space
  - `forward()` - Bidirectional attention + gated fusion
**Lines of Code**: ~150-200  
**Dependencies**: PyTorch, Transformers (DistilBERT)

#### 2. `src/models/enhanced_exvad_model.py`
**Purpose**: Enhanced ExVAD model with CMAF integration  
**Key Classes/Functions**:
- `EnhancedExVADModel` class
  - Combines AEGM (BLIP) + MADM (CLIP) + CMAF
  - Enhanced anomaly detection head (multi-layer + LayerNorm + Sigmoid)
  - `forward()` - Full forward pass with cross-modal fusion
  - `forward_with_attention_weights()` - Return attention for visualization
  - `get_cross_modal_similarity()` - Compute similarity scores
**Lines of Code**: ~200-250  
**Dependencies**: PyTorch, CLIP, BLIP, DistilBERT, CMAF

#### 3. `src/models/aegm.py`
**Purpose**: Anomaly Explanation Generation Module (BLIP-based)  
**Key Classes/Functions**:
- `SimpleAEGM` class
  - `generate_captions()` - Generate image captions
  - `generate_explanation()` - Create anomaly explanations
**Lines of Code**: ~100-150  
**Dependencies**: Transformers (BLIP)

#### 4. `src/models/madm.py`
**Purpose**: Motion Anomaly Detection Module (CLIP-based baseline)  
**Key Classes/Functions**:
- `SimpleMADM` class
  - CLIP ViT-B/32 feature extraction (512D)
  - Baseline anomaly head (logits output)
**Lines of Code**: ~80-120  
**Dependencies**: OpenAI CLIP

#### 5. `src/models/exvad_model.py`
**Purpose**: Baseline ExVAD model (no CMAF)  
**Key Classes/Functions**:
- `ExVADModel` class
  - Combines AEGM + MADM without cross-modal attention
  - For comparison with CMAF-enhanced version
**Lines of Code**: ~150-180  
**Dependencies**: PyTorch, CLIP, BLIP

#### 6. `src/models/temporal_cross_modal_attention.py`
**Purpose**: Temporal extension of CMAF (experimental)  
**Key Classes/Functions**:
- Temporal pooling across frame sequences
- Extended CMAF with temporal dynamics
**Lines of Code**: ~120-150  
**Dependencies**: PyTorch, CMAF

#### 7. `src/models/__init__.py`
**Purpose**: Package initialization  
**Content**: Module exports

---

### Utilities (`src/utils/`)

#### 8. `src/utils/data_loader_new.py`
**Purpose**: Dataset loading and preprocessing  
**Key Classes/Functions**:
- `VideoDataset` class
  - Supports video files (.mp4, .avi, etc.)
  - Supports PNG frame sequences (grouped by name)
  - Returns list of PIL Image frames
  - UCF-Crime dataset handling (14 categories)
- Custom collate function for batching
**Lines of Code**: ~200-250  
**Dependencies**: PyTorch, PIL, OpenCV

#### 9. `src/utils/video_utils.py`
**Purpose**: Video/frame processing utilities  
**Key Functions**:
- `sample_frames()` - Uniform temporal sampling (default: 16 frames)
- `frames_to_tensor()` - Convert PIL frames to PyTorch tensors
- `preprocess_frame()` - Normalize and resize frames
**Lines of Code**: ~80-120  
**Dependencies**: PyTorch, PIL, torchvision

#### 10. `src/utils/evaluation_metrics.py`
**Purpose**: Comprehensive evaluation metrics and visualization  
**Key Functions**:
- `calculate_metrics()` - Precision, Recall, F1, Accuracy
- `compute_roc_auc()` - ROC curve and AUC
- `compute_pr_auc()` - Precision-Recall curve and AUC
- `plot_confusion_matrix()` - Confusion matrix visualization
- `plot_roc_curve()` - ROC curve plotting
- `plot_per_category_performance()` - Category-wise analysis
**Lines of Code**: ~250-300  
**Dependencies**: scikit-learn, matplotlib, numpy

#### 11. `src/utils/threshold_tuning.py`
**Purpose**: Optimal threshold search and selection  
**Key Functions**:
- `find_optimal_threshold()` - Grid search for best threshold
- `evaluate_at_threshold()` - Compute metrics at specific threshold
- `per_category_thresholds()` - Category-specific threshold optimization
**Lines of Code**: ~100-150  
**Dependencies**: numpy, scikit-learn

#### 12. `src/utils/__init__.py`
**Purpose**: Package initialization  
**Content**: Module exports

---

### Training Scripts (`src/`)

#### 13. `src/train.py`
**Purpose**: Baseline ExVAD training script  
**Key Features**:
- Standard training loop
- No cross-modal attention
- Adam optimizer, CrossEntropyLoss
- Checkpoint saving
- Validation evaluation
**Lines of Code**: ~300-400  
**Dependencies**: PyTorch, ExVADModel

#### 14. `src/train_enhanced.py`
**Purpose**: CMAF-enhanced model training script  
**Key Features**:
- Integrates cross-modal attention fusion
- Enhanced anomaly head training
- Learning rate scheduling
- Early stopping
- Comprehensive logging
**Lines of Code**: ~350-450  
**Dependencies**: PyTorch, EnhancedExVADModel, CMAF

#### 15. `src/demo.py`
**Purpose**: Demo script for baseline model  
**Key Features**:
- Quick testing on sample videos
- Result visualization
**Lines of Code**: ~150-200  
**Dependencies**: ExVADModel, video_utils

#### 16. `src/demo_new.py`
**Purpose**: Enhanced demo script  
**Key Features**:
- Improved visualization
- Attention weight display
**Lines of Code**: ~150-200  
**Dependencies**: EnhancedExVADModel

#### 17. `src/test_gpu.py`
**Purpose**: GPU availability and CUDA testing  
**Key Features**:
- Check PyTorch CUDA installation
- Test GPU memory
- Verify model can run on GPU
**Lines of Code**: ~30-50  
**Dependencies**: PyTorch

---

### Root-Level Scripts

#### 18. `demo_cmaf.py` (if exists in root)
**Purpose**: Quick CMAF demo script  
**Key Features**:
- Tests saved CMAF checkpoint
- Supports video files and PNG sequences
- Quick evaluation on sample data (3 per category)
- Saves metrics to `outputs/demo_results.json`
**Lines of Code**: ~200-300  
**Dependencies**: EnhancedExVADModel, data_loader_new

#### 19. `test_enhanced_model.py`
**Purpose**: Enhanced model testing utilities  
**Key Features**:
- Load and test CMAF checkpoints
- Evaluation harness
- Quick sanity checks
**Lines of Code**: ~100-150  
**Dependencies**: EnhancedExVADModel

#### 20. `visualize_cmaf.py`
**Purpose**: CMAF visualization tools  
**Key Features**:
- Attention weight visualization
- Cross-modal similarity heatmaps
- Result plotting and analysis
**Lines of Code**: ~150-200  
**Dependencies**: matplotlib, seaborn, EnhancedExVADModel

#### 21. `scripts/evaluate_saved_models.py` (if exists)
**Purpose**: Compare all saved model checkpoints  
**Key Features**:
- Evaluate multiple checkpoints at once
- Test across threshold range (0.05 to 0.95)
- Generate comparison reports
- Robust checkpoint loading (handles old formats)
**Lines of Code**: ~200-250  
**Dependencies**: EnhancedExVADModel, ExVADModel

---

## Documentation Files

### Technical Documentation

#### 22. `CMAF_EXPLANATION.md`
**Purpose**: Detailed CMAF architecture explanation  
**Content**:
- Mathematical formulation of CMAF
- Architecture diagrams and flow
- Implementation details
- Design rationale and ablation studies
**Word Count**: ~2000-3000 words

#### 23. `IMPLEMENTATION_ROADMAP.md`
**Purpose**: Development timeline and milestones  
**Content**:
- Feature implementation schedule
- Completed milestones
- Future enhancements
- Version history
**Word Count**: ~1000-1500 words

#### 24. `ADVANCED_RESEARCH_DIRECTIONS.md`
**Purpose**: Research extensions and future work  
**Content**:
- Potential improvements
- Novel research directions
- Integration opportunities
- Open challenges
**Word Count**: ~1500-2000 words

#### 25. `PROJECT_STRUCTURE.md`
**Purpose**: Comprehensive project structure for report prep  
**Content**:
- File organization
- Architecture overview
- Usage examples
- Performance summary
**Word Count**: ~3000-4000 words

#### 26. `Readme.md`
**Purpose**: Concise project README  
**Content**:
- Quick start guide
- Installation instructions
- Usage examples
- Key results summary
**Word Count**: ~400-600 words

#### 27. `CMAF_Architecture.png`
**Purpose**: Visual architecture diagram  
**Content**: Graphical representation of CMAF model flow

---

## Configuration Files

#### 28. `requirements.txt`
**Purpose**: Python dependencies  
**Content**:
```
torch>=2.0.0
transformers==4.35.2
clip @ git+https://github.com/openai/CLIP.git
Pillow
opencv-python
numpy
matplotlib
scikit-learn
tqdm
```

#### 29. `.gitignore`
**Purpose**: Git exclusion patterns  
**Excludes**:
- `data/` - Dataset (too large)
- `models_saved/` - Checkpoints (>100MB)
- `outputs/` - Evaluation results
- `.venv/` - Virtual environment
- `__pycache__/` - Python cache

#### 30. `.github/copilot-instructions.md`
**Purpose**: AI agent development guidelines  
**Content**:
- Code standards
- Common operations
- Development workflow

---

## Summary Statistics

### Code Distribution
- **Model Files**: 7 files (~1000 lines total)
- **Utility Files**: 5 files (~800 lines total)
- **Training Scripts**: 3 files (~850 lines total)
- **Demo/Test Scripts**: 4-5 files (~600 lines total)
- **Evaluation Scripts**: 1-2 files (~250 lines total)

**Total Python Code**: ~20 files, ~3500-4000 lines of code

### Documentation
- **Markdown Files**: 6 files
- **Total Documentation**: ~12,000-15,000 words
- **Diagrams**: 1 architecture image

### Configuration
- **Config Files**: 2 files
- **Dependencies**: ~10 major packages

---

## Files NOT in Repository (Excluded)

### Data (Excluded via .gitignore)
- `data/Train/` - Training videos/frames (~800-1000 samples)
- `data/Test/` - Test videos/frames (~290 samples)
- **Total Dataset Size**: Several GB (too large for git)

### Model Checkpoints (Excluded via .gitignore)
- `models_saved/best_model.pth` - Best CMAF checkpoint (~500MB)
- `models_saved/baseline_best_model.pth` - Baseline checkpoint
- `models_saved/best_model_bak.pth` - Backup checkpoint
- `models_saved/best_model_old_architecture.pth` - Legacy checkpoint

### Outputs (Excluded via .gitignore)
- `outputs/demo_results.json` - Demo metrics
- `outputs/demo_visualization.png` - Visual results
- `outputs/model_comparison_results.json` - Comparison data
- `outputs/overall_comparison.csv` - CMAF vs Baseline
- `outputs/*.png` - Various plots and visualizations

---

## For Report Preparation

### Essential Files to Include/Reference
1. **Core CMAF**: `src/models/cross_modal_attention.py`
2. **Enhanced Model**: `src/models/enhanced_exvad_model.py`
3. **Training**: `src/train_enhanced.py`
4. **Evaluation**: `src/utils/evaluation_metrics.py`
5. **Documentation**: All .md files
6. **Architecture**: `CMAF_Architecture.png`

### Key Code Sections to Highlight
1. **Cross-Modal Fusion** (cross_modal_attention.py, lines ~50-120)
2. **Attention Mechanism** (cross_modal_attention.py, forward method)
3. **Enhanced Head** (enhanced_exvad_model.py, anomaly head definition)
4. **Training Loop** (train_enhanced.py, main training function)
5. **Metrics Calculation** (evaluation_metrics.py, calculate_metrics)

### Available Results
- Demo: 87.8% accuracy (41 samples)
- Full Test: 77.24% accuracy (290 samples)
- Improvement over baseline: +25.86%
- Threshold: 0.1 (recommended)

---

## Repository Information
- **GitHub**: [2004Hrishikesh/Ex-VAD](https://github.com/2004Hrishikesh/Ex-VAD)
- **Current Branch**: `readme-update`
- **Main Branches**: `main`, `cmaf-research`, `readme-update`
- **License**: MIT

---

*Generated: November 2025*  
*For: CMAF Comprehensive Report Preparation*  
*All code files are tracked in git and available for review/analysis*
