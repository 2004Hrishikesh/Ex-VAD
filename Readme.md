# Ex-VAD: Explainable Video Anomaly Detection

This project implements an explainable video anomaly detection system that combines visual feature extraction, motion analysis, and natural language explanations.

## Features

- Deep learning-based video anomaly detection
- Explainable AI with natural language descriptions
- Support for multiple anomaly types (abuse, explosion, etc.)
- CLIP and BLIP integration for robust feature extraction
- Data augmentation and balanced training
- Comprehensive evaluation metrics

## Project Structure

```
Ex_VAD/
├── data/                    # Dataset directory
│   ├── videos/             # Video data
│   │   ├── train/         # Training videos
│   │   └── test/          # Testing videos
├── src/                    # Source code
│   ├── models/            # Model implementations
│   │   ├── aegm.py       # Anomaly Explanation Generation Module
│   │   ├── madm.py       # Motion Anomaly Detection Module
│   │   └── exvad_model.py # Combined model
│   ├── utils/            # Utility functions
│   │   ├── data_loader.py # Data loading utilities
│   │   ├── video_utils.py # Video processing utilities
│   │   └── evaluation_metrics.py # Evaluation metrics
│   ├── train.py          # Training script
│   ├── demo.py           # Demo script
│   └── test_gpu.py       # GPU testing utility
├── models_saved/          # Saved model checkpoints
└── outputs/              # Output results and visualizations
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support)
- Additional requirements in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Ex_VAD.git
cd Ex_VAD
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Training:
```bash
python src/train.py
```

2. Demo:
```bash
python src/demo.py
```

## Evaluation Metrics

The system uses comprehensive evaluation metrics:
- ROC curves and AUC-ROC
- Precision-Recall curves and AUC-PR
- F1 Score
- Confusion Matrix
- False Alarm Rate
- Missing Rate

## Model Architecture

The model consists of three main components:
1. Motion Anomaly Detection Module (MADM)
2. Anomaly Explanation Generation Module (AEGM)
3. Combined ExVAD architecture

## Results

Detailed evaluation results and visualizations are saved in the `outputs` directory.

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{exvad2025,
  title={Ex-VAD: Explainable Video Anomaly Detection},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```
