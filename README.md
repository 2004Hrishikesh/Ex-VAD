# Explainable Video Anomaly Detection with Cross-Modal Attention Fusion (CMAF)

Code for the paper **"Enhancing Explainable Video Anomaly Detection using Cross-Modal Attention Fusion Mechanism"**, Hrishikesh Muniraju and Sridhar Raj S, IConSCEPT 2025 (IEEE). [IEEE Xplore](https://ieeexplore.ieee.org/document/11436826)

The model flags anomalous surveillance videos (UCF-Crime) and explains why. It also writes a short caption of what happens in the clip.

![CMAF architecture](assets/cmaf_architecture.png)

## Idea

Visual features alone miss context. A frame of two people close together looks the same whether they are talking or fighting. CMAF adds a language view of the scene and lets the two views attend to each other:

1. **Visual branch:** CLIP ViT-B/32 features for 16 uniformly sampled frames, mean-pooled to 512-d.
2. **Text branch:** BLIP writes captions for the frames. DistilBERT encodes the caption ([CLS], 768-d).
3. **Cross-modal attention fusion** ([`src/models/cross_modal_attention.py`](src/models/cross_modal_attention.py)):
   - both modalities are projected to a shared 512-d space
   - 8-head attention runs in both directions (visual -> text and text -> visual)
   - a learned sigmoid gate decides, per dimension, how much to trust each direction
   - a residual connection adds the fused vector back onto the original CLIP feature, so the baseline signal is never lost
4. **Anomaly head:** a small MLP gives the anomaly score. The caption is returned with the score as the explanation.

## Results

All numbers below are recomputed from the saved per-video scores in [`results/cmaf_per_video_scores_test290.json`](results/cmaf_per_video_scores_test290.json). Anyone can check them without re-running the model. The test set is the UCF-Crime test split: 290 videos (150 normal, 140 anomalous), with video-level labels.

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | Missed anomalies |
|---|---:|---:|---:|---:|---:|---:|
| ExVAD baseline (CLIP + BLIP, no fusion) | 0.807 | 0.792 | 0.814 | 0.803 | 0.898 | 26 |
| **ExVAD + CMAF** | **0.828** | 0.781 | **0.893** | **0.833** | **0.909** | **15** |

- CMAF catches 11 more anomalous videos than the baseline (missed anomalies drop from 26 to 15). The cost is 5 more false alarms.
- The biggest per-class gains are on Shoplifting (16 -> 19 of 21), Burglary (11 -> 13 of 13) and Shooting (19 -> 21 of 23). These are classes where context matters more than appearance. See [`results/tables/cmaf_test290_per_category.csv`](results/tables/cmaf_test290_per_category.csv).
- The paper reports 85.7% with a +7.5 point gain over the baseline, from the evaluation run at the time of publication. The table above is from the checkpoint in this repository.
- The CMAF decision threshold (0.149) was tuned on this test split, and the baseline uses 0.5.

## Repository layout

```
src/
  models/
    aegm.py                   BLIP caption / explanation generator
    madm.py                   CLIP-based anomaly scorer (baseline)
    exvad_model.py            baseline ExVAD (AEGM + MADM)
    cross_modal_attention.py  CMAF module + enhanced scorer   <- main contribution
    enhanced_exvad_model.py   ExVAD + CMAF
  utils/                      data loading, frame sampling, metrics, threshold tuning
  train.py                    train the baseline
  train_enhanced.py           train ExVAD + CMAF
  evaluate_cmaf_comprehensive.py   baseline vs CMAF evaluation
scripts/                      demo, attention visualization, evaluation helpers
notebooks/                    Kaggle notebook version of the full pipeline
results/                      per-video scores and metric tables
temporal_extension/           follow-up work, not part of the paper (see its README)
```

## Running

```bash
pip install -r requirements.txt
# UCF-Crime frames: data/Train/<Class>/*.png and data/Test/<Class>/*.png (not included)

python -m src.train                          # baseline
cd src && python train_enhanced.py && cd ..  # ExVAD + CMAF (settings in the __main__ block)
python -m src.evaluate_cmaf_comprehensive    # evaluation
python scripts/demo_cmaf.py                  # demo with captions as explanations
```

Model checkpoints are about 1.5 GB because they include the CLIP, BLIP and DistilBERT weights, so they are not stored in git.

## Follow-up: temporal extension

[`temporal_extension/`](temporal_extension/) adds a temporal branch (Transformer + BiLSTM + motion encoder) on top of CMAF, with a confidence-gated fusion rule. On the same test split it reaches 0.862 accuracy and 0.856 F1, and cuts false alarms from 35 to 19. It is ongoing work and is not part of the published paper.

## Citation

```bibtex
@inproceedings{muniraju2025cmaf,
  title     = {Enhancing Explainable Video Anomaly Detection using Cross-Modal Attention Fusion Mechanism},
  author    = {Muniraju, Hrishikesh and Raj S, Sridhar},
  booktitle = {2025 International Conference on Signal Processing, Computation, Electronics, Power and Telecommunication (IConSCEPT)},
  year      = {2025},
  publisher = {IEEE},
  url       = {https://ieeexplore.ieee.org/document/11436826}
}
```

## Author

**Hrishikesh Muniraju** (first author): implemented and trained the CMAF module and ran the evaluations. Now an MS CS student at UMass Amherst.
Co-author and advisor: Sridhar Raj S, School of Computer Science and Engineering, VIT Vellore.
