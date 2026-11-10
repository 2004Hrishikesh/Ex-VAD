# Table 4: Temporal Encoder Ablation

All variants use frozen ExVAD+CMAF base + confidence-gated fusion.
Encoder ablation is post-hoc: unwanted encoder outputs are zeroed
in the concatenated feature vector of the jointly-trained checkpoint.

| Temporal Encoders | Acc. | F1 | FP | N-Acc |
| --- | ---: | ---: | ---: | ---: |
| Transformer only | 0.8552 | 0.8467 | 18 | 0.8800 |
| BiLSTM only | 0.8552 | 0.8478 | 19 | 0.8733 |
| Motion Encoder only | 0.8621 | 0.8571 | 20 | 0.8667 |
| Transformer + BiLSTM | 0.8552 | 0.8478 | 19 | 0.8733 |
| Transformer + Motion | 0.8621 | 0.8571 | 20 | 0.8667 |
| BiLSTM + Motion | 0.8586 | 0.8541 | 21 | 0.8600 |
| **All three (Ours)** | **0.8621** | **0.8561** | **19** | **0.8733** |