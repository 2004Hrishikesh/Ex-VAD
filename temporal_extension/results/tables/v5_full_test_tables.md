# V5 Full Test Sweep Tables

## temporal_v5_high_eff.pth (Epoch 11)

| Metric Set | Threshold | Accuracy | Precision | Recall | F1 | Normal Acc | Anomaly Acc | TP | TN | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current (0.44) | 0.440 | 0.8621 | 0.8623 | 0.8500 | 0.8561 | 0.8733 | 0.8500 | 119 | 131 | 19 | 21 |
| Best F1 / Best Acc | 0.440 | 0.8621 | 0.8623 | 0.8500 | 0.8561 | 0.8733 | 0.8500 | 119 | 131 | 19 | 21 |
| Best meeting CMAF recall target | 0.025 | 0.8034 | 0.7268 | 0.9500 | 0.8235 | 0.6667 | 0.9500 | 133 | 100 | 50 | 7 |

## temporal_v5_high_eff_epoch_010.pth (Epoch 10)

| Metric Set | Threshold | Accuracy | Precision | Recall | F1 | Normal Acc | Anomaly Acc | TP | TN | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current (0.44) | 0.440 | 0.8414 | 0.8615 | 0.8000 | 0.8296 | 0.8800 | 0.8000 | 112 | 132 | 18 | 28 |
| Best F1 / Best Acc | 0.380 | 0.8483 | 0.8529 | 0.8286 | 0.8406 | 0.8667 | 0.8286 | 116 | 130 | 20 | 24 |
| Best meeting CMAF recall target | 0.025 | 0.7966 | 0.7341 | 0.9071 | 0.8115 | 0.6933 | 0.9071 | 127 | 104 | 46 | 13 |

## temporal_v5_high_eff_epoch_020.pth (Epoch 20)

| Metric Set | Threshold | Accuracy | Precision | Recall | F1 | Normal Acc | Anomaly Acc | TP | TN | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current (0.44) | 0.440 | 0.8517 | 0.8647 | 0.8214 | 0.8425 | 0.8800 | 0.8214 | 115 | 132 | 18 | 25 |
| Best F1 / Best Acc | 0.375 | 0.8586 | 0.8511 | 0.8571 | 0.8541 | 0.8600 | 0.8571 | 120 | 129 | 21 | 20 |
| Best meeting CMAF recall target | 0.025 | 0.8207 | 0.7619 | 0.9143 | 0.8312 | 0.7333 | 0.9143 | 128 | 110 | 40 | 12 |

## Notes

- Dataset samples: 290 (140 anomaly, 150 normal)
- Evaluation mode: full test split with shared feature cache (same predictions as direct per-checkpoint run, faster execution)
- Consolidated JSON: outputs/v5_full_test_sweeps_shared_cache.json
