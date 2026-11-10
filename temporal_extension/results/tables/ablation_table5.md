# Table 5: Gating Mechanism Ablation

Same checkpoint; different fusion strategies applied at inference time.

| Fusion Strategy | Acc. | F1 | FP | N-Acc |
| --- | ---: | ---: | ---: | ---: |
| Fixed blend (α=0.3, no gate) | 0.8586 | 0.8541 | 21 | 0.8600 |
| Boost only (no suppress gate) | 0.8621 | 0.8561 | 19 | 0.8733 |
| Suppress only (no boost gate) | 0.8552 | 0.8467 | 18 | 0.8800 |
| **Bidirectional gate (Ours)** | **0.8621** | **0.8561** | **19** | **0.8733** |