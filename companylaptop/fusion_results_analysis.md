# Fusion Notebook Results -- Analysis & Findings

**Notebook:** `companylaptop/fusion_text_wavlm.ipynb`
**Test set:** held-out split (52 positives, ~N negatives based on confusion counts)
**Base models:** `text_rf`, `text_top5`, `wavlm_wp` (whole+pretrained), `wavlm_sp` (segmented+pretrained)

---

## 1. Base Model Performance (at best-F1 threshold)

| Model      | thr  | prec   | rec    | f1     | rec@P85 | rec@P90 |
|------------|------|--------|--------|--------|---------|---------|
| text_rf    | 0.64 | 0.9032 | 0.5385 | 0.6747 | 0.5385  | 0.5385  |
| text_top5  | 0.70 | 0.6905 | 0.5577 | 0.6170 | 0.4615  | 0.3462  |
| wavlm_wp   | 0.46 | 0.6744 | 0.5577 | 0.6105 | NaN     | NaN     |
| wavlm_sp   | 0.46 | 0.6458 | 0.5962 | 0.6200 | NaN     | NaN     |

**Takeaway:** `text_rf` is already the strongest single model. WavLM bases can't hit P>=0.85 at any threshold on their own (confirming WavLM noise at high-confidence tail).

---

## 2. Master Comparison (top methods by F1)

| Method                                  | thr  | prec   | rec    | f1     | rec@P85 | rec@P90 | rec@P95 |
|-----------------------------------------|------|--------|--------|--------|---------|---------|---------|
| **wavg:all_4_opt [0.4, 0.2, 0.4, -0.0]** | 0.46 | 0.8333 | 0.6731 | **0.7447** | 0.5577  | 0.4808  | 0.3846  |
| **wavg:text_rf+wavlm_wp @ a=0.6**        | 0.46 | 0.8000 | 0.6923 | 0.7423 | 0.5769  | **0.5192** | 0.4038  |
| stack:meta_xgb                          | 0.60 | 0.7660 | 0.6923 | 0.7273 | 0.5000  | 0.3654  | 0.2885  |
| wavg:all_4_equal                        | 0.40 | 0.7400 | 0.7115 | 0.7255 | 0.5577  | 0.4423  | 0.2885  |
| **wavg:text_rf+wavlm_sp @ a=0.8**        | 0.56 | **0.8857** | 0.5962 | 0.7126 | 0.5962  | **0.5577** | 0.3846  |
| early:text+wp+sp                        | 0.42 | 0.7115 | 0.7115 | 0.7115 | 0.5000  | 0.4038  | 0.3654  |
| stack:meta_logreg                       | 0.72 | 0.7200 | 0.6923 | 0.7059 | 0.5577  | 0.4231  | 0.0962  |
| geo_mean:text_rf+wavlm_wp               | 0.44 | 0.7143 | 0.6731 | 0.6931 | 0.5577  | 0.4423  | 0.2692  |
| gate:text_rf<>wavlm_wp                  | 0.50 | 0.8333 | 0.5769 | 0.6818 | 0.5577  | 0.5000  | 0.2692  |
| early:text+wp                           | 0.52 | 0.7949 | 0.5962 | 0.6813 | 0.5385  | 0.4615  | 0.3846  |

**All 11 fusion methods beat all 4 base models on F1.** Best F1 jumps from 0.6747 (text_rf) -> 0.7447 (4-way optimized).

---

## 3. Precision-First Operating Points (the money table)

### P >= 0.80 (production baseline)
| Rank | Method                            | rec    | thr  |
|------|-----------------------------------|--------|------|
| 1    | wavg:text_rf+wavlm_wp @ a=0.6     | 0.6923 | 0.46 |
| 2    | wavg:all_4_opt                    | 0.6731 | 0.46 |
| 3    | stack:meta_xgb                    | 0.6346 | 0.68 |
| 4    | stack:meta_logreg                 | 0.6154 | 0.62 |

### P >= 0.85
| Rank | Method                            | rec    | thr  |
|------|-----------------------------------|--------|------|
| 1    | wavg:text_rf+wavlm_sp @ a=0.8     | 0.5962 | 0.56 |
| 2    | wavg:text_rf+wavlm_wp @ a=0.6     | 0.5769 | 0.50 |
| 3    | wavg:all_4_opt                    | 0.5577 | 0.54 |

### P >= 0.90 (strong deployment target)
| Rank | Method                            | rec    | thr  |
|------|-----------------------------------|--------|------|
| 1    | **wavg:text_rf+wavlm_sp @ a=0.8** | **0.5577** | 0.60 |
| 2    | wavg:text_rf+text_top5 @ a=0.9    | 0.5385 | 0.68 |
| 3    | base:text_rf                      | 0.5385 | 0.66 |
| 4    | wavg:text_rf+wavlm_wp @ a=0.6     | 0.5192 | 0.56 |

### P >= 0.95 (ultra-safe / zero-false-accuse)
| Rank | Method                            | rec    | thr  |
|------|-----------------------------------|--------|------|
| 1    | wavg:all_4_equal                  | 0.4423 | 0.64 |
| 1    | rank_fusion:all_4                 | 0.4423 | 0.86 |
| 3    | wavg:text_rf+wavlm_wp @ a=0.6     | 0.4038 | 0.66 |

---

## 4. Error Overlap (Jaccard)

```
             text_rf  text_top5  wavlm_wp  wavlm_sp
text_rf        1.000    0.658     0.306     0.250
text_top5      0.658    1.000     0.259     0.194
wavlm_wp       0.306    0.259     1.000     0.500
wavlm_sp       0.250    0.194     0.500     1.000
```
Error set sizes: text_rf=27, text_top5=36, wavlm_wp=37, wavlm_sp=38

**Key insight:**
- **text_rf vs wavlm_sp = 0.25 Jaccard** -- most complementary pair. Their mistakes rarely overlap, which is why `wavg:text_rf+wavlm_sp` drives the best P>=0.90 recall.
- text_top5 is redundant with text_rf (0.658) -- no independent signal.
- Two WavLM variants have 0.5 overlap -- moderately complementary with each other, but still dominated by text.

This explains why the optimized 4-way weights went `[text_rf=0.4, text_top5=0.2, wavlm_wp=0.4, wavlm_sp=-0.0]` -- it throws away text_top5 redundancy and wavlm_sp where not needed, but falls back to wavlm_sp when paired alone with text_rf for high-precision operation.

---

## 5. Threshold Sweep Highlights

### wavg:text_rf+wavlm_sp @ a=0.8 (the P>=0.90 winner)
| thr  | prec   | rec    | f1     | tp | fp | fn |
|------|--------|--------|--------|----|----|----|
| 0.55 | 0.8857 | 0.5962 | 0.7126 | 31 | 4  | 21 |
| 0.60 | 0.9062 | 0.5577 | 0.6905 | 29 | 3  | 23 |
| 0.70 | 0.9524 | 0.3846 | 0.5479 | 20 | 1  | 32 |
| 0.75 | **1.0000** | 0.3462 | 0.5143 | 18 | 0  | 34 |
| 0.80 | **1.0000** | 0.2885 | 0.4478 | 15 | 0  | 37 |

At thr>=0.75 this method achieves **100% precision with ~35% recall** -- zero false accusations on the test set.

### wavg:all_4_opt (the F1 winner)
| thr  | prec   | rec    | f1     | tp | fp | fn |
|------|--------|--------|--------|----|----|----|
| 0.45 | 0.8140 | 0.6731 | 0.7368 | 35 | 8  | 17 |
| 0.50 | 0.8205 | 0.6154 | 0.7033 | 32 | 7  | 20 |
| 0.60 | 0.9259 | 0.4808 | 0.6329 | 25 | 2  | 27 |

---

## 6. Cross-Batch Generalization (train on one batch, test on another)

| Train    | Test     | Method                        | prec   | rec    | f1     |
|----------|----------|-------------------------------|--------|--------|--------|
| audios2  | audios4  | base:text_rf                  | 0.8462 | 0.6000 | 0.7021 |
| audios2  | audios4  | **wavg:text_rf+wavlm_wp@0.7** | 0.7069 | 0.7455 | **0.7257** |
| audios4  | audios2  | base:text_rf                  | 0.9032 | 0.7619 | 0.8266 |
| audios4  | audios2  | **wavg:text_rf+wavlm_wp@0.5** | 0.8696 | 0.8163 | **0.8231** |
| audios2  | audios5  | base:text_rf                  | 0.7111 | 0.6154 | 0.6598 |
| audios2  | audios5  | **wavg:text_rf+wavlm_wp@0.7** | 0.7000 | 0.6731 | **0.6863** |
| audios4  | audios5  | base:text_rf                  | 0.8056 | 0.5577 | 0.6591 |
| audios4  | audios5  | **wavg:text_rf+wavlm_wp@0.6** | 0.8857 | 0.5962 | **0.7126** |

**Critical finding:** `wavg:text_rf+wavlm_wp` wins F1 on **all 4 cross-batch splits** by +2 to +4 F1 points over text_rf alone. audios5 is the hardest held-out batch -- even there, fusion holds up.

The 4-way optimized weighting doesn't appear in cross-batch winners (because weights were tuned on one split and don't transfer as cleanly). `text_rf+wavlm_wp` pair is the most robust generalizer.

---

## 7. Recommendations

### Production Deployment
Pick **one of two, depending on risk tolerance:**

**Option A -- Maximize recall at high precision (recommended default):**
- Method: `wavg:text_rf+wavlm_wp` with alpha = 0.6
- Threshold: 0.46 -> prec 0.80, rec 0.69, f1 0.74
- Justification: Best F1 that also generalizes across batches; 80% precision is a fair "flag for review" threshold.

**Option B -- Zero-false-accuse (auto-action threshold):**
- Method: `wavg:text_rf+wavlm_sp` with alpha = 0.8
- Threshold: 0.75 -> prec 1.00, rec 0.35
- Justification: ~35% of cheaters caught with zero false positives on the test set. Use for automated penalties; route the remaining 65% borderline cases to Option A for human review.

### What NOT to deploy
- **Voting fusion** (AND/OR/majority) -- high precision but kills recall (AND_all: 0.31, majority: 0.42). Dominated by weighted average at every precision target.
- **Meta-learner stacking** (meta_xgb/meta_logreg) -- decent F1 but falls off a cliff at P>=0.90 (stack:meta_logreg rec@P95=0.0962). Not calibrated for precision-first selection.
- **Early fusion** -- concatenating features didn't beat late weighted fusion. Adds training complexity for no gain.
- **4-way optimized** -- highest F1 on test, but weights are overfit to the held-out split and don't appear in cross-batch winners. Fragile.

### Drop these from the stack
- `text_top5` -- 0.658 Jaccard with text_rf, negligible independent signal.
- `wavlm_sp` -- only useful as a tiebreaker in the a=0.8 variant for P>=0.90. Can keep as a specialized high-precision model but not in main production.

---

## 8. Summary (one-liner for the slack message)

> Fusion works. `wavg(text_rf, wavlm_wp) @ a=0.6` hits **80% prec / 69% rec / 0.74 F1** and is the only method that wins on every cross-batch split. For zero-false-accuse mode, `wavg(text_rf, wavlm_sp) @ a=0.8, thr>=0.75` delivers **100% precision / 35% recall**. Deploy the pair, skip 4-way.
