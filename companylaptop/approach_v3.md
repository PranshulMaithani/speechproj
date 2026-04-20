# Approach v3 — Cheating Detection

**Output granularity:** one score per audio file. Never aggregate across a candidate's Q25/Q26/Q27 — cheating can start mid-exam, and downstream review is per-audio.

**Train / test convention:** train on `audios2 + audios4`, test on held-out `audios5` (52 positives). All per-folder caches + GT live next to notebooks (NB_DIR).

---

## Current best (production)

**Method:** late weighted fusion of two base models.

```
fusion_score = 0.6 * text_rf + 0.4 * wavlm_wp
```

| Component | Classifier | Features | Source notebook |
|---|---|---|---|
| `text_rf` | RandomForest (500 trees, depth 8, class_weight='balanced') | 40 handcrafted text features: stylometric + formal_ai + disfluency + pause groups (dropped prosodic / voice_q / perplexity / suspicious) | `text_cheating_detection.ipynb` |
| `wavlm_wp` | XGBoost (400 est, depth 5, lr 0.04, colsample 0.3) | 768-dim mean-pool of frozen `microsoft/wavlm-base-plus` over full audio | `wavlm_4way_comparison.ipynb` |

**Operating points on audios5:**

| Point | Threshold | Precision | Recall | F1 | Use case |
|---|---|---|---|---|---|
| Balanced | 0.46 | 0.80 | 0.69 | 0.74 | Flag-for-review |
| High-precision | 0.56 | ~0.88 | ~0.60 | 0.71 | Tighter review queue |

Expected after GT cleanup of misclassified review set: **~85% / ~75%**.

## Why this configuration

- `text_rf` hits 90% precision on its own — strongest single signal.
- `wavlm_wp` (whole, pretrained) beat finetuned WavLM in the 4-way comparison at this data scale.
- Error Jaccard(text_rf, wavlm_wp) = 0.306 — complementary enough to lift recall at fixed precision.
- This pair is the **only fusion configuration that wins on every cross-batch split** (audios2→4, 4→2, 2→5, 4→5 — see `fusion_results_analysis.md`).

## What was tried and rejected

- PCA on WavLM — destroyed performance (2026-04-15).
- Fine-tuned WavLM — worse than frozen pretrained at 540 labels.
- Meta-learner stacking (logreg / xgb OOF) — falls off at P≥0.90.
- 4-way optimized weights — best test F1 but overfit, doesn't win cross-batch.
- Voting fusion (AND / OR / majority) — dominated by weighted avg.
- Per-candidate aggregation — wrong granularity, rejected (cheating can start mid-exam).

## Reproduce current best

1. Open `fusion_text_wavlm.ipynb`.
2. Ensure `{folder}_features.csv` and `{folder}_whole_pretrained.csv` exist for audios2, audios4, audios5.
3. Run top-to-bottom. The fusion row to pick is **`wavg:text_rf+wavlm_wp @ a=0.6`**.
4. Misclassification review CSV + audio copies land in `checkpoints_fusion/review_audios5/`.

---

## Future — Whisper-medium as a third signal

Whisper-medium encoder is the next signal to wire in. Rationale:
- Trained for ASR → captures both acoustic and semantic cues.
- Independent error modes vs WavLM (prosody-only) and text_rf (transcript-only).
- Likely catches the "smooth reader" case that fools both current models.

### Status

- Whisper-medium encodings are already being extracted in `encoder_comparison.ipynb` (1024-dim mean-pool over 30s chunks, saved to `{folder}_whisper_whole.csv`).
- **Next step:** stand-alone classifier notebook `whisper_classifier.ipynb` that loads those cached embeddings and trains XGB + RF heads, with threshold sweep and rec@P targets. Save per-file probas to `checkpoints_whisper/pred_{tag}.csv` so fusion can pull them in later.
- **After classifier is trained:** edit `fusion_text_wavlm.ipynb` to add Whisper as a third base model. Try 3-way weighted avg on an (α_text, α_wavlm, α_whisper) grid. Keep only if it beats the 2-way baseline under cross-batch validation.

### Files involved

| File | Purpose | Status |
|---|---|---|
| `encoder_comparison.ipynb` | Extract Whisper embeddings (+ WavLM + Wav2Vec2) | exists, running |
| `{folder}_whisper_whole.csv` | 1024-dim mean-pool cache | per folder |
| `whisper_classifier.ipynb` | Dedicated Whisper classifier + proba export | **TO BUILD** |
| `fusion_text_wavlm.ipynb` | Add Whisper as 3rd signal | **TO EDIT LATER** |
