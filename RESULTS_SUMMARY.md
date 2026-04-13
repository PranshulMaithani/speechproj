# Cheating Detection Model: Approaches and Results Summary

## Project Context

**Goal:** Detect when interview candidates use outside help (GPT, articles, pre-written notes, reading aloud) versus genuinely answering questions from memory.

**Input:** 1-minute free speech audio per question

**Production Targets:** 95+ precision, 80+ recall

### Datasets

| Dataset | Samples | Cheating Rate | Role |
|---------|---------|---------------|------|
| audios2 | ~210 | 67% | Initial training |
| audios4 | ~278 | 24% | Additional training (mostly non-cheating) |
| audios5 | ~261 | — | Held-out test set |
| Incoming | ~600 more | — | ~200/week expected |

---

## Approach 1: Weighted Ensemble (Old/Previous)

### Architecture Overview

Three components combined via a manual weight grid search:

```
audio → [Text XGBoost]  ──┐
                           ├── W_TEXT * text_score + W_WAVLM * wavlm_score → threshold
audio → [WavLM XGBoost] ──┘
```

### Component 1: Text XGBoost (41 Handcrafted Features)

Features extracted from Whisper transcripts and raw audio signal.

**Text-Statistic Features (20) — via spaCy on Whisper transcripts:**

| Feature | Description |
|---------|-------------|
| `ttr` | Type-token ratio |
| `mattr` | Moving average TTR |
| `complex_word_rate` | Rate of complex/polysyllabic words |
| `avg_word_length` | Mean word length |
| `n_words` | Total word count |
| `n_unique_words` | Unique word count |
| `avg_sentence_length` | Mean sentence length |
| `std_sentence_length` | Sentence length variability |
| `fragment_rate` | Rate of sentence fragments |
| `n_sentences` | Total sentence count |
| `self_ref_rate` | Self-reference pronoun rate |
| `discourse_marker_rate` | Rate of discourse markers |
| `hedge_rate` | Rate of hedging language |
| `noun_rate` | Noun density |
| `verb_rate` | Verb density |
| `adj_rate` | Adjective density |
| `repetition_rate` | Word repetition rate |
| `repair_rate` | Rate of speech repairs |
| `filler_rate` | Filler word rate |
| `filler_count` | Raw filler word count |

**Pause Features (13) — from word-level Whisper timestamps:**

| Feature | Description |
|---------|-------------|
| `pause_mean` | Mean pause duration |
| `pause_std` | Pause duration standard deviation |
| `pause_median` | Median pause duration |
| `pause_skew` | Pause duration distribution skew |
| `long_pause_rate` | Rate of long pauses |
| `pause_ratio` | Ratio of pausing to speaking time |
| `n_pauses` | Total pause count |
| `pause_regularity` | Regularity of pause intervals |
| `pause_before_content_ratio` | Pauses preceding content words |
| `pause_before_function_ratio` | Pauses preceding function words |
| `mid_phrase_pause_rate` | Pauses within phrases |
| `words_per_sec` | Overall speaking rate |
| `articulation_rate` | Rate during non-pause speech |

**Prosodic Features (8) — from raw audio signal:**

| Feature | Description |
|---------|-------------|
| `f0_mean` | Mean fundamental frequency (pitch) |
| `f0_std` | Pitch standard deviation |
| `f0_range` | Pitch range |
| `f0_skew` | Pitch distribution skew |
| `f0_slope` | Pitch trajectory slope |
| `energy_mean` | Mean signal energy |
| `energy_std` | Energy variability |
| `speaking_rate_std` | Speaking rate variability |

### Component 2: WavLM XGBoost

- **Input:** 768-dimensional embeddings from pretrained WavLM model
- **Nature:** Audio-only, no transcription required
- **Training:** XGBoost classifier on pooled WavLM embeddings

### Component 3: Combination Method

- Manual weight grid search sweeping `W_TEXT` and `W_WAVLM` from 0.0 to 1.0
- Final score: `W_TEXT * text_score + W_WAVLM * wavlm_score`
- Threshold also swept to optimize precision/recall trade-off

---

### Results: Approach 1

**Trained on audios2 only (~210 samples):**

| Model | Precision | Recall |
|-------|-----------|--------|
| Text XGBoost | 83 | 86 |
| WavLM XGBoost | 80 | 83 |
| Weighted Ensemble (best config) | **90** | **86** |

> Note: Best ensemble weights were grid-searched on this same split — results are overfit to the audios2 distribution.

**Retrained on audios2 + audios4 combined (~490 samples):**

| Model | Precision | Recall | Notes |
|-------|-----------|--------|-------|
| Text XGBoost | 75 | 76 | Degraded due to class distribution shift |

**Evaluated on audios5 (held-out test set):**

| Model | Precision | Recall |
|-------|-----------|--------|
| Best weighted config | 84 | 76 |

---

### Problems Identified in Approach 1

1. **Fragile two-step text pipeline:** audio → Whisper transcript → NLP features compounds errors at each step; transcription mistakes directly corrupt features like `filler_rate` and TTR.

2. **Distribution shift sensitivity:** audios2 is 67% cheating; audios4 is 24% cheating. The class imbalance shift caused significant text model degradation when training data was combined.

3. **Manual grid search overfits to training split:** The optimal `W_TEXT`/`W_WAVLM` weights found during search are not guaranteed to generalize across different data distributions.

4. **wav2vec2 ONNX unreliable:** A wav2vec2 ONNX model was also evaluated but found unreliable in practice and was dropped.

---

## Approach 2: Stacked Ensemble with SBERT (New)

### Architecture Overview

Two base models trained with cross-validation, combined via a stacking meta-learner:

```
audio → Whisper → SBERT + pause + prosodic → [Hybrid XGBoost] ──┐
                                                                   ├── OOF predictions → [LogisticRegression meta-learner]
audio → WavLM embeddings            → [WavLM XGBoost]  ──────────┘
```

### Changes from Approach 1

| Aspect | Approach 1 | Approach 2 |
|--------|-----------|-----------|
| Text representation | 20 handcrafted text-stat features | 384-dim SBERT embeddings (all-MiniLM-L6-v2) |
| Filler/pause/prosodic | Included in text model | Kept (orthogonal to SBERT dims) |
| Combination method | Manual weight grid search | Stacking meta-learner (LogisticRegression on OOF predictions) |
| wav2vec2 | Tried and dropped | Not used |

### Component 1: Hybrid XGBoost (407 Features)

| Feature Group | Dimensions | Source |
|---------------|-----------|--------|
| SBERT embeddings | 384 | all-MiniLM-L6-v2 on Whisper transcript |
| Filler features | 2 | `filler_rate`, `filler_count` |
| Pause features | 13 | Same 13 as Approach 1 |
| Prosodic features | 8 | Same 8 as Approach 1 |
| **Total** | **407** | |

- XGBoost config: `colsample_bytree=0.25` (to limit overfitting on high-dim input)

### Component 2: WavLM XGBoost

- Same 768-dim WavLM embeddings as Approach 1 — unchanged

### Component 3: Stacking Meta-Learner

- `LogisticRegression(C=1.0)` trained on out-of-fold (OOF) predictions from both base models
- Intended to replace manual weight grid search with a learned, generalizable combination

---

### Results: Approach 2

**Trained on audios2 + audios4 combined (~490 samples):**

| Model | Notes |
|-------|-------|
| SBERT-only XGBoost (384 features) | Same performance as 41-feature text model — no improvement from embeddings alone |
| Hybrid XGBoost (407 features) | Slight CV improvement over text model (SBERT + pause/prosodic synergy) |

**Evaluated on audios5 (held-out test set):**

| Model | F1 Score | vs. Approach 1 |
|-------|----------|----------------|
| Stacked ensemble | ~0.6 | Significant regression from 84/76 |

---

### Why Performance Degraded

1. **Overfitting from feature-to-sample imbalance:** 407 features on ~490 samples yields roughly a 1:1 feature-to-sample ratio. Even with `colsample_bytree=0.25`, XGBoost learns training-specific patterns that do not generalize.

2. **Meta-learner combines a good signal with an overfit signal:** LogisticRegression stacking cannot fully compensate when one base model (Hybrid XGBoost) is overfit — the bad signal drags the ensemble below the WavLM-only baseline.

3. **SBERT not better than handcrafted at this scale:** 384 generic sentence embedding dimensions require substantially more data (~1000+ samples) to outperform 20 domain-specific handcrafted features. At 490 samples, the embedding space is too large to fit reliably.

4. **OOF predictions appeared decent in CV** (same distribution as training), but failed on audios5 which has a genuinely different sample distribution — a classic sign of in-distribution overfitting.

---

## Results Comparison: Approach 1 vs. Approach 2

| Configuration | Train Data | Test Data | Precision | Recall | F1 (approx.) |
|---------------|-----------|-----------|-----------|--------|--------------|
| Text XGBoost (A1) | audios2 | audios2 CV | 83 | 86 | 84 |
| WavLM XGBoost (A1) | audios2 | audios2 CV | 80 | 83 | 81 |
| Weighted Ensemble (A1) | audios2 | audios2 CV | **90** | **86** | **88** |
| Text XGBoost (A1, retrained) | audios2+4 | audios2+4 CV | 75 | 76 | 75 |
| Weighted Ensemble (A1) | audios2+4 | audios5 | 84 | 76 | 80 |
| Stacked Ensemble (A2) | audios2+4 | audios5 | — | — | ~60 |

---

## Key Takeaway

> **At ~490 samples, compact handcrafted features (41 features) that encode domain knowledge outperform dense embeddings (384 features). The SBERT approach will become viable at ~1000+ samples. Until then, the simpler weighted ensemble from Approach 1 is more robust.**

The fundamental tension is:
- **More expressive features** (SBERT embeddings) require more data to avoid memorization
- **Domain-specific handcrafted features** encode human priors that compensate for small dataset size
- **Distribution shift** (audios2 vs. audios4 class ratios) is the most pressing practical challenge, not model capacity

---

## Scaling Roadmap

As more data arrives (~200 samples/week), the viable approaches expand:

| Data Scale | Recommended Action |
|------------|-------------------|
| ~490 samples (now) | Use Approach 1 (41-feature weighted ensemble). Maintain class-balanced validation. |
| ~1000 samples | Add PCA on SBERT (384 → 32–64 dims) to make hybrid feasible. Optionally add Whisper encoder embeddings as a 3rd base model. |
| ~2000 samples | Fine-tune sentence transformer with SetFit on domain-specific cheating/genuine examples. |
| ~2000–3000 samples | MLP heads over feature representations become viable over XGBoost. |
| ~5000 samples | Full end-to-end fine-tuning of audio/text encoders. |

---

## Signal Philosophy

No single signal type catches all cheating patterns. Different cheating behaviors leave different traces:

| Cheating Type | Best Signal |
|---------------|-------------|
| Reading from GPT/notes verbatim | Pause patterns (unnaturally smooth), prosodic flatness |
| Paraphrasing pre-written content | Text semantics (high coherence, low hedging, low self-reference) |
| Looking up articles mid-answer | Long pauses, irregular speaking rate |
| Genuine spontaneous answer | High filler/repair rate, natural pitch variation, lower lexical density |

This is why multi-signal fusion is required — precision targets above 95% are only achievable when audio, prosodic, and semantic signals are combined and calibrated together.
