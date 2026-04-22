# Models Explained — Cheating Detection

Single reference for every base model that feeds `fusion_text_wavlm.ipynb`. Written 2026-04-22.

## Why multiple models

No single signal catches every cheating style:

- A candidate reading off a screen → **text gives it away** (polished stylometry, low disfluency, formal transitions).
- A candidate paraphrasing from a prepared note → **audio prosody gives it away** (flat pitch, unnatural pause pattern, abnormally regular articulation).
- A fluent speaker who memorised an answer → **neither pure-text nor pure-prosody is enough**; we need a joint acoustic+semantic view (Whisper's ASR encoder captures both).

Fusion combines the models so one signal can cover another's blind spot.

## Naming convention

| Suffix | Meaning |
|---|---|
| `_wp` | **W**hole + **P**retrained. Frozen encoder, mean-pooled over the ENTIRE audio clip into one vector. |
| `_sp` | **S**egmented + **P**retrained. Frozen encoder, fixed-length windows (e.g., 3s), mean + std concatenated per window then aggregated. Keeps local variance information that whole-pooling destroys. |
| `_ft` | **F**ine-**t**uned. Encoder is unfrozen during head training. In our data regime (~540 labels), `_ft` variants consistently underperformed `_wp` / `_sp` — frozen pretrained won. |

Prefixes like `text_`, `wavlm_`, `whisper_` identify the feature source.

## Training protocol (summary)

- **Training data:** `audios2` (n≈210, 67% cheat) + `audios4` (n≈280, 18% cheat). Both batches are always present during training.
- **Threshold/weight selection:** 5-fold stratified CV on `audios4` only — audios2 is always concatenated to the training split, validation split is always `audios4[val_idx]`. This keeps the calibration surface at the deployment-like positive rate (~17%).
- **Class-weight override:** all heads use `scale_pos_weight = (1 − 0.17) / 0.17 ≈ 4.88` regardless of the training fold's actual positive rate. The loss function behaves as if the world were 17% positive, which removes the class-prior-induced bias from audios2.
- **Feature standardisation:** `StandardScaler` fit on the training split, applied to validation/test.
- **Test set:** `audios5` (n≈270, 16% cheat) is scored ONCE at the end at the frozen threshold — never used for tuning.

---

## 1. `text_rf` — wide stylometric RandomForest

**Input features (~40 dims)** — all computed from the ASR transcript of the audio answer:

| group | features |
|---|---|
| `stylometric` | ttr, mattr, mtld, complex_word_rate, avg_word_length, n_words, n_unique_words, avg_sentence_length, std_sentence_length, fragment_rate, n_sentences, self_ref_rate, noun_rate, verb_rate, adj_rate |
| `formal_ai` | formal_transition_count, formal_transition_rate, ai_phrase_count, ai_phrase_rate |
| `disfluency` | filler_rate, filler_count, repetition_rate, repair_rate, discourse_marker_rate, hedge_rate |
| `pause` | pause_mean, pause_std, pause_median, pause_skew, long_pause_rate, pause_ratio, n_pauses, pause_regularity, pause_before_content_ratio, pause_before_function_ratio, mid_phrase_pause_rate, words_per_sec, articulation_rate, initial_pause, longest_pause |

**Head:**
```python
RandomForestClassifier(
    n_estimators=500, max_depth=8, min_samples_leaf=3,
    class_weight={0: 1.0, 1: SPW_DEPLOY},  # SPW_DEPLOY ≈ 4.88
    n_jobs=-1, random_state=42)
```

**Source notebook:** `text_cheating_detection.ipynb` (feature extraction) + this notebook (training).

**Why RF and not XGB here:** on ~540 mostly-continuous handcrafted features with meaningful interactions (e.g., `formal_transition_rate × hedge_rate`), RF's ensembled shallow trees with bagging generalised better than XGBoost in head-to-head tests. The feature set is small enough that RF's O(n_estimators · depth · n_feats) cost is a non-issue.

**Typical behaviour:** high precision, moderate recall. Best single-signal precision in the stack — strong at flagging obvious scripted/read answers, weaker when a cheater's transcript looks stylistically ordinary.

---

## 2. `text_top5` — narrow stylometric XGBoost

**Input features (5 dims):** `mattr`, `mtld`, `avg_word_length`, `ttr`, `formal_transition_count`.

**Head:**
```python
xgb.XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    scale_pos_weight=SPW_DEPLOY, eval_metric='logloss', random_state=42)
```

**Why this variant exists:** fuses better with Whisper than `text_rf` does. Hypothesis: Whisper's ASR encoder already captures most of the disfluency + pause information that's baked into `text_rf`, so adding `text_rf` to Whisper is partially redundant. `text_top5` uses only **lexical-diversity + formality** signals that are genuinely orthogonal to what the acoustic encoder sees. Experimental evidence from post-GT-cleanup audios5: `wavg:text_top5 + whisper_wp` beat `wavg:text_rf + whisper_wp` by ~3 pp F1.

**Source:** features come from the same `{folder}_features.csv` as `text_rf` — `text_top5` just selects 5 columns.

---

## 3. `wavlm_wp` — WavLM whole-audio embedding

**Encoder:** `microsoft/wavlm-base-plus` — 94M-parameter speech transformer pretrained with masked speech modelling on 94k hours (LibriSpeech + GigaSpeech + VoxPopuli). Frozen (no fine-tuning).

**Input:** raw 16 kHz audio for one answer (whole clip, no chunking).

**Embedding:** forward pass → take the transformer's last-hidden-state → **mean-pool over the time dimension** → 768-dim vector per file. Cached to `{folder}_whole_pretrained.csv`.

**Head:**
```python
xgb.XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    scale_pos_weight=SPW_DEPLOY, eval_metric='logloss', random_state=42)
```

**Source notebook:** `wavlm_4way_comparison.ipynb` (produces the cache).

**Why WavLM-base-plus over other speech SSLs:** in the 4-way comparison (WavLM base, WavLM base-plus, HuBERT, Wav2Vec2), base-plus had the best downstream F1 on cheating detection. `base-plus` was additionally pretrained on 75k hours of English data, which helps for English-spoken exam answers.

**Why whole-pool vs segmented:** WP provides a stable summary of the whole answer's prosody. Dominant signal for detecting "reading voice" (monotone pitch contour, unusually regular articulation).

---

## 4. `wavlm_sp` — WavLM segmented embedding

**Encoder:** same as `wavlm_wp` (`microsoft/wavlm-base-plus`, frozen).

**Input:** same raw audio, but chunked into fixed-length segments (3s windows by default).

**Embedding:** per segment mean-pool (768d). Then aggregate across segments by concatenating [segment-mean, segment-std] → **1536-dim vector**. Cached to `{folder}_seg_pretrained.csv`.

**Head:**
```python
xgb.XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.3,   # colsample=0.3 because 1536-dim is high
    min_child_weight=3, scale_pos_weight=SPW_DEPLOY,
    eval_metric='logloss', random_state=42)
```

**Why SP alongside WP:** the `std` component preserves information about WITHIN-answer variability that whole-pooling discards. A candidate who reads a script has much lower segment-to-segment prosodic variance than a spontaneous speaker — `wavlm_sp`'s std-concat makes that directly visible to the head. Complementary to `wavlm_wp` (which captures the global prosodic profile) on error overlap.

---

## 5. `whisper_wp` — Whisper-medium encoder embedding

**Encoder:** `openai/whisper-medium` — 769M-parameter encoder-decoder ASR model trained on 680k hours of weakly-supervised multilingual audio. We use the **encoder only**, frozen.

**Input:** raw 16 kHz audio; if longer than 30s, split into 30s chunks (Whisper's native context length) and embed each.

**Embedding:** encoder last-hidden-state mean-pooled over time within each chunk → 1024d per chunk → mean across chunks → **1024-dim vector per file**. Cached to `{folder}_whisper_whole.csv`.

**Head:** same XGBoost as `wavlm_wp` with `colsample_bytree=0.8`.

**Source notebook:** `encoder_comparison.ipynb` (extraction) + this notebook (training).

**Why Whisper on top of WavLM:**
- **Pretraining objective differs.** WavLM is trained with masked-speech modelling (reconstruct masked audio frames) — it optimises for acoustic structure. Whisper is trained to transcribe audio to text — it must simultaneously represent **acoustic structure AND lexical/semantic content** in the encoder output because the decoder conditions only on encoder states.
- **Failure mode coverage.** The "smooth reader" case fools WavLM (if someone reads well, prosody looks natural) and partially fools text_rf (well-written script reads fluently). Whisper's encoder can pick up subtle mismatches between what is said and how it is said — e.g., smooth prosody paired with vocabulary that's too perfect — that are invisible to either signal alone.
- **Measured error independence.** In cross-batch Jaccard(whisper_wp errors, wavlm_wp errors) and Jaccard(whisper_wp, text_rf), whisper is the most complementary signal we have — fusion with either one lifts F1 beyond the better of the two bases.

**Why medium and not small / large-v3:**
- `small` (244M): too weak — encoder representation is less discriminative on our clean exam-answer audio.
- `medium` (769M): sweet spot — expressive enough, cache fits in RAM, 30s chunking is manageable.
- `large-v3` (1.55B): slightly better embeddings in tests but 3× slower to extract and cache is too large. Not worth the ops cost given how close medium is.

---

## Frozen vs fine-tuned at this scale

We consistently see `_wp` (frozen) > `_ft` (fine-tuned) in this project. Reason:

- With only ~540 labels, fine-tuning a 94M-parameter (WavLM) or 769M-parameter (Whisper) encoder has too few gradient signals to learn anything useful on top of pretraining. It mostly overfits idiosyncrasies of audios2/4 that don't transfer to audios5.
- Frozen-encoder + small classifier head concentrates all capacity on the head, which has orders-of-magnitude fewer parameters than the encoder → learnable with 540 labels.
- Fine-tuning would likely help around ≥2000 labels. Until then, freeze.

Past experiment (2026-04-15): PCA on top of WavLM destroyed performance — the head needs the raw 768-dim space. Keep the full embedding.

---

## Fusion process

All five bases produce per-file probabilities. Fusion combines those probas into ONE final score per file. Below is the exact procedure used by `fusion_text_wavlm.ipynb`.

### Step 1 — Get OOF base-model probas on audios4-CV

Goal: a full set of validation-set probas for every base model, over a deployment-like positive rate.

```
for fold in StratifiedKFold(audios4, n_splits=5):
    train_split  = audios2 ∪ audios4[train_idx]      # audios2 always in-train
    val_split    = audios4[val_idx]
    for each base in {text_rf, text_top5, wavlm_wp, wavlm_sp, whisper_wp}:
        fit head on train_split  (class_weight / SPW = SPW_DEPLOY)
        cv_scores[base][val_idx] = predict_proba(val_split)
```

Result: `cv_scores[base]` — one OOF vector per base, aligned with `cv_y` (audios4 labels). These are the numbers we tune on.

### Step 2 — Base-model standalone metrics on CV

Best-F1 threshold per base is computed on its CV probas (`best_f1_on`). This gives a reference F1 / prec / recall for every base — "what is this model worth alone at deployment-like conditions?" Output: `cv_base_df`.

### Step 3 — Search fusion candidates

Two families of fusion are tried. For each candidate, weights and the decision threshold are picked on CV probas by maximising CV F1.

**(a) Weighted-average (2-way).** For every pair (a, b) of active bases:
```
proba_fused(x) = wa * proba_a(x) + wb * proba_b(x),   wa + wb = 1
```
Sweep `wa ∈ {0.00, 0.05, 0.10, …, 1.00}`. For each `(wa, wb)`, compute best-F1 threshold on `cv_scores[a] + cv_scores[b]` weighted. Keep the `(wa, wb, thr)` that maximises CV F1 for this pair. Record it as `wavg:a+b`.

**(b) Weighted-average (3-way).** For every triple (a, b, c):
```
proba_fused(x) = w1·proba_a + w2·proba_b + w3·proba_c,   w1+w2+w3 = 1
```
Grid over the simplex with step 0.1 (≈66 points). Same best-F1 threshold search. Record as `wavg:a+b+c`.

**(c) Stacking — meta-logreg.** A logistic regression is trained to combine all base probas as features:
```
meta_features  = [proba_text_rf, proba_text_top5, proba_wavlm_wp, proba_wavlm_sp, proba_whisper_wp]
meta_lr        = LogisticRegression(class_weight='balanced', C=1.0).fit(cv_scores, cv_y)
```
To pick a fair threshold for the stack, we run a **nested 5-fold OOF** inside the CV probas: meta-logreg is trained on 4 meta-folds and predicts on the held-out meta-fold; repeat, concatenate → `p_meta_oof`. Best-F1 threshold is taken on `p_meta_oof` (not on in-sample predictions, which would be optimistic).

After all three families run, every candidate has a frozen `(members, weights, thr, cv_f1, cv_prec, cv_rec, cv_rec@P{85,90,95})` tuple. They're sorted by `cv_f1` into `cv_df`.

### Step 4 — Error-overlap sanity (Jaccard)

Before trusting a fusion, verify the pair/trio is actually complementary on CV:
```
errors[base] = { i : (cv_scores[base][i] >= best_thr[base]) != cv_y[i] }
Jaccard(a, b) = |errors[a] ∩ errors[b]| / |errors[a] ∪ errors[b]|
```
Low Jaccard (< ~0.4) → models fail on different examples → fusion should help. High Jaccard → redundancy; fusion will barely move the needle even if it wins CV F1 by chance.

### Step 5 — One-shot eval on audios5

For each frozen candidate:

1. Refit every base on the **full training set** (`audios2 ∪ audios4`, SPW still = `SPW_DEPLOY`). Produces `test_scores[base]` — one proba vector per base on audios5.
2. Apply the candidate's formula:
   - wavg: `proba_test = Σ weight_i · test_scores[member_i]`
   - stack: `proba_test = meta_lr.predict_proba(column_stack(test_scores[members]))[:, 1]`
3. Threshold with the frozen `thr`. Compute prec / rec / F1 / rec@P{85,90,95} on audios5.
4. `gap_f1 = cv_f1 − test_f1`. Flag any candidate with `|gap_f1| > 0.03`.

This is the ONLY time audios5 sees the model. No threshold or weight is chosen from audios5 numbers.

### Step 6 — Isotonic calibration (top CV fusion only)

Even after thresholding, raw fusion scores aren't directly interpretable as probabilities. Isotonic regression corrects this:
```
iso = IsotonicRegression(out_of_bounds='clip').fit(p_cv, cv_y)
p_te_cal = iso.transform(p_te)
```
After calibration, `score_cal >= 0.85` on audios5 should mean observed precision ≈ 0.85. This lets deployment pick thresholds by precision target (e.g., "only flag when cal-score ≥ 0.90") instead of re-measuring each time. The reliability check in cell-22 prints observed precision at threshold grid {0.50 … 0.95} — if the diagonal holds, calibration is trustworthy.

### Step 7 — Cross-batch swap eval (secondary sanity)

Independent of CV. Four splits: `2→4`, `4→2`, `2→5`, `4→5`. For each:
- fit every base on train batch, score test batch
- best-F1 per base and per 2-way wavg

A fusion is considered **robust** only if it wins on every direction, not just `→audios5`. A fusion that only wins on `audios5` is suspicious (either label-leak from audios5-specific GT work, or a fusion that happens to suit audios5's particular error distribution).

### Why weighted-average usually beats stacking here

With ~540 training rows and 5 base probas as meta-features, logistic regression has little data to learn a non-linear combination without overfitting. Weighted-average has 1–2 free parameters per pair and is equivalent to a restricted logreg with tied coefficients — it generalises better at this scale. If we cross ≥2000 labels, stacking starts to dominate.

### Frozen artifacts

After the notebook runs, all of this is persisted:

| File | Contents |
|---|---|
| `checkpoints_fusion/cv_metrics.csv` | every fusion candidate with its CV numbers |
| `checkpoints_fusion/test_oneshot_metrics.csv` | same candidates with audios5 numbers + `gap_f1` |
| `checkpoints_fusion/cross_batch.csv` | swap-eval table |
| `checkpoints_fusion/frozen_configs.json` | `(members, weights, thr)` for every candidate — reproducible, deployable |
| `checkpoints_fusion/audios5_full_predictions.csv` | per-file scores + calibrated scores + per-base probas (for review) |

---

## Output granularity

Every model outputs **one probability per audio file** (one per Q25/Q26/Q27). We do NOT aggregate across a candidate's three answers — cheating can begin mid-exam, and downstream review is per-answer. See `feedback_per_audio_granularity.md` for the reasoning.

## Files on disk

| File | Produced by | Consumed by |
|---|---|---|
| `{folder}_features.csv` | `text_cheating_detection.ipynb` | `text_rf`, `text_top5` |
| `{folder}_whole_pretrained.csv` | `wavlm_4way_comparison.ipynb` | `wavlm_wp` |
| `{folder}_seg_pretrained.csv` | `wavlm_4way_comparison.ipynb` | `wavlm_sp` |
| `{folder}_whisper_whole.csv` | `encoder_comparison.ipynb` | `whisper_wp` |
| `{folder}GT.csv` | manual labelling | all models (labels) |

All caches live next to `fusion_text_wavlm.ipynb` in `companylaptop/`.
