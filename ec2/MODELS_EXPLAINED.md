# Models, Features, and Methods — Cheating Detection Pipeline

## 1. TL;DR

The pipeline classifies each audio recording of a candidate answering an interview question as either spontaneous speech (label 0) or read-aloud cheating (label 1). Every audio is processed independently — scores are never averaged across questions Q25/Q26/Q27 because cheating can start mid-exam. Three signal types are combined: WavLM-base-plus mean-pooled embeddings (768 dimensions, from the last hidden state or hidden state 9), Whisper-medium encoder mean-pooled embeddings (1024 dimensions), and 55 handcrafted text/prosodic/voice-quality features extracted by `full_text_features.py`. XGBoost base classifiers are trained on each feature subset separately, and three weighted-average picks (Tier A) fuse the resulting probability scores. The EC2 port (`xgboost_train.py`) reproduces the Tier A picks faithfully but adds per-client feature standardization, audio augmentation expansion, few-shot adaptation, and multi-layer WavLM support.

---

## 2. The Feature Foundation: 55-feature audios6_eval spec

### 2.1 Origin and portability

`_build_audios6_eval.py` (company-laptop) is the original notebook generator that defined the 55-feature schema and applied it to the audios6 batch. `full_text_features.py` (EC2) is a self-contained port of that exact schema so the same features can be computed in any environment without the notebook scaffolding. The feature names, lexicons, and computation logic are identical; the two files can be cross-checked line-by-line.

Transcripts arrive differently per environment: the company-laptop notebooks call `faster-whisper` with a filler-word prompt (`_build_audios6_eval.py:252`) and store word-level timestamps in JSON. The EC2 scripts consume that JSON (or any compatible format) via `full_text_features.compute_all_features(audio_path, text, words)` where `words` is the list of `{word, start, end}` dicts.

NaN handling: `full_text_features.compute_all_features` returns `0.0` for every feature it cannot compute (library missing, audio too short, transcript too short). In `xgboost_train.py` the `feat_*` columns are additionally `.fillna(0.0)` after `pd.to_numeric(errors="coerce")` (`xgboost_train.py:471`), so any stray string in gt.csv is coerced to zero rather than silently propagating.

The `feat_` prefix convention: every output key of `compute_all_features` is prefixed with `feat_` (`full_text_features.py:393`). Columns in gt.csv that start with `feat_` are recognized as handcrafted features at runtime (`xgboost_train.py:468`).

### 2.2 Feature groups

#### G_DISFLUENCY (6 features)

| Feature | What it measures |
|---|---|
| `feat_filler_rate` | filler tokens / total word count |
| `feat_filler_count` | raw count of filler tokens |
| `feat_repetition_rate` | fraction of bigrams that appear more than once |
| `feat_repair_rate` | count of repair phrases / sentence count |
| `feat_discourse_marker_rate` | count of discourse markers / sentence count |
| `feat_hedge_rate` | count of hedge phrases / sentence count |

Fillers are a hard-coded set: `{um, uh, uh-huh, uhm, umm, hmm, hm, er, ah, ehm, mhm}` (`full_text_features.py:84`). Discourse markers include "you know", "like", "basically", "actually", etc. (`full_text_features.py:86`). Hedges include "i think", "kind of", "sort of", etc. (`full_text_features.py:87`). Repairs are phrase-level: "i mean", "no wait", "sorry i", etc. (`full_text_features.py:90`). Repetitions are detected at the bigram level — any bigram with count > 1 contributes `count - 1` to the rate (`full_text_features.py:199`). Library: pure Python + spaCy (tokenization only; spaCy degrades to regex if unavailable).

#### G_STYLOMETRIC (15 features)

| Feature | What it measures |
|---|---|
| `feat_ttr` | type-token ratio (unique words / total words) |
| `feat_mattr` | moving-average TTR over a 50-word window |
| `feat_mtld` | measure of textual lexical diversity (bidirectional factor count) |
| `feat_complex_word_rate` | fraction of words with >= 3 syllables |
| `feat_avg_word_length` | mean character count per word |
| `feat_n_words` | total word count |
| `feat_n_unique_words` | vocabulary size |
| `feat_avg_sentence_length` | mean words per sentence |
| `feat_std_sentence_length` | standard deviation of sentence lengths |
| `feat_fragment_rate` | fraction of sentences with fewer than 4 words |
| `feat_n_sentences` | sentence count |
| `feat_self_ref_rate` | first-person tokens / word count |
| `feat_noun_rate` | NOUN POS count / all POS count (requires spaCy) |
| `feat_verb_rate` | VERB POS count / all POS count (requires spaCy) |
| `feat_adj_rate` | ADJ POS count / all POS count (requires spaCy) |

MATTR uses a rolling window of 50 words (`full_text_features.py:147`). MTLD uses threshold 0.72 and averages forward + backward passes (`full_text_features.py:153`). Syllable counting is heuristic: vowel-run counting minus trailing silent-e (`full_text_features.py:130`). Without spaCy, POS-based rates (`noun_rate`, `verb_rate`, `adj_rate`) return 0.

#### G_PAUSE (15 features)

A pause is any inter-word gap > 0.05 s (50 ms) as measured from word timestamps (`full_text_features.py:239`).

| Feature | What it measures |
|---|---|
| `feat_pause_mean` | mean pause duration (seconds) |
| `feat_pause_std` | std of pause durations |
| `feat_pause_median` | median pause duration |
| `feat_pause_skew` | skew of pause durations (long-tail = right-skewed) |
| `feat_long_pause_rate` | fraction of pauses > 0.5 s |
| `feat_pause_ratio` | total pause time / total speaking window |
| `feat_n_pauses` | count of pauses |
| `feat_pause_regularity` | std of inter-pause word intervals (irregular = spontaneous) |
| `feat_pause_before_content_ratio` | pauses immediately before a content-POS word / n_pauses |
| `feat_pause_before_function_ratio` | pauses before function-POS word / n_pauses |
| `feat_mid_phrase_pause_rate` | pauses not preceded by a clause-boundary punctuation / n_pauses |
| `feat_words_per_sec` | words / total duration |
| `feat_articulation_rate` | words / (total duration - total pause time) |
| `feat_initial_pause` | gap before the first word (seconds) |
| `feat_longest_pause` | maximum single gap > 0.05 s |

`pause_before_content_ratio` and `pause_before_function_ratio` use spaCy POS tags on the word immediately after the gap; they fall back to 0 without spaCy (`full_text_features.py:259–270`). Content POS: NOUN, VERB, ADJ, ADV, PROPN. Function POS: DET, ADP, CONJ, CCONJ, SCONJ, PRON, AUX, PART (`full_text_features.py:99–100`).

#### G_SUSPICIOUS (2 features)

A "suspicious gap" is a pause of 0.3–0.8 s that does not follow a clause-boundary character (`.`, `!`, `?`) — the assumption being this gap range corresponds to someone glancing at a script mid-utterance, not to natural sentence-end breath (`full_text_features.py:275–277`).

| Feature | What it measures |
|---|---|
| `feat_suspicious_gap_count` | count of suspicious gaps |
| `feat_suspicious_gap_ratio` | suspicious gaps / total word count |

#### G_FORMAL_AI (4 features)

Detects formal-essay and AI-generated phrases that are characteristic of scripted content.

| Feature | What it measures |
|---|---|
| `feat_formal_transition_count` | count of formal connectives |
| `feat_formal_transition_rate` | per-100-words rate |
| `feat_ai_phrase_count` | count of AI-style boilerplate phrases |
| `feat_ai_phrase_rate` | per-100-words rate |

Formal transitions list (18 phrases): "furthermore", "moreover", "however", "therefore", "additionally", "consequently", "nevertheless", "hence", "thus", "in conclusion", "firstly", "secondly", "thirdly", "in summary", "to summarize", "in essence", "overall", "ultimately" (`full_text_features.py:91–94`).

AI phrases list (14 phrases): "it is important to note", "it is worth noting", "plays a crucial role", "plays a vital role", "delve into", "a wide range of", "on the other hand", "in other words", etc. (`full_text_features.py:95–98`).

#### G_PROSODIC (8 features)

Library: librosa. Audio is loaded at 16 kHz, mono, capped at 120 s (`full_text_features.py:319`).

| Feature | What it measures |
|---|---|
| `feat_f0_mean` | mean fundamental frequency (Hz), voiced frames only |
| `feat_f0_std` | std of F0 |
| `feat_f0_range` | max - min F0 |
| `feat_f0_skew` | skew of F0 distribution |
| `feat_f0_slope` | linear trend of F0 over time (positive = rising, negative = falling) |
| `feat_energy_mean` | mean RMS energy |
| `feat_energy_std` | std of RMS energy |
| `feat_speaking_rate_std` | std of voiced-frame density across 2 s windows |

F0 uses `librosa.pyin` with fmin=75 Hz, fmax=500 Hz, frame_length=2048 (`full_text_features.py:323`). NaN frames (unvoiced) are excluded before computing statistics. `speaking_rate_std` measures how variable the density of voiced activity is across the recording — read-aloud speech tends to be more uniform.

#### G_VOICE_Q (3 features)

Library: parselmouth (Python wrapper for Praat). If parselmouth is not installed, all three return 0.

| Feature | What it measures |
|---|---|
| `feat_jitter_local` | cycle-to-cycle F0 period variation |
| `feat_shimmer_local` | cycle-to-cycle amplitude variation |
| `feat_hnr_mean` | harmonics-to-noise ratio (dB) — higher = cleaner voice |

Praat calls: `To PointProcess (periodic, cc)` with F0 range 75–500 Hz; then `Get jitter (local)` with max period factor 1.3; `Get shimmer (local)` with max amplitude factor 1.6 (`full_text_features.py:349–354`). Read-aloud speech tends to have lower jitter/shimmer (more controlled) and higher HNR.

#### G_PERPLEXITY (2 features)

Library: HuggingFace `transformers` + `gpt2`. If unavailable, both return 0.

| Feature | What it measures |
|---|---|
| `feat_mean_perplexity` | mean per-sentence GPT-2 perplexity (exp of cross-entropy loss) |
| `feat_burstiness` | variance of per-sentence perplexity values |

Each sentence is encoded separately (up to 20 sentences, truncated at 256 tokens); cross-entropy loss is computed with `torch.no_grad()` (`full_text_features.py:375`). Low mean perplexity means the text is predictable to GPT-2 — scripted/AI-generated answers score lower. High burstiness means uneven complexity across sentences, which can indicate pasted-in chunks of different origin.

---

## 3. Tier A: the Original Fusion Strategy

Source: `_build_audios6_tier_a.py` (company-laptop). Generates `audios6_tier_a.ipynb`.

### 3.1 Base Models

Six base models are trained on the TRAIN_POOL (audios2 + audios4 + audios5):

| Name | Features | Algorithm |
|---|---|---|
| `text_top20_xgb` | Top 20 text features by XGB importance on TRAIN_POOL | XGBClassifier |
| `text_all_xgb` | All 55 text features | XGBClassifier |
| `text_stylo_xgb` | 15 stylometric features only | XGBClassifier |
| `whisper_wp_xgb` | Whisper-medium mean-pool (1024d) | XGBClassifier |
| `wavlm_wp_xgb` | WavLM-base-plus mean-pool (768d, last hidden state) | XGBClassifier |
| `whisper_wp_rf` | Whisper-medium mean-pool (1024d) | RandomForestClassifier |

Base model registry defined at `_build_audios6_tier_a.py:401–408`.

XGB hyperparameters (`make_xgb`, `_build_audios6_tier_a.py:187`):

```
n_estimators     = 400
max_depth        = 4
learning_rate    = 0.05
subsample        = 0.8
colsample_bytree = 0.3 if n_feats > 500 else 0.8
min_child_weight = 3
scale_pos_weight = SPW_DEPLOY  (= 4.882)
eval_metric      = 'logloss'
```

`SPW_DEPLOY = (1 - 0.17) / 0.17 = 4.882` (`_build_audios6_tier_a.py:49`). This encodes an assumed deployment base rate of 17% cheating. Setting `scale_pos_weight` to this value tells XGB to treat each cheating positive as worth 4.88 negatives in its gradient updates, matching what the model will see in production where only ~1 in 6 candidates cheats.

Random Forest hyperparameters (`make_rf`, `_build_audios6_tier_a.py:197`):
`n_estimators=400, min_samples_leaf=2, class_weight={0:1.0, 1:SPW_DEPLOY}`.

The top-20 feature ranking uses the same XGB on TRAIN_POOL (all text features) and takes the top 20 by `feature_importances_` (`_build_audios6_tier_a.py:390–396`).

### 3.2 The Three Picks

Picks defined at `_build_audios6_tier_a.py:468–474`:

| Pick | Members | Weights | Frozen threshold | Rationale |
|---|---|---|---|---|
| `pick1` | text_top20_xgb + whisper_wp_xgb + wavlm_wp_xgb | 0.20, 0.44, 0.36 | 0.59 | Mixed view: text + two acoustic models |
| `pick2` | text_all_xgb + wavlm_wp_xgb | from `frozen_configs.json` | from `frozen_configs.json` | Text + acoustic, weights searched from prior CV run |
| `pick3` | text_top20_xgb + text_stylo_xgb + whisper_wp_rf | 0.12, 0.16, 0.72 | 0.59 | Text-heavy + RF for robustness against overfitting |

Pick2 weights and threshold are read at runtime from `checkpoints_everything/frozen_configs.json` or `checkpoints_fusions/frozen_configs.json` (`_build_audios6_tier_a.py:438–463`). If neither file exists, defaults of `[0.5, 0.5]` and threshold 0.5 are used.

### 3.3 OOF Fusion + Threshold Tuning

`a4_oof` (`_build_audios6_tier_a.py:226`): for each base model, runs 5-fold StratifiedGroupKFold on audios4 (grouped by candidate_id), training each fold on (audios2 + the audios4 training folds). Produces out-of-fold predictions on every audios4 row. These OOF predictions serve as the calibration surface for pick weight search and threshold tuning — they are representative of held-out performance without using audios6 labels.

`sweep_thr` (`_build_audios6_tier_a.py:242`): grid search from 0.01 to 0.99 in steps of 0.01; returns the threshold that maximizes F1 on the OOF predictions.

`bootstrap_ci` (`_build_audios6_tier_a.py:262`): 1000-iteration bootstrap of F1 difference between two pick configurations for statistical comparison.

### 3.4 Top-K% Rank-Based Decision Rule (§3, Tier A1)

Instead of using a single frozen threshold, this rule adapts to batch-level prior shift. For each evaluation:

1. Draw a calibration slice (sizes tested: 30, 50, 80, 120 rows) from the new batch, stratified by label.
2. Estimate `K_hat = mean(y_slice)` — the observed cheating rate in the slice.
3. On the remaining rows, flag the top-K_hat% by score as positive (`_build_audios6_tier_a.py:578–580`).

This is repeated across 20 random seeds and the F1s are averaged. The rationale: if a new client has a different cheating prevalence (say 25% instead of 17%), a frozen threshold tuned at 17% will systematically under-flag. K_hat adjusts the decision boundary to match the batch's own prior. Tested against `frozen_thr`, `cv_thr`, and oracle threshold (`_build_audios6_tier_a.py:595–599`).

### 3.5 Per-Question Deviation Features (§4, Tier A2)

For each text feature `f` and each question `q`, compute over honest-labelled TRAIN_POOL rows:

```
PQ_MEAN[f][q] = mean(f | question_id == q, label == 0)
PQ_STD[f][q]  = std(f  | question_id == q, label == 0)
```

Then augment every row with two derived features (`_build_audios6_tier_a.py:660–667`):

```
f__dev  = raw_value - PQ_MEAN[q]           # deviation from honest baseline at this question
f__z    = (raw_value - PQ_MEAN[q]) / PQ_STD[q]   # z-score
```

This triples the text feature count (raw + `__dev` + `__z`) and is designed to control for question-content effects — a formal transition word might appear at normal rates in Q25 but be suspicious in Q26. The eta-squared decomposition at `_build_audios6_tier_a.py:817–836` measures how much question_id explains variance in each feature.

**This feature set is NOT in the EC2 port.** The EC2 port drops it because the new clients (audios6/ALLSTAR) do not share the same question pool as audios2/4/5. Computing PQ_MEAN/PQ_STD against audios2/4/5 questions and applying it to a different client's questions would introduce noise rather than signal.

### 3.6 Frozen Configs

`checkpoints_everything/frozen_configs.json` and `checkpoints_fusions/frozen_configs.json` store the best fusion weights and thresholds found in prior CV search runs. The Tier A script reads these at runtime to initialize pick2 (`_build_audios6_tier_a.py:438–445`). This allows reusing a prior weight search result without re-running the full grid search.

---

## 4. The EC2 Port: `ec2/xgboost_train.py`

### 4.1 What's Carried Over

Same `_make_xgb` hyperparameters (`xgboost_train.py:112–122`):

```
n_estimators     = 400
max_depth        = 4
learning_rate    = 0.05
subsample        = 0.8
colsample_bytree = 0.3 if n_feats > 500 else 0.8
min_child_weight = 3
scale_pos_weight = SPW_DEPLOY  (4.882, when pos_weight mode is active)
eval_metric      = 'logloss'
```

Three Tier A picks reproduced exactly in `PICK_DEFS` (`xgboost_train.py:89–105`):

```python
"tierA_pick1": members=["text_top20","whisper","wavlm"],  weights=[0.20,0.44,0.36]
"tierA_pick2": members=["text_all","wavlm"],              weights=[0.50,0.50]
"tierA_pick3": members=["text_top20","text_stylo","whisper"], weights=[0.12,0.16,0.72]
```

Same six conceptual base model types (text_all, text_stylo, text_top20, whisper, wavlm, everything). The top-20 feature ranking is computed on the training set using XGB importance (`xgboost_train.py:268–280`).

### 4.2 What's Different from Tier A

**Per-WavLM-layer variant expansion.** Tier A used only `wavlm_wp_xgb` (last hidden state). The EC2 port loops over `WAVLM_LAYERS = ["last", "9"]` (`_data_pipeline.py:41`). Hidden state 9 is an intermediate transformer layer of WavLM that encodes different acoustic properties than the final layer — empirically it sometimes captures prosodic structure that the last layer (which is optimized toward downstream tasks) discards. Every WavLM-dependent base and pick is trained twice: once with the last-layer embedding, once with layer-9.

**No Random Forest base.** `whisper_wp_rf` was replaced with a second `whisper_xgb` invocation in pick3. The RF added complexity without consistent benefit on the new data distribution.

**No per-question deviation features.** Dropped because the new clients do not share question pools with audios2/4/5 (see §3.5).

**Per-client feature standardization** (`xgboost_train.py:486–489`, implemented in `_data_pipeline.per_client_standardize`). Centers each client's WavLM, Whisper, and optionally `feat_*` arrays on that client's own mean and std, computed across all of that client's rows (unsupervised — no labels). Rationale: audios2/4/5 come from one production client; audios6 is a different company with a different microphone chain, codec, and candidate pool. WavLM and Whisper mean-pools sit at different absolute locations in feature space per client. Without standardization, the model can learn "which client is this" as a shortcut. With it, the model sees approximately zero-mean unit-variance features within each client's coordinate system.

**Few-shot adaptation** (`--fewshot_frac`). Carves `fewshot_frac` of test-batch candidates into the training set (candidate-disjoint with the remaining test set). Simulates a realistic deployment scenario where a small labelled sample from the new client is available before going live. Implemented in `build_splits` (`_data_pipeline.py:131–146`).

**Class balance modes** (`--class_balance`). Four options:
- `sampler`: per-sample weights using sklearn "balanced" formulation: `w_pos = N/(2*pos)`, `w_neg = N/(2*neg)`. Per-sample weights average 1.0, so XGB's `min_child_weight=3` accumulates 3 units per leaf as expected.
- `pos_weight`: sets `scale_pos_weight = neg/pos` in XGB.
- `both`: both of the above simultaneously.
- `none`: natural class distribution.

The `sampler` mode had a critical bug in an earlier version: per-sample weights were computed as `0.5/class_count`, which is correct for class totals but sub-unit per-sample. With `min_child_weight=3` in XGB, no leaf could accumulate 3 units of hessian, so no splits occurred, every tree stayed at the base score, and AUC was 0.5 across all variants. The fix uses the sklearn formulation (`xgboost_train.py:127–146`).

**Augmentation expansion.** Text-only variants (`expand_train=False`) do not expand the training set with augmented audio because `feat_*` values do not change with noise/pitch/VTLP augmentation — repeating the same feature vector just overweights those rows in XGB without adding information. WavLM/Whisper-bearing variants expand the training rows (`expand_train=True`), stacking original + each augmentation variant vertically with the same labels (`xgboost_train.py:241–261`).

### 4.3 The Variant Matrix

Layer-independent bases (run once, `layer='n/a'`):
- `whisper_xgb` — XGB on Whisper-medium encoder mean-pool (1024d). Always runs.
- `text_all_xgb` — XGB on all 55 `feat_*` columns. **Skipped if `--use_text_features=false`.**
- `text_stylo_xgb` — XGB on the 15 stylometric `feat_*` columns. **Skipped if `--use_text_features=false`.**
- `text_top20_xgb` — XGB on the 20 `feat_*` columns ranked highest by XGB importance on the training set. **Skipped if `--use_text_features=false`.**

Layer-dependent bases (run once per layer in `{last, 9}`):
- `wavlm_xgb_last`, `wavlm_xgb_l9` — XGB on WavLM mean-pool for that layer (768d). Always runs.
- `everything_xgb_last`, `everything_xgb_l9` — XGB on `[wavlm | whisper | feat]` concatenation. Always runs; the `feat` block is omitted when `--use_text_features=false`.

Picks (weighted-average of base probabilities). `needs_wavlm=True` means the pick depends on a WavLM-layer member and is therefore run once per layer:
- `tierA_pick1_last`, `tierA_pick1_l9` — fuses a text base + whisper + wavlm.
- `tierA_pick2_last`, `tierA_pick2_l9` — fuses a text base + wavlm.
- `tierA_pick3` — fuses two text bases + whisper. Layer-independent (no wavlm member), so runs once.

The `LayerMatrices` class (`xgboost_train.py:174`) is built once per WavLM layer and holds the full concatenated `[wavlm | whisper | feat]` matrix for `orig` and each augmentation. Variants then column-slice their needed subset via `build_block_cols` (`xgboost_train.py:215`), avoiding repeated memory allocation.

### 4.3.1 What each pick actually contains, with `--use_text_features` on vs off

The picks are weighted-average fusions of base-model probabilities. When `--use_text_features=false`, the text-only bases (`text_all_xgb`, `text_stylo_xgb`, `text_top20_xgb`) are skipped, so picks that named them as members lose those members. The surviving members have their weights renormalized (sum to 1.0). A pick with fewer than 2 surviving members is **skipped entirely** — it would just duplicate a base row. The skip logic lives at `xgboost_train.py:567–571`.

| Pick | With `--use_text_features=true` | With `--use_text_features=false` |
|---|---|---|
| **tierA_pick1**<br>(per WavLM layer) | `text_top20_xgb × 0.20  +  whisper_xgb × 0.44  +  wavlm_xgb × 0.36` | `whisper_xgb × 0.55  +  wavlm_xgb × 0.45`<br>(renormalized from 0.44/0.36; text_top20 dropped) |
| **tierA_pick2**<br>(per WavLM layer) | `text_all_xgb × 0.50  +  wavlm_xgb × 0.50` | **SKIPPED**<br>(only wavlm survives → 1 member; the row simply repeats the wavlm base) |
| **tierA_pick3**<br>(layer-independent — `whisper_xgb` is the only non-text member) | `text_top20_xgb × 0.12  +  text_stylo_xgb × 0.16  +  whisper_xgb × 0.72` | **SKIPPED**<br>(only whisper survives → 1 member; the row simply repeats the whisper base) |

So `--use_text_features=false` changes the pick output substantially:

- **text=on:** 5 pick variants exist in `summary2.csv` — `tierA_pick1_last`, `tierA_pick1_l9`, `tierA_pick2_last`, `tierA_pick2_l9`, `tierA_pick3`.
- **text=off:** 2 pick variants exist — `tierA_pick1_last`, `tierA_pick1_l9` (degenerated into pure acoustic 2-way fusions). Both `tierA_pick2_*` and `tierA_pick3` disappear from the summary file.

The thresholds quoted in §3.2 (`frozen_thr = 0.59`) come from the original Tier A notebooks; the EC2 port does NOT use those frozen thresholds. Instead, every pick has its threshold re-tuned by sweeping F1 on the validation set inside `xgboost_train.py` (`sweep_threshold` at `_data_pipeline.py:481`). The threshold lives in `summary2.csv` as `val_thr`; the test-set best-F1 threshold also gets reported as `test_best_thr` for comparison. This means under `--use_text_features=false` the pick1 thresholds may be different from those under `--use_text_features=true` even if you'd expect the score scale to be similar — the val-set distribution shifts when text bases disappear.

#### Variant count cheat-sheet for `summary2.csv`

| Configuration | Bases | Picks | Total variants |
|---|---|---|---|
| `--use_text_features=true` | 4 layer-independent (`whisper`, `text_all`, `text_stylo`, `text_top20`) + 2×2 layer-dependent (`wavlm`, `everything` each × 2 layers) = **8 base rows** | 5 (pick1 × 2 layers, pick2 × 2 layers, pick3 once) | **13** |
| `--use_text_features=false` | 1 layer-independent (`whisper`) + 2×2 layer-dependent (`wavlm`, `everything` each × 2 layers, `feat` block dropped) = **5 base rows** | 2 (pick1 × 2 layers; pick2 and pick3 dropped) | **7** |

If you see 7 rows in `summary2.csv` you ran with text features off. If you see 13, text features were on. The base-row count is also a quick sanity check that the run finished — if it's less than expected, check the log for "no usable columns ... skipping" or "only N viable member" warnings.

### 4.4 Output

`summary2.csv` — one row per variant. Key columns:

| Column | Meaning |
|---|---|
| `variant` | e.g. `tierA_pick1_last` |
| `val_thr` | threshold that maximizes F1 on validation set |
| `val_f1@thr` | F1 on val at `val_thr` (what was optimized) |
| `test_best_f1` | F1 on test at its own oracle threshold |
| `test_auc` | AUROC on test |
| `test_ap` | average precision on test |
| `recall@p50` | recall when precision >= 0.50 |
| `recall@p80` | recall when precision >= 0.80 |
| `recall@p85` | recall when precision >= 0.85 |
| `recall@p90` | recall when precision >= 0.90 |
| `recall@p95` | recall when precision >= 0.95 |
| `ind_f1`, `ind_r@p{80,85,90,95}` | same metrics on IND-region test rows |
| `php_f1`, `php_r@p{80,85,90,95}` | same metrics on PHP-region test rows |

The `recall@p*` columns answer "how many cheaters do we catch if we require X% precision?" — a more operationally useful question than threshold-swept F1 alone. The threshold is tuned on validation, not test, so `val_thr` is the one you would deploy.

---

## 5. Shared Module: `ec2/_data_pipeline.py`

Both `neural_baseline_train.py` and `xgboost_train.py` import this module. Every split, metric, and standardization is guaranteed identical between the two scripts so `summary1.csv` (NN) and `summary2.csv` (XGB) are directly comparable on the same rows.

### 5.1 `load_gt_and_filter`

Reads `gt.csv`, casts the `batch` column to `str` (`_data_pipeline.py:247`). This is critical: pandas auto-infers numeric batch IDs like `"2676"` as `int64`. When the ID is stored as int, `gt["batch"].isin({"2676","2677"})` returns all-False, silently dropping ALLSTAR from every split. The cast prevents this. Also drops rows where `label` is not in `{0, 1}` and applies the `--min_duration` filter on `duration_sec`.

### 5.2 `build_splits`

Two modes:

**Mode A** (no `test_batches` argument): 5-fold StratifiedGroupKFold on the full train pool. Fold 0 = test, fold 1 = val, remaining = train. Candidate-disjoint by `group_id`. Used when there is no separate held-out batch.

**Mode B** (`test_batches` specified): test set = rows from the specified batches (optionally filtered by `--test_region_filter`). Val set = one fold from the train pool that excludes all test-batch group_ids. If a region filter is active and the same-region subset of the train pool has >= 30 rows with both classes, val is drawn from that region subset so the val distribution matches test (`_data_pipeline.py:173–185`).

`train_only_batches`: rows from these batches are always forced into train and never appear in val or test. Used for auxiliary labeled data (e.g. ALLSTAR batches 2676/2677) that should enrich training but not contaminate the evaluation.

`fewshot_frac > 0`: randomly selects `fewshot_frac` of test-batch candidates (candidate-disjoint with remaining test), adds their rows to train. Those candidates are also excluded from the val pool to prevent the adaptation candidates from appearing in both train and val (`_data_pipeline.py:161–163`).

### 5.3 `assert_no_group_leak`

Verifies that the `group_id` sets of train, val, and test are fully disjoint (`_data_pipeline.py:204–208`). Called after every `build_splits` invocation. Prevents the silent data leakage that occurs when the same candidate appears in both train and test.

### 5.4 `per_client_standardize`

Maps each row's `batch` to a client label via `CLIENT_MAP`, then for each client independently computes `mean` and `std` across that client's rows and applies `(x - mean) / (std + 1e-6)` in-place to the WavLM cache, Whisper cache, and optionally `feat_arr` (`_data_pipeline.py:315–360`). Operates on features only — labels are never touched. Clients with fewer than 5 rows are skipped with a warning.

**CLIENT_MAP** (`_data_pipeline.py:225–232`):
- audios2, audios4, audios5 → client `A` (same production environment)
- audios6 → client `B` (different company, mic chain, region, question pool)
- 2676, 2677 → `ALLSTAR`

### 5.5 `compute_metrics`

Returns a dict with:
- `thr0.5`: precision/recall/F1 at the fixed 0.5 threshold
- `best_f1`: F1-maximizing threshold from a grid of 91 points (0.05 to 0.95) plus the corresponding precision, recall, threshold
- `topk`: top-K% decision (K = observed base rate in the test set)
- `auc`, `ap`: AUROC and average precision
- `recall_at_precision`: for each target precision in `{0.50, 0.80, 0.85, 0.90, 0.95}`, the maximum recall achievable while maintaining at least that precision, plus the operating threshold (`_data_pipeline.py:494–522`)

The recall@precision values are computed from the full precision-recall curve via `sklearn.metrics.precision_recall_curve`. If a precision target is unachievable, recall is recorded as 0.0 and threshold as NaN.

### 5.6 `extract_region_metrics`

Pulls `best_f1.f1` and `recall@p{80,85,90,95}` for a single region out of the `per_region` block (`_data_pipeline.py:605–617`). Returns NaN-filled dicts when the region is absent from the test set, so summary CSV rows always have the same columns regardless of which regions appear in a given run.

### 5.7 Diagnostic helpers

`log_split_breakdown`: for each split (train/val/test), logs total n, pos/neg counts, per-batch row counts, per-region row counts, and the bucket breakdown (IND/PHP/ALLSTAR/OTHER in candidates and audios). Run immediately after `assert_no_group_leak` to catch problems like "only 3 PHP audios in val" or "ALLSTAR silently dropped".

`log_variant_prelude`: single-line bucket summary logged before each model fit, so the log file is self-explanatory per variant without cross-referencing the split log.

`bucket_breakdown`: categorizes rows into IND (India region), PHP (Philippines region), ALLSTAR (batches 2676/2677), or OTHER, returning per-bucket audio and candidate counts plus positive/negative label counts.

---

## 6. Quick Reference

### Feature groups (8 groups, 55 features total)

| Group | Count | Library |
|---|---|---|
| G_DISFLUENCY | 6 | spaCy (degrades to regex), pure Python |
| G_STYLOMETRIC | 15 | spaCy for POS; pure Python for lexical |
| G_PAUSE | 15 | word timestamps from faster-whisper; spaCy for POS |
| G_SUSPICIOUS | 2 | word timestamps |
| G_FORMAL_AI | 4 | pure Python substring matching |
| G_PROSODIC | 8 | librosa (pyin F0, RMS energy) |
| G_VOICE_Q | 3 | parselmouth / Praat |
| G_PERPLEXITY | 2 | HuggingFace transformers (GPT-2) |

### Tier A vs EC2 model registry

| Component | Tier A name | EC2 name | Notes |
|---|---|---|---|
| Text all XGB | `text_all_xgb` | `text_all_xgb` | Identical |
| Text stylometric XGB | `text_stylo_xgb` | `text_stylo_xgb` | Identical |
| Text top-20 XGB | `text_top20_xgb` | `text_top20_xgb` | EC2 ranks on train set, Tier A on TRAIN_POOL |
| Whisper XGB | `whisper_wp_xgb` | `whisper_xgb` | Identical embedding |
| WavLM last XGB | `wavlm_wp_xgb` | `wavlm_xgb_last` | Same layer |
| WavLM layer 9 XGB | (absent) | `wavlm_xgb_l9` | EC2 addition |
| Everything | (absent) | `everything_xgb_{last,l9}` | EC2 addition |
| Whisper RF | `whisper_wp_rf` | (dropped) | Replaced by whisper XGB in pick3 |
| Dev/z augmented bases | `text_*_aug` (§4) | (absent) | Dropped — question pools differ |

### Pick definitions and weights

| Pick | Members | Weights | Frozen threshold | Where defined |
|---|---|---|---|---|
| pick1 / tierA_pick1 | text_top20 + whisper + wavlm | 0.20, 0.44, 0.36 | 0.59 | `_build_audios6_tier_a.py:469`; `xgboost_train.py:90` |
| pick2 / tierA_pick2 | text_all + wavlm | from frozen_configs.json | from frozen_configs.json | `_build_audios6_tier_a.py:470`; `xgboost_train.py:95` |
| pick3 / tierA_pick3 | text_top20 + text_stylo + whisper* | 0.12, 0.16, 0.72 | 0.59 | `_build_audios6_tier_a.py:472`; `xgboost_train.py:100` |

\* Tier A uses `whisper_wp_rf` (Random Forest); EC2 uses `whisper_xgb`. Weights and frozen threshold are the same.
