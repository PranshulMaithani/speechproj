# Per-Audio Cheating Detection — Methodology & Current System

*A multi-signal classifier that flags a single spoken answer as genuine vs. assisted/cheated.*

This document describes everything in the current approach: the problem framing, the
multi-signal feature pipeline, the model, the training/evaluation protocol, the two
models we finalised, and how the whole thing is kept reproducible. It is written to be
read straight into a presentation.

---

## 1. Problem framing

- **Unit of decision = one audio answer.** Each candidate answers multiple questions
  (e.g. Q25/Q26/Q27). We score **every audio independently** and never average a
  candidate's scores together. Reason: cheating can begin part-way through an exam — a
  candidate can answer Q25 honestly and read Q27 off a screen. Aggregating to a single
  per-candidate number would hide exactly the signal we care about.
- **Binary label per audio:** `1 = cheating / assisted`, `0 = genuine`.
- **No single signal is sufficient.** Different cheating styles leave different traces —
  reading from a screen flattens prosody and removes disfluency; AI-generated answers
  are stylometrically "too clean"; copying introduces unnatural pauses. So the design is
  explicitly **multi-signal fusion**: acoustic self-supervised embeddings + ASR-encoder
  embeddings + handcrafted linguistic/prosodic features, fused into one classifier.

---

## 2. Data

| Batch | Client | Region | Notes |
|-------|--------|--------|-------|
| audios2, audios4, audios5 | Client A | IND | Same production client (mic chain, candidate pool) |
| audios6 | Client B | IND + **PHP** | Different client — different mic, codec, candidate & question pool. Only batch with PHP region. |
| audios7 | new | — | Newest batch; used as a **held-out validity check** (never trained on). |
| 2676, 2677 (ALLSTAR) | auxiliary | — | Extra supervised acoustic signal. **Forced into train only**, never val/test. |
| casual | auxiliary | — | Casual-speech batch, **train-only** for Model 1. |

Key structural facts that drive the whole design:

- **Group = candidate (`group_id`).** All splits are *candidate-disjoint* — a candidate's
  audios are never split across train/val/test. This is enforced by
  `StratifiedGroupKFold` and a hard `assert_no_group_leak` check. Without it the model
  would "recognise the speaker" instead of detecting cheating.
- **Client shift is the hardest problem.** audios2/4/5 (Client A) and audios6 (Client B)
  sit at different absolute locations in embedding space. A model trained on a mixed
  random split and tested on a *held-out client* drops ~19 F1 points vs. an in-mix
  split — i.e. most of the apparent accuracy is "which client is this," not "is this
  cheating." Every protocol decision below is about measuring and controlling this.

---

## 3. The multi-signal feature pipeline

Each audio is turned into one feature vector by concatenating **three signal families**.
WavLM and Whisper are computed on the cloud (T4 GPU) from anonymised `.npy` waveforms;
the 55 handcrafted features are computed on the company laptop from transcripts.

```
  [ WavLM-base-plus mean-pool (768) | Whisper-medium encoder mean-pool (1024) | feat_* (55) ]
                                   = 1847-d raw vector
                       → StandardScaler → (optional PCA) → MLP → sigmoid
```

### 3a. WavLM-base-plus — self-supervised acoustic (768-d)
- `microsoft/wavlm-base-plus`, 16 kHz, mean-pooled over time.
- We cache **two layers**: `last` (final encoder output, the standard baseline) and
  **layer 9** (`hidden_states[9]`) — mid-network layers are known to carry the most
  *paralinguistic* information (prosody, voice quality, speaker effort), which is exactly
  the cheating-relevant content. Layer 9 is what Model 2 uses.

### 3b. Whisper-medium encoder — ASR acoustic-linguistic (1024-d)
- `openai/whisper-medium` **encoder** output, mean-pooled. Captures content/pronunciation
  structure the ASR model attends to. (Encoder only — we never use the decoder text here.)

### 3c. 55 handcrafted features (`feat_*`) — interpretable linguistic + prosodic
Computed on the laptop from a **rich timestamped transcript** (faster-whisper `small`
with a filler-priming prompt + word-level timestamps + VAD). The word timestamps are what
make the pause/rate features possible. Eight signal groups:

| Group | # | Examples | What it catches |
|-------|---|----------|-----------------|
| Disfluency | 6 | filler_rate, repetition_rate, repair_rate, hedge_rate | Genuine speech is *dis*fluent; reading/AI is too clean |
| Stylometric | 15 | ttr, mattr, mtld, avg/std sentence length, noun/verb/adj rate, self_ref_rate | Vocabulary richness & syntactic fingerprint of scripted vs. spontaneous |
| Pause | 15 | pause_mean/std/skew, long_pause_rate, pause_before_content/function, mid_phrase_pause_rate, articulation_rate | Reading vs. thinking — *where* and *how regularly* someone pauses |
| Suspicious gaps | 2 | suspicious_gap_count, suspicious_gap_ratio | Long silences consistent with looking something up |
| Formal / AI phrasing | 4 | formal_transition_rate, ai_phrase_rate | "Furthermore… in conclusion…" markers of AI/scripted text |
| Prosodic | 8 | f0_mean/std/range/skew/slope, energy_mean/std, speaking_rate_std | Flat, monotone delivery of read text |
| Voice quality | 3 | jitter_local, shimmer_local, hnr_mean | Micro-instability of natural voice (Praat/parselmouth) |
| Perplexity | 2 | mean_perplexity, burstiness | GPT-2 perplexity — AI text is unusually low-perplexity |

> Optional dependencies degrade gracefully with a loud warning: spaCy `en_core_web_sm`
> (POS rates), parselmouth/Praat (jitter/shimmer/HNR), GPT-2 (perplexity). If a dep is
> missing those columns are zero-filled — so the **same 55 columns in the same order**
> are produced for every batch, which is what keeps batches comparable.

### 3d. Data augmentation (train only)
8 cached variants per audio: `orig, noise, pitch, speed, gain, air, vtlp, combo`
(Gaussian SNR, ±2-semitone pitch, time-stretch, gain, air absorption, vocal-tract-length
perturbation, and a stochastic mix). **Augmentation expands the training matrix only —
val/test always use `orig`.** The scaler/PCA are fit on the aug-expanded train set.

---

## 4. The model

A compact **MLP head** on top of the frozen, cached features. The embeddings are *not*
fine-tuned — they are extracted once and reused, which makes the whole sweep cheap and
deterministic.

```python
Linear → BatchNorm1d → ReLU → Dropout   (per hidden layer)  →  Linear→1  →  sigmoid
```

- Loss: BCE-with-logits, **label smoothing 0.05**, gradient clip 1.0.
- Optimiser: AdamW + cosine LR schedule, early stopping on **val F1** (patience 10).
- **Class imbalance** handled with a `WeightedRandomSampler` (balanced minibatches in
  expectation) — chosen over `pos_weight` after testing both.

### Architecture variants
| Arch | Hidden | Dropout | Weight decay | Role |
|------|--------|---------|--------------|------|
| `default` | 512→256→128 | 0.40 | 5e-4 | Full-capacity MLP |
| `tiny` | 128 | 0.55 | 5e-3 | Small, heavily regularised |
| `linear` | (none) | 0.00 | 1e-2 | Logistic-regression sanity baseline — if it matches `tiny`, the data is essentially linear and the MLP is just memorising client artifacts |

---

## 5. Training & evaluation protocol

### 5a. Two split modes (the core methodology)
Both are **candidate-disjoint** and stratified by label.

- **Mode A — "20pct" (random in-mix split).** No held-out batch:
  `StratifiedGroupKFold(5)` over all candidates → fold0 = test, fold1 = val, folds 2–4 =
  train. Optimistic upper bound — train and test share clients.
- **Mode B — "a6" (held-out client).** `--test_batches audios6`: **test is fixed to
  audios6** for every seed; the remaining pool is re-partitioned into train/val. This is
  the realistic, harder number — it measures generalisation to a *different client*.

`--train_only_batches` (ALLSTAR `2676,2677`, and `casual` for Model 1) appends those rows
to train **after** the split so they can never leak into val/test.

### 5b. Metrics — we report the whole operating curve, not one number
For per-audio screening, **precision at the alert threshold matters more than raw F1** (a
false cheating flag is costly). So `compute_metrics` reports:
- AUC, Average Precision (AP),
- F1 at 0.5 and **best-F1 with its threshold**,
- **Recall @ precision = {50, 80, 85, 90, 95}%** — "how many cheaters do we still catch
  if we hold false-alarm rate to X%." This is the headline operating-point metric.
- Per-region (IND / PHP) and per-batch breakdowns.

### 5c. Fluke-proofing — `neural_baseline_multiseed.py`
A single run can win by a lucky split. To choose an **architecture** rather than a lucky
seed, we sweep every variant over **N seeds** (default 42–46) and report **mean ± std**
per variant:
- 20pct mode: each seed reshuffles *all* candidates.
- a6 mode: test stays audios6; each seed reshuffles only train/val.
- The ranking key is `avg_best_f1` (mean over seeds); low `std` + high `avg` = genuinely
  good architecture. `best_f1_seed` records which seed produced a variant's top run so a
  suspiciously-high single number can be identified as a fluke.

### 5d. The variant grid
**30 variants = 3 archs × 2 WavLM layers (`last`, `9`) × 5 PCA settings**
(`full, pca98, pca95, pca93, pca90`). The scaler + PCA are fit on train only; `feat_*`
are concatenated *before* PCA so even the pca90 variant still contains the text features
in compressed form.

---

## 6. The two finalised models

We did **not** pick one model — we keep **two**, one per protocol, because they answer
different questions. Both were re-run with `--export_artifacts true` so the exact weights,
scaler, PCA and an `inference_meta.json` are stored and downloadable.

| | **Model 1** | **Model 2** |
|---|---|---|
| Variant | `default_last_pca98` | `tiny_l9_pca95` |
| Architecture | `default` (512→256→128, dropout 0.40, wd 5e-4) | `tiny` (128, dropout 0.55, wd 5e-3) |
| WavLM layer | `last` | **layer 9** (paralinguistic) |
| PCA | keep 98% variance | keep 95% variance |
| Extra train data | **+ casual** batch (train-only) | none |
| Protocol | **20pct** (random in-mix) | **a6** (held-out audios6 client) |
| Run dir | `m1_casual_20pct/default_last_pca98/` | `m2_nocasual_a6/tiny_l9_pca95/` |
| Reads as | best-case ceiling, full capacity | conservative cross-client number |

**Why two:**
- **Model 1 (20pct, full capacity)** answers *"how well can we separate cheating from
  genuine when we have labelled data from this client?"* — the in-domain ceiling.
- **Model 2 (a6, tiny + layer 9)** answers *"how well do we hold up on a client we never
  trained on?"* The tiny, heavily-regularised head on the paralinguistic layer-9 features
  is what survived the held-out-client test best — a big MLP on the `last` layer mostly
  memorised client identity. On a6 the model is **AUC-strong but threshold-sensitive**
  (≈0.92 AUC with best-F1 ≈0.70) — it ranks cheaters well but the decision threshold does
  not transfer cleanly across clients, which is the open problem (see §9).

> Context: an earlier **stacked-ensemble** attempt underperformed (~0.6 F1), so we
> reverted to this fused-feature + per-protocol-model approach, which is stronger and far
> easier to reason about.

### Final evaluation — `evaluate_final_models.py`
Reloads each model's stored `model.pt` + `scaler` + `pca` and produces one workbook:
- **Mode A (own test set):** re-predicts and **verifies** the reproduced probabilities
  equal the saved `pred_score` (`max|recon−saved| < 1e-3` ⇒ bit-exact reproduction).
- **Mode B (`--test_batch audios7`):** scores the **same finalised weights** on a brand-new
  batch neither model trained on, with features rebuilt **in the model's exact training
  feature order**. This is the generalisation/validity check.
- Output `evaluate_final_models.xlsx`: `summary` + per-model `_preds` / `_sweep` (full
  confusion + precision/recall/F1 at every threshold) / `_by_region`, plus
  `predictions_<tag>.csv`.

**How to read the audios7 check:** compare `auc / ap / recall@p90` on audios7 vs. each
model's own-test row. Similar ⇒ the model holds up on new data; a large drop ⇒ it does
not generalise to audios7.

---

## 7. Reproducibility design

Every finalised number is re-derivable:

- **Frozen embeddings.** `extract_embeddings.py` computes WavLM/Whisper once into
  `embeddings_cache.npz`, keyed by filename, and **stamps the model IDs** into the cache.
  It refuses to mix caches built with different models (prevents silently comparing
  base-plus vs. large features). `orig` embeddings are deterministic; augs are seeded.
- **Self-contained (variant, seed).** In the multiseed runner the split is a function of
  `seed` alone, and the model init/sampler/shuffle is reseeded from
  `model_seed = sha256(f"{seed}:{variant}")` right before training. Re-running
  `--seeds <s> --variants <v>` reproduces that one model **bit-for-bit on CPU**, regardless
  of what else ran alongside it.
- **Ledger written every run:** `splits/seed_<seed>.json` (exact train/val/test candidate
  + filename membership), `per_run.csv` (every model's metrics + `model_seed`),
  `summary_mean_std.csv` (the fluke-proof ranking), and `inference_meta.json` per model
  (feature order, dims, model IDs, threshold, reproduce command).

---

## 8. Data / PII boundary (non-negotiable)

| Stays on company laptop (never leaves) | Goes to the cloud (EC2) |
|---|---|
| Raw `.wav` / listenable audio | Anonymised `.npy` waveforms (encoded, no real IDs) |
| Transcripts (contain PII) | `gt.csv` with only `feat_*` numbers + labels |
| `cid_mapping.json` (real CID → `G_NNNNN`) | Embedding caches |

Transcription **and** the 55 text features are computed on the laptop; only anonymised
numeric features + encoded waveforms are uploaded. Real candidate IDs are mapped to
`G_NNNNN` via a mapping file that never leaves the laptop. No cloud script ever reads a
transcript or a wav.

---

## 9. Known limitations & current open problems

- **Cross-client threshold transfer (the main one).** On the held-out-client protocol the
  model ranks well (high AUC) but the F1-optimal threshold doesn't transfer — score
  *scale* shifts per client. Mitigations in the codebase: per-client feature
  standardisation (unsupervised, features-only) and a few-shot adaptation mode (label a
  small slice of a new client before going live). This is the active iteration target.
- **PHP region is thin.** audios6 is the only PHP source, so PHP per-region metrics rest
  on a small sample.
- **Handcrafted-feature dependencies.** Missing spaCy/Praat/GPT-2 silently zero-fills
  those columns; the laptop environment must have all three for full-strength features.

---

## 10. End-to-end pipeline (one glance)

```
COMPANY LAPTOP                                   CLOUD (EC2 + T4 GPU)
─────────────                                    ───────────────────
audios7/*.wav + audios7GT.csv
   │ extract_features_batch.py
   │   (faster-whisper small + word timestamps
   │    → rich transcript → 55 feat_*)
   ▼
audios7_features.csv  (LOCAL, has PII-free numbers)
   │ neural_baseline_prep.py
   │   (anonymise CID→G_NNNNN, build gt.csv, encode npy)
   ▼
gt.csv + anonymised audios7 .npy  ──upload via bucket──▶  extract_embeddings.py
                                                            (WavLM + Whisper → cache,
                                                             incremental, stamped)
                                                                   │
                                          neural_baseline_multiseed.py (choose arch)
                                          neural_baseline_train.py --export_artifacts
                                                                   │
                                          evaluate_final_models.py (--test_batch audios7)
                                                                   ▼
                                              evaluate_final_models.xlsx + predictions CSVs
```

Operational runbooks for each step live in `COMMANDS_ADD_BATCH.txt` (add a batch),
`COMMANDS_CACHE_FIX.txt` (diagnose/stamp/extend the cache), and `COMMANDS_ANALYSIS.txt`
(final evaluation + audios7 validity check).

---

### Where the live numbers come from
This document fixes the **methodology and configuration**. The actual headline metrics for
a presentation should be quoted from the latest run outputs:
- `runs/.../summary_mean_std.csv` — per-architecture mean ± std (fluke-proof ranking).
- `runs/final_eval/evaluate_final_models.xlsx` — the two finalised models on their own
  test sets (`summary` sheet: AUC, AP, best-F1, recall@p80/85/90/95).
- `runs/final_eval_audios7/evaluate_final_models.xlsx` — the same two models on audios7
  (the generalisation check).
