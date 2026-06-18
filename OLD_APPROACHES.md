# Earlier Approaches (Pre-V3) — Detailed Documentation

*The two approaches that came before the current multi-signal neural system, why each
was built, exactly how it worked, what it scored, and why we moved on.*

This is the historical companion to `METHODOLOGY.md` (which documents the **current**
system). Read this for the "how we got here" half of the presentation — it shows the
problem was attacked from two angles, both hit the same conceptual wall, and the lessons
fed directly into today's design.

---

## Chronology (read this first)

| When | Folder | What it was |
|------|--------|-------------|
| **Early March 2026** | `old/` | **Approach 1** — fine-tune wav2vec2 / WavLM / Whisper to classify **read vs spontaneous** speech on ALLSSTAR, then convert per-window predictions into a **`read_ratio`** per file and flag "reading" as a proxy for cheating. CPU/ONNX inference pipeline (`predict_cpu.py`). |
| **March 2026** | `old/`, `old2/` | **Approach 2** — **XGBoost** over handcrafted features. First an interpretable prosodic baseline (`old/src/models/train_xgboost.py`), then a **multi-signal ensemble** (`old2/train_xgboost_ensemble.py`) that fuses text + pause + prosodic features **with the wav2vec2 model's own scores**. This is the direct ancestor of the current fusion idea. |
| **May–June 2026** | `ec2/`, `companylaptop/` | **Current V3** — per-audio cheating classifier on real Mettl labels, fusing WavLM + Whisper embeddings + 55 handcrafted features, two finalised models. Documented in `METHODOLOGY.md`. |

> Both early approaches were trained on **ALLSSTAR** (the `2676` / `2677` corpora), *not*
> on company exam audio, and they predicted **"is this read speech?"** — a *proxy* for
> cheating. The wall they both hit: **read ≠ cheating.** That realisation is what created
> the current approach, which trains on actual per-audio cheating labels.

---

## Shared foundation — the ALLSSTAR read/spontaneous corpus

Both approaches share one dataset and one labelling scheme, built by
`old/src/data/build_manifest.py`:

- **`2676/` → `label = spontaneous` (0)**, **`2677/` → `label = read` (1)**. These are the
  same two ALLSSTAR batches that survive into the current system as the **ALLSTAR
  train-only auxiliary** (`2676,2677`).
- **Filename schema** `ALL_{speakerID}_{Gender}_{L1}_{L2}_{Task}.wav` is parsed into
  metadata: speaker, gender, **L1 (native language / accent)**, task. Task codes encode
  the elicitation style (e.g. `QNA` spontaneous Q&A vs read passages like `LPP`, `HT1/2`,
  `DHR`, `NWS`).
- **~1,050 files (≈699 read / ≈351 spontaneous).**
- **Speaker-stratified split (70/15/15)** by speaker, stratified on L1 — all of one
  speaker's files stay in one split (no speaker leak), and accents are balanced across
  splits. **Accent oversampling** weights (e.g. HIN ×3, GUJ ×3, IND ×2) push the model
  toward the South-Asian target population.

---

# Approach 1 — Read-vs-Spontaneous acoustic model → `read_ratio`

**Folder:** `old/`  ·  **Core hypothesis:** *cheaters read pre-written answers, so detecting
reading detects cheating.*

### 1.1 Model architecture (`old/src/models/train_*.py`)
A pretrained speech encoder (frozen feature extractor + partially frozen transformer) with
a small classification head on top of a **mean-pooled** embedding:

```
raw waveform (16 kHz, N-sec window)
  → Wav2Vec2 / WavLM encoder            (CNN feature-extractor FROZEN;
                                         first 6 transformer layers FROZEN)
  → mean-pool over time                 → (768,)
  → Linear(768, 256) → ReLU → Dropout(0.3) → Linear(256, 2)
  → softmax → [P(spontaneous), P(read)]
```

- **Encoders tried:** `facebook/wav2vec2-base` (primary), `microsoft/wavlm-base-plus`
  (the WavLM swap — same pipeline, one-line config change), and a Whisper-medium encoder
  variant (`train_whisper_cls.py`, takes 80-bin log-mel spectrograms).
- **Window sizes swept:** 5 / 7.5 / **10** / 12.5 / 15 seconds — one model per window size
  (`train_wav2vec2_10sec.py` etc.). The 10-sec model is what the user remembers as
  "trained on ALLSTAR 10-sec data."
- **Training (from `old/configs/config.yaml`):** AdamW, lr 2e-5, warmup 10%, fp16,
  `CrossEntropyLoss` with **class weights**, accent-weighted sampler, early stopping on
  val-F1, grad-clip 1.0. Whisper variant: lr 1e-3, hidden 512.

### 1.2 The `read_ratio` inference pipeline (`old/predict_cpu.py`)
A **self-contained CPU/ONNX** script ("runs on a potato laptop", no PyTorch needed). Per
file:

1. **Load + normalise** audio to 16 kHz mono (cap 120 s).
2. **VAD** — **Silero VAD** (with an RMS-energy fallback) to find speech segments, so the
   model **never sees silence**.
3. **VAD-gated windowing** — slide an N-sec window (adaptive hop ≈ 0.4×window, floor 2.5 s)
   over *speech only*.
4. **Per-window inference** — ONNX-INT8 model → `[P(spont), P(read)]` per window; argmax →
   `read` / `spontaneous` (optional confidence gating).
5. **Temporal median smoothing** over the window label sequence (kills isolated flips).
6. **Merge** consecutive same-label windows into a **segment timeline**; enforce a minimum
   segment length (3 s).
7. **`read_ratio` = (# windows labelled `read`) / (# speaking windows).**
8. **Decision:** `overall_label = "read" if read_ratio ≥ read_threshold else "spontaneous"`;
   `cheating_suspected = (overall_label == "read")`. Threshold **0.45**, tuned on company
   data (default had been 0.50).

Output JSON per file: verdict, overall confidence, `read_ratio`, a **segment-by-segment
timeline** (so you can see *which part* was read), and every window prediction. This
per-segment, per-file granularity is the ancestor of today's **per-audio** rule.

> `next_steps_report.md` (March 2026) documents the planned refinements that were never
> fully pursued because the approach was superseded: threshold calibration, test-time
> augmentation (averaging over window offsets), **duration-weighted voting** (weight each
> window's vote by its length × confidence instead of one-window-one-vote), a WavLM
> retrain, and eventually a multi-scale shared-encoder fusion model.

### 1.3 Results — read/spontaneous classification on ALLSSTAR test
| Model | Window | Test F1 | Test Acc |
|-------|--------|---------|----------|
| wav2vec2 | 5 s | **0.982** | 0.975 |
| wav2vec2 | 10 s | 0.976 | 0.967 |
| wav2vec2 | 15 s | 0.974 | 0.965 |
| WavLM-base-plus | 5 s | 0.978 | 0.970 |
| wav2vec2 "biased" (single-neuron, thr 0.65) | 5 s | 0.978 | 0.971 |

On the **read/spontaneous task it is near-perfect (~0.97–0.98 F1)** and very confident
(read-segment mean P(read) ≈ 0.97, spontaneous ≈ 0.04). Exported to ONNX-INT8 (~122 MB,
~142 ms / 5-s window on CPU).

### 1.4 Why we moved on
- **On company exam audio it was only ~94% precision / 76% recall** — 1 in 4 read cases
  missed — despite near-perfect ALLSSTAR scores. The domain gap (mic, codec, real exam
  speech) was large.
- **The fatal conceptual flaw: "read vs spontaneous" is a weak proxy for "cheating."**
  - A fluent, honest candidate can sound like a reader → false positive.
  - Someone **paraphrasing** GPT/notes sounds spontaneous → false negative.
  - The model has **no notion of cheating in a Mettl exam context** — only of reading.
- **GT label noise** — the 219-file company eval set was being relabelled; small set,
  every mislabel moved recall ~0.5 pt.

These three points are exactly what the current system fixes: train on **real cheating
labels**, fuse **multiple signals** (so no single proxy decides), keep **per-audio**
output, and measure on a **held-out client**.

---

# Approach 2 — XGBoost over features (the fusion ancestor)

**Folders:** `old/` (baseline) and `old2/` (ensemble). XGBoost was chosen as the
**interpretable** counterpart to the black-box acoustic model — it gives feature
importances, trains in seconds, and lets you ablate signal groups.

### 2a. Interpretable prosodic baseline — `old/src/models/train_xgboost.py`
- **Input:** pre-extracted per-window handcrafted features (`window_features.csv`) — MFCC,
  prosody, pause, energy. Standard-scaled (XGBoost doesn't need it, but it stabilises).
- **Model:** `XGBClassifier` (500 trees, depth 6, lr 0.05, subsample 0.8, `hist`), accent +
  class sample weights, same speaker-stratified splits as Approach 1.
- **Outputs:** model + scaler + **top-30 feature-importance plot** + confusion matrix +
  per-accent accuracy breakdown.
- **Result (ALLSSTAR test):** **F1 0.879, AUC 0.883**, precision 0.923, recall 0.840. Solid,
  interpretable, but clearly below the acoustic model — pure handcrafted features alone
  don't separate read/spontaneous as well as the learned encoder.

### 2b. Multi-signal ensemble — `old2/train_xgboost_ensemble.py`  ← the key one
This is the **fusion** model and the conceptual seed of the entire current approach. It
trains XGBoost on **44 features in four groups**:

| Group | # | Examples |
|-------|---|----------|
| **Text** | 20 | filler_rate, repetition_rate, repair_rate, ttr/mattr, sentence-length stats, self_ref_rate, discourse_marker_rate, hedge_rate, noun/verb/adj rate |
| **Pause** | 13 | pause_mean/std/median/skew, long_pause_rate, pause_regularity, pause_before_content/function_ratio, mid_phrase_pause_rate, articulation_rate |
| **Prosodic** | 8 | f0_mean/std/range/skew/slope, energy_mean/std, speaking_rate_std |
| **Acoustic-model scores** | 3 | **`wav2vec2_read_ratio`, `wav2vec2_mean_p_read`, `wav2vec2_max_p_read`** |

The last group is the bridge: **Approach 1's output (the read_ratio + read-probability
stats) is fed in as features** alongside the linguistic/prosodic signals — an early,
hand-built version of "fuse the acoustic model with the text model."

- **Model:** XGBoost (500 trees, depth 6, lr 0.05, `scale_pos_weight` for imbalance,
  early stopping 30 rounds), 5-fold CV, **per-group ablation** (Text-only / Pause-only /
  Prosodic-only / Text+Pause / All), and a **threshold-sensitivity sweep** (0.3–0.8).
- **Result (test):** **F1 0.972, precision 0.999, recall 0.946, AUC 0.995**, CV F1
  0.969 ± 0.004 — the **best of all the early models**, beating the standalone acoustic
  model on precision/AUC.
- **Feature importance — the acoustic scores dominate:** `wav2vec2_read_ratio` (0.37),
  `wav2vec2_max_p_read` (0.31), `wav2vec2_mean_p_read` (0.26) together ≈ **94%** of total
  importance. **Reading:** fusion helped, but mostly by *re-using the acoustic model's
  verdict*; the handcrafted text/pause/prosodic features added the remaining few points
  of precision. A text-only variant (`--no-wav2vec2`) is supported for the case where the
  acoustic scores aren't available.

### 2c. "XGBoost over the embeddings" — where that lives now
In these **old** folders, XGBoost ran over **handcrafted features + acoustic *scores*** —
not over the raw neural embedding vectors. The idea of running **XGBoost directly over the
WavLM/Whisper embeddings** became part of the **current** pipeline:
`ec2/xgboost_train.py` (companion to the neural baseline, writes `summary2.csv` so NN vs
XGB are comparable on identical rows). Its variants include `wavlm_xgb` (768-d), `whisper_xgb`
(1024-d), `everything_xgb` (WavLM+Whisper+feat_*), and weighted-average "picks". So the
lineage is: **handcrafted-feature XGBoost (old) → feature+score ensemble (old2) →
embedding XGBoost head (current).**

---

## How these two approaches produced the current system

| Early idea | Became, in V3 |
|------------|---------------|
| Read-vs-spontaneous proxy label | **Real per-audio cheating label** (0/1) |
| One acoustic model decides | **Multi-signal fusion** (WavLM + Whisper + 55 feat_*) — no single proxy decides |
| wav2vec2-base fine-tuned end-to-end | **Frozen WavLM-base-plus + Whisper-medium embeddings** + small MLP head (cheap, reproducible) |
| Hand-built ensemble (text+pause+prosodic + acoustic scores) | The **55 `feat_*`** handcrafted features (same 8 signal groups, expanded) concatenated with embeddings |
| `read_ratio` over VAD windows; per-segment timeline | **Per-audio** scoring kept; never aggregate across a candidate's questions |
| Speaker-stratified ALLSSTAR split | **Candidate-disjoint, client-held-out** protocols (20pct vs a6) to measure real generalisation |
| ALLSSTAR `2676`/`2677` as the whole dataset | Same `2676`/`2677` kept as **train-only ALLSTAR auxiliary** acoustic signal |

**One-line takeaway for the talk:** *the early work proved we could detect reading almost
perfectly and that fusing signals beats any single model — but it also proved that
"reading" is the wrong target. V3 keeps the fusion lesson and the per-audio granularity,
and swaps the proxy label for genuine cheating labels measured under client shift.*

---

## Limitations of the old approaches (so the contrast is honest)
- **Proxy target.** Everything was trained on read-vs-spontaneous, not cheating — high
  ALLSSTAR scores overstate real-world performance (94P/76R on company data).
- **No client-shift measurement.** Splits were speaker-stratified within one corpus; there
  was no held-out-client test, so generalisation to a new exam client was never measured.
- **Acoustic-score leakage in importance.** The "fusion" ensemble's strength came ~94% from
  the acoustic model's own output; the handcrafted features contributed little on this
  proxy task (they matter more on the real cheating task).
- **Small, noisy eval set.** 219 company files, partially relabelled — numbers were
  unstable.

### Pointers (files referenced)
- Approach 1 train: `old/src/models/train_wav2vec2_10sec.py` (+ 5/7.5/12.5/15 s,
  `train_wavLM_5sec.py`, `train_whisper_cls.py`)
- Approach 1 inference: `old/predict_cpu.py`; data/labels: `old/src/data/build_manifest.py`;
  config: `old/configs/config.yaml`; roadmap: `old/next_steps_report.md`
- Approach 2 baseline: `old/src/models/train_xgboost.py`
- Approach 2 ensemble: `old2/train_xgboost_ensemble.py`; narrative:
  `old2/BRAINSTORMING_AND_DOCUMENTATION.md`
- Current XGBoost-over-embeddings: `ec2/xgboost_train.py`
- Result JSONs quoted above: `old/checkpoints/*_results.json`,
  `old2/checkpoints_ensemble/ensemble_results.json`
