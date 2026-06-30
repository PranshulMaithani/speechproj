# Presentation — Rich Slide Content (Slides 4 → 20)

Detailed, presentation-grade content for each slide so you have plenty of material to
**compress onto the actual slide**. Follows the slide numbering in
`PRESENTATION_OUTLINE.md`. For each slide you get:

- **Message** — the single takeaway (put a version of this in the slide title)
- **Full explanation** — the complete, correct version of the idea
- **Slide bullets** — richer than they need to be; trim to 4–6 short lines
- **Key facts/numbers** — exact values to keep accurate on stage
- **Say this** — a ~20–40s spoken script
- **Visual** — what to draw
- **If asked** — a Q&A safety net

> Live metrics are marked `‹fill: …›`. Pull them from `summary_mean_std.csv` and
> `evaluate_final_models.xlsx` so nothing on stage is invented. Documented numbers (old
> approaches, the a6 ≈0.92 AUC figure) are stated inline.

---

## Slide 4 — What we tried before (the ~10%)

**Message:** Two earlier approaches worked on a *proxy* task and hit the same wall —
*reading is not the same as cheating* — which is exactly why the current system exists.

**Full explanation.**
Before the current system we attacked the problem from two angles, both trained on the
public **ALLSTAR** corpus (`2676` = spontaneous, `2677` = read), **not** on real exam audio.

- **Approach 1 — read-vs-spontaneous fine-tuning → "read ratio."** We fine-tuned
  wav2vec2 / WavLM / Whisper to classify short windows as *read* vs *spontaneous*, then
  converted per-window predictions into a single **`read_ratio`** per file (fraction of
  speaking windows labelled "read") and flagged a high read-ratio as "probably cheating."
  Inference ran on CPU via ONNX + Silero-VAD, sweeping window sizes (5 / 7.5 / 10 / 12.5 /
  15 s). It was *excellent at its proxy task* — but the proxy was the problem.
- **Approach 2 — XGBoost fusion.** First an interpretable prosodic baseline, then a
  **multi-signal ensemble** that fused text + pause + prosodic features **with the
  wav2vec2 model's own read-vs-spont scores**. This is the direct ancestor of today's
  fusion idea — but its feature importance was dominated by the single acoustic proxy
  score, so it was really "read detection" wearing a fusion costume.

**Both hit the same wall: read ≠ cheating.** A candidate can read and be honest (nervous,
slow) or speak fluently and still be cheating (memorised an AI answer). The proxy didn't
transfer to real candidate data. That realisation forced the current design: **train on
actual per-audio cheating labels, and fuse genuinely complementary signals.**

**Slide bullets (compress):**
- Approach 1: fine-tune for read-vs-spontaneous → `read_ratio` proxy
- Approach 2: XGBoost fusing features **+ the acoustic proxy score**
- Both trained on ALLSTAR (public), not real exam audio
- Strong on the proxy, **but read ≠ cheating** → didn't transfer
- Lessons → real cheating labels + true multi-signal fusion

**Key facts/numbers:**
- Approach 1 best proxy result: **wav2vec2 @ 5 s window, F1 ≈ 0.982** (read-vs-spont on ALLSTAR)
- On real company data it behaved as a *read detector*: ≈ **94% precision / ≈76% recall**
- Approach 2b ensemble: **F1 ≈ 0.972, AUC ≈ 0.995** — but **~94% of importance = the wav2vec2 score**
- ALLSTAR: ~1,050 files (~699 read / ~351 spontaneous)

**Say this:** "We first tried to detect *reading* — fine-tuning speech models to tell read
from spontaneous speech, then flagging anyone who 'reads too much.' It scored ~0.98 on that
task. But on real candidates it was just a read detector — and reading isn't cheating. A
second XGBoost approach fused more features, but ~94% of its decision still came from that
one acoustic score. Both taught us the same lesson: train on actual cheating labels and fuse
signals that are genuinely different. That's the system I'll show next."

**Visual:** two "tried" cards (Approach 1 / Approach 2) → red "read ≠ cheating" stamp → arrow
into the current-system diagram.

**If asked:** ALLSTAR survives today as the **train-only auxiliary** (`2676,2677`) — we kept
the data, dropped the proxy framing.

---

## Slide 5 — Current system at a glance

**Message:** One audio becomes three complementary signal families, fused into one vector,
scored by one small model — and the heavy parts are computed once and frozen.

**Full explanation.**
Every audio answer is turned into a single feature vector by concatenating three families:
a **self-supervised acoustic** embedding (WavLM), an **ASR-encoder** embedding (Whisper),
and **55 interpretable linguistic/prosodic** features. That 1847-dimensional vector is
standardised, optionally compressed with PCA, and fed to a compact MLP that outputs a
cheating probability. Crucially, **WavLM and Whisper are extracted once into a cache and
never fine-tuned** — so the entire model search is cheap, deterministic, and reproducible.

**Slide bullets:**
- WavLM mean-pool → **768**
- Whisper encoder mean-pool → **1024**
- Handcrafted features → **55**
- Concatenate → **1847-d** → scale → PCA → MLP → **P(cheat)**
- Embeddings **frozen** (extracted once, cached)

**Key facts/numbers:** `768 + 1024 + 55 = 1847`. Output is a per-audio probability in [0,1].

**Say this:** "Here's the spine of the whole talk. Each audio goes through three signal
families — an acoustic model, an ASR encoder, and 55 hand-built features. We stack them into
one 1847-dimensional vector, standardise it, compress it, and a small neural net turns it
into a cheating probability. The big models are run once and frozen — that's what makes
everything after this fast and reproducible."

**Visual:** the left-to-right pipeline diagram (three coloured inputs merging → 1847 → scale
→ PCA → MLP → P). Use this exact diagram, expanded, on slides 7–11.

**If asked:** Why freeze instead of fine-tune? Limited labels + we want a cheap 30-variant
sweep; frozen features make each run deterministic and bit-reproducible.

---

## Slide 6 — Data & the real challenge

**Message:** The hard problem isn't "cheating vs genuine" — it's generalising from one
client to a *different* client, because that shift dominates the feature space.

**Full explanation.**
Our data spans several batches. **audios2/4/5 are one production client (Client A, IND
region); audios6 is a different client (Client B)** — different microphone chain, codec,
candidate pool, question pool, and the only batch with the **PHP** region. **audios7** is the
newest batch, held out entirely as a validity check. **ALLSTAR (2676/2677)** and a
**casual-speech** batch are auxiliary, forced into training only.

Two structural rules drive everything:
- **Candidate-disjoint splits.** The group key is the *candidate*; a person's audios never
  span train/val/test. Otherwise the model "recognises the speaker" instead of detecting
  cheating — enforced by `StratifiedGroupKFold` + a hard `assert_no_group_leak`.
- **Client shift is the dominant effect.** Client A and Client B sit at different absolute
  locations in embedding space. A model trained on a mixed split but tested on a *held-out
  client* drops **~19 F1 points** — meaning most of the apparent accuracy is "which client
  is this," not "is this cheating." Every later protocol decision is about measuring and
  containing this.

**Slide bullets:**
- Client A = audios2/4/5 (IND) · Client B = audios6 (IND + PHP, new mic/codec)
- audios7 = held-out validity batch (never trained on)
- ALLSTAR + casual = train-only auxiliary
- Splits are **candidate-disjoint** (group = candidate)
- Held-out client costs **~19 F1 points** ← the core challenge

**Key facts/numbers:** ~19 F1-point drop on held-out client; audios6 is the only PHP source.

**Say this:** "A quick map of the data. Three batches are one client; audios6 is a totally
different client — different mic, codec, people. We always split by candidate, so the model
can't just memorise a voice. And here's the punchline: if we train mixed but test on a
client we've never seen, we lose about 19 F1 points. So a lot of 'accuracy' is really the
model guessing which client it's looking at. Containing that is what the rest of the method
is about."

**Visual:** two clusters in 2-D embedding space (A vs B) with a dashed "shift" arrow; a
candidate icon spanning train/test crossed out to show the disjoint rule.

**If asked:** Mitigations exist (per-client standardisation, few-shot adaptation) — covered on
slide 20.

---

## Slide 7 — Signal family 1: WavLM (self-supervised acoustic)

**Message:** WavLM captures *how* something is said, and a mid-network layer captures the
speaking-style cues that matter for cheating better than the final layer.

**Full explanation.**
We use `microsoft/wavlm-base-plus` at 16 kHz, mean-pooled over time into a **768-d** vector.
We cache **two layers**: `last` (the final encoder output — the standard baseline) and
**layer 9** (`hidden_states[9]`). Mid-network layers of self-supervised speech models are
known to carry the most **paralinguistic** information — prosody, voice quality, speaking
effort — which is exactly the cheating-relevant content (read or memorised speech sounds
flatter and more effortful-or-too-smooth). Final-layer features drift toward the
pre-training objective and lose some of that. Model 2 uses layer 9 for this reason; we keep
both and let the sweep choose.

**Slide bullets:**
- `wavlm-base-plus`, 16 kHz, mean-pooled → **768-d**
- Cache two layers: **`last`** and **layer 9**
- Layer 9 = paralinguistics (prosody, voice quality, effort)
- "How it's said," not "what was said"
- Model 2 runs on layer 9

**Say this:** "The first signal is WavLM — a self-supervised speech model. It encodes *how*
something is said. We pull two layers: the last one, and layer 9 in the middle. The middle
of these networks is where speaking style lives — prosody, effort, voice quality — and that's
exactly what changes when someone reads or recites. Our second model leans entirely on that
layer-9 signal."

**Visual:** a transformer stack with layer 9 highlighted, small tags "prosody / effort /
voice quality"; "last" also marked.

**If asked:** mean-pool over time (not CLS) because there's no sentence token; we pool with
length weighting across chunks.

---

## Slide 8 — Signal family 2: Whisper encoder (ASR acoustic-linguistic)

**Message:** Whisper's *encoder* adds the content/pronunciation structure WavLM doesn't —
and we deliberately never touch its decoded text.

**Full explanation.**
We take the **encoder** of `openai/whisper-medium`, mean-pooled into a **1024-d** vector.
The ASR encoder represents the acoustic-linguistic structure the model attends to when
transcribing — pronunciation, phonetic content, delivery — which is complementary to WavLM's
self-supervised view. We use the **encoder only**; we never decode text here. The text
signal is captured separately and *interpretably* by the 55 handcrafted features (next
slide), so the embedding side stays purely acoustic.

**Slide bullets:**
- `whisper-medium` **encoder**, mean-pooled → **1024-d**
- Content/pronunciation structure the ASR attends to
- **Encoder only — decoder/text never used here**
- Complementary to WavLM's self-supervised view

**Say this:** "Second signal: the encoder of Whisper, the ASR model. It captures the
content-and-pronunciation structure the model uses to transcribe — a different view from
WavLM. Note we only use the encoder; we never use Whisper's text output here. The text side
is handled separately, in a way we can actually interpret, which is the next slide."

**Visual:** Whisper drawn as encoder (green, "used") + decoder (greyed, "not used").

**If asked:** medium, not large, for cost/throughput on a single T4; the encoder dim is 1024.

---

## Slide 9 — Signal family 3: 55 handcrafted features (interpretable)

**Message:** 55 hand-built features encode the linguistic and prosodic tells a careful human
reviewer would notice — and they're fully interpretable.

**Full explanation.**
Computed on the laptop from a **rich, word-timestamped transcript** (faster-whisper `small`
with a filler-priming prompt + word timestamps + VAD). Word-level timing is what makes the
pause/rate features possible. The 55 features fall into **eight groups**, each catching a
different cheating tell:

| Group | # | Examples | Catches |
|---|---|---|---|
| Disfluency | 6 | filler_rate, repetition_rate, repair_rate, hedge_rate | genuine speech is *dis*fluent; read/AI is too clean |
| Stylometric | 15 | TTR, MATTR, MTLD, sentence-length mean/std, POS rates, self_ref_rate | scripted vs spontaneous vocabulary/syntax fingerprint |
| Pause | 15 | pause mean/std/skew, long_pause_rate, pause_before_content/function, articulation_rate | *where* and *how regularly* someone pauses |
| Suspicious gaps | 2 | suspicious_gap_count, suspicious_gap_ratio | long silences consistent with looking something up |
| Formal / AI phrasing | 4 | formal_transition_rate, ai_phrase_rate | "furthermore… in conclusion…" AI/scripted markers |
| Prosodic | 8 | f0 mean/std/range/skew/slope, energy mean/std, speaking_rate_std | flat, monotone delivery of read text |
| Voice quality | 3 | jitter_local, shimmer_local, hnr_mean | micro-instability of a natural voice |
| Perplexity | 2 | mean_perplexity, burstiness | GPT-2 perplexity — AI text is unusually low-perplexity |

A robustness detail worth a sentence: optional dependencies (spaCy / Praat-parselmouth /
GPT-2) **degrade gracefully** — if one is missing those columns are zero-filled, so **the
same 55 columns in the same order** are produced for every batch. That column stability is
what keeps batches comparable across the whole pipeline.

**Slide bullets:**
- From a word-timestamped transcript (faster-whisper)
- 8 groups: Disfluency · Stylometric · Pause · Suspicious gaps · Formal/AI · Prosodic · Voice quality · Perplexity
- Interpretable: filler rate, MTLD, long-pause rate, AI-phrase rate, f0 slope, jitter, GPT-2 perplexity
- Same 55 columns, same order, every batch

**Key facts/numbers:** 55 features, 8 groups; GPT-2 perplexity + burstiness for AI text.

**Say this:** "Third family — 55 features we built by hand from a word-level transcript.
They're grouped into eight kinds of tell: disfluency, vocabulary richness, pause patterns,
suspicious gaps, AI-style phrasing, prosody, voice quality, and text perplexity. The nice
thing is they're interpretable — 'this answer pauses too regularly and is too low-perplexity'
is something you can actually explain. And they're computed identically for every batch, so
nothing drifts."

**Visual:** 8 labelled tiles (icon + one feature each), colour-matched to the "handcrafted"
family colour.

**If asked:** these are computed on the laptop (PII) and only the *numbers* are uploaded —
see the PII slide.

---

## Slide 10 — Fusion & preprocessing

**Message:** Concatenate the three families, standardise, then PCA — turning 1847 raw numbers
into one clean, comparable vector without leaking test information.

**Full explanation.**
The three blocks are concatenated into a single **1847-d** vector, passed through a
`StandardScaler`, then an optional `PCA`. Both the scaler and PCA are **fit on the training
set only** (and on the augmentation-expanded training set, at that) — never on val/test, so
there's no leakage. The handcrafted features are concatenated **before** PCA, so even the
aggressive pca90 variant still carries the text signal in compressed form. PCA both denoises
the high-dim embeddings and makes the 30-variant search cheap.

**Slide bullets:**
- `[768 | 1024 | 55] = 1847-d`
- `StandardScaler` → optional `PCA` (variance kept: full/98/95/93/90)
- Scaler + PCA fit on **train only** (no leakage)
- feat_* concatenated **before** PCA → kept even at pca90

**Say this:** "We stack the three families into one 1847-dimensional vector, standardise it,
and optionally run PCA. Everything is fit on the training data only — the scaler and PCA
never see val or test. And we add the hand features before PCA, so even when we compress
hard, the linguistic signal survives."

**Visual:** three coloured bars stacking into one long bar, then shrinking through a PCA
funnel.

**If asked:** PCA variance is a swept hyperparameter — slide 12.

---

## Slide 11 — The classifier

**Message:** On frozen features a small MLP is enough; we treat *capacity* as a knob and keep
a linear sanity baseline to catch memorisation.

**Full explanation.**
The head is a compact MLP: `Linear → BatchNorm → ReLU → Dropout` per hidden layer, then
`Linear → 1 → sigmoid`. Training uses BCE-with-logits with **label smoothing 0.05**, gradient
clipping 1.0, AdamW with a cosine LR schedule, and **early stopping on validation F1**
(patience 10). Class imbalance is handled with a **`WeightedRandomSampler`** (balanced
minibatches in expectation), chosen over `pos_weight` after testing both. We define three
architectures:

| Arch | Hidden | Dropout | Weight decay | Role |
|---|---|---|---|---|
| `default` | 512→256→128 | 0.40 | 5e-4 | full-capacity MLP |
| `tiny` | 128 | 0.55 | 5e-3 | small, heavily regularised |
| `linear` | none | 0.00 | 1e-2 | logistic-regression sanity baseline |

The `linear` model is a deliberate tripwire: if it matches `tiny`, the data is essentially
linear in this feature space and a big MLP is just memorising client artifacts — which is
precisely what we observed on the held-out client.

**Slide bullets:**
- MLP: `Linear→BN→ReLU→Dropout … →Linear(1)→sigmoid`
- BCE + label-smoothing 0.05, grad-clip 1.0, AdamW + cosine
- Early stop on **val F1**; `WeightedRandomSampler` for imbalance
- 3 archs: **default** (512→256→128) · **tiny** (128) · **linear** (sanity baseline)

**Say this:** "Because the features are frozen, the model itself can be small. It's a short
MLP with the usual regularisation, early-stopped on validation F1, with balanced minibatches
to handle the class imbalance. We run three sizes — a full one, a tiny heavily-regularised
one, and a plain linear baseline. That linear one is a tripwire: if it ties the others, the
problem is basically linear and a big network is just memorising the client."

**Visual:** three architecture cards side by side with their "role" line.

**If asked:** label smoothing + dropout 0.4–0.55 because the held-out-client setting punishes
overconfident memorisation.

---

## Slide 12 — The 30-variant sweep

**Message:** We don't *guess* the architecture — we sweep 30 and let the evaluation pick.

**Full explanation.**
The variant grid is **30 = 3 architectures × 2 WavLM layers (`last`, `9`) × 5 PCA settings
(`full, pca98, pca95, pca93, pca90`)**. Because embeddings are frozen and cached, training 30
heads is cheap. This converts "which architecture?" from an opinion into a measured ranking,
and it's the search space the two finalised models were selected from.

**Slide bullets:**
- **30 variants = 3 archs × 2 layers × 5 PCA**
- Cheap because embeddings are frozen
- Turns architecture choice into a *measured* ranking
- The 2 finalised models come from this grid

**Key facts/numbers:** 3 × 2 × 5 = 30.

**Say this:** "Rather than pick a model by intuition, we sweep all 30 combinations of size,
WavLM layer, and PCA level. It's cheap because the embeddings are already cached, and it
means the final choice is something we measured, not guessed."

**Visual:** a 3×2×5 grid of cells; foreshadow by lightly circling two cells (the winners).

**If asked:** the sweep is run under both protocols and across seeds — next two slides.

---

## Slide 13 — Evaluation protocol: two split modes (core methodology)

**Message:** We report two numbers on purpose — an in-domain *ceiling* and a cross-client
*floor* — because either one alone would mislead.

**Full explanation.**
Both modes are candidate-disjoint and label-stratified.

- **Mode A — "20pct" (random in-mix).** `StratifiedGroupKFold(5)` over all candidates →
  fold 0 = test, fold 1 = val, folds 2–4 = train. Train and test share clients, so this is
  the **optimistic ceiling**: "given labelled data from this client, how well can we
  separate?"
- **Mode B — "a6" (held-out client).** Test is **fixed to all of audios6** (Client B); the
  rest is re-partitioned into train/val. This is the **realistic floor**: "how well do we
  hold up on a client we never trained on?"

Train-only auxiliaries (ALLSTAR `2676,2677`, and `casual` for Model 1) are appended **after**
the split, so they can never leak into val/test.

**Slide bullets:**
- **20pct (Mode A):** in-mix, candidate-disjoint → optimistic **ceiling**
- **a6 (Mode B):** test = all audios6, a **held-out client** → realistic **floor**
- Both candidate-disjoint + label-stratified
- Aux data appended **after** the split (no leak)
- The gap between them = the client-shift cost

**Say this:** "We deliberately report two numbers. The first, '20pct,' mixes clients in
train and test — that's the best case, the ceiling. The second, 'a6,' holds out an entire
client as the test set — that's the realistic, harder number. Reporting both is honest: one
tells you the potential, the other tells you what survives a new client."

**Visual:** two split diagrams side by side; in Mode B, audios6 boxed off as "held-out test."

**If asked:** a6 has no region filter — every audios6 row (IND and PHP) is in the test set.

---

## Slide 14 — Metrics: the whole operating curve

**Message:** For screening, "how many cheaters do we catch at a fixed false-alarm rate"
matters more than a single F1.

**Full explanation.**
A false accusation is costly, so we don't optimise one threshold blindly. `compute_metrics`
reports AUC and Average Precision; F1 at 0.5 and **best-F1 with its threshold**; and the
headline operating-point metric, **Recall @ precision = {50, 80, 85, 90, 95}%** — i.e. "how
many cheaters do we still catch if we hold precision (low false alarms) at X%." We also break
metrics down **per region (IND / PHP)** and per batch.

**Slide bullets:**
- AUC, Average Precision (AP)
- F1@0.5 and **best-F1 + its threshold**
- **Recall @ precision = 50/80/85/90/95%** ← the headline
- Per-region (IND/PHP) + per-batch breakdowns

**Say this:** "Because a wrong flag is expensive, the metric we lead with is recall at a fixed
precision — 'if we only allow, say, a 10% false-alarm rate, what fraction of cheaters do we
still catch?' That's far more useful to an operations team than a bare F1, and we report it
at several precision targets, broken down by region."

**Visual:** a PR curve with the precision=90% operating point marked and annotated "catch
‹X›% of cheaters at ≤10% false alarms."

**If asked:** AUC measures ranking quality regardless of threshold; recall@precision measures
the chosen operating point — both matter, see slide 18.

---

## Slide 15 — Fluke-proofing: multi-seed

**Message:** A single run can win by a lucky split, so we rank architectures across 5 seeds
by mean ± std.

**Full explanation.**
We run every variant across **seeds 42–46** and report **mean ± std** per variant, ranking by
`avg_best_f1` (high mean + low std = genuinely good, not lucky). In 20pct mode each seed
reshuffles all candidates; in a6 mode the test stays audios6 and only train/val reshuffle.
For reproducibility, the model init/sampler is seeded with `model_seed = sha256(seed:variant)`
right before training, so any single `(variant, seed)` re-runs **bit-for-bit**, and we record
which seed produced a variant's best run — so a suspiciously high single number is caught as a
fluke rather than shipped.

**Slide bullets:**
- Every variant × **seeds 42–46** → **mean ± std**
- Rank by `avg_best_f1`; low std = genuinely good
- 20pct: reshuffle all · a6: test fixed, reshuffle train/val
- `model_seed = sha256(seed:variant)` → bit-exact re-runs

**Say this:** "Any one split can get lucky. So we run every architecture across five seeds and
rank by the *average*, watching the spread. A high average with a low spread is a real winner;
a single high number with a big spread is a fluke — and we log exactly which seed produced it
so we can tell the difference."

**Visual:** bar chart of variants with mean ± std error bars; the two winners highlighted.

**If asked:** "best_f1_seed" in the summary names the lucky seed; reproduce any model with
`--seeds <s> --variants <v>`.

---

## Slide 16 — Data augmentation study

**Message:** Augmentation is *searched*, not assumed — and it only ever expands the training
set, never evaluation.

**Full explanation.**
We cache 8 augmented variants per audio — `noise, pitch, speed, gain, air, vtlp, combo,
codec` (Gaussian-SNR noise, ±2-semitone pitch, time-stretch, gain, air absorption,
vocal-tract-length perturbation, a stochastic mix, and codec). **Augmentation expands only
the training matrix; val/test always use `orig`,** and the scaler/PCA are refit on the
expanded train. Rather than dumping all augs in, we measured which subset actually helps via
several searches: per-aug *singles*, *leave-one-out*, *greedy forward selection* on
validation, and an *exhaustive* size-3-to-6 combination sweep — each ranked by held-out F1.

**Slide bullets:**
- 8 augs: noise, pitch, speed, gain, air, vtlp, combo, codec
- **Train-only**: val/test stay `orig`; scaler/PCA refit on expanded train
- Searched: singles · leave-one-out · greedy · exhaustive size-3–6
- Best subset chosen by held-out F1 — `‹fill: best subset + lift from leaderboard›`

**Say this:** "We don't just throw every augmentation in. We cache eight of them, apply them
only to training data — evaluation always uses the original audio — and then actually search
for which combination helps, from single augs up to exhaustive size-three-to-six sweeps,
ranked on held-out F1."

**Visual:** a small leaderboard snippet (best subset vs none vs all), `‹fill from
aug_strategy / aug_combo leaderboard›`.

**If asked:** orig is always included in training alongside the chosen augs; "size-3" means
orig + 3 augmentations.

---

## Slide 17 — The two finalised models

**Message:** We keep two models, not one — each answers a different question (best-case
ceiling vs cross-client robustness).

**Full explanation.**

| | **Model 1** | **Model 2** |
|---|---|---|
| Variant | `default_last_pca98` (+ casual) | `tiny_l9_pca95` |
| Architecture | default (512→256→128, drop 0.40, wd 5e-4) | tiny (128, drop 0.55, wd 5e-3) |
| WavLM layer | `last` | **layer 9** |
| PCA | keep 98% | keep 95% |
| Extra train data | **+ casual** (train-only) | none |
| Protocol | **20pct** (in-mix) | **a6** (held-out client) |
| Reads as | best-case ceiling | conservative cross-client |

**Model 1** is the in-domain ceiling: full capacity, last layer, plus the casual batch as
extra train-only data. **Model 2** is the cross-client survivor: a tiny, heavily-regularised
head on the **paralinguistic layer-9** features — a big MLP on the `last` layer mostly
memorised client identity, so the small model on layer 9 generalised best to the held-out
client. Context worth one line: an earlier **stacked ensemble** underperformed (~0.6 F1), so
we reverted to this clearer two-model design, which is both stronger and easier to reason
about.

**Slide bullets:**
- **Model 1** `default_last_pca98` (+casual) @ 20pct → ceiling
- **Model 2** `tiny_l9_pca95` @ a6 → cross-client floor
- Two questions: "potential here" vs "survives a new client"
- (Earlier stacked ensemble ~0.6 F1 → reverted to this)

**Say this:** "We didn't crown one model — we kept two, because they answer different
questions. Model 1 is full-capacity on an in-mix split: the best case. Model 2 is a tiny,
heavily-regularised model on the layer-9 features, evaluated on a held-out client: the
honest, conservative number. A bigger model there just memorised the client, so small won."

**Visual:** the side-by-side comparison table; a small note "stacked ensemble ~0.6 F1 →
reverted."

**If asked:** both are exported with weights + scaler + PCA + `inference_meta.json`, so they
re-run and can be scored on any new batch.

---

## Slide 18 — Results

**Message:** Strong in-domain; on a held-out client it *ranks* cheaters well (high AUC) but
the decision threshold is the catch.

**Full explanation.**
Quote real numbers from `evaluate_final_models.xlsx`. The known qualitative result: on the
held-out-client protocol the model is **AUC-strong but threshold-sensitive** — it ranks
cheaters well (**≈0.92 AUC, best-F1 ≈0.70 on a6**) but the F1-optimal threshold doesn't
transfer cleanly from one client to another. The audios7 row is the true generalisation check
(same weights, brand-new batch).

**Slide table (fill from the sheet):**
| Model / protocol | AUC | AP | best-F1 | R@p90 | R@p95 |
|---|---|---|---|---|---|
| Model 1 — 20pct | ‹fill› | ‹fill› | ‹fill› | ‹fill› | ‹fill› |
| Model 2 — a6 | ≈0.92 | ‹fill› | ≈0.70 | ‹fill› | ‹fill› |
| Model 2 — audios7 (held-out) | ‹fill› | ‹fill› | ‹fill› | ‹fill› | ‹fill› |

**Say this:** "On an in-mix split the separation is strong. On a held-out client, the model
still *ranks* cheaters well — about 0.92 AUC — but the best threshold doesn't transfer
cleanly across clients, so the F1 sits around 0.70. The score scale shifts per client. That
ranking-vs-threshold gap is the one honest weakness, and it's exactly what we're working on
next."

**Visual:** the results table + a small bar comparing 20pct vs a6 best-F1 to show the
client-shift gap.

**If asked:** "How do you verify these aren't cherry-picked?" — every number is reproduced by
reloading the stored weights (`max|recon−saved| < 1e-3`), and the architecture was chosen by
the multi-seed mean, not a single run.

---

## Slide 19 — Reproducibility & the PII boundary

**Message:** Every number is re-derivable, and no raw audio or transcript ever leaves the
laptop — the system is both auditable and privacy-safe.

**Full explanation.**

- **Reproducible.** `extract_embeddings.py` computes WavLM/Whisper once into a cache keyed by
  filename and **stamped with the model IDs** (it refuses to mix caches from different
  models). Each `(variant, seed)` re-runs bit-for-bit. Every run writes a ledger:
  `splits/seed_*.json` (exact candidate membership), `per_run.csv` (metrics + model_seed),
  `summary_mean_std.csv` (the fluke-proof ranking), and `inference_meta.json` per model
  (feature order, dims, model IDs, threshold, reproduce command).
- **PII boundary (non-negotiable):**

| Stays on laptop (never leaves) | Goes to cloud (EC2) |
|---|---|
| raw `.wav` / listenable audio | anonymised `.npy` waveforms |
| transcripts (contain PII) | `gt.csv` — only `feat_*` numbers + labels |
| `cid_mapping.json` (real ID → `G_NNNNN`) | embedding caches |

Transcription and the 55 text features run on the laptop; only anonymised numbers and encoded
waveforms go to the cloud, and **no cloud script ever reads a transcript or a wav.**

**Slide bullets:**
- Frozen, model-ID-stamped embedding cache; per-(variant,seed) bit-exact re-runs
- Ledger every run: splits + per_run + summary + inference_meta
- Laptop keeps wavs / transcripts / real-ID map
- Cloud gets only anonymised `.npy` + `gt.csv` (feat_* + labels)

**Say this:** "Two things make this trustworthy. First, it's fully reproducible — embeddings
are cached and stamped, and any single model re-runs bit-for-bit, with a ledger of exactly who
was in train, val, and test. Second, the privacy boundary: all the audio and transcripts stay
on the company laptop; only anonymised numbers and encoded waveforms go to the cloud. No cloud
script ever sees a wav or a transcript."

**Visual:** a laptop ⟷ cloud diagram with a dashed boundary line and "no wav/transcript
crosses."

**If asked:** real IDs map to `G_NNNNN` locally; the mapping file never leaves the laptop.

---

## Slide 20 — Limitations, next steps & takeaways

**Message:** It works in-domain; cross-client *threshold transfer* is the open problem we're
actively solving — and the whole thing is reproducible and privacy-safe.

**Full explanation.**

- **Open problems.** (1) *Cross-client threshold transfer* — the model ranks well on a new
  client but the F1-optimal threshold shifts, because score *scale* moves per client. (2)
  *Thin PHP sample* — audios6 is the only PHP source, so PHP per-region metrics rest on few
  samples. (3) *Handcrafted-feature dependencies* — missing spaCy/Praat/GPT-2 silently
  zero-fills those columns, so the laptop environment must have all three for full strength.
- **In progress.** Per-client feature standardisation (unsupervised, features-only); few-shot
  client adaptation (label a small slice of a new client before going live); the
  augmentation-strategy search.
- **Three takeaways.** (1) Per-audio + multi-signal fusion beats any single signal. (2) Honest
  two-protocol evaluation exposes client shift instead of hiding it. (3) Everything is
  reproducible and PII-safe.

**Slide bullets:**
- Open: cross-client **threshold transfer** (main) · thin PHP · feature deps
- In progress: per-client standardisation · few-shot adaptation · aug search
- Takeaways: multi-signal > single · two-protocol honesty · reproducible + PII-safe

**Say this:** "To close honestly: in-domain it works well. The open problem is that on a brand
new client, the ranking holds but the threshold drifts — so we're working on per-client
calibration and a small bit of few-shot adaptation. The three things I'd want you to remember:
fusing multiple signals per audio beats any single trick; we evaluate on two protocols so
client shift can't hide; and every result is reproducible and privacy-safe."

**Visual:** a three-column "Done / In progress / Next" roadmap; end on this, not text.

**If asked:** the few-shot mode already exists in the split builder (`fewshot_frac`) — label a
small slice of a new client, candidate-disjoint, to recalibrate.

---

### Appendix (Q&A back-pocket)
- **A1** Full 55-feature table (8 groups in detail) — from `METHODOLOGY.md` §3c.
- **A2** End-to-end pipeline diagram (laptop ↔ cloud) — `METHODOLOGY.md` §10.
- **A3** Augmentation leaderboards in full — `aug_strategy` / `aug_combo` outputs.
- **A4** Reproducibility ledger example — `inference_meta.json` + `splits/seed_*.json`.

*Source of truth: `METHODOLOGY.md` (current system) and `OLD_APPROACHES.md` (slide 4). Pull
live metrics from `summary_mean_std.csv` and `evaluate_final_models.xlsx`.*
