# Per-Audio Cheating Detection — 20-Slide Presentation Outline

A ready-to-build deck for a ~15–20 minute talk. **~90% covers the current system**;
one slide covers earlier approaches. Each slide below gives: the **on-slide headline**
(make this the takeaway, not a topic), **what goes on the slide**, a **visual idea**,
and **speaker notes** (the detail you *say*, not show).

> Numbers marked `‹fill: …›` should be pulled from your latest run outputs so you never
> present a fabricated figure:
> - `runs/.../summary_mean_std.csv` — per-architecture mean ± std (fluke-proof ranking)
> - `runs/final_eval/evaluate_final_models.xlsx` → `summary` sheet — the 2 models on their own test sets
> - `runs/final_eval_audios7/evaluate_final_models.xlsx` — the 2 models on the held-out audios7 batch

---

## How to make this deck impactful (read first)

1. **One idea per slide, and the title states it.** Use *assertion–evidence* design: the
   title is a full-sentence claim ("No single signal catches every cheating style"), the
   body is the evidence (a diagram/chart), not a topic label ("Signals").
2. **Show, don't list.** Replace bullet walls with one diagram, one chart, or one table.
   If a slide is only text, you're narrating — move that text to the speaker notes.
3. **Progressive reveal.** Build the 1847-d feature vector one block at a time; build the
   pipeline diagram left-to-right. Don't drop the final architecture on slide 5.
4. **A consistent visual language.** Pick one colour per signal family (WavLM / Whisper /
   handcrafted) and reuse it on every slide. Candidate = one icon throughout.
5. **Lead with the problem and the hard part.** Spend the first third on *why this is
   hard* (per-audio, client shift). Earned credibility makes the results land.
6. **Be honest about limits.** The "threshold doesn't transfer across clients" slide is
   what makes a technical audience trust the rest. Don't hide it — frame it as the next
   target.
7. **Numbers = operating points, not just F1.** "We catch 70% of cheaters while holding
   false alarms to 10%" beats "AUC 0.92" for a non-ML audience.
8. **6×6 / big fonts.** ≤6 bullets, ≤6 words each, 24pt+. The audience reads *or* listens,
   never both.
9. **Narrative arc (the funnel):** Problem → Why hard → Our idea → How we built it → How
   we measured it honestly → What works → What's next.

**Suggested timing:** Intro/problem (slides 1–4) ≈ 4 min · System & features (5–11) ≈ 7 min ·
Protocol & evaluation (12–16) ≈ 5 min · Results & close (17–20) ≈ 4 min.

---

## Slide 1 — Title

- **Title:** *Per-Audio Cheating Detection: A Multi-Signal Approach*
- **Subtitle:** Flagging a single spoken answer as genuine vs. assisted
- Your name · Mercer | Mettl · date
- **Visual:** one clean hero image (a waveform morphing into a ✓/✗), product-neutral.
- **Notes:** "I'll walk through how we detect cheating in spoken assessment answers — the
  problem, the approach, and how we made sure the numbers are real."

## Slide 2 — The problem

- **Headline:** *Remote spoken assessments are easy to game — and a wrong accusation is costly.*
- On slide: candidate answers Q25/Q26/Q27 → some genuine, some read off a screen / AI-generated / copied.
- **Visual:** a candidate icon with three answer bubbles, one flagged red.
- **Notes:** Cheating styles differ — reading from a screen, pasting an AI answer, copying.
  False positives are expensive (you're accusing a real person), so **precision matters as
  much as recall**. This shapes every later decision.

## Slide 3 — Core thesis

- **Headline:** *No single signal catches every cheating style — so we fuse several, per audio.*
- Two design commitments:
  - **Per-audio decision** (never average a candidate's answers — cheating can start mid-exam)
  - **Multi-signal fusion** (acoustic + ASR + linguistic/prosodic)
- **Visual:** Venn / funnel of 3 signal families → one decision.
- **Notes:** Reading flattens prosody and removes disfluency; AI text is "too clean";
  copying adds unnatural pauses. Each leaves a *different* trace, so one feature family is
  never enough. And because cheating can begin on Q27 after an honest Q25, we score **every
  audio independently**.

## Slide 4 — What we tried before *(the ~10%)*

- **Headline:** *Two earlier approaches taught us what doesn't generalise.*
- **(1) Read-vs-spontaneous fine-tuning** (wav2vec2/WavLM/Whisper on ALLSTAR, "read ratio").
  Near-perfect on the proxy task (‹F1 ≈ 0.98 on ALLSTAR›) — but **reading ≠ cheating**, and
  it didn't transfer to real candidate data.
- **(2) XGBoost over embeddings + acoustic scores.** Strong on paper (‹AUC ≈ 0.99›) but
  **~94% of its importance was the read-vs-spont score** — a single proxy in disguise.
- **Takeaway →** we needed *true multi-signal fusion* and *honest cross-client evaluation*.
- **Visual:** two small "tried" cards with a red "didn't generalise" stamp → arrow to the current approach.
- **Notes:** Keep this to ~90 seconds. The point isn't the old numbers; it's *why* they
  motivated the current design. (Detail lives in `OLD_APPROACHES.md`.)

## Slide 5 — Current system at a glance

- **Headline:** *One audio → three signal families → one fused vector → one probability.*
- **Visual (the spine of the talk):**
  ```
  audio ─► WavLM mean-pool (768) ─┐
          Whisper enc. mean-pool (1024) ─┼─► concat 1847-d ─► scale ─► PCA ─► MLP ─► P(cheat)
          55 handcrafted feat_* ─┘
  ```
- **Notes:** "Everything for the rest of the talk hangs on this diagram — I'll expand each
  block." Embeddings are **extracted once and frozen** (not fine-tuned), which makes the
  whole search cheap and reproducible.

## Slide 6 — Data & the real challenge

- **Headline:** *The hard problem isn't cheating vs. genuine — it's client A vs. client B.*
- On slide (small table): Client A = audios2/4/5 (IND) · Client B = audios6 (IND+PHP, new mic/codec) ·
  audios7 = held-out validity · ALLSTAR + casual = train-only aux.
- Two rules: **candidate-disjoint splits** (group = candidate); **client shift costs ~19 F1 points**.
- **Visual:** two clusters in embedding space (A vs. B) with a dashed "shift" arrow.
- **Notes:** If a candidate's audios leak across train/test the model just "recognises the
  speaker." And a model trained mixed but tested on a *held-out client* drops ~19 F1 — most
  apparent accuracy is "which client is this," not "is this cheating." Every protocol choice
  later is about controlling this.

## Slide 7 — Signal family 1: WavLM (self-supervised acoustic)

- **Headline:** *WavLM hears how it was said — and a mid-network layer hears it best.*
- `microsoft/wavlm-base-plus`, 16 kHz, mean-pooled → 768-d. We cache **`last` and layer 9**.
- **Layer 9 carries paralinguistics** (prosody, voice quality, effort) — exactly the
  cheating-relevant content; it's what Model 2 uses.
- **Visual:** transformer stack with layer 9 highlighted; tiny "prosody/effort" tags.
- **Notes:** Self-supervised models encode *how* something is said. Final-layer features
  drift toward the pre-training objective; the middle of the network is where speaking
  style lives — so we keep both and let the sweep decide.

## Slide 8 — Signal family 2: Whisper encoder (ASR acoustic-linguistic)

- **Headline:** *Whisper's encoder adds the content/pronunciation structure WavLM misses.*
- `openai/whisper-medium` **encoder** output, mean-pooled → 1024-d. **Encoder only — no decoder text.**
- **Visual:** Whisper split into encoder (used, green) and decoder (greyed out).
- **Notes:** The ASR encoder captures the structure the model attends to when transcribing —
  complementary to WavLM's self-supervised view. We deliberately avoid the decoded text
  here (the text signal is captured separately, interpretably, in the next family).

## Slide 9 — Signal family 3: 55 handcrafted features (interpretable)

- **Headline:** *55 interpretable features encode the linguistics a human reviewer would notice.*
- 8 groups (show as a labelled grid): Disfluency · Stylometric · Pause · Suspicious gaps ·
  Formal/AI phrasing · Prosodic · Voice quality · Perplexity.
- One example each: filler_rate · MTLD · long_pause_rate · suspicious_gap_ratio ·
  ai_phrase_rate · f0_slope · jitter · GPT-2 perplexity.
- **Visual:** 8 tiles, each with an icon + one feature name.
- **Notes:** Computed from a rich word-timestamped transcript (faster-whisper). Genuine
  speech is *dis*fluent and irregularly paced; read/AI text is too clean, too low-perplexity,
  too evenly paused. Missing optional deps (spaCy/Praat/GPT-2) zero-fill **the same 55
  columns in the same order**, so batches stay comparable.

## Slide 10 — Fusion & preprocessing

- **Headline:** *Concatenate → standardise → PCA: 1847 numbers become one comparable vector.*
- `[768 | 1024 | 55] = 1847-d` → `StandardScaler` → optional PCA (fit on **train only**).
- `feat_*` are concatenated **before** PCA, so even pca90 keeps the text signal compressed.
- **Visual:** three coloured bars stacking into one, then shrinking through PCA.
- **Notes:** Scaler and PCA are fit on the (augmentation-expanded) **training set only** — no
  leakage. PCA both denoises and makes the 30-variant search cheap.

## Slide 11 — The classifier

- **Headline:** *A small MLP on frozen features — capacity is a knob, not a default.*
- `Linear→BatchNorm→ReLU→Dropout … →Linear(1)→sigmoid`; BCE + label-smoothing 0.05,
  AdamW + cosine, early-stop on **val F1**, `WeightedRandomSampler` for imbalance.
- 3 archs: **default** (512→256→128) · **tiny** (128, heavy reg.) · **linear** (LR sanity baseline).
- **Visual:** the 3 arch cards side by side with their "role".
- **Notes:** Frozen embeddings → a light head is enough. The `linear` baseline is a tripwire:
  if it matches `tiny`, the data is essentially linear and a big MLP is just memorising
  client artifacts — which is exactly what we saw on the held-out client.

## Slide 12 — The 30-variant sweep

- **Headline:** *We don't guess the model — we sweep 30 and let the protocol pick.*
- **30 = 3 archs × 2 WavLM layers (`last`,`9`) × 5 PCA (`full/98/95/93/90`).**
- **Visual:** a 3×2×5 grid; later slides will circle the two winners.
- **Notes:** Cheap because embeddings are frozen. This turns "which architecture?" from an
  opinion into a measured ranking.

## Slide 13 — Evaluation protocol: two split modes *(core methodology)*

- **Headline:** *We report two numbers on purpose: an in-domain ceiling and a cross-client floor.*
- **20pct (Mode A):** candidate-disjoint StratifiedGroupKFold; train & test share clients →
  **optimistic ceiling.**
- **a6 (Mode B):** test = **all of audios6 (a held-out client)** → **realistic, harder floor.**
- Train-only aux (ALLSTAR, casual) appended **after** the split — can't leak.
- **Visual:** two split diagrams side by side, audios6 boxed off in Mode B.
- **Notes:** One protocol would be misleading. 20pct says "given labelled data from this
  client, how well can we separate?"; a6 says "how well do we hold up on a client we never
  trained on?" The gap between them *is* the client-shift story from slide 6.

## Slide 14 — Metrics: the whole operating curve

- **Headline:** *For screening, "recall at fixed precision" matters more than raw F1.*
- We report: AUC, AP, F1@0.5, **best-F1 + its threshold**, and **Recall @ precision =
  50/80/85/90/95%**, plus per-region (IND/PHP) breakdowns.
- **Visual:** a PR curve with the p90 operating point marked; annotate "catch X% of cheaters
  at ≤10% false alarms."
- **Notes:** A false accusation is costly, so we fix precision and ask how many cheaters we
  still catch. That's the number an operations team actually cares about.

## Slide 15 — Fluke-proofing: multi-seed

- **Headline:** *A single run can win on luck — so we rank architectures over 5 seeds.*
- Every variant × seeds 42–46 → **mean ± std**; rank by `avg_best_f1`, low std = genuinely good.
- 20pct: each seed reshuffles all candidates · a6: test fixed, train/val reshuffled.
- **Visual:** a bar chart of variants with error bars (mean ± std).
- **Notes:** `model_seed = sha256(seed:variant)` makes each (variant, seed) reproducible
  bit-for-bit, and we record which seed gave a variant's best run — so a suspiciously high
  single number is caught as a fluke, not shipped.

## Slide 16 — Data augmentation study

- **Headline:** *Augmentation is searched, not assumed — and only ever expands training.*
- 8 cached augs (`noise, pitch, speed, gain, air, vtlp, combo, codec`); **val/test stay
  `orig`.** We ran singles / leave-one-out / greedy / exhaustive size-3–6 searches.
- **Visual:** a small leaderboard snippet (best aug subset vs. none) — `‹fill from
  aug_strategy / aug_combo leaderboard›`.
- **Notes:** Augmentation expands the train matrix (scaler/PCA refit on it); evaluation rows
  are never augmented. We measured which subset actually helps rather than throwing all of
  them in.

## Slide 17 — The two finalised models

- **Headline:** *We keep two models — they answer two different questions.*
- **Visual (side-by-side table):**
  | | Model 1 | Model 2 |
  |---|---|---|
  | Variant | `default_last_pca98` (+casual) | `tiny_l9_pca95` |
  | Protocol | 20pct (in-mix) | a6 (held-out client) |
  | Reads as | best-case ceiling | conservative cross-client |
- **Notes:** Model 1 = the in-domain ceiling with full capacity. Model 2 = the tiny,
  heavily-regularised head on **layer-9** features that survived the held-out-client test —
  a big MLP on `last` mostly memorised client identity. (An earlier stacked ensemble
  underperformed at ~0.6 F1, so we reverted to this clearer, stronger design.)

## Slide 18 — Results

- **Headline:** *Strong in-domain; ranks cheaters well across clients, threshold is the catch.*
- **Visual (fill from your sheets):**
  | Model / protocol | AUC | AP | best-F1 | R@p90 | R@p95 |
  |---|---|---|---|---|---|
  | Model 1 — 20pct | ‹fill› | ‹fill› | ‹fill› | ‹fill› | ‹fill› |
  | Model 2 — a6 | ≈0.92 | ‹fill› | ≈0.70 | ‹fill› | ‹fill› |
  | Model 2 — audios7 (held-out) | ‹fill› | ‹fill› | ‹fill› | ‹fill› | ‹fill› |
- **Notes:** On the held-out client the model is **AUC-strong but threshold-sensitive** —
  it ranks cheaters well, but the decision threshold doesn't transfer cleanly across
  clients. Quote real numbers from `evaluate_final_models.xlsx`; the audios7 row is the
  honest generalisation check.

## Slide 19 — Reproducibility & the PII boundary

- **Headline:** *Every number is re-derivable, and no raw audio or transcript leaves the laptop.*
- **Reproducible:** frozen, model-ID-stamped embedding cache; per-(variant,seed) bit-exact
  re-runs; split ledger + `inference_meta.json` per model.
- **PII boundary (table):** Laptop = wavs, transcripts, real-ID map · Cloud = anonymised
  `.npy`, `gt.csv` (feat_* + labels), caches.
- **Visual:** a laptop ⟷ cloud diagram with a dashed boundary; "no transcript/wav crosses".
- **Notes:** Transcription and the 55 text features run on the laptop; only anonymised
  numbers and encoded waveforms go to the cloud. Real IDs map to `G_NNNNN` locally. This is
  non-negotiable and also what makes the work auditable.

## Slide 20 — Limitations, next steps & takeaways

- **Headline:** *It works in-domain; cross-client threshold transfer is the open problem we're solving.*
- **Open problems:** cross-client threshold transfer (main) · thin PHP sample · handcrafted-feature deps.
- **In progress:** per-client feature standardisation · few-shot client adaptation · the aug-strategy search.
- **Three takeaways:** (1) per-audio + multi-signal fusion beats any single signal;
  (2) honest two-protocol evaluation exposes client shift; (3) everything is reproducible
  and PII-safe.
- **Visual:** a simple "done / in-progress / next" three-column roadmap.
- **Notes:** Close on the funnel you opened with: hard problem → fused multi-signal design →
  measured honestly → strong in-domain, clear next target. End on the roadmap, not a wall
  of text.

---

### Appendix slides (optional, keep in back pocket for Q&A)
- A1 — Full 55-feature table (the 8 groups in detail).
- A2 — The end-to-end pipeline diagram (laptop ↔ cloud, from `METHODOLOGY.md` §10).
- A3 — Augmentation leaderboard in full (`aug_strategy` / `aug_combo` outputs).
- A4 — Reproducibility ledger example (`inference_meta.json` + `splits/seed_*.json`).

*Source of truth for all content: `METHODOLOGY.md` (current system) and `OLD_APPROACHES.md`
(slide 4). Pull live metrics from the run sheets named at the top of this file.*
