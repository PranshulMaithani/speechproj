# Presentation Content — Part 2: Fusion → Models → Training → Pros/Cons

Rich slide content for the second half of the deck. Same format as
`PRESENTATION_CONTENT.md`: per slide you get the **message/title**, **slide bullets**,
a **"say this"** script, and a **diagram**. Grounded in `METHODOLOGY.md`.

---

# SECTION A — Fusion & Preprocessing (1 slide)

## Slide A1 — Fusing the three signals into one vector
**Title:** *Concatenate the three families → standardize → PCA → the vector the model sees*

**Slide bullets:**
- **Fuse:** stack the three embeddings end-to-end → `768 (WavLM) + 1024 (Whisper) + 55 (handcrafted) = 1847-d`.
- **Standardize:** `StandardScaler` → every dimension zero-mean / unit-variance (so no single big-scale feature dominates). **Fit on TRAIN only.**
- **PCA (optional):** keep **90–98%** of the variance → **denoises** the high-dim embeddings and **compresses** them. Also fit on train only.
- **feat_\* are concatenated BEFORE PCA** → even the aggressive pca90 variant still carries the linguistic signal in compressed form.
- Output = one clean, comparable vector fed to the MLP.

**Say this (~30s):** "Fusion is just concatenation — we glue the three signal families into one 1847-dimensional vector. But raw, the dimensions live on wildly different scales, so we standardize every dimension to zero mean and unit variance. Then optional PCA keeps 90–98% of the variance, which both denoises the embeddings and shrinks them. The one rule that matters: the scaler and PCA are fit on training data only, so no test information leaks in."

**Diagram (Mermaid — paste starting at `flowchart`):**
```
flowchart LR
  W["WavLM<br>768-d"] --> C["Concatenate<br>768 + 1024 + 55 = 1847-d"]
  H["Whisper<br>1024-d"] --> C
  F["Handcrafted<br>55-d"] --> C
  C --> S["StandardScaler<br>zero mean, unit variance<br>fit on TRAIN only"]
  S --> P["PCA (optional)<br>keep 90 to 98 percent variance<br>denoise + compress"]
  P --> V["Final feature vector<br>into the MLP"]
  S:::proc
  P:::proc
  classDef proc fill:#f0fdf4,stroke:#4ade80,color:#14532d
```

**Key line to land:** "Fusion = concatenate; the real work is standardize + PCA, both fit on train only."

---

# SECTION B — Model Architectures (2 slides)

Both are compact **MLP heads** on top of the frozen, fused features (embeddings are never
fine-tuned). Same building block everywhere: `Linear → BatchNorm → ReLU → Dropout`, ending in
`Linear → 1 → sigmoid`. Loss = BCE + label-smoothing 0.05, AdamW + cosine LR, gradient-clip
1.0, early-stopping on validation F1, class imbalance handled by a `WeightedRandomSampler`.

## Slide B1 — Model 1 architecture: full-capacity MLP
**Title:** *Model 1 — the full-capacity head (our in-domain ceiling)*

**Slide bullets:**
- **Hidden layers: 512 → 256 → 128** (3 blocks), then `→ 1 → sigmoid`.
- **Dropout 0.40**, **weight decay 5e-4** — moderate regularization.
- **~0.4–1.1M parameters** (scales with input dim: ~1.1M on the full 1847-d, ~0.4M after PCA).
- **Represents:** maximum modelling capacity — "given labelled data from this client, how well can we separate cheating from genuine?" The **best-case ceiling.**
- Finalised as `default_last_pca98` (WavLM `last` layer, PCA-98) on the in-mix (20pct) protocol.

**Say this:** "Model 1 is the full-capacity head — three hidden layers, 512 down to 128, a few hundred thousand to about a million parameters depending on how hard we compress. It's the model we use when we have labelled data from the same client, so it answers 'what's the best we can do in-domain' — the ceiling."

**Diagram:** simple box chain — `fused vector → [512] → [256] → [128] → sigmoid → P(cheat)`, each box labelled "Linear+BN+ReLU+Dropout(0.40)".

## Slide B2 — Model 2 architecture: compact, heavily regularized MLP
**Title:** *Model 2 — the compact, regularized head (our cross-client survivor)*

**Slide bullets:**
- **Hidden layer: 128** (single block), then `→ 1 → sigmoid`.
- **Dropout 0.55**, **weight decay 5e-3** — **heavy** regularization (much stronger than Model 1).
- **~0.06–0.24M parameters** — an order of magnitude smaller than Model 1.
- **Represents:** deliberate *low* capacity — "how well do we hold up on a client we never trained on?" A big model on this task mostly **memorised client identity**; the small, regularized head **generalised best**.
- Finalised as `tiny_l9_pca95` (WavLM **layer 9**, PCA-95) on the held-out-client (a6) protocol.

**Say this:** "Model 2 is the opposite design choice — a single small hidden layer with very heavy regularization, ten-times fewer parameters. We made it small on purpose: on the held-out-client test, a large model just memorised which client it was looking at. The tiny, heavily-regularized head on the layer-9 features is what actually generalised — so this is our honest cross-client number."

**Diagram:** `fused vector → [128] → sigmoid → P(cheat)`, box labelled "Linear+BN+ReLU+Dropout(0.55)". Put it **beside** the Model 1 diagram at the same scale so the size difference is visually obvious.

> **Contrast to state out loud:** bigger isn't better here. Model 1 (big) wins in-domain;
> Model 2 (small) wins across clients — which is why we keep both.

---

# SECTION C — Training Procedure (2 slides)

## Slide C1 — We didn't guess — we swept the choices
**Title:** *Every design choice was measured: architecture × WavLM layer × PCA × aux data*

**Slide bullets:**
- Too many knobs to guess, so we **swept a grid** (embeddings are frozen/cached → cheap):
  - **Architecture:** full-capacity vs compact-regularized.
  - **WavLM layer:** `last` vs **layer 9** (paralinguistic).
  - **PCA variance:** `full / 98 / 95 / 93 / 90%` — traded compression vs signal retention.
  - **Auxiliary data:** with/without the **casual** batch; **ALLSTAR** forced train-only.
- Each variant scored under **both protocols** (in-mix 20pct AND held-out-client a6).
- Selection metric: **recall @ fixed precision** + best-F1, not a single accuracy number.

**Say this:** "Rather than pick an architecture by intuition, we swept the real choices — model size, which WavLM layer, how aggressively to compress with PCA, and whether to add auxiliary data. Because the embeddings are cached, running the whole grid is cheap, and every variant is scored under both the easy in-mix split and the hard held-out-client split."

**Diagram:** a small grid/tree: `Architecture (2) × Layer (2) × PCA (5) × Aux (±casual)` → "scored on 20pct + a6." *(Optional footnote: a plain linear baseline was also run purely as a memorisation tripwire — drop from the slide if you like.)*

## Slide C2 — Multi-seed: choosing a winner, not a lucky split
**Title:** *We rank by the average over 5 seeds — so the winner isn't a fluke*

**Slide bullets:**
- A single train/val/test split can win by **luck**. So every variant is run over **5 seeds** and reported as **mean ± std**.
- **20pct:** each seed reshuffles all candidates. **a6:** test stays the held-out client; only train/val reshuffle.
- **Rank by average best-F1**; **low std + high mean = a genuinely good design** (not a lucky seed).
- Fully reproducible: init is seeded from `hash(seed, variant)` → any single model re-runs **bit-for-bit**.

**Say this:** "One split can flatter a model. So we run every candidate across five seeds and rank by the average, watching the spread. A high average with a low spread is a real winner; a single high number with a big spread is a fluke. And because we seed everything deterministically, any model we pick can be re-run bit-for-bit."

**Diagram:** a bar chart of a few variants with **mean ± std error bars**; the chosen one highlighted. (Reuse from your `summary_mean_std.csv` when you have live numbers.)

---

# SECTION D — Advantages (1 slide)

**Title:** *Why this design works*
- **Multi-signal fusion** beats any single signal — different cheating styles leave different traces.
- **Per-audio decision** — catches cheating that starts mid-exam (no averaging away the signal).
- **Interpretable features alongside embeddings** — decisions can be explained, which matters for accusations.
- **Frozen, cached embeddings** — the whole sweep is cheap, deterministic, and **reproducible**.
- **Honest two-protocol evaluation** — exposes client shift instead of hiding it.
- **Robust training** — class-imbalance sampler, augmentation, multi-seed selection.
- **PII-safe by construction** — no audio/transcript leaves the laptop.

**Say this:** "The wins: fusing several signals catches more cheating styles than any one trick; deciding per audio catches mid-exam cheating; the hand features keep it explainable; frozen embeddings make it cheap and reproducible; and evaluating on two protocols keeps us honest about generalisation."

---

# SECTION E — Limitations & open problems (1 slide)

**Title:** *What it doesn't solve yet*
- **Cross-client threshold transfer (main).** On a new client the model **ranks** cheaters well (high AUC) but the F1-optimal **threshold shifts** — score scale moves per client.
- **Thin PHP region.** Only one batch (audios6) has PHP, so PHP metrics rest on a small sample.
- **Handcrafted-feature dependencies.** Missing spaCy/Praat/GPT-2 silently zero-fills columns — full strength needs all three.
- **Limited real labels.** We lean on proxy auxiliary data (ALLSTAR, casual) because labelled exam cheating is scarce.
- **Embeddings frozen, not fine-tuned.** A deliberate cost/repro trade-off — fine-tuning could add accuracy but loses cheapness/determinism.

**In progress:** per-client feature standardisation · few-shot client adaptation · augmentation-strategy search.

**Say this:** "The honest limits: in-domain it works, but on a brand-new client the ranking holds while the threshold drifts — that's the main open problem. PHP data is thin, the hand features need their tools installed, and we lean on proxy data because real cheating labels are scarce. We're actively working on per-client calibration and light few-shot adaptation."

---

# SECTION F — Results *(you fill in)*
Placeholder — drop in the numbers from `evaluate_final_models.xlsx` / `summary_mean_std.csv`:
- Model 1 (20pct): AUC / AP / best-F1 / recall@p90 / p95
- Model 2 (a6): AUC / AP / best-F1 / recall@p90 / p95
- Model 2 on audios7 (held-out validity check)
Suggested visual: a small table + a bar comparing 20pct vs a6 best-F1 (the client-shift gap).

---

*Source of truth: `METHODOLOGY.md` (system) + `EMBEDDING_EXTRACTION.md` (encoders). Part 1
slide content: `PRESENTATION_CONTENT.md` / `PRESENTATION_OUTLINE.md`.*
