# Neural Baseline Plan — Per-Audio Cheating Detection

## Goal

Iterate the per-audio cheating detector beyond the frozen-encoder + XGBoost ceiling (a4/a5 F1 ~0.74, a6 F1 ~0.61, a6 AUC 0.82). Per-audio output is the deployment requirement; never aggregate across Q25/Q26/Q27.

## Diagnosis (from prior iterations)

- **AUC=0.82, F1=0.61 on a6** → encoder embeddings carry signal but the head's decision boundary is wrong on a structural subset of the data.
- **IND F1 < PHP F1** consistently across all conditions.
- **Top FPs are real model errors** (audio sounds honest, model says cheat). Top FNs are real model errors (audio sounds cheating, model misses). GT audit closed; residual gap is not label noise.
- **Working hypothesis:** WavLM-base-plus was pretrained on ~94k hours of mostly L1 English. L2 Indian English is OOD for it. IND honest audio lands in unusual regions of embedding space, which the head learned to associate with "cheat" (because PHP-dominant training data has no L2-IND honest examples there). The bottleneck is encoder-level domain mismatch, not head capacity.

## Constraints

- T4 GPU (16 GB) for training; CPU laptop for inference.
- Permission obtained to convert all audios to .npy and upload.
- ~1000 labeled audios across a2/a4/a5/a6, plus unlabeled a3 + a6-IND-unlabeled + ALLSTAR Indian-L1 subsets for unsupervised pretraining.
- Per-audio output, no cross-question aggregation.

---

## Staged plan

### Stage 0 (current) — Baseline neural fusion with frozen encoders

**Question this stage answers:** does a simple MLP on concatenated frozen embeddings beat the existing weighted XGBoost stack? If yes, the rest of the staged plan has a workable head to attach to. If no, more capacity is not the answer.

- Concat WavLM-base-plus (768d, mean-pool) + Whisper-medium encoder (1024d, mean-pool) = 1792d.
- Variants: no PCA, PCA-98%-variance, PCA-95%, PCA-90%.
- MLP head: 512 → 256 → 128 → 1 with BN + dropout.
- 60 / 20 / 20 train / val / test split, **speaker-wise stratified** (no candidate leaks across splits).
- Compare against current frozen-encoder XGBoost stack on the same test split.

**Note on PCA:** Prior work showed PCA on WavLM destroyed XGBoost performance (768 → 80 dropped F1 from ~0.85 to ~0.58). PCA + MLP may behave differently because MLP can re-expand the bottleneck and PCA acts as a regularizer rather than a fixed feature selector. Including the no-PCA variant as the control. If PCA variants underperform clearly, we drop them.

**Decision rule:** if any variant beats current best (a4/a5 F1 ~0.74) on the held-out 20% by ≥2 F1 points, neural head is the new baseline. If not, encoder is the bottleneck (as diagnosed) and we go straight to Stage 1.

### Stage 1 — Continued pretraining (CPT) of WavLM on unlabeled IND audio

**Question this stage answers:** does in-domain encoder adaptation move IND-honest embeddings closer to PHP-honest in WavLM space, and does that close the IND F1 gap?

- Pool: a3 IND + a6 IND + ALLSTAR Indian L1 subsets. Target 10–30 hours.
- MLM-style continued pretraining, 5–10s chunks, batch 16, lr 5e-5, ~10–20k steps. ~6–12 T4 hours.
- **Gate (do this before retraining anything):** extract embeddings for held-out IND-honest, IND-cheat, PHP-honest, PHP-cheat with original vs CPT'd WavLM. Measure CORAL distance and t-SNE separation. IND-honest must move closer to PHP-honest. If it doesn't, abort CPT and go to Stage 3 directly.

### Stage 2 — Re-extract embeddings, retrain Stage 0 baseline

If Stage 1 gate passes, re-run Stage 0 with CPT'd WavLM. Single variable: encoder. If a6 F1 jumps ≥0.04, encoder shift was the dominant bottleneck. If not, stop pushing this hypothesis and reconsider.

### Stage 3 — Partial fine-tuning with GRL (gated)

Only if Stage 2 helps but doesn't close enough of the gap.

- Houlsby adapters (bottleneck 256) inserted in each WavLM transformer block. Adapters preferred over top-6-unfreeze at this label scale.
- Cheat head + region head with GRL between encoder pool and region head. λ schedule: 0 for 200 steps, ramp to 0.05 over 500, hold. Conservative ceiling because region label correlates with cheat label.
- Pooling: mean + std concat.
- Augmentation pipeline: codec sim (Opus 16k, AMR-NB), MUSAN noise (5–15 dB SNR), RIR conv, SpecAugment. Skip pitch / VTLP — they distort paralinguistic features that carry cheat signal.
- LR: encoder 2e-5, head 1e-3, layer-wise decay 0.95, AdamW wd 0.01, cosine.

### Stage 4 — Fusion swap

If audio NN logit beats audio XGBoost on a4-CV by ≥2 F1, swap into the existing weighted-avg fusion (0.6 text + 0.4 audio). Keep text head as XGBoost — end-to-end multimodal cross-attention is not justified at 1000 labels.

### Stage 5 — Calibration

Per-domain (IND vs PHP) temperature scaling on a held-out cal split. Top-K% rank decision rule for threshold transfer.

---

## What we explicitly don't do

- Full fine-tuning of all WavLM layers. Validated negative at 540 labels; 1000 labels is still small.
- Cross-attention multimodal fusion. Not at this data scale.
- Stacking ensembles. Validated negative.
- ALLSTAR's read-vs-spontaneous as cheat labels. Acoustically adjacent but not identical. Use ALLSTAR for accent coverage in CPT only.
- Fine-tuning Whisper. Whisper's representations are ASR-tuned (semantic-rich, paralinguistic-poor). Fine-tuning probably degrades the linguistic-content signal.
- Per-candidate aggregation across questions. Per-audio output is non-negotiable.

---

## Pipeline split

- **Company laptop:** raw audio prep, candidate-ID anonymization, npy + gt csv generation. Output is what gets uploaded; cid mapping stays local.
- **EC2 (T4):** encoder extraction (frozen, Stage 0), MLP training across PCA variants, evaluation, logging.

Scripts: `companylaptop/neural_baseline_prep.py`, `ec2/neural_baseline_train.py`.
