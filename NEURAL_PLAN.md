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

## Pipeline (current)

Three scripts, three roles. Embedding extraction is the slow step and runs **once**; the train script is fast (cache load + 10 MLP fits) and intended to be re-run repeatedly with different `--train_batches` / `--test_batches` / `--use_augs` / `--min_duration` combinations.

### 1. `companylaptop/neural_baseline_prep.py` — company laptop (mercer-mettl only)

- Reads `audios{2..6}/<filename>.wav` + sibling `<batch>GT.csv` (+ optional `<batch>_features.csv`, `<batch>_transcripts.json`).
- Anonymizes candidate IDs to `G_NNNNN` (mapping kept in `local/cid_mapping.json`, never uploaded).
- Resamples to 16 kHz mono, writes `upload/audio_npy/<gid>_<qid>.npy`.
- Writes `upload/gt.csv` with: group_id, question_id, batch, label, region, duration_sec, npy_filename, feat_\*.
- Transcripts and real CIDs are scrubbed before save; only numeric features cross the air gap.
- ALLSTAR is **not** handled here anymore — see (1b) below.

### 1b. `ec2/allstar_prep.py` — EC2 (ALLSTAR is public, no PII concern)

- Downloads `2676.zip` and `2677.zip` from `Pransfrance/speechproj-models` via `huggingface_hub.hf_hub_download` and unzips them locally.
- Walks each `ALL_<L1>_ENG_<TASK>/` subfolder and parses `ALL_<spkID>_<M|F>_<L1>_ENG_<TASK>.wav` filenames.
- **Label is derived from the task code suffix**, not from the 2676/2677 top folder — both folders mix tasks. Read (1): `ST1..ST4, LPP, NWS`. Spontaneous (0): `QNA, HT1, HT2, DHR`. Unmapped task codes are skipped with a count.
- Slices each audio into 3 non-overlapping random 40–60s segments (deterministic with a seeded RNG).
- Transcribes each segment with **faster-whisper medium** (word timestamps + VAD).
- Computes the **full 55-feature audios6_eval set** via `ec2/full_text_features.py` (disfluency + stylometric + pause + suspicious + formal_AI + prosodic + voice_quality + perplexity). Same `feat_*` schema as mercer-mettl rows.
- Appends rows to `<UPLOAD>/gt.csv` in place. Segment npys land in the same `<UPLOAD>/audio_npy/` as mercer-mettl. ALLSTAR group_ids live in their own `AS_NNNNN` namespace and cannot collide with mercer-mettl `G_NNNNN`. A separate `allstar_speaker_mapping.json` tracks raw spkID → AS_NNNNN.
- Flushes gt.csv + transcripts.json every 100 segments so a crash doesn't lose work; re-running picks up where it left off.

### 2. `ec2/extract_embeddings.py` — T4, runs once per dataset state

- Input: the uploaded `upload/` folder (gt.csv + audio_npy/).
- For every row × every aug in `--augs` × every WavLM layer in `--wavlm_layers`:
  - Loads waveform → applies aug → WavLM mean-pool per layer (768 each) + Whisper encoder mean-pool (1024).
- Output: a single `embeddings_cache.npz` with `wavlm_<layer>_<aug>` / `whisper_<aug>` matrices aligned to gt's `npy_filename` order, plus `wavlm_layers` / `aug_names` / `filenames` arrays.
- **Incremental**: re-running with the same gt + augs + layers is a no-op. Adding a new audio file, a new aug, or a new layer only extracts the missing combinations. Old caches that only stored a single last-layer matrix (`wavlm_<aug>`) are auto-migrated to `wavlm_last_<aug>`.
- Default layers: `last` (encoder output) + `9` (often best for paralinguistic tasks). WavLM-base-plus has 12 transformer layers, so valid integer tags are 0–12 (where 0 is the embedding output and 12 == last).
- Available augmentations: `orig`, `noise`, `pitch`, `speed`, `gain`, `air`, `vtlp` (custom VTLP, since audiomentations doesn't ship it), `combo`. Optional `rir`, `bgnoise`, `codec` if external corpora + libs are present.

### 3. `ec2/neural_baseline_train.py` — T4, re-run per experiment

- Loads `embeddings_cache.npz` (does not call any encoder).
- Applies `--min_duration` filter to drop short / noisy rows from **both** train and test before splits.
- Split modes (always speaker-disjoint via `group_id`):
  - **Mode A** — empty `--test_batches`: `StratifiedGroupKFold(5)` on `--train_batches` minus `--train_only_batches` → fold 0 test, fold 1 val, folds 2-4 train. Candidate-disjoint, label-stratified.
  - **Mode B** — explicit `--test_batches`: test = those batches (optionally `--test_region_filter IND`). Train pool = `--train_batches` minus `--train_only_batches` minus any candidate that leaks into the test set. Val drawn from the same-region subset of the train pool when large enough, otherwise mixed-region with a warning.
  - After either mode, **`--train_only_batches` rows are appended to train**. By default this is `2676,2677` so ALLSTAR auxiliary data is always train-only.
- For each row in train: concat WavLM (selected layer) + Whisper + handcrafted `feat_*` from gt.csv → standardize on train only. The standardizer / PCA are fit **per WavLM layer** (since the WavLM block is different) and reused across all 5 PCA settings for that layer.
- `--use_augs`: expands the **training** matrix only. `val` and `test` always use `orig` embeddings.
- Trains **20 variant heads** = 2 architectures × 2 WavLM layers × 5 PCA settings:
  - **Architectures**
    - `default` — 512 → 256 → 128 → 1, dropout 0.40, wd 5e-4.
    - `tiny` — 128 → 1, dropout 0.55, wd 5e-3.
  - **WavLM layer**
    - `last` — encoder output (standard baseline).
    - `l9` — `hidden_states[9]`, often the strongest paralinguistic layer.
  - **PCA on the standardized concat** — fit on train only, applied to val + test.
    - `full` (no PCA, control)
    - `pca98`, `pca95`, `pca93`, `pca90`
- All variants: AdamW + cosine schedule, label smoothing 0.05, gradient clip 1.0, BCE-with-logits, early-stop on val F1 (patience 10).
- **Class balancing** (`--class_balance`, default `sampler`):
  - `sampler` — `WeightedRandomSampler` so every TRAIN minibatch is class-balanced in expectation. Replaces the old `pos_weight` reweighting.
  - `pos_weight` — natural shuffling, `pos_weight = neg/pos` in BCE-with-logits.
  - `both` — sampler + pos_weight (usually over-corrects; available for ablation).
  - `none` — natural distribution (sanity-check baseline).
  - Val/test always use natural distribution regardless. The split-summary log line prints the actual pos/neg percentages so the imbalance is visible up front.
- Outputs per variant: `predictions.csv`, `metrics.json`. Plus a top-level `summary.csv` with `variant, arch, wavlm_layer, pca, val_f1, test_auc/ap/f1@0.5/best_f1/topk_f1`, recall@precision (p50/p80/p90/p95), and best/p80/p90 thresholds.

**Note on PCA:** Prior work showed PCA on WavLM destroyed XGBoost performance (768 → 80 dropped F1 from ~0.85 to ~0.58). PCA + MLP may behave differently because the MLP can re-expand the bottleneck and PCA acts as a regularizer rather than a fixed feature selector. The `full` variants are the controls. If PCA variants underperform clearly across both archs and both layers, drop them in later iterations.

**Note on text features inside PCA:** `feat_*` columns are concatenated with WavLM + Whisper **before** standardization and PCA, so every variant — including the most-compressed `pca90` — trains on a representation that still includes the text features (subject to PCA mixing). For ALLSTAR rows `feat_*` is zero (no transcripts), which acts as an "auxiliary, no L2 markers" indicator.

---

## Staged research plan

### Stage 0 (current) — Baseline neural fusion with frozen encoders

**Question this stage answers:** does any of the 10 variants beat the existing weighted XGBoost stack on the same speaker-disjoint test split? If yes, neural head is the new baseline for downstream stages. If no, encoder is the bottleneck (as diagnosed) and we go straight to Stage 1.

**Decision rule:** if any variant beats current best (a4/a5 F1 ~0.74) on the held-out 20% by ≥2 F1 points, switch baselines. If not, proceed to Stage 1.

### Stage 1 — Continued pretraining (CPT) of WavLM on unlabeled IND audio

**Question this stage answers:** does in-domain encoder adaptation move IND-honest embeddings closer to PHP-honest in WavLM space, and does that close the IND F1 gap?

- Pool: a3 IND + a6 IND + ALLSTAR Indian L1 subsets. Target 10–30 hours.
- MLM-style continued pretraining, 5–10s chunks, batch 16, lr 5e-5, ~10–20k steps. ~6–12 T4 hours.
- **Gate (do this before retraining anything):** extract embeddings for held-out IND-honest, IND-cheat, PHP-honest, PHP-cheat with original vs CPT'd WavLM. Measure CORAL distance and t-SNE separation. IND-honest must move closer to PHP-honest. If it doesn't, abort CPT and go to Stage 3 directly.

### Stage 2 — Re-extract embeddings, retrain Stage 0 baseline

If Stage 1 gate passes, re-run `extract_embeddings.py` with the CPT'd WavLM (point it at the new checkpoint) and re-run the train script with the same flags. Single variable: encoder. If a6 F1 jumps ≥0.04, encoder shift was the dominant bottleneck. If not, stop pushing this hypothesis and reconsider.

### Stage 3 — Partial fine-tuning with GRL (gated)

Only if Stage 2 helps but doesn't close enough of the gap.

- Houlsby adapters (bottleneck 256) inserted in each WavLM transformer block. Adapters preferred over top-6-unfreeze at this label scale.
- Cheat head + region head with GRL between encoder pool and region head. λ schedule: 0 for 200 steps, ramp to 0.05 over 500, hold. Conservative ceiling because region label correlates with cheat label.
- Pooling: mean + std concat.
- Augmentation pipeline: codec sim (Opus 16k, AMR-NB), MUSAN noise (5–15 dB SNR), RIR conv, SpecAugment. Skip pitch / VTLP at this stage — they distort paralinguistic features that carry cheat signal once the encoder is actually being updated.
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

## File map

```
allstar/
    upload_to_hf.py             (optional) push local 2676/2677 to a HF dataset repo
companylaptop/
    neural_baseline_prep.py     air-gapped mercer-mettl prep + anonymization
    text_features.py            dependency-free transcript -> feat_* features
                                 (the lite 18-col version; only used as a fallback)
ec2/
    allstar_prep.py             download from HF + segment + faster-whisper +
                                 full 55-feature audios6 spec; appends to gt.csv
    full_text_features.py       the 55-feature pipeline ported from audios6_eval
    extract_embeddings.py       cold encoder pass over orig + every aug
                                 (WavLM last + layer 9, Whisper). Incremental.
    neural_baseline_train.py    cache -> 20 variant MLPs -> summary.csv
    requirements.txt
NEURAL_PLAN.md                  this doc
```

---

## Typical command flow

See `COMMANDS.txt` at the project root for the full copy-pasteable version with
placeholders. Brief outline:

```powershell
# (A) Company laptop -- mercer-mettl prep + upload
python companylaptop\neural_baseline_prep.py
# upload data\neural_prep_out\upload\  to EC2 (scp/rsync)

# (B) EC2 -- install + ALLSTAR + extract + train
pip install -r ec2/requirements.txt
python -m spacy download en_core_web_sm
huggingface-cli login

python ec2/allstar_prep.py --out_dir /opt/data/upload          # downloads + transcribes + feats
python ec2/extract_embeddings.py \                              # WavLM last + l9, Whisper, all augs
    --data_dir /opt/data/upload \
    --out_path /opt/data/upload/embeddings_cache.npz \
    --augs orig,noise,pitch,speed,gain,air,vtlp,combo \
    --wavlm_layers last,9
python ec2/neural_baseline_train.py \                           # 20 variants per run
    --data_dir /opt/data/upload \
    --out_dir  /opt/runs/r1 \
    --train_batches audios2,audios4,audios5,2676,2677 \
    --train_only_batches 2676,2677 \
    --test_batches audios6 --test_region_filter IND \
    --min_duration 5.0 --use_augs all
```
