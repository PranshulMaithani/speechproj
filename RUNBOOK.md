# RUNBOOK — Running the Per-Audio Cheating-Detection Pipeline

End-to-end, chronological instructions for the **current decided method**: from a fresh set
of audios → features → embeddings → trained models → evaluation → ablations.

There are **two ways** to run the training half, documented as two parts:

- **Part A — legacy multi-script flow** (what we actually used to produce the finalised
  models): a chain of purpose-built scripts, one per stage.
- **Part B — unified flow**: the two consolidated scripts (`train_pipeline.py`,
  `evaluate_models.py`).

The **laptop-side feature/prep steps (Stage 1) are identical for both parts** — only the
training/eval half differs.

---

## 0. Orientation

### The PII boundary (non-negotiable)
| Stays on the **laptop** (never leaves) | Goes to the **cloud (EC2)** |
|---|---|
| raw `.wav`, transcripts, real candidate IDs | anonymised `.npy` waveforms, `gt.csv` (feat_* + labels), embedding caches |

Transcription **and** the 55 `feat_*` features are computed on the laptop; only anonymised
numbers + encoded waveforms are uploaded. **No cloud script ever reads a wav or a transcript.**

### Companion docs (deep detail for specific stages)
| Doc | Use it for |
|---|---|
| `COMMANDS_ADD_BATCH.txt` | the exact laptop→bucket→EC2 new-batch commands (source for Part A Stage 1–2) |
| `COMMANDS_CACHE_FIX.txt` | diagnosing / stamping / extending the embedding cache |
| `COMMANDS_ANALYSIS.txt` | the finalised M1/M2 training + analysis-workbook commands |
| `COMMANDS_CASUAL.txt` | adding the Casual-Conversations auxiliary batch |
| `COMMANDS_LARGE.txt` | building a large-model cache (wavlm-large / whisper-large-v3) |
| `COMMANDS_AUG_SAMPLES.txt` | augmentation ablation sample commands |
| `METHODOLOGY.md` | what the system is + the two finalised models |
| `EMBEDDING_EXTRACTION.md` | how WavLM/Whisper embeddings are produced |

### The two finalised models (the target of the whole pipeline)
- **Model 1** = `default_last_pca98` **+ casual**, on the **20pct** split → run dir `m1_casual_20pct`.
- **Model 2** = `tiny_l9_pca95` (no casual), on the **a6** split → run dir `m2_nocasual_a6`.

### Placeholders used below
`<DATA_DIR>` = EC2 data dir (e.g. `/home/ubuntu/nn/data`) holding `gt.csv` + `audio_npy/` +
the cache · `<BASE_CACHE>` = the 768-d base cache (`embeddings_cache_base.npz`, wavlm-base-plus
+ whisper-medium) · `<BUCKET>` = your S3 bucket · `<RUNS>` = `/home/ubuntu/nn/runs`.

---

# PART A — Legacy multi-script flow (chronological)

The order of scripts, start to finish:

```
LAPTOP:  extract_features_batch.py  →  neural_baseline_prep.py  →  (stage + upload zip)
EC2:     extract_embeddings.py  →  neural_baseline_multiseed.py  (pick arch)
         →  neural_baseline_train.py --export_artifacts  (freeze M1 & M2)
         →  evaluate_final_models.py  (score, incl. new-batch validity)
         →  [optional] aug_ablation* / aug_strategy_20pct / aug_combo_ablation / xgboost_train
```

## Stage 1 — LAPTOP: new audios → anonymised gt.csv + npy

**1.0 Lay out the batch** (`audios<N>/` with files `<realCID>_<qid>.wav`, plus
`audios<N>GT.csv` with `filename,label[,region]`). For a single-region batch, either add a
`region` column or set `DEFAULT_REGION_BY_BATCH["audios<N>"]` in `neural_baseline_prep.py`.

**1.1 Transcribe + compute the 55 features** (needs faster-whisper, spaCy `en_core_web_sm`,
parselmouth, transformers):
```bash
python companylaptop/extract_features_batch.py --batch audios7
# GPU: ... --device cuda --compute_type float16
# -> audios7_transcripts.json + audios7_features.csv   (LOCAL, PII-bearing)
```

**1.2 Prep → anonymised `gt.csv` + `.npy`** (auto-discovers the new batch; rebuilds gt.csv
for ALL batches):
```bash
python companylaptop/neural_baseline_prep.py
# KEEP data/neural_prep_out/local/cid_mapping.json  (so candidate IDs stay stable)
```

**1.3 Stage ONLY the new npy + the new gt.csv into one zip** (PowerShell — see
`COMMANDS_ADD_BATCH.txt` A3 for the exact snippet), then **1.4 upload**:
```powershell
aws s3 cp D:\audios7_upload.zip s3://<BUCKET>/
```

## Stage 2 — EC2: unzip + build embeddings (incremental)

**2.1 Pull + unzip** (replaces `gt.csv`, adds the new npy; back up first):
```bash
cd <DATA_DIR> && cp gt.csv gt.csv.bak
aws s3 cp s3://<BUCKET>/audios7_upload.zip . && unzip -o audios7_upload.zip -d <DATA_DIR>
```

**2.2 Extract WavLM+Whisper embeddings** — only the new files get encoded (incremental).
Model IDs / augs / layers MUST match the cache being extended:
```bash
cd /home/ubuntu/nn && source venv/bin/activate
tmux new -s emb    # survive SSH drops
python ec2/extract_embeddings.py \
    --data_dir <DATA_DIR> --out_path <BASE_CACHE> \
    --augs orig,noise,pitch,speed,gain,air,vtlp,combo --wavlm_layers last,9
# watch: "files needing extraction = <new-batch count>"
```
(Model-ID mismatch error → see `COMMANDS_CACHE_FIX.txt` / the TROUBLESHOOTING block in
`COMMANDS_ADD_BATCH.txt`.)

**2.3 Verify cache aligns with gt** (must be 0 missing, 0 all-zero) — snippet in
`COMMANDS_ADD_BATCH.txt` B3.

## Stage 3 — EC2: TRAIN

**3a — Choose the architecture (multi-seed, fluke-proof).** Ranks every variant over 5 seeds
by mean ± std → `summary_mean_std.csv`:
```bash
# 20pct protocol (reshuffle all candidates each seed):
python ec2/neural_baseline_multiseed.py \
    --data_dir <DATA_DIR> --cache <BASE_CACHE> \
    --out_dir <RUNS>/multiseed_20pct \
    --train_batches audios2,audios4,audios5,audios6 \
    --seeds 42,43,44,45,46
# a6 protocol (test fixed to audios6, reshuffle train/val):
#   ... --out_dir <RUNS>/multiseed_a6 --train_batches audios2,audios4,audios5,audios6 --test_batches audios6
```

**3b — Freeze the two finalised models** with exported weights (each run writes
`<out_dir>/<variant>/{model.pt, scaler.joblib, pca.joblib, inference_meta.json,
predictions.csv}`):
```bash
# Model 1: default_last_pca98 + casual @ 20pct
python ec2/neural_baseline_train.py \
    --data_dir <DATA_DIR> --cache <BASE_CACHE> --out_dir <RUNS>/m1_casual_20pct \
    --train_batches audios2,audios4,audios5,audios6 --train_only_batches casual \
    --test_batches "" --min_duration 30 --use_text_features true --use_augs all \
    --per_client_standardize false --export_artifacts true --dump_full_predictions true

# Model 2: tiny_l9_pca95, no casual @ a6 (test = all audios6)
python ec2/neural_baseline_train.py \
    --data_dir <DATA_DIR> --cache <BASE_CACHE> --out_dir <RUNS>/m2_nocasual_a6 \
    --train_batches audios2,audios4,audios5,audios6 --train_only_batches "" \
    --test_batches audios6 --min_duration 30 --use_text_features true --use_augs all \
    --per_client_standardize false --export_artifacts true --dump_full_predictions true
```
> This runs the full 30-variant sweep with one global seed (no `--variants`), which
> reproduces the finalised numbers exactly; `evaluate_final_models.py` then picks the
> `default_last_pca98` / `tiny_l9_pca95` sub-dirs. (See `COMMANDS_ANALYSIS.txt`.)

## Stage 4 — EC2: EVALUATE

**4a — Score both finalised models on their own test sets** (verifies the reloaded weights
reproduce the saved predictions to < 1e-3):
```bash
python ec2/evaluate_final_models.py \
    --data_dir <DATA_DIR> --cache <BASE_CACHE> --runs_root <RUNS> \
    --out_dir <RUNS>/final_eval
    # override dirs if named differently: --m1_dir m1_casual_20pct --m2_dir m2_nocasual_a6
```

**4b — Validity check on a brand-new batch** (same stored weights, fresh data neither model
trained on):
```bash
python ec2/evaluate_final_models.py \
    --data_dir <DATA_DIR> --cache <BASE_CACHE> --runs_root <RUNS> \
    --out_dir <RUNS>/final_eval_audios7 --test_batch audios7
```
Output workbook `evaluate_final_models.xlsx`: `summary` + per-model `_preds` / `_sweep` /
`_by_region`. Read: audios7 metrics close to own-test ⇒ it generalises; a big drop ⇒ it doesn't.

## Stage 5 — EC2: ABLATIONS (optional, any order)

| Script | Question it answers | Sample |
|---|---|---|
| `aug_strategy_20pct.py` | best aug **subset** for Model 1, split reshuffled over 5 seeds (singles/LOO/greedy/leaderboard) | `--data_dir <DATA_DIR> --cache <BASE_CACHE> --out_dir <RUNS>/aug_strategy_20pct --seeds 42,43,44,45,46` |
| `aug_combo_ablation.py` | exhaustive **every size-3..6 combination**, single seed, one model | `--data_dir <DATA_DIR> --cache <BASE_CACHE> --out_dir <RUNS>/aug_combo_m1 --model 1 --seed 42` (see `aug_combo_ablation_RUN.txt`) |
| `aug_ablation.py` | leave-one-out / greedy on a **frozen** split, multi-seed avg | `--data_dir <DATA_DIR> --cache <BASE_CACHE> --out_dir <RUNS>/aug_ablation --seeds 42,43,44,45,46` |
| `aug_ablation2.py` | LOO **anchored** to the exact original exported weights | `--data_dir <DATA_DIR> --cache <BASE_CACHE> --out_dir <RUNS>/aug_ablation2 --runs_root <RUNS>` |
| `xgboost_train.py` | gradient-boosted baseline over the same embeddings (companion to the NN) | shares `_data_pipeline.py`; same `--data_dir/--cache/--out_dir/--train_batches/...` |

More aug-ablation examples: `COMMANDS_AUG_SAMPLES.txt`.

---

# PART B — Unified flow (`train_pipeline.py` + `evaluate_models.py`)

Same **Stage 1 (laptop) and the upload** as Part A — nothing changes there. The unified
scripts replace **Stages 2–5 on EC2**:

| Legacy (Part A) | Unified (Part B) |
|---|---|
| `extract_embeddings.py` + `neural_baseline_multiseed.py` + `neural_baseline_train.py --export_artifacts` | **`train_pipeline.py`** (guarded extract → train grid×seeds → export all + threshold workbooks) |
| `evaluate_final_models.py` | **`evaluate_models.py`** (discovers any saved model folder, scores it) |

## Stage 2/3 — TRAIN (one command)

Write a job config once (copy `ec2/configs/train_job.example.yaml`), then:
```bash
python ec2/train_pipeline.py --config ec2/configs/train_job.example.yaml \
    --out_dir <RUNS>/retrain_20pct
```
What it does, in order (each stage skippable):
1. **Coverage + guarded extract** — prints a per-batch table (audio present? cached? augs/
   layers covered?) and only runs `extract_embeddings.py` if audio is present **and**
   something's missing. On EC2 (no audio) it auto-skips and uses the cache.
2. Load gt + cache; build one reshuffled split per seed (`splits/seed_*.json`).
3. Train every `variant × seed`, exporting the full artifact folder + **both** val and test
   predictions per model.
4. Aggregate → `summary/per_run.csv`, `summary_mean_std.csv`, `best_models.csv`.
5. Per-variant **threshold workbook** (`models/<variant>/threshold_sweep.xlsx`): a `seed_<n>`
   sheet (val/test/combined P-R-F1 at every threshold) + a `robustness` sheet lining all 5
   seeds up with mean/min/std-F1 and recommended thresholds.

Key config fields (all overridable on the CLI): `train_batches`, `test_batches`
(`[]`→20pct, `[audios6]`→a6), `train_only_batches` (e.g. `casual`), `use_augs`, `variants`
(`all` / `grid` / explicit names), `archs`/`layers`/`pca` (for `grid`), `seeds`,
`min_duration`, `do_extract` (`auto`/`true`/`false`).

Output tree:
```
<out_dir>/
  config_resolved.json · log_train.txt · splits/seed_<seed>.json
  models/<variant>/seed_<seed>/{model.pt, scaler.joblib, pca.joblib, inference_meta.json,
                                predictions.csv, predictions_val.csv, metrics.json}
  models/<variant>/threshold_sweep.xlsx
  summary/{per_run.csv, summary_mean_std.csv, best_models.csv, training_summary.xlsx}
```

## Stage 4 — TEST / INFERENCE (any saved model, any data)

`evaluate_models.py` discovers every folder under `--models_root` that has
`model.pt + inference_meta.json` and scores it — no model-specific code.

```bash
# score every saved model on its OWN test set (verifies reload == saved scores):
python ec2/evaluate_models.py \
    --models_root <RUNS>/retrain_20pct/models \
    --data_dir <DATA_DIR> --cache <BASE_CACHE> \
    --out_dir <RUNS>/retrain_20pct/eval_own

# score selected models on a fresh held-out batch (audios7):
python ec2/evaluate_models.py \
    --models_root <RUNS>/retrain_20pct/models \
    --models "default_last_pca98/seed_42,tiny_l9_pca95/seed_44" \
    --data_dir <DATA_DIR> --cache <BASE_CACHE> --test_batch audios7 \
    --out_dir <RUNS>/eval_audios7
```
Outputs: `per_model/<id>/{predictions.csv, threshold_sweep.csv, metrics.json}`,
`eval_summary.csv`, `eval_summary.xlsx`. Use `--threshold` to also record a chosen operating
threshold per row.

---

## Picking a model — any variant is usable (not just M1/M2)

**M1/M2 are a choice, not a limit.** A training run freezes **every** variant it trains
(`neural_baseline_train.py` runs the full sweep; `train_pipeline.py` runs whatever `variants:`
lists), each into its own self-describing folder:

```
<variant>/  →  model.pt · scaler.joblib · pca.joblib · inference_meta.json
```

At the artifact level all variants are equal — nothing marks one as "finalised." `M1`
(`default_last_pca98`) and `M2` (`tiny_l9_pca95`) are known only by **convention** (the run-dir
names), the **hardcoded specs in `evaluate_final_models.py`**, and **`METHODOLOGY.md` §6**.

So in future retraining you can pick **any** variant's weights and use it — after evaluation:

1. Read **`summary/summary_mean_std.csv`** — it ranks every variant by avg best-F1 ± std over
   the seeds. High mean + low std = genuinely good, not a lucky seed.
2. **Match the protocol to your use case:** in-domain (same client, you have labels) → the
   20pct ranking, favour full-capacity; new client → the a6 ranking, favour the small /
   layer-9 model (a big model on `last` memorises client identity). Skip the `linear`
   variants — they're sanity baselines.
3. Confirm the pick on held-out data with `evaluate_models.py`, then that folder **is** your
   model. (Optionally copy it to a stable path like `runs/finalised/<variant>/`.)

**Three guardrails so a swapped model "just works":**
- **Move the whole folder as a unit** — the `scaler`/`pca` were fit on that run's train data;
  never mix a `model.pt` with another run's scaler/PCA.
- **Same feature schema at inference** — same base cache (same WavLM/Whisper IDs → same dims)
  and the same 55 `feat_*` columns in the same order. This is enforced: `inference_meta.json`
  stamps the dims + `feat_cols` and the loader checks `in_dim`, so a mismatch **errors** rather
  than silently mispredicting.
- **Re-check the decision threshold on the target data** — weights transfer cleanly, the
  F1-optimal threshold may not (the cross-client caveat). Use the threshold sweep to set it.

## Scoring one model — what `--models_root` is

`--models_root` is the **directory `evaluate_models.py` searches** (recursively), not a file.
It treats **every folder containing `model.pt` + `inference_meta.json`** as a model to score —
that's how it can sweep many at once. To score just **one** model, point the root straight at
a folder that holds that one bundle:

```bash
# runs/mypick/<variant>/  contains: model.pt, scaler.joblib, pca.joblib, inference_meta.json
python ec2/evaluate_models.py \
    --models_root runs/mypick \
    --data_dir <DATA_DIR> --cache <BASE_CACHE> --test_batch audios7 \
    --out_dir runs/eval_mypick
```

Or keep a shared root and narrow with the substring filter: `--models "default_l9_pca95/seed_42"`.
The only requirement is the folder holds the **complete bundle** (`model.pt`, `scaler.joblib`,
`inference_meta.json`, **and `pca.joblib` if the model used PCA**). `evaluate_models.py` does
**not** care about the `runs_root`/`run_dir` training layout — only about finding
`model.pt + inference_meta.json` under `--models_root`.

---

## Quick decision guide

- **New batch of audios?** → Part A Stage 1 (laptop) + upload, always. Then either Part A
  Stages 2–4 **or** Part B (one `train_pipeline.py` run + `evaluate_models.py`).
- **Reproduce the exact finalised M1/M2 numbers?** → Part A Stage 3b + 4a (the sweep with one
  global seed is what those numbers came from).
- **New retraining experiment (pick the best of many variants, get threshold workbooks)?** →
  Part B (`train_pipeline.py`).
- **Score an existing model on new data?** → `evaluate_models.py` (Part B) or
  `evaluate_final_models.py --test_batch` (Part A) for the two finalised models specifically.
- **Which augs help?** → Part A Stage 5 (`aug_strategy_20pct.py` / `aug_combo_ablation.py`).

*Source of truth for commands: the `COMMANDS_*.txt` files + each script's `--help`. System
description: `METHODOLOGY.md`.*
