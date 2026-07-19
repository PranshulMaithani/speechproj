# Reference — Unified TRAINING (`train_pipeline.py` + `wav_pipeline.py`)

Every parameter of the **one-command** training path. Two entry points:

- **`pipeline/wav_pipeline.py`** — full **WAV → trained models** (transcribe → anonymise/npy →
  embeddings → train). Use when you start from raw audio. `wav_pipeline_gpu.py` is the identical
  GPU-transcription copy.
- **`ec2/train_pipeline.py`** — **cache/gt → trained models** (the training core the wav pipeline
  calls in its last stage). Use directly when `gt.csv` + `audio_npy/` already exist.

**PII:** the wav pipeline anonymises real CID → encoded id, keeps `local/cid_mapping.json`
local, and only `.npy` + `gt.csv` move downstream. Raw wavs never leave.

---

## A. `pipeline/wav_pipeline.py` — WAV → everything

### Where to put your data (modular; no code edits to add a batch)
Drop side by side **inside `companylaptop/`**:
```
companylaptop/audios<N>/<realCID>_<qid>.wav     ← raw audio, flat folder
companylaptop/audios<N>GT.csv                   ← columns: filename,label[,region]
```
`--batches auto` discovers every `audios<N>/` that has a matching `<batch>GT.csv`, so adding
audios8/9/… is just dropping the folder + its GT csv.

### Stages (run in this order; any subset via `--stages`)
`features` (transcribe + 55 feat_*) → `prep` (anonymise → npy + gt.csv + cid_mapping) →
`embed` (WavLM+Whisper cache) → `train` (variant×seed + threshold books).

### Parameters
| Flag | Default | Meaning |
|---|---|---|
| `--audio_root` | `companylaptop/` | folder holding `audios<N>/` + `audios<N>GT.csv` |
| `--batches` | `auto` | `auto` = discover; or comma list e.g. `audios7,audios8` |
| `--stages` | `features,prep,embed,train` | comma subset to run, in order |
| `--force` | off | pass `--force` to the features + embed stages |
| `--dry_run` | off | print each stage's command without executing |
| `--transcribe_device` | `cpu` (gpu copy: `cuda`) | faster-whisper device |
| `--compute_type` | `int8` (gpu copy: `float16`) | faster-whisper compute type |
| `--transcribe_model` | `""` | override the transcription model — **leave empty** to keep feature parity |
| `--cache` | `data/neural_prep_out/upload/embeddings_cache_base.npz` | embedding cache to build/extend |
| `--augs` | `orig,noise,pitch,speed,gain,air,vtlp,combo` | augs to cache |
| `--wavlm_layers` | `last,9` | WavLM layers to extract |
| `--wavlm_id` / `--whisper_id` | `""` → base-plus / medium | override encoder checkpoints |
| `--data_dir` | `data/neural_prep_out/upload` | gt.csv + audio_npy dir (prep's output) |
| `--out_dir` | `data/runs/retrain_from_wav` | train_pipeline output dir |
| `--train_config` | `""` | YAML job file passed to `train_pipeline.py` (see §B) |

### Run
```bash
python pipeline/wav_pipeline.py --batches auto \
    --out_dir data/runs/retrain_from_wav --train_config ec2/configs/train_job.example.yaml
# GPU transcription (faster): same flags, use wav_pipeline_gpu.py
python pipeline/wav_pipeline_gpu.py --batches auto --out_dir data/runs/retrain_from_wav_gpu \
    --train_config ec2/configs/train_job.example.yaml
# re-train only (skip transcribe/prep/embed):
python pipeline/wav_pipeline.py --stages train --out_dir ...
```
`wav_pipeline_gpu.py` reuses `wav_pipeline.py`'s logic and only flips Stage-1 defaults to
`cuda`/`float16` — same model, prompt, output schema, features, embeddings, and training.

---

## B. `ec2/train_pipeline.py` — cache/gt → models (the training core)

Config via a **YAML job file** (`--config`) whose fields any CLI flag overrides. Stages:
0 guarded extract → 1 load → 2 splits per seed → 3 train grid×seed (export bundles + val/test
predictions) → 4 aggregate → 5 per-variant threshold workbooks.

### Config fields / CLI flags (every one is `--<key>`)
| Key | Default | Meaning |
|---|---|---|
| `data_dir` | *(required)* | folder with `gt.csv` (+ `audio_npy/`) |
| `cache` | `<data_dir>/embeddings_cache.npz` | embedding cache |
| `out_dir` | *(required)* | run output dir |
| `audio_subdir` | `audio_npy` | where Stage-0 extract looks for waveforms |
| `do_extract` | `auto` | `auto` (extract only if audio present + something missing) / `true` / `false` |
| `wavlm_id` / `whisper_id` | base-plus / medium | encoder ids (only used if extraction runs) |
| `train_batches` | `audios2,audios4,audios5,audios6` | train pool |
| `test_batches` | `[]` | `[]` → **20pct**; `[audios6]` → **a6** |
| `test_region_filter` | `""` | region-restrict the a6 test set (e.g. `PHP`) |
| `train_only_batches` | `[casual]` | forced train-only aux (e.g. `casual`, `2676,2677`) |
| `use_augs` | `all` | `all` / comma list / `""` (none) |
| `seeds` | `42,43,44,45,46` | each seed reshuffles the split AND seeds init |
| `variants` | `all` | `all` (30) / `grid` (archs×layers×pca below) / explicit names |
| `archs` | `default,tiny,linear` | used when `variants: grid` |
| `layers` | `last,9` | used when `variants: grid` |
| `pca` | `full,pca98,pca95,pca93,pca90` | used when `variants: grid` |
| `min_duration` | 30.0 | drop rows shorter than this |
| `use_text_features` | `true` | include the 55 `feat_*` |
| `per_client_standardize` | `false` | center features per client (unsupervised) |
| `class_balance` | `sampler` | `sampler` / `pos_weight` / `both` / `none` |
| `batch_size` / `epochs` / `lr` | 64 / 60 / 1e-3 | training knobs |
| `thr_step` | 0.01 | threshold-workbook granularity |
| `--config` | — | (CLI only) path to the YAML job file |

Sample config: `ec2/configs/train_job.example.yaml`. To freeze **specific** configs only, set
`variants:` to an explicit list (each `variant × seed` is exported).

### Outputs (`out_dir`)
```
config_resolved.json · log_train.txt · splits/seed_<seed>.json
models/<variant>/seed_<seed>/{model.pt, scaler.joblib, pca.joblib, inference_meta.json,
                              predictions.csv (test), predictions_val.csv, metrics.json}
models/<variant>/threshold_sweep.xlsx     ← per-seed val/test/combined sweep + robustness sheet
summary/{per_run.csv, summary_mean_std.csv, best_models.csv, training_summary.xlsx}
```
Every model folder is a **complete, self-describing bundle** → score any with
`evaluate_models.py` (see `REF_EVAL_uniscript.md`). Pick a winner from `summary_mean_std.csv`.

### Run directly (cache already built)
```bash
python ec2/train_pipeline.py --config ec2/configs/train_job.example.yaml \
    --data_dir <UP> --cache <CACHE> --out_dir data/runs/retrain_20pct
```

*See also: `RUNBOOK.md` Part B, `METHODOLOGY.md`.*
