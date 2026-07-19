# Reference — Unified TRAINING (`ec2/train_pipeline.py`)

**One universal training script.** It optionally turns **raw wavs → anonymised npy + gt.csv**
(auto-detecting new audio folders), builds the WavLM/Whisper embedding cache, then trains the
variant × seed grid — all from a single command and one config. Run it from raw audio on a
jump server, or from an existing `gt.csv` + cache on EC2. For evaluation see
`REF_EVAL_uniscript.md`; for the older per-stage scripts see `REF_TRAIN_multiscript.md`.

**PII:** Stage I anonymises real CID → encoded id, keeps `local/cid_mapping.json` local, and
only `.npy` + `gt.csv` move downstream. Raw wavs never move.

---

## Where to put your data (modular — no code edits to add a batch)
Drop, side by side, **inside `companylaptop/`** (or set `audio_root`):
```
companylaptop/audios<N>/<realCID>_<qid>.wav     ← raw audio, flat folder
companylaptop/audios<N>GT.csv                   ← columns: filename,label[,region]
```
`label`: `read/cheating/scripted/yes/1` → **1**; `spontaneous/genuine/no/0` → **0**.
Stage I auto-discovers any `audios<N>/` that has a matching `<batch>GT.csv`, so adding
audios8/9/… needs **zero code changes** — just the folder + its GT csv, then re-run.
- Single-region batch, no `region` column → add `"audios<N>": "IND"` in
  `neural_baseline_prep.DEFAULT_REGION_BY_BATCH`.
- Mixed-region batch → put a `region` column in its GT csv.

## The stages (in order)
| Stage | What it does | Skippable via |
|---|---|---|
| **I ingest** | auto-detect folders → transcribe new batches (faster-whisper, CPU/GPU) → anonymise → `npy` + `gt.csv` | `ingest` |
| **0 extract** | build/extend the WavLM+Whisper embedding cache (incremental) | `do_extract` |
| **1 load** | load gt + cache | — |
| **2 splits** | one reshuffled split per seed (`splits/seed_*.json`) | — |
| **3 train** | every `variant × seed`; export full bundle + val/test predictions | — |
| **4 aggregate** | `per_run.csv`, `summary_mean_std.csv`, `best_models.csv` | — |
| **5 threshold** | per-variant `threshold_sweep.xlsx` (seed sheets + robustness) | — |

`ingest = auto` runs Stage I **only** if a discovered batch is new (not in `gt.csv`) or is
missing its features csv; otherwise it skips straight to training. On EC2 (no `audios<N>/`
folders) it auto-skips, so the same script also serves the pure train-from-cache path.

## Config fields / CLI flags (every one is also `--<key>`)
Config via a YAML job file (`--config`); any CLI flag overrides it. Sample:
`ec2/configs/train_job.example.yaml`.

**Stage I — wav ingest**
| Key | Default | Meaning |
|---|---|---|
| `ingest` | `auto` | `auto` (run if new folders/features) / `true` (always) / `false` (skip; train from gt/cache) |
| `audio_root` | `""` → `companylaptop/` | folder holding `audios<N>/` + `audios<N>GT.csv` |
| `transcribe_device` | `cpu` | `cpu` \| `cuda` — **`cuda` = fast GPU transcription** |
| `transcribe_compute_type` | `int8` | faster-whisper compute type (cuda: `float16`) |
| `transcribe_model` | `""` | override transcription model — **leave empty** to keep feature parity with audios2..6 |
| `retranscribe` | `false` | re-transcribe every discovered batch (`--force`) |

**Paths / embeddings**
| Key | Default | Meaning |
|---|---|---|
| `data_dir` | `data/neural_prep_out/upload` | gt.csv + audio_npy dir. Default = the ingest output; **override for a ready EC2 dir** |
| `cache` | `<data_dir>/embeddings_cache.npz` | embedding cache to build/extend |
| `out_dir` | `data/runs/train` | run output dir |
| `audio_subdir` | `audio_npy` | where Stage-0 extract reads waveforms |
| `do_extract` | `auto` | `auto` (extract only if audio present + missing) / `true` / `false` |
| `wavlm_id` / `whisper_id` | base-plus / medium | encoder ids (only used if extraction runs) |

**Data selection / training**
| Key | Default | Meaning |
|---|---|---|
| `train_batches` | `audios2,audios4,audios5,audios6` | train pool |
| `test_batches` | `[]` | `[]` → **20pct**; `[audios6]` → **a6** |
| `test_region_filter` | `""` | region-restrict the a6 test set (e.g. `PHP`) |
| `train_only_batches` | `[casual]` | forced train-only aux (e.g. `casual`, `2676,2677`) |
| `use_augs` | `all` | `all` / comma list / `""` |
| `seeds` | `42,43,44,45,46` | each seed reshuffles the split AND seeds init |
| `variants` | `all` | `all` (30) / `grid` (archs×layers×pca) / explicit list |
| `archs` / `layers` / `pca` | 3 / `last,9` / 5 | used when `variants: grid` |
| `min_duration` | 30.0 | drop rows shorter than this |
| `use_text_features` | `true` | include the 55 `feat_*` |
| `per_client_standardize` | `false` | center features per client |
| `class_balance` | `sampler` | `sampler` / `pos_weight` / `both` / `none` |
| `batch_size` / `epochs` / `lr` | 64 / 60 / 1e-3 | training knobs |
| `thr_step` | 0.01 | threshold-workbook granularity |
| `--config` | — | (CLI only) YAML job file |

## Outputs (`out_dir`)
```
config_resolved.json · log_train.txt · splits/seed_<seed>.json
models/<variant>/seed_<seed>/{model.pt, scaler.joblib, pca.joblib, inference_meta.json,
                              predictions.csv (test), predictions_val.csv, metrics.json}
models/<variant>/threshold_sweep.xlsx     ← per-seed val/test/combined sweep + robustness sheet
summary/{per_run.csv, summary_mean_std.csv, best_models.csv, training_summary.xlsx}
```
Plus (when Stage I runs): `data/neural_prep_out/upload/{gt.csv, audio_npy/}` and
`data/neural_prep_out/local/cid_mapping.json` (**LOCAL** — keep it so encoded ids stay stable).
Every model folder is a **self-describing bundle** → score any with `evaluate_models.py`.

## Run
```bash
# From raw wavs (jump server): drop audios<N>/ + GT in companylaptop/, then:
python ec2/train_pipeline.py --config ec2/configs/train_job.example.yaml \
    --out_dir data/runs/retrain_from_wav
# ...with fast GPU transcription:
python ec2/train_pipeline.py --config ec2/configs/train_job.example.yaml \
    --transcribe_device cuda --transcribe_compute_type float16 \
    --out_dir data/runs/retrain_from_wav_gpu

# Train-only from an existing EC2 gt + cache (no audio present -> ingest auto-skips,
# or force it off):
python ec2/train_pipeline.py --config ec2/configs/train_job.example.yaml \
    --data_dir /home/ubuntu/nn/data --cache /home/ubuntu/nn/data/embeddings_cache_base.npz \
    --ingest false --out_dir data/runs/retrain_20pct

# Re-train only, reuse existing gt+cache+embeddings:
python ec2/train_pipeline.py --ingest false --do_extract false --out_dir ...
```

*See also: `RUNBOOK.md` Part B, `METHODOLOGY.md`, `EMBEDDING_EXTRACTION.md`.*
