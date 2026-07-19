# Reference — Multi-Script TRAINING (legacy flow)

Every parameter of every script in the **legacy multi-script training** path, in run order,
so anyone can reproduce or extend it. This is the flow that produced the finalised Model 1 /
Model 2. For the one-command version see `REF_TRAIN_uniscript.md`.

**PII rule:** raw wavs + transcripts + real IDs stay on the laptop/jump server; only the
anonymised `.npy` + `gt.csv` move to EC2. The mapping `cid_mapping.json` is kept LOCAL.

### Run order
```
LAPTOP:  extract_features_batch.py (per batch)  →  neural_baseline_prep.py
EC2:     extract_embeddings.py  →  neural_baseline_multiseed.py (pick arch)
                                →  neural_baseline_train.py --export_artifacts (freeze)
```

---

## Where to put your data (the modular convention)

Drop, side by side, **inside `companylaptop/`** (flat or nested both work — the scripts
`rglob`):
```
companylaptop/audios<N>/<ciid>/<ciid>_<q>.wav   ← raw audio (nested per-candidate, q=1..27)
companylaptop/audios<N>GT.csv                   ← columns: filename,label[,region]
```
By default these live in `companylaptop/`. To instead keep them in the **data dir** (with the
npy — as the unified `train_pipeline.py` does), set env vars: `PREP_AUDIO_ROOT=<data_dir>`
(read by prep) + `--audio_root <data_dir>` (for `extract_features_batch.py`), and
`PREP_UPLOAD_DIR=<data_dir>` / `PREP_LOCAL_DIR=<data_dir>/local` so prep writes `gt.csv` +
`audio_npy/` there and keeps `cid_mapping.json` local.

**Only Q25/26/27 are processed** (→ npy → embeddings). `extract_features_batch.py` and
`neural_baseline_prep.py` read the env var `KEEP_QUESTIONS` (default `25,26,27`; set
`KEEP_QUESTIONS=all` to keep every question). To reclaim disk, delete the other questions
with `companylaptop/prune_questions.py` (dry-run by default; `--apply` to delete):
```bash
python companylaptop/prune_questions.py --batch audios8 --apply
```
`label` accepts `read/cheating/scripted/yes/1` → **1 (cheat)** and
`spontaneous/genuine/no/0` → **0**. Adding audios8, audios9, … needs **no code change** —
`discover_batches()` picks up any `audios<N>/` that has a matching `<batch>GT.csv`.
- Single-region batch with no `region` column → add `"audios<N>": "IND"` (or `PHP`) to
  `DEFAULT_REGION_BY_BATCH` in `neural_baseline_prep.py`.
- Mixed-region batch → put a `region` column in its GT csv.

---

## Stage 1 — `companylaptop/extract_features_batch.py`
Transcribe (rich word-timestamped) + compute the 55 handcrafted `feat_*`. Run **once per batch**.

| Flag | Default | Meaning |
|---|---|---|
| `--batch` | *(required)* | batch name / folder, e.g. `audios7` |
| `--audio_root` | `""` → the script's dir (`companylaptop/`) | where `<batch>/` and `<batch>GT.csv` live |
| `--only` | `both` | `both` \| `transcribe` \| `features` — run one stage only |
| `--model` | `WHISPER_TRANSCRIBE_MODEL` (small) | transcription model. **Keep the default** — changing it diverges the features from audios2..6 |
| `--device` | `cpu` | `cpu` \| `cuda` (GPU is much faster) |
| `--compute_type` | `int8` | faster-whisper compute type (`int8` cpu; `float16` gpu) |
| `--force` | off | re-transcribe / recompute even if outputs exist |

**Reads:** `<audio_root>/<batch>/*.wav`. **Writes (LOCAL):** `<batch>_transcripts.json`,
`<batch>_features.csv`. GPU: `--device cuda --compute_type float16`.

## Stage 2 — `companylaptop/neural_baseline_prep.py`
Anonymise (real CID → encoded group id), encode wav → npy, build `gt.csv`. **No CLI args** —
it auto-discovers batches and uses these module constants:

| Constant | Default | Meaning |
|---|---|---|
| `AUDIO_ROOT` | the script dir (`companylaptop/`) | where `audios<N>/` + `<batch>GT.csv` live |
| `OUTPUT_ROOT` | `../data/neural_prep_out` | output root |
| `BATCHES` | `audios2..audios7` | baseline list; **auto-discovery adds any other `audios<N>/` with a GT csv** |
| `TEXT_LABEL_MAP` | read/cheating→1, spontaneous/genuine→0 | label string → int |
| `DEFAULT_REGION_BY_BATCH` | audios2–5 = IND | region fallback when GT has no region column |
| `TARGET_SR` | 16000 | resample rate for the npy |

**Writes:** `data/neural_prep_out/upload/gt.csv` + `upload/audio_npy/<group_id>_<qid>.npy`
(→ EC2), and `data/neural_prep_out/local/cid_mapping.json` (**KEEP LOCAL** — never upload).
Prefers `<batch>_features.csv` for the `feat_*` columns. **Keep `cid_mapping.json`** so a
candidate's encoded id stays stable across re-runs.

## Stage 3 — `ec2/extract_embeddings.py`
WavLM + Whisper mean-pool embeddings for every npy × every aug → one stamped `.npz` cache.
**Incremental** — only missing `(file, aug, layer)` combos are computed.

| Flag | Default | Meaning |
|---|---|---|
| `--data_dir` | *(required)* | folder with `gt.csv` + `audio_npy/` (prep's `upload/`) |
| `--out_path` | *(required)* | the cache `.npz` to build/extend |
| `--wavlm_id` | `microsoft/wavlm-base-plus` | WavLM checkpoint (base-plus=768d; large=1024d) |
| `--whisper_id` | `openai/whisper-medium` | Whisper checkpoint (medium=1024d encoder) |
| `--augs` | `orig,noise,pitch,speed,gain,air,vtlp,combo` | augs to cache (`orig` always included) |
| `--wavlm_layers` | `last,9` | WavLM layer tags to extract (`last` always forced in) |
| `--rir_dir` | `""` | optional impulse-response dir → enables `rir` aug |
| `--noise_dir` | `""` | optional background-noise dir → enables `bgnoise` aug |
| `--seed` | 42 | seeds the (seeded) augmentations |
| `--force` | off | ignore existing cache, re-extract from scratch |

**Model-ID safety:** the cache is stamped; it refuses to append if the passed model IDs differ
from what's stored (no mixing 768d base with 1024d large). Keep base/large caches at separate
paths. See `COMMANDS_CACHE_FIX.txt`.

## Stage 4a — `ec2/neural_baseline_multiseed.py` (choose the architecture)
Runs every variant × N seeds → mean ± std ranking (fluke-proof). Use it to pick the arch.

| Flag | Default | Meaning |
|---|---|---|
| `--data_dir` | *(required)* | gt.csv + audio_npy dir |
| `--out_dir` | *(required)* | run output dir |
| `--cache` | `<data_dir>/embeddings_cache.npz` | embedding cache |
| `--train_batches` | `audios2,audios4,audios5,audios6` | batches in the train pool |
| `--test_batches` | `""` | empty → **20pct** (reshuffle all); e.g. `audios6` → **a6** (fixed test) |
| `--test_region_filter` | `""` | restrict the test set to a region (e.g. `PHP`) |
| `--train_only_batches` | `""` | batches forced into train only (e.g. `casual`, `2676,2677`) |
| `--min_duration` | 0.0 | drop rows shorter than this (finalised used **30**) |
| `--use_augs` | `""` | augs added to train (`all` / comma list / empty) |
| `--use_text_features` | `true` | include the 55 `feat_*` |
| `--per_client_standardize` | `false` | center features per client (unsupervised) |
| `--seeds` | `42,43,44,45,46` | seeds to average over |
| `--variants` | `""` (all 30) | restrict to named variants |
| `--export_artifacts` | `true` | save model.pt + scaler + pca + inference_meta per (seed, variant) |
| `--batch_size` / `--epochs` / `--lr` | 64 / 60 / 1e-3 | training knobs |
| `--class_balance` | `sampler` | `sampler` \| `pos_weight` \| `both` \| `none` |

**Key outputs:** `summary_mean_std.csv` (the ranking), `per_run.csv`, `splits/seed_*.json`,
`multiseed_summary.xlsx`, and per-(seed,variant) artifact folders.

## Stage 4b — `ec2/neural_baseline_train.py` (freeze the finalised models)
Trains the **full 30-variant sweep** with one global seed (no `--variants` flag exists) and,
with `--export_artifacts true`, exports **every** variant's bundle into `<out_dir>/<variant>/`.

| Flag | Default | Meaning |
|---|---|---|
| `--data_dir` | *(required)* | gt.csv + audio_npy dir |
| `--out_dir` | *(required)* | run output dir |
| `--cache` | `<data_dir>/embeddings_cache.npz` | embedding cache |
| `--train_batches` | `audios2,audios4,audios5,audios6,2676,2677` | train pool |
| `--test_batches` | `""` | empty → 20pct; `audios6` → a6 |
| `--test_region_filter` | `""` | region-restrict the test set |
| `--train_only_batches` | `2676,2677` | forced train-only (M1 used `casual`; M2 used `""`) |
| `--min_duration` | 0.0 | drop short rows (finalised = **30**) |
| `--use_augs` | `""` | augs added to train (finalised = `all`) |
| `--use_text_features` | `true` | include `feat_*` |
| `--per_client_standardize` | `false` | per-client centering |
| `--fewshot_frac` | 0.0 | (a6 only) carve a fraction of the test client into train |
| `--dump_full_predictions` | `false` | write per-file predictions incl. per-aug columns |
| `--export_artifacts` | `false` | **set `true`** to save model.pt/scaler/pca/inference_meta per variant |
| `--batch_size` / `--epochs` / `--lr` | 64 / 60 / 1e-3 | training knobs |
| `--seed` | 42 | global seed |
| `--class_balance` | `sampler` | imbalance handling |

**The two finalised runs** (from `COMMANDS_ANALYSIS.txt`):
```bash
# Model 1: default_last_pca98 + casual @ 20pct
python ec2/neural_baseline_train.py --data_dir <UP> --cache <CACHE> --out_dir <RUNS>/m1_casual_20pct \
    --train_batches audios2,audios4,audios5,audios6 --train_only_batches casual \
    --test_batches "" --min_duration 30 --use_augs all --use_text_features true \
    --per_client_standardize false --export_artifacts true --dump_full_predictions true
# Model 2: tiny_l9_pca95 @ a6
python ec2/neural_baseline_train.py --data_dir <UP> --cache <CACHE> --out_dir <RUNS>/m2_nocasual_a6 \
    --train_batches audios2,audios4,audios5,audios6 --train_only_batches "" \
    --test_batches audios6 --min_duration 30 --use_augs all --use_text_features true \
    --per_client_standardize false --export_artifacts true --dump_full_predictions true
```
`default_last_pca98` from the first run = Model 1; `tiny_l9_pca95` from the second = Model 2.
Every other variant is also frozen and pickable — see `RUNBOOK.md` "Picking a model".

## Notes & gotchas
- **`neural_baseline_train.py` has NO `--variants` flag** — it always trains the full
  30-variant sweep with one global seed (that's what reproduces the finalised numbers). With
  `--export_artifacts true` it exports **every** variant's bundle into `<out_dir>/<variant>/`,
  so you pick the model afterwards (any variant is usable — see `RUNBOOK.md` "Picking a model").
- **`--train_only_batches` rows are appended AFTER the split** (M1's `casual`, ALLSTAR
  `2676,2677`) so they can never leak into val/test.
- **`--min_duration 30`** and **`--use_augs all`** are part of the finalised recipe — keep them
  to reproduce M1/M2.
- **Cache model-ID stamp:** `extract_embeddings.py` refuses to append if the passed
  `--wavlm_id/--whisper_id` differ from what's stored (no mixing 768-d base with 1024-d large).
  Keep base/large caches at separate `--out_path`s. See `COMMANDS_CACHE_FIX.txt`.
- **`neural_baseline_prep.py` has no CLI** — it's driven by module constants + the env vars
  `KEEP_QUESTIONS` / `PREP_AUDIO_ROOT` / `PREP_UPLOAD_DIR` / `PREP_LOCAL_DIR` (above).
- **`--test_batches ""`** = 20pct; **`--test_batches audios6`** = a6. `--fewshot_frac` (a6 only)
  carves a slice of the test client into train.

*See also: `RUNBOOK.md` (chronological), `COMMANDS_ADD_BATCH.txt`, `COMMANDS_ANALYSIS.txt`.*
