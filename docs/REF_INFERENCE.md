# Reference — INFERENCE (`inference/run_inference.py`)

Self-contained scoring of raw audio with already-trained model bundles: from wavs all the
way to a per-model cheat probability + decision, in one command. No labels, no training,
nothing uploaded — pure local inference.

```
wav -> npy (16 kHz) -> transcript -> 55 features -> WavLM+Whisper embeddings
    -> per model: concat 1847-d -> scaler -> pca -> MLP -> sigmoid
    -> apply that model's threshold.txt -> CHEAT (1) / GENUINE (0)
```

## Folder layout (create under `inference/`)
```
inference/
  audios/    your audio, ANY names. Each immediate child dir = one "audio group"
             (audios1/, teamA/, ...). Inside a group there may be candidate subfolders
             OR direct wavs — it RECURSES and scores only *_25 / *_26 / *_27 wavs.
             Loose wavs directly under audios/ form the group "(root)".
  models/    one subfolder per model, ANY name. Each MUST contain:
                 model.pt   scaler.joblib   inference_meta.json   threshold.txt
                 pca.joblib   (only if the model used PCA)
  data/      cache (npy / transcript / features / embeddings) — auto-created, reused
             across models and re-runs.
  results/   inference_results.xlsx + per-model CSVs — auto-created.
  run_inference.py
```

## Model bundle (what each `models/<name>/` needs)
These are exactly the artifacts `train_pipeline.py` / the training runs export, **plus a
`threshold.txt`** you add:
| File | Required | Purpose |
|---|---|---|
| `model.pt` | yes | MLP weights |
| `scaler.joblib` | yes | StandardScaler fit at train time |
| `inference_meta.json` | yes | arch (`hidden`,`dropout`), `wavlm_layer` (`last`/`9`), `feat_cols`, `in_dim` — drives feature order + MLP shape |
| `threshold.txt` | yes | a single float, e.g. `0.74` — the decision threshold |
| `pca.joblib` | if PCA used | PCA fit at train time (present iff the model used PCA) |

A folder missing any required file is **skipped with a warning** (not fatal), so a bad model
never stops the others.

## Audio conventions
- **Group** = the first path component under `audios/` (e.g. `audios1`); loose wavs → `(root)`.
- **Recursion:** all wavs under a group are found (candidate subfolders included).
- **Question filter:** only files whose name ends `_25` / `_26` / `_27` are scored
  (`--keep_questions all` scores every question).
- Names are used **as-is** (no anonymisation — this is local inference, nothing is uploaded).

## Parameters
| Flag | Default | Meaning |
|---|---|---|
| `--inference_dir` | this script's dir | folder holding `audios/ data/ models/ results/` |
| `--keep_questions` | `25,26,27` | only these qids are scored (`all` = every question) |
| `--transcribe_device` | `cpu` | `cpu` \| `cuda` — faster-whisper device for the transcript step |
| `--transcribe_compute_type` | `int8` | faster-whisper compute type (cuda: `float16`) |
| `--transcribe_model` | `""` | override transcription model (`""` keeps the training default → feature parity) |
| `--force` | off | recompute features + embeddings even if cached in `data/` |

## Pipeline stages
1. **Discover** audio (groups + `_25/26/27` filter) and models (folders with the required files).
2. **Features** — transcribe (faster-whisper) + compute the 55 `feat_*` via the training
   scripts' own functions → `data/features.csv` (shared by all models).
3. **Embeddings** — WavLM (`last` + `9`) + Whisper mean-pools per wav → cached in `data/emb/`
   and the npy in `data/npy/` (computed once, reused by every model).
4. **Predict** — per model: build `concat[ wavlm[meta.layer] | whisper | feat_* ]` in the
   model's `feat_cols` order → `scaler` → `pca` (if any) → MLP → sigmoid → probability.
   `in_dim` is checked against `inference_meta.json` (a mismatch errors, not silent).
5. **Decision + Excel** — `result = 1 (CHEAT)` iff `probability >= threshold`, else `0 (GENUINE)`.

## Outputs (`results/`)
- **`inference_results.xlsx`** — a `summary` sheet (per model: threshold, n_audios, n_cheat,
  n_genuine, mean_probability) **+ one sheet per model**, columns:
  `audio_group · audio_name · question · path · probability · threshold · result · decision`.
- **`inference_<model>.csv`** — the same per-model table as CSV.
- **`inference_summary.csv`** — the summary as CSV.

## Run
```bash
# CPU transcription:
python inference/run_inference.py
# fast GPU transcription:
python inference/run_inference.py --transcribe_device cuda --transcribe_compute_type float16
# recompute the cache (after changing audio or forcing fresh features/embeddings):
python inference/run_inference.py --force
```

## Add a model / add audio (no code changes)
- **New model:** copy its exported bundle into `inference/models/<any_name>/` and drop a
  `threshold.txt` (one float) beside it. Re-run — it gets its own sheet.
- **New audio:** drop a folder (or loose wavs) under `inference/audios/`. Re-run — a new
  `audio_group` appears in every model's sheet.

## Notes & gotchas
- **Feature-name prefix:** `inference_meta.feat_cols` are `feat_<name>`; the features csv has
  bare `<name>`. Lookup is prefix-robust and missing features **zero-fill** (matches training).
- **Layer per model:** Model 1 uses WavLM `last`, Model 2 uses `9` — both are always extracted,
  and each model picks the one in its meta, so mixed models in one run are fine.
- **Caching:** features + embeddings are computed once and reused across all models; use
  `--force` to recompute (e.g. after replacing audio with the same names).
- **Heavy deps** (torch, transformers, faster-whisper, spaCy, parselmouth) are imported lazily
  inside the compute stages — folder/model discovery works without them, but a real run needs
  them installed (the jump-server env).
- **Threshold transfer:** the number in `threshold.txt` is only as good as the data it was
  tuned on; on a very different client, re-tune it (see the threshold sweeps in the training
  workbooks / `evaluate_models.py`).

*See also: `REF_EVAL_uniscript.md` (metrics-oriented evaluation with labels), `RUNBOOK.md`.*
