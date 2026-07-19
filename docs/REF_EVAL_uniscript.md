# Reference — Unified EVALUATION (`evaluate_models.py`)

Every parameter of the generic model tester. Unlike `evaluate_final_models.py` (hardwired to
the two finalised models — see `REF_EVAL_selected_models.md`), this scores **any** saved model
folder, so you can evaluate a whole training run or a single hand-picked variant.

## What it does
Discovers every folder under `--models_root` that contains **`model.pt` + `inference_meta.json`**,
reloads each (`scaler` + optional `pca` + weights), rebuilds ORIG features in that model's own
stored feature order, predicts, and writes metrics + a full threshold sweep + per-region
breakdown. No model-specific code — the bundle is self-describing.

Two modes:
- **default — own test set:** scores each model on its own `predictions.csv` rows and verifies
  the reload reproduces the saved `pred_score`.
- **`--test_batch X`:** scores every discovered model on a fresh held-out batch neither trained on.

## Parameters
| Flag | Default | Meaning |
|---|---|---|
| `--models_root` | *(required)* | folder searched **recursively** for `model.pt` + `inference_meta.json` |
| `--data_dir` | *(required)* | folder with `gt.csv` |
| `--out_dir` | *(required)* | where results are written |
| `--cache` | `<data_dir>/embeddings_cache.npz` | embedding cache used to rebuild features |
| `--models` | `""` | comma **substring** filter on model id (e.g. `default_last_pca98,seed_42`); empty = all discovered |
| `--test_batch` | `""` | comma batch(es) to score every model on as fresh held-out data; empty = each model's own test set |
| `--min_duration` | 30.0 | (test_batch mode) drop eval rows shorter than this — match training |
| `--threshold` | *(none)* | optional decision threshold to ALSO record per row (adds a `pred@override` column) |
| `--thr_step` | 0.01 | threshold-sweep granularity (0.01 → 101 rows) |

## Model discovery — what counts as a model
A folder is scored if it has **both** `model.pt` and `inference_meta.json`. Its **id** is the
folder path relative to `--models_root` (e.g. `default_last_pca98/seed_42`). To score **one**
model, point `--models_root` straight at a folder holding one bundle; to score a **subset**,
keep a shared root and use `--models`. The bundle must be complete: `model.pt`,
`scaler.joblib`, `inference_meta.json`, **and `pca.joblib` if the model used PCA**.
`evaluate_models.py` ignores the training `runs_root`/`run_dir` layout — only the bundle matters.

## Outputs (`--out_dir`)
| File | Contents |
|---|---|
| `per_model/<id>/predictions.csv` | per-row scores + decisions at stored-thr / best-F1-thr / 0.5 (+ `pred@override`, + saved score in own-test mode) |
| `per_model/<id>/threshold_sweep.csv` | precision/recall/F1/confusion at every threshold |
| `per_model/<id>/metrics.json` | full `compute_metrics` dump |
| `eval_summary.csv` | one row per model — AUC, AP, best-F1+thr, stored-thr, F1@stored/0.5, recall@p80/90/95, repro diff |
| `eval_summary.xlsx` | `summary` + per-model `_sweep` / `_region` sheets |

## Example commands
```bash
# score every model of a training run on their own test sets (repro check)
python ec2/evaluate_models.py \
    --models_root data/runs/retrain_20pct/models \
    --data_dir <UP> --cache <CACHE> --out_dir data/runs/retrain_20pct/eval_own

# score selected models on a fresh held-out batch
python ec2/evaluate_models.py \
    --models_root data/runs/retrain_20pct/models \
    --models "default_last_pca98/seed_42,tiny_l9_pca95/seed_44" \
    --data_dir <UP> --cache <CACHE> --test_batch audios7 \
    --out_dir data/runs/eval_audios7

# score ONE hand-picked model (point the root at its folder or parent)
python ec2/evaluate_models.py \
    --models_root data/runs/mypick --data_dir <UP> --cache <CACHE> \
    --out_dir data/runs/eval_mypick --test_batch audios7
```

## How to read it
- **repro diff ≈ 0** (own-test) ⇒ the reloaded weights reproduce the saved predictions.
- Rank models in `eval_summary.csv` by best-F1 / recall@p90 for your protocol; confirm the
  winner on held-out data before adopting it.
- Weights transfer cleanly across data, but the **decision threshold may not** (cross-client
  caveat) — use `--threshold` / the sweep to set the operating point on the target data.

## Picking a model to deploy
Any variant a training run produced is a drop-in — copy its whole bundle to a stable path
(e.g. `data/runs/finalised/<variant>/`) and point `--models_root` at it. See `RUNBOOK.md`
"Picking a model" for the selection guidance (match capacity to in-domain vs cross-client).

*See also: `REF_TRAIN_uniscript.md`, `RUNBOOK.md` Part B.*
