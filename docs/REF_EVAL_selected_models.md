# Reference — Selected-Model EVALUATION (`evaluate_final_models.py`)

Every parameter of the legacy evaluator that scores the **two finalised models** from their
stored weights. For scoring *any* model folder (the unified tester) see `REF_EVAL_uniscript.md`.

## What it does
Reloads each finalised model's `model.pt` + `scaler.joblib` + `pca.joblib`, rebuilds the ORIG
features in the model's exact training feature order, predicts, and writes detailed metrics +
a full threshold sweep + per-region breakdown into one workbook. Two modes:

- **(A) default — own test set:** predicts on each run's `predictions.csv` rows and **verifies**
  the reloaded weights reproduce the saved `pred_score` (`max|recon−saved| < 1e-3` ⇒ exact).
- **(B) `--test_batch X`:** scores the **same stored weights** on a brand-new batch (e.g.
  `audios7`) neither model trained on — the generalisation/validity check.

The two models are **hardcoded** (variant + arch + layer + run dir):
`model1_20pct` = `default_last_pca98` @ `m1_casual_20pct`; `model2_a6` = `tiny_l9_pca95` @
`m2_nocasual_a6`. (To score *other* variants use `evaluate_models.py`.)

## Parameters
| Flag | Default | Meaning |
|---|---|---|
| `--data_dir` | *(required)* | folder with `gt.csv` (and the cache unless `--cache` given) |
| `--out_dir` | *(required)* | where the workbook + per-row CSVs are written |
| `--cache` | `<data_dir>/embeddings_cache.npz` | embedding cache used to rebuild features |
| `--runs_root` | `/home/ubuntu/nn/runs` | folder holding the finalised run dirs |
| `--m1_dir` | `m1_casual_20pct` | run dir of finalised Model 1 (`runs_root/<m1_dir>/default_last_pca98/`) |
| `--m2_dir` | `m2_nocasual_a6` | run dir of finalised Model 2 (`runs_root/<m2_dir>/tiny_l9_pca95/`) |
| `--test_batch` | `""` | comma batch(es) to score BOTH models on as fresh held-out data; empty → own test sets |
| `--min_duration` | 30.0 | (test_batch mode) drop eval rows shorter than this — **match training (30)** |
| `--thr_step` | 0.01 | threshold-sweep granularity (0.01 → 101 rows) |

## Inputs it expects
Each model's run dir must contain the exported bundle (from a training run with
`--export_artifacts true`):
```
<runs_root>/<run_dir>/<variant>/{model.pt, scaler.joblib, pca.joblib, inference_meta.json, predictions.csv}
```
Plus `gt.csv` + the matching embedding cache in `--data_dir` / `--cache`. If a model used a
different WavLM/Whisper cache, point `--cache` at that one (dims are checked; a mismatch errors).

## Outputs (`--out_dir`)
`<tag>` = `model1_20pct` / `model2_a6` (mode A) or `<model>_on_<batch>` (mode B).
| File | Contents |
|---|---|
| `final_eval_summary.csv` | both models side by side — AUC, AP, best-F1+thr, F1@0.5, recall@p50/80/85/90/95, thresholds, repro diff |
| `predictions_<tag>.csv` | per-row scores + hard decisions at best-thr / 0.5 / p90-thr (+ saved score in mode A) |
| `evaluate_final_models.xlsx` | `summary` + per-model `_preds` / `_sweep` (P/R/F1/confusion at every threshold) / `_by_region` |

## Example commands
```bash
# own test sets (repro check)
python ec2/evaluate_final_models.py \
    --data_dir <UP> --cache <CACHE> --runs_root <RUNS> --out_dir <RUNS>/final_eval

# validity on a new batch (same weights, fresh data)
python ec2/evaluate_final_models.py \
    --data_dir <UP> --cache <CACHE> --runs_root <RUNS> \
    --out_dir <RUNS>/final_eval_audios7 --test_batch audios7

# your run dirs are named differently:
#   ... --m1_dir m1_casual_20pct --m2_dir m2_nocasual_a6
```

## How to read it
- **repro_max_abs_diff ≈ 0** (mode A) ⇒ the stored weights reproduce the saved predictions.
- **`--test_batch` (mode B):** compare `auc / ap / recall@p90` on the new batch vs each model's
  own-test row — similar ⇒ it generalises; a big drop ⇒ it doesn't hold up on that batch.
- The `_sweep` sheet gives precision/recall at every threshold — pick the row at your target
  precision (recall@precision is the headline operating point). Remember the cross-client
  **threshold-transfer caveat**: re-tune the threshold on the target data.

*See also: `RUNBOOK.md` Stage 4, `METHODOLOGY.md` §6.*
