# Slide — Best Model 1 (finalised): configuration & results

*The full-capacity, in-domain model. This slide states exactly what Model 1 is, how it was
trained, and how it scored.*

**One-liner (slide subtitle):** *`default_last_pca98` + casual, on the 20pct split — our
best-case, in-domain ceiling.*

---

## Configuration (put this as a table on the slide)

| Setting | Value |
|---|---|
| **Variant** | `default_last_pca98` |
| **Architecture** | `default` MLP — **512 → 256 → 128 → sigmoid** |
| **Regularization** | dropout **0.40**, weight decay **5e-4**, label smoothing 0.05, grad-clip 1.0 |
| **WavLM layer** | `last` |
| **Whisper** | medium encoder (1024-d) |
| **Handcrafted features** | all 55 (`use_text_features = true`) |
| **Feature vector** | `768 + 1024 + 55 = 1847-d` → StandardScaler → **PCA (98% variance)** |
| **Augmentations** | `all` (train-only; val/test use `orig`) |
| **Extra train data** | **+ casual** batch (train-only; never in val/test) |
| **Protocol** | **20pct** — Mode A, 60/20/20 candidate-disjoint `StratifiedGroupKFold` on audios2/4/5/6 |
| **Class imbalance** | `WeightedRandomSampler` |
| **Optimiser / schedule** | AdamW, cosine LR, lr 1e-3, batch 64, up to 60 epochs, early-stop on **val F1** (patience 10) |
| **min_duration** | 30 s (shorter clips dropped) |
| **Seed** | 42 · `per_client_standardize = false` |
| **Run dir** | `runs/m1_casual_20pct/default_last_pca98/` |
| **Parameters** | ~0.4–1.1M (scales with post-PCA input dim) |

**What it represents:** *"Given labelled data from this client, how well can we separate
cheating from genuine?"* — the **in-domain ceiling**, full modelling capacity.

---

## Feature → model pipeline (Mermaid diagram for the slide)

```
flowchart LR
  W["WavLM last<br>768-d"] --> C["Concatenate<br>1847-d"]
  H["Whisper encoder<br>1024-d"] --> C
  F["55 handcrafted<br>55-d"] --> C
  C --> S["StandardScaler<br>fit on train only"]
  S --> P["PCA · keep 98 percent"]
  P --> M["MLP 512-256-128<br>dropout 0.40"]
  M --> O["sigmoid to P(cheat)"]
  C:::cat
  S:::proc
  P:::proc
  M:::proc
  O:::out
  classDef cat stroke:#22d3ee,fill:#ecfeff
  classDef proc stroke:#818cf8,fill:#eef2ff
  classDef out stroke:#f87171,fill:#fef2f2
```

---

## Results (fill from your run sheet)

Pull real numbers from `runs/final_eval/evaluate_final_models.xlsx` (`summary` sheet) or
`summary_mean_std.csv`. Table template:

| Metric | Value |
|---|---|
| Test candidates / audios | ‹fill› |
| Prevalence (cheat %) | ‹fill› |
| **AUC** | ‹fill› |
| **Average Precision (AP)** | ‹fill› |
| **Best-F1** (and its threshold) | ‹fill› @ ‹thr› |
| F1 @ 0.5 | ‹fill› |
| **Recall @ precision 80%** | ‹fill› |
| **Recall @ precision 90%** | ‹fill› |
| **Recall @ precision 95%** | ‹fill› |
| Per-region — IND (best-F1 / R@p90) | ‹fill› |
| Per-region — PHP (best-F1 / R@p90) | ‹fill› |

**Headline number to lead with:** *"At precision 90% (≤10% false alarms), Model 1 catches
‹X›% of cheating answers"* — the recall@precision line is the one an operations team cares
about most.

> Reproducibility note you can say aloud: these numbers are reproduced by reloading the
> stored `model.pt` + scaler + PCA (`evaluate_final_models.py`), and the reproduced
> probabilities match the saved predictions to < 1e-3 — i.e. the exact weights, not a re-run.

---

## Speaker notes (~30s)
"Model 1 is our full-capacity, in-domain model. It fuses WavLM's last layer, Whisper's
encoder, and the 55 hand features into an 1847-dimensional vector, compresses it with PCA to
98% variance, and runs a 512-256-128 MLP. It's trained on all four Mettl batches plus the
casual auxiliary set, on the in-mix 20pct split — so it answers 'what's the best we can do
when we have labelled data from this client.' On that setting it reaches ‹AUC›, and at 90%
precision it still catches ‹X›% of cheaters."

---

*Config source of truth: `COMMANDS_ANALYSIS.txt` (Model 1 command) + `METHODOLOGY.md` §6.
Live metrics: `evaluate_final_models.xlsx`. Companion: `SLIDE_MODEL2.md` (if you make one for
the held-out-client model).*
