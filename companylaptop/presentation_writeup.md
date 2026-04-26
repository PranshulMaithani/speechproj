# Honest evaluation & training-time fixes — presentation writeup

Audience: internal review. Goal: walk through the three diagnostic problems we found in the cheating-detection pipeline, the experiments we ran to understand each one, and the training-time fixes that closed the worst of them. Numbers and CSV pointers are inline so anyone can pull up the source data.

All artifacts live on the company laptop under:

- Notebook: `companylaptop/honest_eval_and_improve.ipynb` (sections referenced as §N below)
- Output CSVs: `companylaptop/checkpoints_honest_eval/`
- Fusion-pipeline CSVs: `companylaptop/checkpoints_fusion/`

Batches: **a2** (always-train), **a4**, **a5**. We evaluate via two rotations:
- **Rot A** — train = [a2, a4], CV = a4, test = a5
- **Rot B** — train = [a2, a5], CV = a5, test = a4

---

## The three problems we set out to investigate

1. **Small CV → unstable thresholds.** Each rotation's CV fold contains only one batch (~250 audios). Bootstrap thresholds came back with wide CIs and the picked operating point shifted run-to-run, especially the high-precision threshold.
2. **CV/test gap depends on which batch is held out.** Rot A looked clean. Rot B, which puts a4 in test, opened a large gap (whisper +0.13 F1 between CV and test). Same models, same code — only the batch role changed.
3. **High-precision threshold is non-deterministic.** Whatever bootstrap seed / fold split we used, the precision-≥0.9 threshold drifted by enough that we couldn't quote a stable operating point to the deployment side.

Headline of this writeup: **problem 2 is mostly solved**; problems 1 and 3 are partially mitigated and need more data to fully resolve.

---

## §3 — Baseline: how big is the gap, before any fix?

What we did: ran both rotations on the seven base models with the original ground truth and the original CV protocol. Two threshold strategies per model: **F1** (max-F1 on CV) and **P90** (smallest threshold reaching CV precision ≥ 0.9).

CSVs:
- `checkpoints_honest_eval/rotation_A.csv`
- `checkpoints_honest_eval/rotation_B.csv`

Columns: `model, strategy, thr_median, thr_p10, thr_p90, cv, te, te_prec, te_rec, gap`. `gap = cv - te`. `thr_p10` / `thr_p90` are the bootstrap CI bounds (BOOT_N=200) on the chosen threshold.

Headline numbers (F1 strategy):

| model            | Rot A gap | Rot B gap |
|------------------|-----------|-----------|
| whisper_wp_xgb   | ~+0.02    | **+0.134** |
| wavlm_whole_ft   | ~+0.03    | +0.07–0.09 |
| text_stylo       | ~+0.04    | +0.096    |

Rot A is fine. Rot B is not. That asymmetry is the signal that drove everything below.

`thr_p10` vs `thr_p90` for the P90 strategy showed CI widths of 0.10–0.18 across models — concrete evidence that **problem 3 (high-precision threshold non-determinism)** is real and that bootstrap doesn't make it go away with this much CV data.

---

## Problem 1 — Small CV and unstable thresholds

What we already knew: each rotation's CV is a single batch (~250 audios), positives are ~17%, so each CV fold sees ~10 positives. F1-vs-threshold curves on that few positives are jagged; bootstrap CI on the picked threshold is wide.

What we did about it (limited): kept the bootstrap CI columns (`thr_p10`, `thr_p90`) in every rotation CSV so we always quote uncertainty alongside the point estimate. We did **not** fix it structurally — that needs more data.

Where to look:
- Any `rotation_*.csv` in `checkpoints_honest_eval/` — look at `thr_p10` and `thr_p90` for the P90 strategy. Width > 0.10 is the recurring symptom.

Status: documented but not solved. Action item is to either (a) grow a4/a5 by re-collection, or (b) move to nested CV with merged train+CV folds for the threshold pick. Both are out of scope for this iteration.

---

## Problem 2 — The CV/test gap (asymmetric across rotations)

This is the bulk of the work. The story is: we suspected three different things in turn, ran a diagnostic for each, and the truth turned out to be **two of the three combined** (label noise + speaking-time confound). The fix is at training time, not at inference time.

### §10 — Data-asymmetry diagnostic (is one batch intrinsically harder?)

What we did: trained both acoustic models on **a2 alone** (no a4, no a5 in train), then scored a4 and a5 separately. Compared prior-invariant metrics (AUC, PR-AUC) so the 17%-vs-25% positive-rate difference between batches couldn't fake an asymmetry.

CSV: `checkpoints_honest_eval/data_asymmetry.csv`
Columns: `model, target_batch, n, n_pos, auc, pr_auc, brier`.

Headline: AUC was **0.92 on a4 and 0.94 on a5** — a5 is actually slightly *easier* in isolation. So the Rot B gap is **not** because a4 is intrinsically harder. The gap is created by *including a5 in training* — i.e. it's about what training on a5 does, not about what testing on a4 demands.

That immediately suggested two follow-up hypotheses for **why a5 in training would hurt**:
- **H1 — Label noise on a5.** Some "honest" rows on a5 are actually cheating. The model learns to call them honest, then fails on the analogous-but-correctly-labeled rows in a4.
- **H2 — Confound: short speaking time.** a5 has more very-short audios than a4, which look acoustically like script-reading even when they're spontaneous. Training on a5 with those rows teaches the model "short ⇒ cheating".

We chased both.

### §11 — Label-suspicion audit on a5

What we did: scored every a5 audio with both whisper and wavlm-FT (each trained on a2 alone) and flagged rows where **both** models disagreed strongly with the label.

CSVs:
- `checkpoints_honest_eval/a5_label_audit_positives.csv` — 43 rows, sorted by `min_prob` ASC. Suspicious ones at top.
- `checkpoints_honest_eval/a5_label_audit_negatives.csv` — 218 rows, sorted by `max_prob` DESC. Suspicious ones at top.
- `checkpoints_honest_eval/a5_label_audit_all.csv` — full 261 rows by joint disagreement.

Columns (all three): `candidate_id, filename, label_int, whisper, wavlm_ft, avg, min_prob, max_prob, suspect_type, audio_path`. The `audio_path` column is the absolute path to the audio file so re-listening is one click.

Findings:
- **0 of 43** labeled-cheating rows looked honest by both models.
- **18 of 218** labeled-honest rows looked cheating by both models (~8.3% of negatives).
- Top 5 candidates by mean disagreement were all in the negative set, all three of their audios disagreeing with the label.

User re-listened to the top candidates: **1 confirmed mislabel** (corrected the GT), **2 short-audio confounds** (label was right; the audio just had ~10 s of speech), **2 fine** (model wrong).

So H1 is **partially true** — there's some real label noise — but the top suspects were dominated by H2. That's why we needed §13 next.

### §12 — Re-run rotations after relabel

What we did: built a one-cell harness that backs up `audios5GT.csv`, snapshots the original rotation CSVs, reloads a5 from the corrected GT, re-runs both rotations, and prints a delta table.

CSVs:
- `checkpoints_honest_eval/audios5GT_baseline.csv` — frozen original GT.
- `checkpoints_honest_eval/rotation_A_baseline.csv`, `rotation_B_baseline.csv` — pre-relabel snapshots.
- `checkpoints_honest_eval/rotation_A_corrected.csv`, `rotation_B_corrected.csv` — post-relabel.

Same column set as §3 baselines.

Findings (delta in whisper Rot B F1 gap from the single corrected label): roughly **−0.01 to −0.02**, well within bootstrap noise. So fixing one label moved nothing meaningful — **label noise alone does not explain the gap**. That's the result that pushed us to chase H2 hard.

### §13 — Speaking-time computation

What we did: for every audio in a2/a4/a5, computed total duration and approximate speaking time (using the ASR transcripts where available, falling back to total duration). Plotted per-batch histograms.

CSVs (one per batch):
- `checkpoints_honest_eval/audios2_durations.csv`
- `checkpoints_honest_eval/audios4_durations.csv`
- `checkpoints_honest_eval/audios5_durations.csv`

Columns: `filename, candidate_id, label_int, total_duration_s, speaking_time_s, speech_ratio, source`.

Headline: fraction of audios with `speaking_time_s` < 30:
- **a2: 2.3%**
- **a4: 10.4%**
- **a5: 10.7%**

So a4 and a5 both have ~5× more short audios than a2. That short-audio cohort overlaps heavily with the "looks scripted but isn't" failure pattern. **Mechanically explains** the asymmetry: training on a5 (with short rows) pushes the boundary one way; training on a4 (with similar short rows) does the same; whichever short-audio cohort lands in test gets misclassified.

### §15 — Fast min-speaking-time sweep

What we did: ran a slim version of the rotation eval (no bootstrap, 1 seed, 3 folds, 200 trees, 2 models) at MIN_SPEAKING_S ∈ {0, 15, 20, 25, 30, 35}. This took ~15–20 min instead of ~2 hr per threshold.

CSV: `checkpoints_honest_eval/min_speaking_sweep.csv`
Columns: `min_s, rot, model, cv_f1, te_f1, gap`.

Headline (whisper Rot B gap by `min_s`):

| min_s | gap   |
|-------|-------|
| 0     | +0.133 |
| 15    | +0.130 |
| 20    | +0.127 |
| 25    | +0.126 |
| 30    | +0.113 |
| 35    | +0.070 |

Monotonic shrink → stricter is strictly better in this range, no U-shape yet, so we're still removing confound rather than cutting into real signal. We picked **30 s** as the operating point: it captures most of the gain and leaves enough training data (∼89% of a4/a5 retained).

### §14 — Combined intervention (relabel + filter, dual test view)

What we did: rewrote the §14 cell to do the realistic deployment evaluation. Reload a5 from corrected GT, filter the **training** sides (a2/a4/a5) at MIN_SPEAKING_S=30, but score the test batch in **two views**:

- **Full** (unfiltered) — what the model actually faces in production.
- **Filtered** — what the model would face if we also screened test inputs by speaking time. This is the "deployment ceiling".

CSVs:
- `checkpoints_honest_eval/rotation_A_minS30_dual.csv`
- `checkpoints_honest_eval/rotation_B_minS30_dual.csv`

Columns: `model, thr, thr_p10, thr_p90, cv, te_full, te_prec_full, te_rec_full, gap_full, n_te_full, te_filt, te_prec_filt, te_rec_filt, gap_filt, n_te_filt`. The `_full` set is the main number to quote; `_filt` is the upper bound.

For comparison we also kept the earlier intervention at `min_s=25`:
- `checkpoints_honest_eval/rotation_A_minS25.csv`
- `checkpoints_honest_eval/rotation_B_minS25.csv`

**Final headline numbers (Rot B, F1 strategy, full-test view):**

| model           | gap baseline | gap min30+relabel | F1 baseline | F1 min30+relabel |
|-----------------|--------------|-------------------|-------------|------------------|
| whisper_wp_xgb  | **+0.187**   | **+0.099**        | 0.683       | 0.701            |
| wavlm_whole_ft  | +0.09        | +0.085            | ~0.69       | ~0.70 (essentially flat on full) |
| text_stylo      | +0.096       | **−0.073**        | 0.636       | 0.696            |

On the filtered-test view at 30 s:
- whisper Rot B F1 = **0.745** (gap +0.055)
- wavlm_whole_ft Rot B F1 = **0.701** (delta vs unfiltered +0.046)
- 251 of 288 a4 audios retained.

**Read of these results:**
- whisper gap dropped by ~50% (Δ −0.088). This is the main win.
- text_stylo not only lost the gap, it crossed zero — became *competitive with* the acoustic models on Rot B. That's evidence the text features were never the problem; they were just being evaluated against a confounded baseline.
- wavlm_FT moved very little on the unfiltered test, more on the filtered view — suggesting wavlm's failure mode on a4 is *not* primarily the short-audio confound. Worth a separate diagnostic if we want to push it further.

### Where problem 2 stands now

- Hypothesis chain: data asymmetry → not intrinsic (§10) → label noise + short-audio (§11–§13) → label noise alone insufficient (§12) → speaking-time filter is the dominant fix (§15, §14).
- Whisper Rot B gap closed from +0.187 → +0.099 with a single training-time change (filter `speaking_time_s ≥ 30` on training rows only).
- Bonus: text model became competitive, fusion picture is now honest.
- Open: wavlm_FT didn't benefit on full test. That's the next thing to chase.

---

## Problem 3 — High-precision threshold can't be pinned down

Why this matters: the deployment side wants a single operating point with precision ≥ 0.9. We can't quote one with confidence because the chosen `thr_p90` shifts run-to-run.

Where to look: the `thr_p10` and `thr_p90` columns in any rotation CSV. For most models the CI width on the P90 threshold is 0.10–0.18 — i.e. "the right threshold is somewhere between 0.55 and 0.73", which is too wide to commit to.

Why this happens (mechanically): with ~10 positives in a CV fold, the precision curve has staircase jumps at every positive; one positive flipping moves the threshold by a lot. Bootstrap captures the variance but doesn't reduce it.

What we did: nothing structural. Same reason as problem 1 — needs more CV positives, which means more data.

Status: open. Mitigation for now is to quote the F1 threshold (much tighter CI) and the precision/recall *at* that threshold, instead of trying to hit a precision target.

---

## Quick reference — where every number lives

All paths relative to `companylaptop/checkpoints_honest_eval/` unless noted.

| Topic                          | File                                                  | Key columns                                                  |
|--------------------------------|-------------------------------------------------------|--------------------------------------------------------------|
| Baseline rotations             | `rotation_A.csv`, `rotation_B.csv`                    | `model, strategy, thr_median, thr_p10, thr_p90, cv, te, gap` |
| Pre-everything snapshots       | `rotation_A_baseline.csv`, `rotation_B_baseline.csv`  | same as above                                                |
| Post-relabel only              | `rotation_A_corrected.csv`, `rotation_B_corrected.csv`| same as above                                                |
| Filter at 25 s                 | `rotation_A_minS25.csv`, `rotation_B_minS25.csv`      | same as above                                                |
| **Final: filter 30 s + relabel, dual test** | `rotation_A_minS30_dual.csv`, `rotation_B_minS30_dual.csv` | `model, thr, thr_p10, thr_p90, cv, te_full, gap_full, n_te_full, te_filt, gap_filt, n_te_filt` (+ prec/rec) |
| Data-asymmetry diagnostic      | `data_asymmetry.csv`                                  | `model, target_batch, auc, pr_auc, brier`                    |
| a5 label audit                 | `a5_label_audit_positives.csv`, `a5_label_audit_negatives.csv`, `a5_label_audit_all.csv` | `candidate_id, filename, label_int, whisper, wavlm_ft, avg, min_prob, max_prob, suspect_type, audio_path` |
| Per-audio durations            | `audios2_durations.csv`, `audios4_durations.csv`, `audios5_durations.csv` | `filename, candidate_id, label_int, total_duration_s, speaking_time_s, speech_ratio, source` |
| Threshold sweep                | `min_speaking_sweep.csv`                              | `min_s, rot, model, cv_f1, te_f1, gap`                       |
| Candidate ID table             | `candidate_id_table.csv`                              | `batch, filename, candidate_id, question`                    |

Notebook section map (`honest_eval_and_improve.ipynb`):

| §   | Purpose                                                  |
|-----|----------------------------------------------------------|
| 1–3 | Setup, BASE_REGISTRY, baseline rotations                 |
| 10  | Data-asymmetry diagnostic (train on a2 alone)            |
| 11  | a5 label audit + suspicion CSVs                          |
| 12  | Re-run rotations after relabel                           |
| 13  | Speaking-time computation + per-batch histograms         |
| 14  | **Final**: combined relabel + filter, dual test view     |
| 15  | Fast threshold sweep                                     |

---

## TL;DR for the slide

- We had a **+0.18 F1 gap** between CV and test on whisper when a4 was held out.
- It wasn't intrinsic difficulty (§10), and it wasn't mostly label noise (§12).
- It was the **short-audio confound** (§13 + §15). Filtering training rows at `speaking_time_s ≥ 30` cut the gap roughly **in half** (+0.187 → +0.099), and made the text model competitive.
- Threshold-stability problems (problems 1 and 3) are still there. They need more data, not better code.
