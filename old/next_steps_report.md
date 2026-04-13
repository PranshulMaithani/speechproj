# Read vs Spontaneous Speech Classifier — Next Steps Report

**Date:** March 2026  
**Current state:** Wav2Vec2 5sec ONNX model, 94% precision / 76% recall on company data (219 files, 1 min limit), new inference pipeline (`predict_cpu.py`) running with Silero VAD

---

## Where You Are Right Now

You have a working end-to-end system. The inference pipeline has been fully rebuilt with:

- Silero VAD replacing the broken per-window RMS VAD
- VAD-gated windowing (model never sees silence)
- Confidence gating (uncertain windows excluded from voting)
- Adaptive hop size scaling with window length
- Minimum segment length enforcement
- Temporal median smoothing

The 5sec Wav2Vec2 ONNX model is running on the company laptop. The GT labels are partially cleaned (you reviewed ~60/219 files), seniors are finishing the relabelling. That relabelling is the critical dependency for everything that follows.

The core remaining gap is **76% recall** — 1 in 4 read-speech cases is being missed. Everything below is aimed at closing that gap systematically.

---

## Phase 1 — Immediate (This Week, No Retraining)

### 1.1 — Threshold Calibration (Highest Priority)

**Why:** Your current `read_threshold = 0.50` was never tuned to your data. It is an arbitrary default. Moving it to even 0.40 or 0.35 could recover several recall points at acceptable precision cost. This is the single cheapest improvement available.

**When to do it:** Once the seniors finish relabelling. Do not calibrate against partially-cleaned labels — wait for the full clean set. The current run you have going right now will produce the JSON you need.

**How to do it:**

```python
import json
import numpy as np
from sklearn.metrics import precision_recall_curve

# Load the predictions JSON from the current run
with open("outputs/results_audios2.json") as f:
    predictions = json.load(f)

# Your ground truth — filename -> 1 (read) or 0 (spontaneous)
ground_truth = {
    "file001.wav": 1,
    "file002.wav": 0,
    # ... all 219 files after senior relabelling
}

read_ratios, y_true = [], []
for r in predictions:
    if r["filename"] in ground_truth:
        read_ratios.append(r["read_ratio"])
        y_true.append(ground_truth[r["filename"]])

read_ratios = np.array(read_ratios)
y_true = np.array(y_true)

# Print full precision/recall table
precision, recall, thresholds = precision_recall_curve(y_true, read_ratios)
print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 44)
for p, r, t in zip(precision, recall, thresholds):
    f1 = 2 * p * r / (p + r + 1e-8)
    print(f"{t:10.3f} {p:10.1%} {r:10.1%} {f1:10.1%}")

# Find threshold maximising recall while keeping precision >= 90%
best_thresh, best_recall, best_prec = 0.50, 0.0, 0.0
for p, r, t in zip(precision, recall, thresholds):
    if p >= 0.90 and r > best_recall:
        best_recall = r
        best_prec = p
        best_thresh = t

print(f"\nOptimal threshold: {best_thresh:.3f}")
print(f"At this threshold — Precision: {best_prec:.1%}, Recall: {best_recall:.1%}")
```

Then update `DEFAULT_CONFIG` in `predict_cpu.py`:

```python
"read_threshold": 0.38,   # whatever the script outputs
```

**Expected gain:** +4 to +8 recall points. Zero cost.

---

### 1.2 — Test-Time Augmentation (TTA)

**Why:** Running the same audio through the model with slightly different window offsets and averaging the probabilities smooths out cases where a read segment fell awkwardly between two window boundaries. Costs 3x inference time but zero retraining.

**How to add it:** In `predict_cpu.py`, replace the single windowing + inference block in `predict_file` with this:

```python
def predict_with_tta(
    audio, sr, speech_segments, classifier, cfg,
    offsets_sec=(0.0, 0.5, 1.25)
):
    """Run inference with multiple window offsets, average probabilities."""
    window_sec = cfg["window_sec"]
    hop_sec = adaptive_hop(window_sec)
    all_probs_per_offset = []

    for offset in offsets_sec:
        shifted = [
            {"start": max(0.0, s["start"] + offset), "end": s["end"]}
            for s in speech_segments
        ]
        windows = make_vad_gated_windows(
            audio, sr, shifted, window_sec, hop_sec,
            merge_gap_sec=cfg["vad_merge_gap_sec"],
        )
        if not windows:
            continue
        chunks = np.stack([w[0] for w in windows])
        probs = []
        for i in range(0, len(chunks), cfg.get("batch_size", 4)):
            probs.append(classifier.predict_batch(chunks[i:i+4]))
        all_probs_per_offset.append((windows, np.concatenate(probs, axis=0)))

    if not all_probs_per_offset:
        return None, None

    # Use the 0-offset windows as the reference for timestamps
    ref_windows, _ = all_probs_per_offset[0]
    # Average probs across offsets (all have same number of windows at 0-offset)
    avg_probs = np.mean([p for _, p in all_probs_per_offset], axis=0)
    return ref_windows, avg_probs
```

Add a `--tta` flag to the CLI to make it optional since it triples inference time.

**Expected gain:** +1 to +3 recall points.

---

### 1.3 — Duration-Weighted Voting

**Why:** Currently every window gets one vote regardless of how long it is or how confident the model was. A 15-second highly-confident read window should outweigh a 2.5-second borderline one. This is a one-line change in `predict_file`.

**How:** Replace the `read_ratio` calculation block with:

```python
def compute_read_ratio_weighted(window_preds):
    speaking = [wp for wp in window_preds if wp["label"] in ("spontaneous", "read")]
    if not speaking:
        return 0.0
    total_w, read_w = 0.0, 0.0
    for wp in speaking:
        duration = wp["end_sec"] - wp["start_sec"]
        weight = duration * wp["confidence"]
        total_w += weight
        if wp["label"] == "read":
            read_w += weight
    return read_w / total_w if total_w > 0 else 0.0

# Then in predict_file, replace:
# read_ratio = read_count / len(speaking)
# with:
read_ratio = compute_read_ratio_weighted(window_preds)
```

**Expected gain:** +1 to +2 recall points, especially on files with a mix of short and long segments.

---

## Phase 2 — Model Upgrade (After Clean Labels Are Ready)

### 2.1 — WavLM-Base+ Retrain

**Why:** WavLM adds a masked speech denoising objective on top of wav2vec2's contrastive objective during pre-training. On every downstream speech task — speaker classification, emotion, style — WavLM-Base+ consistently outperforms Wav2Vec2-Base by 2–5 points. The architecture is identical so your entire training and export pipeline works unchanged.

**How:** In `configs/config.yaml`, change exactly one line:

```yaml
training:
  wav2vec2:
    model_name: "microsoft/wavlm-base-plus"   # was: "facebook/wav2vec2-base"
```

Everything else — batch size, learning rate, freeze layers, ONNX export, quantisation — stays the same. Run `train_wav2vec2_5sec.py` and then `export_onnx.py` as normal. The exported ONNX file drops into `predict_cpu.py` as a direct replacement.

**What to train:** Just the 5sec model first. Verify it beats the current Wav2Vec2 5sec on your clean GT labels before training other window sizes. Do not assume it will be better — measure it.

**Expected gain:** +2 to +5 recall points.

---

### 2.2 — Prosodic Feature Enrichment for XGBoost

**Why:** Read speech is betrayed not just by how it sounds acoustically but by how *regular* it is. Pitch variance, pause rhythm uniformity, and energy consistency are hard to fake naturally. These features can catch trained presenters or newsreaders whose raw acoustics are deceptively spontaneous-sounding.

**How:** Add these features to `src/features/extract_features.py` alongside your existing ones:

```python
def extract_prosodic_features(audio, sr, f0_min=75, f0_max=400):
    import librosa

    # F0 trajectory via probabilistic YIN
    f0, voiced_flag, _ = librosa.pyin(audio, fmin=f0_min, fmax=f0_max, sr=sr)
    f0_voiced = f0[voiced_flag & ~np.isnan(f0)]

    # Pitch variance — spontaneous speech is much more variable
    f0_std   = np.nanstd(f0_voiced) if len(f0_voiced) > 1 else 0.0
    f0_range = float(np.nanmax(f0_voiced) - np.nanmin(f0_voiced)) if len(f0_voiced) > 1 else 0.0

    # F0 linearity — reading has smoother, more linear pitch contour
    if len(f0_voiced) > 3:
        x = np.arange(len(f0_voiced))
        coeffs = np.polyfit(x, f0_voiced, 1)
        residuals = f0_voiced - np.polyval(coeffs, x)
        f0_linearity = 1.0 / (1.0 + np.std(residuals))
    else:
        f0_linearity = 0.0

    # Pause regularity — reading has more metronomic pauses
    rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]
    silence_mask = rms < np.percentile(rms, 30)
    pauses = []
    count = 0
    for s in silence_mask:
        if s:
            count += 1
        elif count > 0:
            pauses.append(count)
            count = 0
    pause_regularity = 1.0 / (1.0 + np.std(pauses)) if len(pauses) > 2 else 0.0

    # Energy coefficient of variation — monotone reading has low CV
    rms_mean = np.mean(rms)
    energy_cv = np.std(rms) / (rms_mean + 1e-8)

    # ZCR variance — proxy for articulatory variation
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=512, hop_length=256)[0]
    zcr_std = float(np.std(zcr))

    return {
        "prosodic_f0_std":          f0_std,
        "prosodic_f0_range":        f0_range,
        "prosodic_f0_linearity":    f0_linearity,
        "prosodic_pause_regularity": pause_regularity,
        "prosodic_energy_cv":       energy_cv,
        "prosodic_zcr_std":         zcr_std,
    }
```

Retrain XGBoost with these added to the feature vector. Check feature importances after training — if `f0_linearity` and `pause_regularity` are not in the top 10, something is wrong with the computation.

**Expected gain:** +1 to +3 recall points, specifically on trained readers whose acoustic features are controlled.

---

## Phase 3 — Architecture Overhaul (Long Term, High Effort)

This is the ideal end state described in our earlier discussion. Only pursue this once Phases 1 and 2 are complete and you have a stable, clean GT dataset with sufficient size.

### 3.1 — Multi-Scale Shared Encoder with Scale Attention Fusion

**The core idea:** One WavLM encoder, three window sizes (5s, 10s, 15s), three forward passes per training step, fused with a learned attention mechanism that learns which scale is most informative for each speaker/style. Prosodic features are concatenated at the fusion layer rather than voted on separately.

```python
class MultiScaleSpeechClassifier(nn.Module):
    """
    Single WavLM encoder, three temporal scales, learned scale attention.
    Prosodic features fused at classifier input, not at output.
    """
    def __init__(self, model_name="microsoft/wavlm-base-plus",
                 hidden_size=256, num_labels=2, dropout=0.3,
                 prosodic_dim=6):
        super().__init__()

        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        self.encoder.feature_extractor._freeze_parameters()
        enc_dim = self.encoder.config.hidden_size  # 768

        # Scale attention: score each of the 3 window scales
        self.scale_attn = nn.Linear(enc_dim, 1)

        # Prosodic projection: map raw features into encoder space
        self.prosodic_proj = nn.Sequential(
            nn.Linear(prosodic_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )

        # Classifier sees fused encoder output + projected prosodic features
        self.classifier = nn.Sequential(
            nn.Linear(enc_dim + 128, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def encode(self, x):
        """x: (B, T) → (B, 768) mean-pooled"""
        h = self.encoder(x).last_hidden_state
        return h.mean(dim=1)

    def forward(self, windows_5s, windows_10s, windows_15s, prosodic_feats):
        """
        windows_*: (B, T_i) raw waveform at 16kHz
        prosodic_feats: (B, prosodic_dim) hand-crafted features
        """
        e5  = self.encode(windows_5s)    # (B, 768)
        e10 = self.encode(windows_10s)   # (B, 768)
        e15 = self.encode(windows_15s)   # (B, 768)

        # Stack scales: (B, 3, 768)
        stacked = torch.stack([e5, e10, e15], dim=1)

        # Learned scale attention
        attn_scores = self.scale_attn(stacked)          # (B, 3, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        fused = (stacked * attn_weights).sum(dim=1)     # (B, 768)

        # Project and concatenate prosodic features
        prosodic_emb = self.prosodic_proj(prosodic_feats)  # (B, 128)
        combined = torch.cat([fused, prosodic_emb], dim=-1) # (B, 896)

        return self.classifier(combined)  # (B, 2)
```

**Training considerations:**
- Three encoder forward passes per step means ~3x GPU memory. On an 8GB card you will need to reduce batch size to 4 or 6.
- Use gradient checkpointing (`model.encoder.gradient_checkpointing_enable()`) to trade compute for memory.
- Freeze all encoder layers for the first 2 epochs, then unfreeze the top 4 layers. The scale attention and prosodic projection need to stabilise before the encoder starts moving.
- The three window sizes need to be prepared in the dataset: for each training example, yield three crops (5s, 10s, 15s) centred on the same position. Pad shorter crops.

**ONNX export:** This model cannot be exported to a single ONNX file trivially because it has three inputs of different sizes. The cleanest approach is to export the encoder once as a shared ONNX model and call it three times in Python at inference time, then run the attention + classifier as a second lightweight ONNX model. The second model is tiny and fast.

**Expected gain over current system:** +5 to +10 recall points combined with all Phase 1 and 2 improvements. Ceiling is approximately 90%+ recall at 90%+ precision with sufficient training data coverage.

---

### 3.2 — Data Augmentation During Training

**Why it matters:** Your training set likely underrepresents certain reader types — trained presenters, non-native English readers, people who read slowly with natural-sounding pauses. The model has never seen these and will miss them. Augmentation artificially creates harder examples during training.

**Augmentations to add to your dataset class:**

```python
import numpy as np
import librosa

def augment_audio(audio, sr, label):
    """
    Apply augmentations. For read speech, also add deceptive augmentations
    that make read speech harder to detect (forces model to learn deeper cues).
    """
    augmented = audio.copy()

    # Standard augmentations (both classes)
    # 1. Speed perturbation (preserves pitch — use for read speech at varied speeds)
    if np.random.random() < 0.3:
        rate = np.random.uniform(0.85, 1.15)
        augmented = librosa.effects.time_stretch(augmented, rate=rate)

    # 2. Pitch shift (makes read speech sound more natural in pitch)
    if np.random.random() < 0.2:
        steps = np.random.uniform(-2, 2)
        augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=steps)

    # 3. Additive noise (background conditions)
    if np.random.random() < 0.4:
        noise_level = np.random.uniform(0.002, 0.015)
        noise = np.random.randn(len(augmented)) * noise_level
        augmented = augmented + noise

    # 4. Room impulse response simulation (mic distance variation)
    # Simple approximation: slight reverb via echo
    if np.random.random() < 0.2:
        delay = int(sr * np.random.uniform(0.01, 0.05))
        decay = np.random.uniform(0.1, 0.3)
        echo = np.zeros_like(augmented)
        echo[delay:] = augmented[:-delay] * decay
        augmented = np.clip(augmented + echo, -1.0, 1.0)

    # 5. For READ speech specifically: simulate fluent-sounding reading
    #    by slightly regularising the pace (makes it harder to detect)
    if label == 1 and np.random.random() < 0.25:
        # Compress dynamic range slightly — trained readers control energy
        augmented = np.sign(augmented) * np.abs(augmented) ** 0.85

    return augmented
```

The most important augmentation for your use case is #3 (noise) and #4 (room acoustics) because company recordings will have variable acoustic conditions that your training data may not cover well.

---

## Phase 4 — Evaluation Infrastructure

You need this regardless of which model you use. Right now your evaluation is manual and error-prone.

### 4.1 — Automated Evaluation Script

```python
# eval.py — run this after every model change or threshold update
import json
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    precision_recall_curve
)
from pathlib import Path

def evaluate(predictions_json_path, ground_truth_dict, threshold=0.50):
    with open(predictions_json_path) as f:
        predictions = json.load(f)

    y_true, y_pred, read_ratios = [], [], []
    missed_reads = []   # files that were read but predicted spontaneous
    false_alarms = []   # files that were spontaneous but predicted read

    for r in predictions:
        fname = r["filename"]
        if fname not in ground_truth_dict:
            continue
        true_label = ground_truth_dict[fname]
        pred_label = 1 if r["read_ratio"] >= threshold else 0

        y_true.append(true_label)
        y_pred.append(pred_label)
        read_ratios.append(r["read_ratio"])

        if true_label == 1 and pred_label == 0:
            missed_reads.append({
                "filename": fname,
                "read_ratio": r["read_ratio"],
                "overall_confidence": r["overall_confidence"],
            })
        if true_label == 0 and pred_label == 1:
            false_alarms.append({
                "filename": fname,
                "read_ratio": r["read_ratio"],
                "overall_confidence": r["overall_confidence"],
            })

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("=" * 60)
    print(f"EVALUATION — threshold={threshold:.2f}, n={len(y_true)}")
    print("=" * 60)
    print(classification_report(y_true, y_pred,
                                target_names=["spontaneous", "read"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    print(f"\nMissed reads ({len(missed_reads)}):")
    for m in sorted(missed_reads, key=lambda x: x["read_ratio"]):
        print(f"  {m['filename']:<40s} read_ratio={m['read_ratio']:.3f}")

    print(f"\nFalse alarms ({len(false_alarms)}):")
    for m in sorted(false_alarms, key=lambda x: x["read_ratio"], reverse=True):
        print(f"  {m['filename']:<40s} read_ratio={m['read_ratio']:.3f}")

    # PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, read_ratios)
    print(f"\nPR curve (precision >= 90%):")
    for p, r, t in zip(precision, recall, thresholds):
        if p >= 0.90:
            f1 = 2*p*r/(p+r+1e-8)
            print(f"  t={t:.3f}  P={p:.1%}  R={r:.1%}  F1={f1:.1%}")

    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "missed_reads": missed_reads,
        "false_alarms": false_alarms,
    }
```

Run this every time you change the model, threshold, or inference pipeline. Save the output — you need a paper trail to show your seniors what each change achieved.

---

## Dependency Map

Everything below shows what blocks what. Do not skip steps.

```
Senior relabelling complete
        │
        ├─► Calibration script → optimal threshold → update predict_cpu.py
        │         │
        │         └─► Re-run inference on 219 files → new baseline numbers
        │
        ├─► Add TTA to predict_cpu.py → re-run → compare to baseline
        │
        ├─► Add weighted voting → re-run → compare
        │
        └─► WavLM retrain (can start in parallel while waiting for labels)
                  │
                  ├─► Export to ONNX → drop into predict_cpu.py → compare
                  │
                  └─► Add prosodic features → retrain XGBoost branch → compare
                            │
                            └─► Multi-scale architecture (only if above is stable)
```

---

## Realistic Target Numbers

Starting from 94% precision / 76% recall:

| Change | Estimated Recall Gain | Effort |
|---|---|---|
| Threshold calibration | +4 to +8% | 1 hour |
| TTA (3 offsets) | +1 to +3% | 2 hours |
| Duration-weighted voting | +1 to +2% | 30 min |
| WavLM-Base+ retrain | +2 to +5% | 1 training day |
| Prosodic features in XGBoost | +1 to +3% | half day |
| Multi-scale + fusion | +3 to +6% | 1 week |

Realistic ceiling with all changes applied and sufficient clean training data: **~90% recall at ~90%+ precision.** Whether you reach that depends on how well your training set covers the hard cases — trained newsreader-style speakers, non-native readers, and slow deliberate readers with natural-sounding pauses. If those cases are not in the training data, no architecture change will catch them.

---

## Notes on the GT Label Situation

The human error in your GT labels is a real problem that affects every number you produce. A few things to keep in mind:

- **Calibration numbers are only as good as your labels.** If 10% of your GT labels are wrong, your measured recall could be off by several points in either direction.
- **The missed reads list from the evaluation script is useful beyond metrics.** Listen to those files. If they sound genuinely spontaneous to you as well, that tells you the model is correctly uncertain, not wrong. If they are obviously read, that tells you the model has a gap.
- **Consider a three-way label:** read / spontaneous / ambiguous. Files that your seniors disagree on should be marked ambiguous and excluded from metric calculation. Forcing a binary label on genuinely ambiguous audio pollutes your evaluation.
- **The 219 files are a small evaluation set.** At this size, a single mislabelled file moves your recall by ~0.5%. Be cautious about over-interpreting small differences between model versions — use confidence intervals or bootstrap resampling if you want to be rigorous.

---

## Summary — What To Do Next In Order

1. **Wait for the current inference run to finish** — collect `results_audios2.json`
2. **Wait for senior relabelling** — do not calibrate against partially-clean labels
3. **Run calibration script** — get optimal threshold, update `predict_cpu.py`, re-run inference once with the new threshold
4. **Add weighted voting** — 30-minute change, re-run, compare numbers
5. **Add TTA** — if inference time is acceptable on the laptop, enable it
6. **Start WavLM retrain** on your training machine in parallel with steps 3–5
7. **Add prosodic features to XGBoost**, retrain that branch
8. **Evaluate everything against the clean GT** using the evaluation script — produce a table showing before/after for each change
9. **Present that table to seniors** — concrete numbers justify compute budget for Phase 3
10. **Begin multi-scale architecture** only after Phase 1 and 2 are validated and stable
