# Read vs Spontaneous Speech Classifier — Full Project Summary
# For Claude Code Context

---

## Project Goal

Automatically detect whether a speaker is reading from a prepared script or speaking spontaneously during a company assessment/interview. Flagged as "cheating" if reading. Must run on a CPU-only company laptop.

---

## Repository Structure

```
Speech project/
├── src/
│   ├── data/
│   │   ├── dataset.py              # SpeechWindowDataset, build_accent_sampler
│   │   └── audio_utils.py          # load_audio, window_audio, compute_speech_ratio
│   └── models/
│       ├── train_wav2vec2_5sec.py  # Wav2Vec2 5sec training
│       ├── train_wav2vec2_7sec.py  # Wav2Vec2 7.5sec training
│       ├── train_wav2vec2_10sec.py # Wav2Vec2 10sec training
│       ├── train_wavLM_5sec.py     # WavLM-Base+ training (uses WavLMModel not Wav2Vec2Model)
│       └── train_whisper_cls.py    # Whisper Medium + MLP head training
├── configs/
│   └── config.yaml                 # Main config
├── checkpoints/                    # All saved models
├── outputs/                        # Manifests, predictions, results
├── data/
│   ├── read/                       # GigaSpeech audiobook clips (1-min each)
│   └── spontaneous/                # GigaSpeech podcast clips (1-min each)
├── predict_cpu.py                  # Main CPU inference script (self-contained)
├── export_onnx.py                  # Wav2Vec2/WavLM ONNX export
├── export_whisper_onnx.py          # Whisper ONNX export (separate script)
├── download_datasets.py            # Downloads GigaSpeech clips
├── check_and_balance_manifest.py   # Fixes AllStar durations, checks window balance
├── upload_hf.py                    # Uploads models to HuggingFace
└── check_onnx.py                   # Verifies ONNX model input shape
```

---

## Datasets

### Training Data

**AllStar DB (primary)**
- Paths: `2676/` (spontaneous), `2677/` (read)
- 699 train files, 135 val, 216 test
- Native and non-native English speakers
- Clean studio recordings
- Key limitation: hesitation only appears in spontaneous sections

**GigaSpeech (expanded)**
- Source: `speechcolab/gigaspeech` on HuggingFace (requires login + accept terms)
- `source=0` (audiobook) → read speech → `data/read/`
- `source=1` (podcast) → spontaneous → `data/spontaneous/`
- `source=2` (YouTube) → skipped (mixed labels)
- 500 x 1-minute files per class downloaded
- Each file = 23 windows at 5s/2.5s hop

**Combined manifest**: `outputs/manifest_expanded.csv`
- Total: 2050 rows, 77,513 windows
- Train: 30,497 read windows + 28,773 spontaneous windows (well balanced)

### Dataset class important note
`src/data/dataset.py` uses column `duration` NOT `duration_sec`. This was a bug that caused NaN errors — already fixed. Also `build_accent_sampler` uses `row.get("l1", "unknown")` for GigaSpeech rows that have no `l1` column.

---

## Models

### Current model inventory

| Model | Checkpoint | ONNX (quant) | Input | Status |
|---|---|---|---|---|
| Wav2Vec2 5sec (AllStar only) | `wav2vec2_best.pt` (old) | `speech_classifier_quant.onnx` | (B, 80000) | Production baseline |
| Wav2Vec2 5sec (expanded data) | `wav2vec2_5secExpanded.pt` | `speech_classifier_wav2vec2_5sec_quantexpanded_data.onnx` | (B, 80000) | Awaiting company eval |
| Wav2Vec2 7.5sec | `wav2vec2_7sec_best.pt` | `speech_classifier_wav2vec2_7_5sec_quant.onnx` | (B, 120000) | Ready |
| Wav2Vec2 10sec | `wav2vec2_10sec_best.pt` | `speech_classifier_wav2vec2_10sec_quant.onnx` | (B, 160000) | Ready |
| Wav2Vec2 12.5sec | `wav2vec2_12_5sec_best.pt` | `speech_classifier_wav2vec2_12_5sec_quant.onnx` | (B, 200000) | Ready |
| Wav2Vec2 15sec | `wav2vec2_15sec_best.pt` | `speech_classifier_wav2vec2_15sec_quant.onnx` | (B, 240000) | Ready |
| WavLM-Base+ 5sec | `wavlm_5sec_best.pt` | `speech_classifier_wavlm_5sec_quant.onnx` | (B, 80000) | Trained, not production |
| Whisper Medium | `whisper_medium_best.pt` | `speech_classifier_whisper_medium_quant.onnx` | (B, 80, 3000) | Training in progress |

**IMPORTANT**: All models stored at `HuggingFace: Pransfrance/speechproj-models` (private)

### Architecture details

**Wav2Vec2 / WavLM classifier:**
```python
# SpeechClassifier in train_wav2vec2_5sec.py / train_wavLM_5sec.py
encoder = WavLMModel.from_pretrained(model_name)   # WavLM uses WavLMModel not Wav2Vec2Model
classifier = Linear(768, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 2)
# CNN frozen always, first 6 transformer layers frozen
# Mean pooling over time dimension
```

**Whisper classifier:**
```python
# WhisperClassifier in train_whisper_cls.py
# Entire encoder frozen with torch.no_grad() in forward() — MLP head only trains
encoder = WhisperModel (openai/whisper-medium, enc_dim=1024)
# CNN, embed_positions, ALL encoder layers, decoder — all frozen
# torch.no_grad() around encoder forward pass (saves VRAM, enables large batch)
classifier = Linear(1024,512) -> GELU -> Dropout -> Linear(512,256) -> GELU -> Dropout -> Linear(256,2)
# Input: log-mel spectrogram (B, 80, 3000) — always 3000 frames regardless of audio length
```

---

## Config (configs/config.yaml)

```yaml
paths:
  data_root: "D:/GoodProjects/Speech project"
  spontaneous_dir: "2676"
  read_dir: "2677"
  manifest_csv: "outputs/manifest_expanded.csv"   # USE THIS — not manifest.csv
  checkpoints_dir: "checkpoints"

audio:
  sample_rate: 16000
  window_sec: 5.0
  hop_sec: 2.5
  min_speech_ratio: 0.20
  vad_energy_threshold: 0.01
  max_duration_sec: 120

training:
  wav2vec2:
    model_name: "facebook/wav2vec2-base"   # or "microsoft/wavlm-base-plus" for WavLM
    freeze_layers: 6
    hidden_size: 256
    dropout: 0.3
    batch_size: 16
    learning_rate: 0.00002
    warmup_ratio: 0.10
    num_epochs: 15
    patience: 3
    weight_decay: 0.01
    fp16: true

  whisper:
    model_name: "openai/whisper-medium"
    freeze_layers: 6       # NOTE: actually all layers frozen via torch.no_grad()
    hidden_size: 512
    dropout: 0.3
    batch_size: 32         # can be high because encoder frozen with no_grad
    learning_rate: 0.001   # higher than wav2vec2 since only MLP trains
    warmup_ratio: 0.10
    num_epochs: 15
    patience: 4
    weight_decay: 0.01
    fp16: true

accent_weights:
  HIN: 3.0
  GUJ: 3.0
  IND: 2.0
  # ... etc
```

---

## Training Commands

```bash
# Wav2Vec2 5sec (production model)
python -m src.models.train_wav2vec2_5sec --config configs/config.yaml

# WavLM 5sec
python -m src.models.train_wavLM_5sec --config configs/config.yaml

# Whisper Medium (MLP head only — encoder frozen)
python -m src.models.train_whisper_cls --config configs/config.yaml

# Wav2Vec2 7.5sec (config must have window_sec: 7.5, hop_sec: 3.0 before running)
python -m src.models.train_wav2vec2_7sec --config configs/config.yaml
```

---

## Export Commands

```bash
# Wav2Vec2 / WavLM — move old checkpoints out first if needed
# _find_checkpoints globs wav2vec2*_best.pt and wavlm*_best.pt
python export_onnx.py --config configs/config.yaml

# Whisper — separate script
python export_whisper_onnx.py --config configs/config.yaml
# or with explicit checkpoint:
python export_whisper_onnx.py --config configs/config.yaml --checkpoint checkpoints/whisper_medium_best.pt

# Verify ONNX input shape
python check_onnx.py checkpoints/speech_classifier_quant.onnx
```

**IMPORTANT about export_onnx.py:**
- `_find_checkpoints` globs `*.pt` excluding wavlm/whisper/finetuned
- `_infer_window_sec` maps checkpoint tag to window size
- Must have `wavlm_5sec` in tag for WavLM to get 5.0s window
- Whisper export is SEPARATE — uses `export_whisper_onnx.py`
- Whisper dummy input must be `(1, 80, 3000)` — always 3000 frames

---

## Inference (predict_cpu.py)

### Key settings
```python
DEFAULT_CONFIG = {
    "read_threshold": 0.45,   # PRODUCTION VALUE — do not change without new eval
    "min_conf": 0.0,          # disabled — was hurting recall
    "window_sec": 5.0,
    "sample_rate": 16000,
    "temporal_smooth_window": 3,
    "min_segment_sec": 3.0,
    "vad_merge_gap_sec": 1.0,
}
```

### Model aliases
```
5sec / wav2vec2      -> speech_classifier_quant.onnx
7_5sec               -> speech_classifier_wav2vec2_7_5sec_quant.onnx
10sec                -> speech_classifier_wav2vec2_10sec_quant.onnx
12_5sec              -> speech_classifier_wav2vec2_12_5sec_quant.onnx
15sec                -> speech_classifier_wav2vec2_15sec_quant.onnx
wavlm / wavlm_5sec   -> speech_classifier_wavlm_5sec_quant.onnx
whisper              -> speech_classifier_whisper_medium_quant.onnx
```

### Run commands
```bash
# Production baseline (Wav2Vec2 5sec, RMS VAD, threshold 0.45)
python predict_cpu.py --audio audios2/ --model 5sec --output outputs/results_5sec.json --no-silero

# Expanded data model
python predict_cpu.py --audio audios2/ --model checkpoints/speech_classifier_wav2vec2_5sec_quantexpanded_data.onnx --window-sec 5.0 --output outputs/results_expanded.json --no-silero

# Whisper (auto-detected from filename)
python predict_cpu.py --audio audios2/ --model whisper --output outputs/results_whisper.json --no-silero

# WavLM (not production — domain mismatch issue)
python predict_cpu.py --audio audios2/ --model wavlm --output outputs/results_wavlm.json --no-silero
```

### Two classifier classes in predict_cpu.py
- `ONNXClassifier` — for Wav2Vec2/WavLM, takes raw waveform `(B, samples)`
- `WhisperONNXClassifier` — for Whisper, takes log-mel spectrogram `(B, 80, 3000)`, extracts mel internally from waveform using librosa
- Auto-detected from filename: if "whisper" in name → WhisperONNXClassifier

---

## Performance Results

### On company data (219 files, 1-min limit, partially cleaned GT labels)

| Model | Threshold | Precision | Recall | Notes |
|---|---|---|---|---|
| Wav2Vec2 5sec | 50% | 93.6% | 69.4% | |
| Wav2Vec2 5sec | 45% | **94.1%** | **76.1%** | **PRODUCTION BASELINE** |
| Wav2Vec2 5sec | 38% (calibrated) | 91.2% | 77.6% | Bad trade |
| New pipeline + Silero VAD | 51% | 93% | 72.1% | Silero hurts hesitant readers |
| WavLM 5sec (AllStar only) | optimal | 92% | 50% | Domain mismatch |
| Wav2Vec2 expanded data | TBD | TBD | TBD | Awaiting laptop eval |
| Whisper Medium | TBD | TBD | TBD | Training in progress |

### On AllStar test set

| Model | Accuracy | F1 |
|---|---|---|
| Wav2Vec2 5sec (AllStar only) | 96.95% | 97.75% |
| Wav2Vec2 5sec (expanded data) | 97.21% | 97.98% |
| WavLM 5sec | 96.95% | 97.75% |

---

## Critical Problems Discovered

### 1. Domain mismatch (main problem)
AllStar has clean boundaries: hesitation = spontaneous, fluency = read.
Company data: hesitant readers exist, fluent spontaneous speakers exist.
Model learned wrong rules. No inference fix works — needs company-specific training data.

### 2. WavLM is better so it's worse
WavLM learned AllStar patterns more deeply and confidently. Confidently misclassifies hesitant readers as spontaneous. Needs fine-tuning on company data before it can outperform Wav2Vec2.

### 3. Wrong problem definition (most important insight)
Files contain multiple questions. Each question has spontaneous + read portions.
File-level read_ratio of 40-50% = model correctly detecting both behaviors within one file, not confusion.
Real task is SEGMENT-LEVEL detection, not file-level classification.
The segment timeline output is what matters, not overall_label.

### 4. Boundary cases are linguistically ambiguous
Fluent spontaneous speakers sound like readers acoustically.
No acoustic model alone can distinguish them reliably.
Whisper MLP may help via linguistic/prosodic representations.

### 5. GT label quality
Labels manually done by SMEs with human errors.
~60/219 files reviewed and corrected.
Seniors finishing full relabelling.
Do NOT calibrate threshold until full clean labels are available.

---

## What Was Tried and Why It Failed/Helped

| Change | Result | Reason |
|---|---|---|
| Silero VAD | Worse recall | Clips hesitant speech — correct behavior but wrong for this data |
| Confidence gate (0.65) | Worse recall | Silences weak read votes from hesitant windows |
| Calibrated threshold 38% | Minor gain | 1.5% recall for 2.9% precision loss — bad trade |
| WavLM training | 50% recall on company | Deeper learning of wrong patterns |
| GigaSpeech expanded data | Same AllStar perf | Adding clean read/spont doesn't help with hesitant readers |
| Disabled Silero (--no-silero) | Better | Old RMS VAD accidentally passes more content through |
| 10sec windows | Worse than 5sec | Spans transitions, blurs read/spontaneous signal in one window |

---

## Current Pipeline Flow

```
Audio file
    ↓
load_audio() — peak normalize
    ↓
RMS VAD (global threshold) — get speech segments
    ↓
make_vad_gated_windows() — 5s windows, 2.5s hop, speech only
    ↓
ONNXClassifier.predict_batch() — Wav2Vec2 ONNX inference
    ↓
per-window label + confidence
    ↓
median_filter smoothing (window=3)
    ↓
_merge_segments() → enforce_min_segment_length(3s)
    ↓
read_ratio vote → overall_label at threshold 0.45
    ↓
JSON output with segments + window_predictions
```

---

## What Needs to Happen Next

### Immediate
1. Evaluate expanded Wav2Vec2 on company data — does GigaSpeech help?
2. Evaluate Whisper Medium on company data once training finishes
3. Get senior timestamp labels (when does reading start/stop in each file)
4. Change evaluation to segment-level, not file-level

### Manager meeting case (staging access)
Experiment sequence to present:
1. AllStar only → 94.1%/76.1% on concentrated data
2. Expanded data (+ GigaSpeech) → TBD (probably similar)
3. Whisper MLP → TBD
4. Evidence: "More general data doesn't fix it. Company-specific fine-tuning is the only remaining lever."

### After staging access
- Fine-tune WavLM-Base+ on company boundary cases (hesitant readers + fluent spontaneous)
- Target: 40-50 confirmed files per class from company data
- Mix ratio: 1 company per 2-3 AllStar in fine-tuning set
- Lower LR (5e-6), more frozen layers (8), fewer epochs (5)
- Re-enable Silero VAD after fine-tuning — will then help not hurt

### Long term architecture
Multi-scale WavLM (5s + 10s + 15s) with scale attention fusion + prosodic features concatenated at classifier. See full_project_report.md for complete code.

---

## Known Bugs and Fixes Applied

| Bug | Fix |
|---|---|
| `dataset.py` reads `duration_sec` but manifest has `duration` | Changed to `row["duration"]` |
| `build_accent_sampler` crashes on GigaSpeech rows (no `l1` column) | Changed to `row.get("l1", "unknown")` |
| AllStar manifest has NaN durations | `check_and_balance_manifest.py` reads actual audio files |
| WavLM export used `Wav2Vec2Model` not `WavLMModel` | Import `WavLMModel` from transformers |
| 7.5sec model was actually 5sec | Config `window_sec` was 5.0 during training — needs retrain |
| `torchcodec` required for audio decoding | Added `decode=False` + librosa manual decode |
| TEDLIUM uses deprecated loading script | Switched to GigaSpeech |
| Whisper export fails with 500 frames | Whisper always needs 3000 frames — fixed dummy input |
| GradScaler deprecated API | Use `torch.amp.GradScaler("cuda")` not `torch.cuda.amp.GradScaler()` |

---

## HuggingFace

Repo: `Pransfrance/speechproj-models` (private)

Upload script: `upload_hf.py`

Login: `huggingface-cli login` (token from https://huggingface.co/settings/tokens)

---

## Important Files and What They Do

| File | Purpose |
|---|---|
| `predict_cpu.py` | Self-contained laptop inference — copy this + ONNX to laptop |
| `export_onnx.py` | Exports Wav2Vec2/WavLM to ONNX + INT8 quantize |
| `export_whisper_onnx.py` | Exports Whisper to ONNX + INT8 quantize |
| `check_onnx.py` | Verifies ONNX model input shape — run after every export |
| `download_datasets.py` | Downloads GigaSpeech, slices into 1-min files, builds manifest |
| `check_and_balance_manifest.py` | Fixes missing AllStar durations, shows window balance |
| `upload_hf.py` | Uploads models to HuggingFace |
| `evaluation.ipynb` | Precision/recall curves, confusion matrices, error analysis |
| `src/data/dataset.py` | SpeechWindowDataset — reads `duration` column (not `duration_sec`) |
| `src/models/train_wavLM_5sec.py` | WavLM training — uses `WavLMModel` import |
| `src/models/train_whisper_cls.py` | Whisper MLP training — encoder fully frozen with `torch.no_grad()` |

---

## Key Numbers to Remember

| Metric | Value |
|---|---|
| Production baseline | 94.1% precision / 76.1% recall |
| Production threshold | 0.45 |
| Production model | Wav2Vec2 5sec, `speech_classifier_quant.onnx` |
| Precision floor (never go below) | 90% |
| Boundary zone | 0.38-0.55 read_ratio |
| AllStar WavLM F1 | 97.75% |
| Company WavLM recall | ~50% (domain mismatch) |
| Total train windows | 59,270 (30,497 read + 28,773 spontaneous) |
| Company files | 219 files, 1-minute limit |
