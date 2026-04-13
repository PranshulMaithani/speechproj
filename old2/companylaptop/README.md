# Cheating Detection in Speech Assessments

Detects whether candidates are reading from prepared/AI-generated answers during assessments.

## Quick Start

```bash
# 1. Setup (from speechtry/ root)
cd speechtry
pip install -r requirements.txt

# 2. Transcribe all audio files (run once, ~1-2 hours on CPU)
python -m approach1.transcribe

# 3. Extract text + acoustic features (~10-15 min)
python -m approach1.extract_features

# 4. Train classifier (runs 3 experiments: text-only, acoustic-only, combined)
python -m approach1.train

# 5. Evaluate with error analysis
python -m approach1.evaluate

# 6. Predict on new files
python -m approach1.predict --audio path/to/audio_or_folder
```

## Expected Directory Structure

```
speechtry/
├── audios2/                           # 219 cheating-concentrated audios
│   └── <candidate_id>/
│       ├── <candidate_id>_25.wav
│       ├── <candidate_id>_26.wav
│       └── <candidate_id>_27.wav
├── audios4/                           # 288 clearly labeled audios
│   └── <candidate_id>/
│       ├── <candidate_id>_25.wav
│       ├── <candidate_id>_26.wav
│       └── <candidate_id>_27.wav
├── outputs/
│   └── audio2inference+sme.xlsx       # Ground truth labels
│       Sheet "Combined"         -> audios2 labels (filename, filepath, RevGT+MyVerdict)
│       Sheet "wav2vec2audios4"  -> audios4 labels (filename, filepath, RevGT+MyVerdict)
├── approach1/                         # <-- THIS CODE
│   ├── config.py
│   ├── transcribe.py
│   ├── text_features.py
│   ├── acoustic_features.py
│   ├── extract_features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── requirements.txt
├── README.md
└── old/                               # Previous code (move manually)
```

## Step-by-Step Guide

### Step 1: Setup Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

**First time only:** Whisper will download model weights (~500MB for "small") on first run.

### Step 2: Transcribe Audio Files

```bash
# Transcribe everything (resumes if interrupted -- skips already-done files)
python -m approach1.transcribe

# Only transcribe audios2:
python -m approach1.transcribe --audio-dir audios2

# Use a larger/smaller Whisper model:
python -m approach1.transcribe --model medium   # better accuracy, slower
python -m approach1.transcribe --model tiny     # fastest, lower accuracy

# Force re-transcribe:
python -m approach1.transcribe --force
```

Transcripts are saved as JSON in `outputs/approach1/transcripts/`.
This is the slowest step (~10-20s per file on CPU). It's resumable.

### Step 3: Extract Features

```bash
# Full extraction (text + acoustic features)
python -m approach1.extract_features

# Text features only (fast, good for quick validation)
python -m approach1.extract_features --skip-acoustic
```

Output: `outputs/approach1/features.csv` with ~70 features per file.

### Step 4: Train Classifier

```bash
# Full run: trains text-only, acoustic-only, and combined models
python -m approach1.train

# Quick validation of linguistic signal only:
python -m approach1.train --text-only

# Skip hyperparameter search (faster, uses defaults):
python -m approach1.train --no-search
```

This trains both LogisticRegression (baseline) and XGBoost for each feature set.
Prints confusion matrices, feature importance, and precision/recall.

**Key output to check:**
- If text-only F1 > 0.60 -> linguistic signal is real, proceed with combined model
- If text-only F1 < 0.55 -> linguistic signal is weak, may need neural approach

### Step 5: Evaluate

```bash
python -m approach1.evaluate
```

Shows:
- Full classification report
- Threshold sweep (precision/recall at different thresholds)
- Error analysis (which files are misclassified and why)
- Feature distribution comparison between cheating and genuine

### Step 6: Predict on New Files

```bash
# Single file:
python -m approach1.predict --audio path/to/audio.wav

# Folder of files:
python -m approach1.predict --audio new_audios/

# With specific model:
python -m approach1.predict --audio new_audios/ --model outputs/approach1/models/xgboost_combined.pkl

# Custom output:
python -m approach1.predict --audio new_audios/ --output results.json
```

Output JSON includes verdict (CHEATING/UNCERTAIN/GENUINE), probability, transcript,
and key feature signals for interpretability.

## Configuration

Edit `approach1/config.py` to adjust:

- **`LABEL_MAP`**: Maps your Excel label values to binary (1=cheating, 0=genuine).
  Add any label variants you see in warnings during feature extraction.
- **`WHISPER_MODEL`**: Default "small". Use "medium" for better accuracy or "tiny" for speed.
- **`GT_SHEETS`**: Sheet names and column names from your Excel file.
- **`TEST_RATIO` / `VAL_RATIO`**: Split ratios (default 15%/15%).

## Troubleshooting

**"unmapped label" warnings during feature extraction:**
Your Excel has label values not in `LABEL_MAP`. Add them to `config.py`.

**"transcript not found" errors:**
Run Step 2 (transcribe) first. Or check that filenames in Excel match actual wav filenames.

**Low text-only accuracy (<55%):**
The linguistic signal may be weak for your data. This means the problem is more acoustic
than linguistic. Try the neural approach (Approach 2) instead.

**Whisper produces bad transcripts:**
Try `--model medium` for better accuracy. Check a few transcripts manually in
`outputs/approach1/transcripts/`.
