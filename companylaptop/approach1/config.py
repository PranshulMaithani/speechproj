"""
Configuration for Approach 1: Whisper ASR + Text/Acoustic Features + XGBoost
"""
from pathlib import Path

# ── Paths (relative to speechtry/ root) ──────────────────────────────────
# Adjust SPEECHTRY_ROOT if running from a different working directory
SPEECHTRY_ROOT = Path(__file__).resolve().parent.parent  # speechtry/

AUDIO_DIRS = {
    "audios2": SPEECHTRY_ROOT / "audios2",
    "audios4": SPEECHTRY_ROOT / "audios4",
}

GT_EXCEL = SPEECHTRY_ROOT / "outputs" / "audio2inference+sme.xlsx"
GT_SHEETS = {
    "audios2": {"sheet": "Combined", "filename_col": "filename", "path_col": "filepath", "label_col": "RevGT+MyVerdict"},
    "audios4": {"sheet": "wav2vec2audios4", "filename_col": "filename", "path_col": "filepath", "label_col": "RevGT+MyVerdict"},
}

OUTPUT_DIR = SPEECHTRY_ROOT / "outputs" / "approach1"
TRANSCRIPTS_DIR = OUTPUT_DIR / "transcripts"
FEATURES_CSV = OUTPUT_DIR / "features.csv"
MODEL_DIR = OUTPUT_DIR / "models"

# ── Audio ─────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
MAX_DURATION_SEC = 65  # slightly above 1-min to avoid clipping

# ── Whisper ───────────────────────────────────────────────────────────────
WHISPER_MODEL = "small"  # "tiny", "base", "small", "medium"

# ── Labels ────────────────────────────────────────────────────────────────
# Map raw GT labels to binary. Add mappings as needed based on your Excel.
LABEL_MAP = {
    # cheating variants
    "read": 1,
    "Read": 1,
    "READ": 1,
    "cheating": 1,
    "Cheating": 1,
    "cheater": 1,
    "Cheater": 1,
    "reading": 1,
    "Reading": 1,
    "yes": 1,
    "Yes": 1,
    "Y": 1,
    "1": 1,
    1: 1,
    # not cheating variants
    "spontaneous": 0,
    "Spontaneous": 0,
    "not cheating": 0,
    "Not Cheating": 0,
    "Not cheating": 0,
    "no": 0,
    "No": 0,
    "N": 0,
    "0": 0,
    0: 0,
    "genuine": 0,
    "Genuine": 0,
}

# ── Train / test split ───────────────────────────────────────────────────
TEST_RATIO = 0.15
VAL_RATIO = 0.15
RANDOM_SEED = 42
