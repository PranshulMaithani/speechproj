"""
Prosodic & Acoustic Feature Extraction
========================================
Extracts handcrafted features from audio windows for the XGBoost baseline.

Feature groups:
  1. Pitch (F0) statistics
  2. Speaking rate & rhythm
  3. Pause characteristics
  4. Spectral features (MFCCs, spectral tilt, centroid)
  5. Energy/intensity patterns
  6. Voice quality (jitter, shimmer)

These features capture the acoustic differences between read and spontaneous
speech that go beyond simple fluency/hesitation detection.
"""

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from typing import Optional


def _safe_stat(arr: np.ndarray, func, default: float = 0.0) -> float:
    """Safely compute a statistic, returning default on empty/nan."""
    if arr is None or len(arr) == 0:
        return default
    val = func(arr)
    return default if np.isnan(val) else float(val)


# ========================
# 1. PITCH (F0) FEATURES
# ========================

def extract_pitch_features(audio: np.ndarray, sr: int, f0_min: int = 75, f0_max: int = 500) -> dict:
    """
    Extract pitch contour features using Praat via parselmouth.

    Key insight: Read speech has MORE REGULAR pitch patterns (predictable
    declination, consistent pitch resets at sentence starts). Spontaneous
    speech has more variable pitch with more extreme excursions.
    """
    snd = parselmouth.Sound(audio, sampling_frequency=sr)
    pitch = call(snd, "To Pitch", 0.0, f0_min, f0_max)

    # Get F0 values (voiced frames only)
    f0_values = pitch.selected_array["frequency"]
    voiced = f0_values[f0_values > 0]

    features = {}

    if len(voiced) < 3:
        # Not enough voiced frames
        features.update({
            "f0_mean": 0.0, "f0_std": 0.0, "f0_min": 0.0, "f0_max": 0.0,
            "f0_range": 0.0, "f0_range_semitones": 0.0, "f0_cv": 0.0,
            "f0_slope_mean": 0.0, "f0_slope_std": 0.0,
            "f0_rising_ratio": 0.0, "f0_voiced_ratio": 0.0,
            "f0_reset_count": 0.0, "f0_median": 0.0, "f0_iqr": 0.0,
            "f0_skewness": 0.0, "f0_kurtosis": 0.0,
        })
        return features

    features["f0_mean"] = float(np.mean(voiced))
    features["f0_std"] = float(np.std(voiced))
    features["f0_min"] = float(np.min(voiced))
    features["f0_max"] = float(np.max(voiced))
    features["f0_range"] = features["f0_max"] - features["f0_min"]
    features["f0_median"] = float(np.median(voiced))
    features["f0_cv"] = features["f0_std"] / features["f0_mean"] if features["f0_mean"] > 0 else 0.0

    # Range in semitones (perceptually meaningful)
    if features["f0_min"] > 0:
        features["f0_range_semitones"] = 12 * np.log2(features["f0_max"] / features["f0_min"])
    else:
        features["f0_range_semitones"] = 0.0

    # IQR (robust measure of spread)
    features["f0_iqr"] = float(np.percentile(voiced, 75) - np.percentile(voiced, 25))

    # Higher-order moments
    from scipy.stats import skew, kurtosis
    features["f0_skewness"] = float(skew(voiced))
    features["f0_kurtosis"] = float(kurtosis(voiced))

    # F0 dynamics: slopes between consecutive voiced frames
    f0_diff = np.diff(voiced)
    features["f0_slope_mean"] = float(np.mean(np.abs(f0_diff)))
    features["f0_slope_std"] = float(np.std(f0_diff))
    features["f0_rising_ratio"] = float(np.sum(f0_diff > 0) / len(f0_diff))

    # Voiced ratio (how much of the audio has voicing)
    features["f0_voiced_ratio"] = float(len(voiced) / len(f0_values)) if len(f0_values) > 0 else 0.0

    # Pitch resets: sudden jumps > 1 std dev (common at sentence starts in reading)
    threshold = features["f0_std"] * 1.5
    features["f0_reset_count"] = float(np.sum(np.abs(f0_diff) > threshold))

    return features


# ========================
# 2. SPEAKING RATE & RHYTHM
# ========================

def extract_rhythm_features(audio: np.ndarray, sr: int) -> dict:
    """
    Speaking rate and rhythm features.

    Key insight: Read speech tends to have MORE REGULAR rhythm (lower nPVI,
    more consistent syllable durations). Spontaneous speech has more variable
    rhythm, with lengthening, contractions, and irregular timing.
    """
    features = {}

    # Onset detection as proxy for syllable nuclei
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onsets = librosa.onset.onset_detect(y=audio, sr=sr, onset_envelope=onset_env, units="time")

    if len(onsets) < 2:
        features.update({
            "syllable_rate": 0.0, "ioi_mean": 0.0, "ioi_std": 0.0,
            "ioi_cv": 0.0, "npvi": 0.0, "rhythm_regularity": 0.0,
            "onset_count": float(len(onsets)),
        })
        return features

    # Inter-onset intervals (proxy for syllable timing)
    ioi = np.diff(onsets)

    duration = len(audio) / sr
    features["syllable_rate"] = float(len(onsets) / duration) if duration > 0 else 0.0
    features["ioi_mean"] = float(np.mean(ioi))
    features["ioi_std"] = float(np.std(ioi))
    features["ioi_cv"] = features["ioi_std"] / features["ioi_mean"] if features["ioi_mean"] > 0 else 0.0
    features["onset_count"] = float(len(onsets))

    # nPVI (normalized Pairwise Variability Index)
    # Higher nPVI = more variable rhythm (more spontaneous-like)
    if len(ioi) >= 2:
        pvi_pairs = []
        for i in range(len(ioi) - 1):
            denom = (ioi[i] + ioi[i + 1]) / 2
            if denom > 0:
                pvi_pairs.append(abs(ioi[i] - ioi[i + 1]) / denom)
        features["npvi"] = float(np.mean(pvi_pairs) * 100) if pvi_pairs else 0.0
    else:
        features["npvi"] = 0.0

    # Rhythm regularity: autocorrelation of onset envelope at the syllable rate
    if features["syllable_rate"] > 0:
        expected_period = int(sr / features["syllable_rate"] / 256)  # in frames
        if expected_period > 0 and expected_period < len(onset_env):
            autocorr = np.correlate(onset_env, onset_env, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]
            if expected_period < len(autocorr):
                # Normalized autocorrelation at expected syllable period
                features["rhythm_regularity"] = float(
                    autocorr[expected_period] / (autocorr[0] + 1e-10)
                )
            else:
                features["rhythm_regularity"] = 0.0
        else:
            features["rhythm_regularity"] = 0.0
    else:
        features["rhythm_regularity"] = 0.0

    return features


# ========================
# 3. PAUSE FEATURES
# ========================

def extract_pause_features(
    audio: np.ndarray,
    sr: int,
    energy_threshold: float = 0.01,
) -> dict:
    """
    Pause pattern features.

    Key insight: Both read and spontaneous speech have pauses, but the
    DISTRIBUTION differs. Read speech has pauses at syntactic boundaries
    (more regular). Spontaneous speech has pauses for planning (variable
    lengths, not always at boundaries). Bad cheaters who inject random
    pauses create unusual pause distributions that don't match either
    pattern cleanly.
    """
    from src.data.audio_utils import simple_vad

    vad_mask = simple_vad(audio, sr, energy_threshold)
    frame_dur = 256 / sr  # default hop_length

    features = {}

    # Find pause segments (consecutive False in VAD)
    pauses = []
    in_pause = False
    pause_start = 0

    for i, is_speech in enumerate(vad_mask):
        if not is_speech and not in_pause:
            pause_start = i
            in_pause = True
        elif is_speech and in_pause:
            pause_dur = (i - pause_start) * frame_dur
            if pause_dur >= 0.15:  # Only count pauses >= 150ms
                pauses.append(pause_dur)
            in_pause = False

    features["pause_count"] = float(len(pauses))
    features["total_pause_duration"] = float(sum(pauses))
    total_dur = len(audio) / sr
    features["pause_ratio"] = features["total_pause_duration"] / total_dur if total_dur > 0 else 0.0

    if len(pauses) > 0:
        pauses_arr = np.array(pauses)
        features["pause_mean"] = float(np.mean(pauses_arr))
        features["pause_std"] = float(np.std(pauses_arr))
        features["pause_max"] = float(np.max(pauses_arr))
        features["pause_cv"] = (
            features["pause_std"] / features["pause_mean"]
            if features["pause_mean"] > 0 else 0.0
        )
        # Short vs long pause ratio
        short = np.sum(pauses_arr < 0.5)
        long = np.sum(pauses_arr >= 0.5)
        features["short_long_pause_ratio"] = float(short / (long + 1))
    else:
        features.update({
            "pause_mean": 0.0, "pause_std": 0.0, "pause_max": 0.0,
            "pause_cv": 0.0, "short_long_pause_ratio": 0.0,
        })

    # Pause position: are pauses evenly distributed or clustered?
    if len(pauses) >= 3:
        pause_positions = []
        idx = 0
        for i, is_speech in enumerate(vad_mask):
            if not is_speech and (i == 0 or vad_mask[i - 1]):
                pause_positions.append(i * frame_dur / total_dur)  # Normalized position
        if len(pause_positions) >= 2:
            inter_pause = np.diff(pause_positions)
            features["pause_regularity"] = float(np.std(inter_pause))
        else:
            features["pause_regularity"] = 0.0
    else:
        features["pause_regularity"] = 0.0

    return features


# ========================
# 4. SPECTRAL FEATURES
# ========================

def extract_spectral_features(audio: np.ndarray, sr: int, n_mfcc: int = 13) -> dict:
    """
    Spectral features: MFCCs, spectral tilt, centroid variability.

    Key insight: Read speech often has a more consistent spectral profile
    across the utterance. Spontaneous speech shows more spectral variation
    due to changing articulation patterns, word choices, and emotional
    coloring.
    """
    features = {}

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    for i in range(n_mfcc):
        features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
        features[f"mfcc_{i}_std"] = float(np.std(mfccs[i]))

    # Delta MFCCs (dynamics)
    delta_mfccs = librosa.feature.delta(mfccs)
    for i in range(min(5, n_mfcc)):  # First 5 delta coefficients
        features[f"delta_mfcc_{i}_mean"] = float(np.mean(delta_mfccs[i]))
        features[f"delta_mfcc_{i}_std"] = float(np.std(delta_mfccs[i]))

    # Spectral centroid (brightness/darkness variation)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"] = float(np.std(centroid))
    features["spectral_centroid_cv"] = (
        features["spectral_centroid_std"] / features["spectral_centroid_mean"]
        if features["spectral_centroid_mean"] > 0 else 0.0
    )

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"] = float(np.std(rolloff))

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    features["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
    features["spectral_bandwidth_std"] = float(np.std(bandwidth))

    # Spectral contrast (measures spectral "peakiness")
    try:
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features["spectral_contrast_mean"] = float(np.mean(contrast))
        features["spectral_contrast_std"] = float(np.std(contrast))
    except Exception:
        features["spectral_contrast_mean"] = 0.0
        features["spectral_contrast_std"] = 0.0

    # Spectral flatness (tonality measure)
    flatness = librosa.feature.spectral_flatness(y=audio)[0]
    features["spectral_flatness_mean"] = float(np.mean(flatness))
    features["spectral_flatness_std"] = float(np.std(flatness))

    # Zero-crossing rate (correlates with noise content)
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    return features


# ========================
# 5. ENERGY FEATURES
# ========================

def extract_energy_features(audio: np.ndarray, sr: int) -> dict:
    """
    Energy/intensity pattern features.

    Key insight: Read speech has smoother energy contours with gradual
    declination. Spontaneous speech has more abrupt energy changes.
    """
    features = {}

    rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]

    features["energy_mean"] = float(np.mean(rms))
    features["energy_std"] = float(np.std(rms))
    features["energy_max"] = float(np.max(rms))
    features["energy_cv"] = (
        features["energy_std"] / features["energy_mean"]
        if features["energy_mean"] > 0 else 0.0
    )

    # Energy dynamics
    energy_diff = np.diff(rms)
    features["energy_diff_mean"] = float(np.mean(np.abs(energy_diff)))
    features["energy_diff_std"] = float(np.std(energy_diff))

    # Energy contour regularity (autocorrelation at typical breath-group length)
    if len(rms) > 50:
        autocorr = np.correlate(rms - np.mean(rms), rms - np.mean(rms), mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        if autocorr[0] > 0:
            # Check regularity at ~2-3 second intervals (typical for reading)
            check_lag = min(int(2.0 * sr / 256), len(autocorr) - 1)
            features["energy_autocorr_2s"] = float(autocorr[check_lag] / autocorr[0])
        else:
            features["energy_autocorr_2s"] = 0.0
    else:
        features["energy_autocorr_2s"] = 0.0

    # Dynamic range
    if features["energy_mean"] > 0:
        features["dynamic_range_db"] = float(
            20 * np.log10(features["energy_max"] / (features["energy_mean"] + 1e-10))
        )
    else:
        features["dynamic_range_db"] = 0.0

    return features


# ========================
# 6. VOICE QUALITY
# ========================

def extract_voice_quality_features(audio: np.ndarray, sr: int) -> dict:
    """
    Voice quality features: jitter and shimmer.

    These measure micro-perturbations in pitch and amplitude.
    Reading aloud tends to produce more controlled phonation.
    """
    snd = parselmouth.Sound(audio, sampling_frequency=sr)
    features = {}

    try:
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)

        # Jitter (pitch perturbation)
        features["jitter_local"] = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        features["jitter_rap"] = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)

        # Shimmer (amplitude perturbation)
        features["shimmer_local"] = call(
            [snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )
        features["shimmer_apq3"] = call(
            [snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )

        # HNR (Harmonics-to-Noise Ratio)
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        features["hnr_mean"] = call(harmonicity, "Get mean", 0, 0)

    except Exception:
        features.update({
            "jitter_local": 0.0, "jitter_rap": 0.0,
            "shimmer_local": 0.0, "shimmer_apq3": 0.0,
            "hnr_mean": 0.0,
        })

    # Replace NaN/undefined with 0
    for k, v in features.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            features[k] = 0.0

    return features


# ========================
# COMBINED EXTRACTION
# ========================

def extract_all_features(
    audio: np.ndarray,
    sr: int,
    f0_min: int = 75,
    f0_max: int = 500,
    n_mfcc: int = 13,
    energy_threshold: float = 0.01,
) -> dict:
    """
    Extract all feature groups from an audio window.

    Returns a flat dictionary of ~90+ features.
    """
    features = {}

    features.update(extract_pitch_features(audio, sr, f0_min, f0_max))
    features.update(extract_rhythm_features(audio, sr))
    features.update(extract_spectral_features(audio, sr, n_mfcc))
    features.update(extract_energy_features(audio, sr))
    features.update(extract_pause_features(audio, sr, energy_threshold))
    features.update(extract_voice_quality_features(audio, sr))

    return features


def get_feature_names() -> list[str]:
    """Return ordered list of all feature names for consistency."""
    # Generate from a dummy extraction
    dummy = np.random.randn(16000 * 5).astype(np.float32) * 0.01  # 5s of noise
    feats = extract_all_features(dummy, 16000)
    return sorted(feats.keys())
