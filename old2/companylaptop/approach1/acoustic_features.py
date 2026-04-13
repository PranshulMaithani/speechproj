"""
Acoustic feature extraction from audio files.
Uses only librosa (no parselmouth dependency for portability).

Captures prosodic/acoustic signal:
- Pitch patterns (via librosa.pyin)
- Speaking rate & rhythm
- Pause characteristics
- Energy patterns
- Spectral features
"""
import numpy as np
import librosa


def _safe_stat(arr, func, default=0.0):
    if arr is None or len(arr) == 0:
        return default
    val = func(arr)
    return default if np.isnan(val) else float(val)


def extract_pitch_features(audio: np.ndarray, sr: int) -> dict:
    """Pitch (F0) features using librosa pyin."""
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio, fmin=75, fmax=500, sr=sr, frame_length=2048,
    )
    voiced = f0[~np.isnan(f0)] if f0 is not None else np.array([])

    feats = {}
    if len(voiced) < 3:
        for k in ["f0_mean", "f0_std", "f0_range", "f0_range_semitones", "f0_cv",
                   "f0_slope_mean", "f0_slope_std", "f0_voiced_ratio", "f0_iqr"]:
            feats[k] = 0.0
        return feats

    feats["f0_mean"] = float(np.mean(voiced))
    feats["f0_std"] = float(np.std(voiced))
    feats["f0_range"] = float(np.max(voiced) - np.min(voiced))
    feats["f0_cv"] = feats["f0_std"] / feats["f0_mean"] if feats["f0_mean"] > 0 else 0.0
    feats["f0_iqr"] = float(np.percentile(voiced, 75) - np.percentile(voiced, 25))

    if np.min(voiced) > 0:
        feats["f0_range_semitones"] = float(12 * np.log2(np.max(voiced) / np.min(voiced)))
    else:
        feats["f0_range_semitones"] = 0.0

    # F0 dynamics
    f0_diff = np.diff(voiced)
    feats["f0_slope_mean"] = float(np.mean(np.abs(f0_diff)))
    feats["f0_slope_std"] = float(np.std(f0_diff))

    # Voiced ratio
    total_frames = len(f0) if f0 is not None else 1
    feats["f0_voiced_ratio"] = float(len(voiced) / total_frames)

    return feats


def extract_rhythm_features(audio: np.ndarray, sr: int) -> dict:
    """Speaking rate and rhythm features."""
    feats = {}
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onsets = librosa.onset.onset_detect(y=audio, sr=sr, onset_envelope=onset_env, units="time")

    duration = len(audio) / sr
    feats["onset_count"] = float(len(onsets))
    feats["syllable_rate"] = float(len(onsets) / duration) if duration > 0 else 0.0

    if len(onsets) < 2:
        feats.update({"ioi_mean": 0.0, "ioi_std": 0.0, "ioi_cv": 0.0, "npvi": 0.0})
        return feats

    ioi = np.diff(onsets)
    feats["ioi_mean"] = float(np.mean(ioi))
    feats["ioi_std"] = float(np.std(ioi))
    feats["ioi_cv"] = feats["ioi_std"] / feats["ioi_mean"] if feats["ioi_mean"] > 0 else 0.0

    # nPVI
    if len(ioi) >= 2:
        pvi = []
        for i in range(len(ioi) - 1):
            denom = (ioi[i] + ioi[i+1]) / 2
            if denom > 0:
                pvi.append(abs(ioi[i] - ioi[i+1]) / denom)
        feats["npvi"] = float(np.mean(pvi) * 100) if pvi else 0.0
    else:
        feats["npvi"] = 0.0

    return feats


def extract_pause_features(audio: np.ndarray, sr: int) -> dict:
    """Pause pattern features using RMS energy VAD."""
    feats = {}
    rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]
    frame_dur = 256 / sr

    # Dynamic threshold (same as audio_utils.simple_vad)
    rms_max = float(np.max(rms))
    if rms_max < 1e-8:
        feats.update({k: 0.0 for k in [
            "pause_count", "pause_ratio", "pause_mean", "pause_std",
            "pause_max", "pause_cv", "short_long_pause_ratio",
        ]})
        return feats

    noise_floor = np.percentile(rms, 10)
    signal_peak = np.percentile(rms, 95)
    thresh = noise_floor + (signal_peak - noise_floor) * 0.15
    thresh = max(thresh, rms_max * 0.05)
    speech_mask = rms > thresh

    # Find pauses
    pauses = []
    in_pause = False
    pause_start = 0
    for i, is_speech in enumerate(speech_mask):
        if not is_speech and not in_pause:
            pause_start = i
            in_pause = True
        elif is_speech and in_pause:
            dur = (i - pause_start) * frame_dur
            if dur >= 0.15:
                pauses.append(dur)
            in_pause = False

    total_dur = len(audio) / sr
    feats["pause_count"] = float(len(pauses))
    feats["pause_ratio"] = sum(pauses) / total_dur if total_dur > 0 else 0.0

    if pauses:
        p = np.array(pauses)
        feats["pause_mean"] = float(np.mean(p))
        feats["pause_std"] = float(np.std(p))
        feats["pause_max"] = float(np.max(p))
        feats["pause_cv"] = feats["pause_std"] / feats["pause_mean"] if feats["pause_mean"] > 0 else 0.0
        short = np.sum(p < 0.5)
        long = np.sum(p >= 0.5)
        feats["short_long_pause_ratio"] = float(short / (long + 1))
    else:
        feats.update({"pause_mean": 0.0, "pause_std": 0.0, "pause_max": 0.0,
                       "pause_cv": 0.0, "short_long_pause_ratio": 0.0})
    return feats


def extract_energy_features(audio: np.ndarray, sr: int) -> dict:
    """Energy/intensity features."""
    feats = {}
    rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]

    feats["energy_mean"] = float(np.mean(rms))
    feats["energy_std"] = float(np.std(rms))
    feats["energy_max"] = float(np.max(rms))
    feats["energy_cv"] = feats["energy_std"] / feats["energy_mean"] if feats["energy_mean"] > 0 else 0.0

    energy_diff = np.diff(rms)
    feats["energy_diff_mean"] = float(np.mean(np.abs(energy_diff)))
    feats["energy_diff_std"] = float(np.std(energy_diff))

    return feats


def extract_spectral_features(audio: np.ndarray, sr: int, n_mfcc: int = 13) -> dict:
    """Spectral features: MFCCs and spectral statistics."""
    feats = {}
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    for i in range(n_mfcc):
        feats[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
        feats[f"mfcc_{i}_std"] = float(np.std(mfccs[i]))

    # Delta MFCCs (first 5)
    delta = librosa.feature.delta(mfccs)
    for i in range(min(5, n_mfcc)):
        feats[f"delta_mfcc_{i}_mean"] = float(np.mean(delta[i]))
        feats[f"delta_mfcc_{i}_std"] = float(np.std(delta[i]))

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    feats["spectral_centroid_mean"] = float(np.mean(centroid))
    feats["spectral_centroid_std"] = float(np.std(centroid))

    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    feats["spectral_rolloff_mean"] = float(np.mean(rolloff))

    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["zcr_std"] = float(np.std(zcr))

    return feats


def extract_all_acoustic_features(audio: np.ndarray, sr: int) -> dict:
    """Extract all acoustic feature groups from raw audio."""
    feats = {}
    feats.update(extract_pitch_features(audio, sr))
    feats.update(extract_rhythm_features(audio, sr))
    feats.update(extract_pause_features(audio, sr))
    feats.update(extract_energy_features(audio, sr))
    feats.update(extract_spectral_features(audio, sr))
    return feats
