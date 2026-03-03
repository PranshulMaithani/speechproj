"""
Audio Processing Utilities
============================
Windowing, VAD (Voice Activity Detection), resampling, and audio I/O.
"""

import numpy as np
import librosa
import soundfile as sf
from dataclasses import dataclass


@dataclass
class AudioWindow:
    """A single audio window with metadata."""
    audio: np.ndarray       # Raw waveform samples
    sr: int                 # Sample rate
    start_sec: float        # Start time in original file
    end_sec: float          # End time in original file
    speech_ratio: float     # Fraction of window that is speech (by energy)
    is_valid: bool          # True if speech_ratio >= min threshold


def load_audio(path: str, target_sr: int = 16000, max_duration: float = 120.0) -> tuple[np.ndarray, int]:
    """
    Load audio, convert to mono, resample to target_sr.
    Peak-normalizes so that VAD thresholds are format-agnostic
    (handles wav, m4a, mp3, webm — audioread varies in output amplitude).
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress audioread/soundfile fallback warnings
        audio, sr = librosa.load(path, sr=target_sr, mono=True, duration=max_duration)

    # Peak-normalize: guarantees consistent amplitude regardless of codec/encoder
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak * 0.95
    return audio, sr


def compute_rms_energy(audio: np.ndarray, frame_length: int = 512, hop_length: int = 256) -> np.ndarray:
    """Compute frame-level RMS energy."""
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    return rms


def simple_vad(
    audio: np.ndarray,
    sr: int,
    energy_threshold: float = 0.01,
    frame_length: int = 512,
    hop_length: int = 256,
    min_speech_dur: float = 0.1,
) -> np.ndarray:
    """
    Simple energy-based Voice Activity Detection.

    Returns:
        Boolean mask at frame level (True = speech, False = silence).
    """
    rms = compute_rms_energy(audio, frame_length, hop_length)

    # Fully relative threshold — independent of format/codec/amplitude level.
    # Works equally on wav, m4a, mp3, webm etc. even though audioread and
    # soundfile output different absolute energy levels for the same content.
    rms_max = float(np.max(rms))
    if rms_max < 1e-8:          # Genuinely flat signal (pure silence)
        return np.zeros(len(rms), dtype=bool)
    noise_floor = np.percentile(rms, 10)       # Bottom 10% = background noise
    signal_peak = np.percentile(rms, 95)        # Top 5% = loudest speech frames
    # Threshold at 15% of the way from noise floor to signal peak
    dynamic_thresh = noise_floor + (signal_peak - noise_floor) * 0.15
    # Require at least 5% of peak so silence gaps between utterances are
    # correctly labelled (safe after peak-normalization in load_audio).
    dynamic_thresh = max(dynamic_thresh, rms_max * 0.05)
    speech_mask = rms > dynamic_thresh

    # Smooth: fill short gaps (< min_speech_dur)
    min_frames = int(min_speech_dur * sr / hop_length)
    smoothed = speech_mask.copy()

    # Fill short silence gaps within speech
    in_speech = False
    gap_start = 0
    for i in range(len(smoothed)):
        if smoothed[i]:
            if in_speech and (i - gap_start) < min_frames:
                smoothed[gap_start:i] = True
            in_speech = True
        else:
            if in_speech:
                gap_start = i
                in_speech = False

    return smoothed


def compute_speech_ratio(
    audio: np.ndarray,
    sr: int,
    energy_threshold: float = 0.01,
) -> float:
    """Compute fraction of audio that contains speech."""
    vad_mask = simple_vad(audio, sr, energy_threshold)
    if len(vad_mask) == 0:
        return 0.0
    return float(vad_mask.sum()) / len(vad_mask)


def window_audio(
    audio: np.ndarray,
    sr: int,
    window_sec: float = 5.0,
    hop_sec: float = 2.5,
    min_speech_ratio: float = 0.20,
    energy_threshold: float = 0.01,
) -> list[AudioWindow]:
    """
    Segment audio into overlapping windows.

    Args:
        audio: Raw waveform (mono, already at target sr)
        sr: Sample rate
        window_sec: Window length in seconds
        hop_sec: Hop size in seconds
        min_speech_ratio: Minimum speech fraction to keep a window
        energy_threshold: RMS threshold for VAD

    Returns:
        List of AudioWindow objects (both valid and invalid flagged)
    """
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    total_samples = len(audio)

    windows = []
    start = 0

    while start < total_samples:
        end = min(start + window_samples, total_samples)
        chunk = audio[start:end]

        # Pad if chunk is too short (last window)
        if len(chunk) < window_samples // 2:
            break  # Skip very short tail chunks
        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)), mode="constant")

        speech_ratio = compute_speech_ratio(chunk, sr, energy_threshold)

        windows.append(
            AudioWindow(
                audio=chunk,
                sr=sr,
                start_sec=start / sr,
                end_sec=end / sr,
                speech_ratio=speech_ratio,
                is_valid=speech_ratio >= min_speech_ratio,
            )
        )

        start += hop_samples

    return windows


def get_speaking_segments(
    audio: np.ndarray,
    sr: int,
    energy_threshold: float = 0.01,
    min_segment_dur: float = 0.3,
    hop_length: int = 256,
) -> list[dict]:
    """
    Get contiguous speech segments from audio.

    Returns:
        List of dicts with 'start_sec', 'end_sec', 'duration_sec'
    """
    vad_mask = simple_vad(audio, sr, energy_threshold, hop_length=hop_length)
    frame_dur = hop_length / sr

    segments = []
    in_speech = False
    seg_start = 0.0

    for i, is_speech in enumerate(vad_mask):
        t = i * frame_dur
        if is_speech and not in_speech:
            seg_start = t
            in_speech = True
        elif not is_speech and in_speech:
            duration = t - seg_start
            if duration >= min_segment_dur:
                segments.append({
                    "start_sec": round(seg_start, 3),
                    "end_sec": round(t, 3),
                    "duration_sec": round(duration, 3),
                })
            in_speech = False

    # Handle trailing speech
    if in_speech:
        t = len(vad_mask) * frame_dur
        duration = t - seg_start
        if duration >= min_segment_dur:
            segments.append({
                "start_sec": round(seg_start, 3),
                "end_sec": round(t, 3),
                "duration_sec": round(duration, 3),
            })

    return segments
