"""
Basic audio loading/preprocessing (Phase 10).

Loads an audio file, resamples to a fixed rate, and pads/trims to a
fixed clip length so every clip produces a feature vector of the same
shape regardless of the original file's length.
"""

import numpy as np
import librosa

from utils.config import SIREN_SAMPLE_RATE, SIREN_CLIP_SECONDS


def load_audio(path: str) -> np.ndarray:
    """Load an audio file as a mono waveform at SIREN_SAMPLE_RATE."""
    waveform, _ = librosa.load(path, sr=SIREN_SAMPLE_RATE, mono=True)

    target_len = SIREN_SAMPLE_RATE * SIREN_CLIP_SECONDS
    if len(waveform) > target_len:
        waveform = waveform[:target_len]
    elif len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))

    return waveform
