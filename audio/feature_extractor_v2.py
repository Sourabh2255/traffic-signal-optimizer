"""
Audio Feature Extraction for Siren Classifier V2.

Matches training preprocessing specifications:
- Audio loaded as mono at 22,050 Hz.
- Window size: up to 5 seconds (padded or trimmed to 5.0 seconds).
- 20 MFCC coefficients: mean and standard deviation of each (40 features).
- Spectral centroid: mean and standard deviation (2 features).
- Spectral rolloff: mean and standard deviation (2 features).
- Zero-crossing rate: mean and standard deviation (2 features).
- Exact feature count: 46 features.
- Exact feature order preserved:
    [mfcc_0_mean .. mfcc_19_mean,
     mfcc_0_std  .. mfcc_19_std,
     spectral_centroid_mean, spectral_centroid_std,
     spectral_rolloff_mean,  spectral_rolloff_std,
     zero_crossing_rate_mean, zero_crossing_rate_std]
"""

import os
from typing import Union, List
import io
import numpy as np
import librosa

from utils.config import (
    SIREN_V2_SAMPLE_RATE,
    SIREN_V2_CLIP_SECONDS,
    SIREN_V2_N_MFCC,
    SIREN_V2_EXPECTED_FEATURES,
)


def load_audio_v2(
    audio_source: Union[str, io.BytesIO, bytes],
    sr: int = SIREN_V2_SAMPLE_RATE,
    max_duration: float = SIREN_V2_CLIP_SECONDS,
) -> np.ndarray:
    """
    Load an audio file/source as mono at `sr` (22,050 Hz) and pad/trim to `max_duration` (5.0s).

    Args:
        audio_source: Path to audio file, BytesIO buffer, or raw bytes.
        sr: Target sampling rate (default 22,050 Hz).
        max_duration: Target duration in seconds (default 5.0s).

    Returns:
        1D numpy array representing mono audio waveform of length sr * max_duration.

    Raises:
        FileNotFoundError: If audio file path does not exist.
        ValueError: If audio file is empty or corrupted.
    """
    if isinstance(audio_source, str):
        if not os.path.exists(audio_source):
            raise FileNotFoundError(f"Audio file not found: {audio_source}")
        if os.path.getsize(audio_source) == 0:
            raise ValueError(f"Audio file is empty (0 bytes): {audio_source}")

    if isinstance(audio_source, bytes):
        if len(audio_source) == 0:
            raise ValueError("Audio byte buffer is empty (0 bytes).")
        audio_source = io.BytesIO(audio_source)

    try:
        waveform, _ = librosa.load(audio_source, sr=sr, mono=True)
    except Exception as e:
        raise ValueError(f"Failed to decode audio source: {e}") from e

    if waveform is None or len(waveform) == 0:
        raise ValueError("Decoded audio waveform is empty.")

    target_len = int(sr * max_duration)
    if len(waveform) > target_len:
        waveform = waveform[:target_len]
    elif len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))

    return waveform.astype(np.float32)


def extract_v2_features_from_waveform(
    waveform: np.ndarray,
    sr: int = SIREN_V2_SAMPLE_RATE,
) -> np.ndarray:
    """
    Extract exactly 46 audio features from a preprocessed waveform:
    - 20 MFCC means (indices 0..19)
    - 20 MFCC standard deviations (indices 20..39)
    - spectral centroid mean and std (indices 40..41)
    - spectral rolloff mean and std (indices 42..43)
    - zero crossing rate mean and std (indices 44..45)

    Returns:
        1D numpy array of shape (46,) and dtype float64.
    """
    if waveform is None or len(waveform) == 0:
        raise ValueError("Cannot extract features from empty waveform.")

    # 1. 20 MFCC coefficients
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=SIREN_V2_N_MFCC)
    mfcc_mean = mfcc.mean(axis=1)  # shape (20,)
    mfcc_std = mfcc.std(axis=1)    # shape (20,)

    # 2. Spectral centroid
    cent = librosa.feature.spectral_centroid(y=waveform, sr=sr)
    cent_mean = float(np.mean(cent))
    cent_std = float(np.std(cent))

    # 3. Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)
    rolloff_mean = float(np.mean(rolloff))
    rolloff_std = float(np.std(rolloff))

    # 4. Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=waveform)
    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))

    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        np.array([
            cent_mean, cent_std,
            rolloff_mean, rolloff_std,
            zcr_mean, zcr_std,
        ], dtype=np.float64),
    ]).astype(np.float64)

    if len(features) != SIREN_V2_EXPECTED_FEATURES:
        raise ValueError(
            f"Feature vector must contain exactly {SIREN_V2_EXPECTED_FEATURES} values, "
            f"got {len(features)}"
        )

    return features


def extract_v2_features(
    audio_source: Union[str, io.BytesIO, bytes, np.ndarray],
    sr: int = SIREN_V2_SAMPLE_RATE,
    max_duration: float = SIREN_V2_CLIP_SECONDS,
) -> np.ndarray:
    """
    Full pipeline: loads/preprocesses audio and extracts the 46-value feature vector.

    Args:
        audio_source: Audio file path, BytesIO buffer, raw bytes, or preprocessed waveform array.
        sr: Sample rate (default 22,050 Hz).
        max_duration: Maximum window length (default 5.0 seconds).

    Returns:
        1D numpy array of shape (46,).
    """
    if isinstance(audio_source, np.ndarray):
        waveform = audio_source
        target_len = int(sr * max_duration)
        if len(waveform) > target_len:
            waveform = waveform[:target_len]
        elif len(waveform) < target_len:
            waveform = np.pad(waveform, (0, target_len - len(waveform)))
    else:
        waveform = load_audio_v2(audio_source, sr=sr, max_duration=max_duration)

    return extract_v2_features_from_waveform(waveform, sr=sr)


def get_v2_feature_names() -> List[str]:
    """Return ordered list of the 46 feature names matching model training order."""
    mfcc_means = [f"mfcc_{i}_mean" for i in range(SIREN_V2_N_MFCC)]
    mfcc_stds = [f"mfcc_{i}_std" for i in range(SIREN_V2_N_MFCC)]
    spectral_features = [
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_rolloff_mean",
        "spectral_rolloff_std",
        "zero_crossing_rate_mean",
        "zero_crossing_rate_std",
    ]
    return mfcc_means + mfcc_stds + spectral_features
