"""
MFCC feature extraction (Phase 10).

MFCCs (Mel-Frequency Cepstral Coefficients) are a standard, explainable
audio feature for simple classifiers like Random Forest / SVM — a good
fit for a college Data Science project (vs. a deep audio model, which
would be overkill and harder to explain in a viva).

For each clip we take the MEAN and STANDARD DEVIATION of each of the
N_MFCC coefficients across time, producing a fixed-length feature vector
per clip (2 * N_MFCC values).
"""

import numpy as np
import librosa

from utils.config import SIREN_SAMPLE_RATE, N_MFCC
from audio.audio_processor import load_audio


def extract_mfcc_features(path: str) -> np.ndarray:
    """Return a 1D feature vector: [mfcc_means..., mfcc_stds...]."""
    waveform = load_audio(path)
    mfcc = librosa.feature.mfcc(y=waveform, sr=SIREN_SAMPLE_RATE, n_mfcc=N_MFCC)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def feature_names() -> list:
    means = [f"mfcc_{i}_mean" for i in range(N_MFCC)]
    stds = [f"mfcc_{i}_std" for i in range(N_MFCC)]
    return means + stds
