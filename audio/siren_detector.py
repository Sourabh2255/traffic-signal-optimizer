"""
Siren detection wrapper (Phase 11) — loads the trained classifier and
exposes a simple predict(path) -> {is_siren, confidence} interface for
the Flask app.
"""

import os
import joblib
import numpy as np

from utils.config import SIREN_MODEL_PATH, SIREN_CONFIDENCE_THRESHOLD
from audio.feature_extractor import extract_mfcc_features


class SirenDetector:
    _model = None

    def __init__(self):
        if SirenDetector._model is None:
            if not os.path.exists(SIREN_MODEL_PATH):
                raise FileNotFoundError(
                    "Siren model not found. Run: python -m ml.generate_siren_dataset "
                    "then python -m ml.train_siren_model"
                )
            SirenDetector._model = self._load_or_retrain()
        self.model = SirenDetector._model

    @staticmethod
    def _load_or_retrain():
        """
        Load the saved classifier; if it cannot be loaded (e.g. saved with a
        different scikit-learn version), retrain from the bundled siren /
        no_siren clips instead of crashing.
        """
        try:
            return joblib.load(SIREN_MODEL_PATH)
        except Exception as err:
            print(f"[siren] Could not load saved siren model ({type(err).__name__}); retraining...")
            from ml.train_siren_model import train_and_evaluate
            train_and_evaluate()
            return joblib.load(SIREN_MODEL_PATH)

    def predict(self, audio_path: str) -> dict:
        features = extract_mfcc_features(audio_path).reshape(1, -1)
        proba = self.model.predict_proba(features)[0]
        siren_confidence = float(proba[1])  # class 1 = siren

        return {
            "is_siren": siren_confidence >= SIREN_CONFIDENCE_THRESHOLD,
            "confidence": round(siren_confidence, 3),
        }
