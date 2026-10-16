"""
Green-time prediction (Phase 7) — connects YOLO-derived features to the
trained Random Forest model and clamps the output to [MIN_GREEN, MAX_GREEN].

    YOLO counts -> features -> Random Forest -> raw prediction -> clamp -> green_time
"""

import os
import joblib
import pandas as pd

from utils.config import TRAFFIC_MODEL_PATH, TRAFFIC_MODEL_FEATURES, MIN_GREEN, MAX_GREEN


class GreenTimePredictor:
    _model = None

    def __init__(self):
        if GreenTimePredictor._model is None:
            if not os.path.exists(TRAFFIC_MODEL_PATH):
                raise FileNotFoundError(
                    "Traffic model not found. Run: python ml/train_traffic_model.py"
                )
            GreenTimePredictor._model = self._load_or_retrain()
        self.model = GreenTimePredictor._model

    @staticmethod
    def _load_or_retrain():
        """
        Load the saved model. A .pkl saved with a different scikit-learn
        version can fail to load, so in that case retrain from the bundled
        simulation dataset (takes a few seconds) instead of crashing.
        """
        try:
            return joblib.load(TRAFFIC_MODEL_PATH)
        except Exception as err:
            print(f"[predictor] Could not load saved traffic model ({type(err).__name__}); retraining...")
            from ml.train_traffic_model import train_and_evaluate
            train_and_evaluate()
            return joblib.load(TRAFFIC_MODEL_PATH)

    def predict(self, features: dict) -> dict:
        """
        Args:
            features: dict containing all keys in TRAFFIC_MODEL_FEATURES

        Returns:
            {"raw_prediction": float, "final_green_time": float}
        """
        row = pd.DataFrame([[features[f] for f in TRAFFIC_MODEL_FEATURES]], columns=TRAFFIC_MODEL_FEATURES)
        raw = float(self.model.predict(row)[0])
        final = max(MIN_GREEN, min(MAX_GREEN, raw))
        return {
            "raw_prediction": round(raw, 1),
            "final_green_time": round(final, 1),
        }
