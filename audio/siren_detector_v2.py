"""
Siren Audio Classifier V2.

Loads the newly trained SVM Pipeline model (models/siren_model_v2.joblib)
and performs inference on audio recordings using 46 MFCC and spectral features.

Classes:
- 'ambulance_siren'
- 'fire_truck_siren'
- 'negative_sound'

Safety & Integrity Rules:
1. Missing model files, corrupt files, empty audio, or inference errors MUST NOT
   be interpreted as confirmed sirens (is_siren=False, confidence=0.0).
2. The loaded model is a scikit-learn Pipeline that already includes a StandardScaler.
   Raw 46-value feature vectors are passed directly without double-scaling.
3. Feature vector length is strictly validated to be exactly 46 before prediction.
"""

import os
from typing import Optional, Dict, Any, Union
import logging
import joblib
import numpy as np

from utils.config import (
    SIREN_V2_CONFIDENCE_THRESHOLD,
    SIREN_V2_CLASSES,
    SIREN_V2_EXPECTED_FEATURES,
    get_siren_model_v2_path,
)
from audio.feature_extractor_v2 import extract_v2_features

logger = logging.getLogger(__name__)


class SirenDetectorV2:
    """
    Siren Classifier V2 wrapping the trained Pipeline(StandardScaler, SVC).
    """
    _cached_model = None
    _cached_path = None

    def __init__(self, model_path: Optional[str] = None):
        resolved_path = model_path or get_siren_model_v2_path()
        self.model_path = resolved_path

        if not os.path.exists(resolved_path):
            raise FileNotFoundError(
                f"Siren Model V2 not found at '{resolved_path}'. "
                "Ensure 'siren_model_v2.joblib' exists in the 'models/' folder."
            )

        if SirenDetectorV2._cached_model is None or SirenDetectorV2._cached_path != resolved_path:
            SirenDetectorV2._cached_model = self._load_model(resolved_path)
            SirenDetectorV2._cached_path = resolved_path

        self.model = SirenDetectorV2._cached_model

    @staticmethod
    def _load_model(path: str):
        """Load joblib pipeline and validate its attributes."""
        try:
            model = joblib.load(path)
        except Exception as err:
            logger.exception("Failed to load Siren Model V2 from %s: %s", path, err)
            raise RuntimeError(f"Could not load Siren Model V2: {err}") from err

        if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
            raise ValueError(f"Loaded object from {path} lacks predict/predict_proba methods.")

        return model

    @classmethod
    def is_available(cls, model_path: Optional[str] = None) -> bool:
        """Check whether the V2 model file exists and is accessible."""
        path = model_path or get_siren_model_v2_path()
        return bool(path and os.path.exists(path))

    @classmethod
    def get_model_info(cls, model_path: Optional[str] = None) -> Dict[str, Any]:
        """Return metadata about the V2 model."""
        path = model_path or get_siren_model_v2_path()
        available = cls.is_available(path)
        info = {
            "model_version": "v2",
            "model_path": path,
            "available": available,
            "expected_features": SIREN_V2_EXPECTED_FEATURES,
            "classes": SIREN_V2_CLASSES,
            "pipeline": None,
        }
        if available:
            try:
                m = cls(path).model
                info["pipeline"] = str(m)
                info["classes"] = list(getattr(m, "classes_", SIREN_V2_CLASSES))
            except Exception as e:
                info["error"] = str(e)
        return info

    def predict(
        self,
        audio_source: Union[str, bytes],
        threshold: float = SIREN_V2_CONFIDENCE_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Run inference on the provided audio source.

        Args:
            audio_source: File path or raw bytes of audio file.
            threshold: Minimum confidence score to confirm a siren (default: 0.50).

        Returns:
            Dictionary with:
            - 'is_siren': bool (True only if confirmed siren class >= threshold)
            - 'confidence': float (confidence in [0.0, 1.0])
            - 'predicted_class': str ('ambulance_siren', 'fire_truck_siren', 'negative_sound')
            - 'probabilities': dict of class -> float
            - 'siren_type': str or None ('ambulance_siren', 'fire_truck_siren', None)
            - 'vehicle_type': str ('Ambulance', 'Fire Truck', 'None')
            - 'model_version': 'v2'
            - 'error': str or None
            - 'status': 'success' or 'error'
        """
        # Default safe error response: errors are NEVER interpreted as sirens
        safe_negative_response = {
            "is_siren": False,
            "confidence": 0.0,
            "predicted_class": "negative_sound",
            "probabilities": {
                "ambulance_siren": 0.0,
                "fire_truck_siren": 0.0,
                "negative_sound": 1.0,
            },
            "siren_type": None,
            "vehicle_type": "None",
            "model_version": "v2",
            "features_count": 0,
            "error": None,
            "status": "error",
        }

        # 1. Input validation
        if audio_source is None:
            res = dict(safe_negative_response)
            res["error"] = "No audio source provided."
            return res

        if isinstance(audio_source, str):
            if not os.path.exists(audio_source):
                res = dict(safe_negative_response)
                res["error"] = f"Audio file not found: {audio_source}"
                return res
            if os.path.getsize(audio_source) == 0:
                res = dict(safe_negative_response)
                res["error"] = f"Audio file is empty (0 bytes): {audio_source}"
                return res

        if isinstance(audio_source, bytes) and len(audio_source) == 0:
            res = dict(safe_negative_response)
            res["error"] = "Audio byte buffer is empty (0 bytes)."
            return res

        # 2. Audio preprocessing & feature extraction (exactly 46 features)
        try:
            features = extract_v2_features(audio_source)
        except Exception as err:
            logger.warning("Feature extraction failed: %s", err)
            res = dict(safe_negative_response)
            res["error"] = f"Audio preprocessing failed: {err}"
            return res

        if len(features) != SIREN_V2_EXPECTED_FEATURES:
            res = dict(safe_negative_response)
            res["error"] = (
                f"Feature vector length mismatch: expected {SIREN_V2_EXPECTED_FEATURES}, "
                f"got {len(features)}"
            )
            return res

        # 3. Pipeline prediction (StandardScaler + SVC)
        # Note: features are passed unscaled; pipeline's scaler applies transformation
        try:
            X = features.reshape(1, -1)
            pred_class = str(self.model.predict(X)[0])
            proba = self.model.predict_proba(X)[0]
            classes = list(self.model.classes_)
            proba_dict = {str(c): float(p) for c, p in zip(classes, proba)}
        except Exception as err:
            logger.exception("Model V2 inference failed: %s", err)
            res = dict(safe_negative_response)
            res["error"] = f"Inference failure: {err}"
            return res

        # 4. Map output classes & determine siren confirmation
        if pred_class in ("ambulance_siren", "fire_truck_siren"):
            class_conf = float(proba_dict.get(pred_class, 0.0))
            is_siren = bool(class_conf >= threshold)
            siren_type = pred_class
            vehicle_type = "Ambulance" if pred_class == "ambulance_siren" else "Fire Truck"
            confidence = class_conf
        else:
            # negative_sound or unknown non-siren class
            is_siren = False
            siren_type = None
            vehicle_type = "None"
            confidence = float(proba_dict.get(pred_class, 1.0))

        return {
            "is_siren": is_siren,
            "confidence": round(confidence, 3),
            "predicted_class": pred_class,
            "probabilities": {k: round(v, 4) for k, v in proba_dict.items()},
            "siren_type": siren_type,
            "vehicle_type": vehicle_type,
            "model_version": "v2",
            "features_count": len(features),
            "error": None,
            "status": "success",
        }
