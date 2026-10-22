"""
Audio processing and siren classification module.
Provides both original SirenDetector (V1) and newly integrated SirenDetectorV2.
"""

from typing import Optional
from audio.siren_detector import SirenDetector
from audio.siren_detector_v2 import SirenDetectorV2
from utils.config import SIREN_ACTIVE_VERSION


def get_siren_detector(version: Optional[str] = None):
    """
    Factory to retrieve the appropriate siren detector based on version.
    Defaults to SIREN_ACTIVE_VERSION ('v1') to preserve production behavior.
    Falls back gracefully if the requested version is unavailable.

    Args:
        version: 'v1' or 'v2'. If None, uses SIREN_ACTIVE_VERSION.
    """
    selected_version = (version or SIREN_ACTIVE_VERSION).lower()
    if selected_version == "v2":
        if SirenDetectorV2.is_available():
            try:
                return SirenDetectorV2()
            except Exception:
                return SirenDetector()
        return SirenDetector()

    return SirenDetector()


__all__ = ["SirenDetector", "SirenDetectorV2", "get_siren_detector"]
