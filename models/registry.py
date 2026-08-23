"""Factory: Modellname (String) -> VideoFeatureExtractor-Instanz.
Zentrale Stelle, um die Matrix in run_experiment.py generisch zu halten."""

from __future__ import annotations

from .base import VideoFeatureExtractor


def get_extractor(name: str) -> VideoFeatureExtractor:
    if name == "dinov2":
        from .dinov2 import DINOv2Extractor
        return DINOv2Extractor()
    if name == "slowfast":
        from .slowfast import SlowFastExtractor
        return SlowFastExtractor()
    if name == "vjepa":
        from .vjepa import VJEPAExtractor
        return VJEPAExtractor()
    if name == "dismo":
        from .dismo import DisMoExtractor
        return DisMoExtractor()
    raise ValueError(f"Unbekanntes Modell: {name}")
