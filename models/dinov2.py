"""
DINOv2 als Appearance-Baseline: reines Bildmodell, kein Zeitbezug.
Embedding = Mean-Pool der CLS-Token über alle Frames.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import VideoFeatureExtractor


class DINOv2Extractor(VideoFeatureExtractor):
    name = "dinov2_vitb14"

    def __init__(self, device: str | None = None, variant: str = "dinov2_vitb14"):
        # Für CPU-Smoke-Tests empfiehlt sich variant="dinov2_vits14" (klein,
        # ~5x schneller auf CPU als die Base-Variante) - für die finalen
        # GPU-Läufe bei "dinov2_vitb14" bleiben, wie im Betreuer-Setup vorgesehen.
        self.name = variant
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            torch.hub.load("facebookresearch/dinov2", variant)
            .to(self.device)
            .eval()
        )
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        x = frames.to(self.device)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return x

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        # model_input: [T, 3, 224, 224] -> DINOv2 Forward pro Frame
        per_frame_features = self.model(model_input)  # [T, D]
        return per_frame_features.mean(dim=0)          # Mean-Pooling über Zeit
