"""
Gemeinsames Interface für alle Modelle. JEDE Modell-Klasse kapselt ihr
eigenes Preprocessing + Forward-Pass + Pooling zu EINEM Clip-Embedding.
Der Rest der Pipeline (Perturbationen, Retrieval, Metriken) bleibt für
alle Modelle identisch -> fairer Vergleich, austauschbare Modelle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class VideoFeatureExtractor(ABC):
    name: str = "base"

    @abstractmethod
    def preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [T, C, H, W] float 0..1 (roh, nach Perturbation)
        -> modellspezifisches Input-Format (Resize, Normalisierung,
        ggf. Resampling der Frame-Anzahl)."""

    @abstractmethod
    def forward(self, model_input) -> torch.Tensor:
        """-> ein Embedding-Vektor, shape [D]."""

    @torch.no_grad()
    def embed(self, frames: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(frames)
        emb = self.forward(x)
        return F.normalize(emb.float(), dim=-1)  # L2-Norm -> Cosine = Dot-Product
