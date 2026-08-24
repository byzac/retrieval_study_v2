"""
DisMo Wrapper - vollwertiges Kernmodell (nicht mehr Stretch-Goal).

Nutzt den offiziell bereitgestellten Motion-Extractor über torch.hub
(https://github.com/CompVis/DisMo, README-Abschnitt "Motion Extraction").
Das ist NUR der Motion-Encoder, nicht das CogVideoX-Motion-Transfer-Modell
- genau der Teil, der für uns als Repräsentationsmodell relevant ist.

Wichtige Eigenheiten laut Repo, die das preprocess() berücksichtigt:
  - Input-Format: [B, T, H, W, C], Werte in [-1, 1] (NICHT [B, C, T, H, W]
    wie die meisten anderen Video-Modelle!)
  - DisMo wurde auf Clips der Länge 8 trainiert; `forward_sliding()`
    berechnet pro Zeitschritt eine Bewegungsrepräsentation und läuft
    dafür in einem Sliding-Window über den Clip. Die Ausgabe hat T-4
    Zeitschritte (max. Prädiktionsdistanz 4 beim Training).
  - Wir mitteln über diese T-4 Motion-Embeddings zu einem einzigen
    Clip-Level-Vektor (analog zum Mean-Pooling bei DINOv2/V-JEPA), um
    mit dem gemeinsamen Retrieval-Protokoll kompatibel zu bleiben.

Hypothese für die Diskussion (Kap. 6): DisMo sollte von allen Modellen
am selektivsten reagieren - starker Embedding-Shift bei temporalen
Störungen (frame_repeat, shuffle), kaum Shift bei rein visuellen
Störungen (blur), weil die Repräsentation explizit von Appearance
entkoppelt trainiert wurde.
"""

from __future__ import annotations

import torch

from .base import VideoFeatureExtractor


class DisMoExtractor(VideoFeatureExtractor):
    """
    ACHTUNG Auflösung: Das offizielle Beispiel im Repo nutzt 256x256
    (torch.rand((B, T, 256, 256, 3))), während config.FRAME_SIZE = 224
    ist (wie für DINOv2/V-JEPA üblich). Vor dem ersten echten Lauf
    unbedingt prüfen, ob DisMo mit 224x224 klarkommt (ViT-Backbones mit
    Patch-Embedding sind oft auflösungsflexibel) oder ob hier explizit
    auf 256x256 resized werden muss - siehe model.patch_size o.ä. im
    geladenen Objekt, oder einfach testweise mit 256 durchlaufen lassen.
    """

    name = "dismo_motion_extractor_large"

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            torch.hub.load("CompVis/DisMo", "motion_extractor_large", trust_repo=True)
            .to(self.device)
            .eval()
        )

    def preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [T, C, H, W] float in [0, 1] -> DisMo erwartet [B, T, H, W, C] in [-1, 1]
        x = frames.to(self.device)
        x = x.permute(0, 2, 3, 1)          # [T, H, W, C]
        x = x.mul(2).sub(1)                 # [0,1] -> [-1,1]
        return x.unsqueeze(0)               # [1, T, H, W, C]

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            motion_embeddings = self.model.forward_sliding(model_input)  # [1, T-4, D]
        return motion_embeddings.mean(dim=1).squeeze(0)  # Mean-Pool über Zeit -> [D]
