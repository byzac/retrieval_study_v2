"""
SlowFast (PyTorchVideo, torch.hub) als 3D-CNN mit explizitem Motion-Pfad.
Slow-Pathway: wenige Frames, hohe Kanalzahl -> Appearance/Semantik.
Fast-Pathway: viele Frames, wenig Kanäle -> feine Bewegungsdynamik.

WICHTIG: Nutzt den letzten Layer VOR dem Klassifikations-Head als
Embedding (Global-Average-Pool-Feature, 2304-dim bei slowfast_r50).
Falls das pytorchvideo-Repo seine Hub-API ändert, muss ggf. der Layer-
Name in `_register_hook` angepasst werden (`print(self.model)` hilft).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import VideoFeatureExtractor

# Standard-Normalisierung wie im offiziellen PyTorchVideo-Tutorial
_MEAN = [0.45, 0.45, 0.45]
_STD = [0.225, 0.225, 0.225]
_SIDE_SIZE = 256
_CROP_SIZE = 256
_NUM_FRAMES = 32     # Fast-Pathway-Länge
_ALPHA = 4           # Slow-Pathway = jedes 4. Frame der Fast-Pathway


class _PackPathway(torch.nn.Module):
    """Erzeugt aus einem [C, T, H, W]-Clip die zwei SlowFast-Inputs
    [slow_pathway, fast_pathway], wie vom Modell erwartet."""

    def forward(self, frames: torch.Tensor) -> list[torch.Tensor]:
        fast = frames
        slow_idx = torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // _ALPHA).long()
        slow = torch.index_select(frames, 1, slow_idx)
        return [slow, fast]


class SlowFastExtractor(VideoFeatureExtractor):
    name = "slowfast_r50"

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
            .to(self.device)
            .eval()
        )
        self._features = {}
        self._register_hook()
        self.pack_pathway = _PackPathway()

    def _register_hook(self):
        """Hook auf den Pooling-Layer VOR dem finalen FC-Klassifikationskopf,
        um das Embedding statt der Klassenlogits zu bekommen."""

        def hook(module, inp, out):
            self._features["pooled"] = out

        # blocks[-1] ist der ResNetBasicHead; .pool ist das Pooling davor
        target_layer = self.model.blocks[-1].pool
        target_layer.register_forward_hook(hook)

    def preprocess(self, frames: torch.Tensor):
        # frames: [T, C, H, W] (T = N_FRAMES_RAW aus config) -> [C, T, H, W]
        x = frames.permute(1, 0, 2, 3).to(self.device)

        x = F.interpolate(
            x.unsqueeze(0), size=(_NUM_FRAMES, _SIDE_SIZE, _SIDE_SIZE),
            mode="trilinear", align_corners=False,
        ).squeeze(0)

        mean = torch.tensor(_MEAN).view(3, 1, 1, 1).to(self.device)
        std = torch.tensor(_STD).view(3, 1, 1, 1).to(self.device)
        x = (x - mean) / std

        slow, fast = self.pack_pathway(x)
        return [slow.unsqueeze(0), fast.unsqueeze(0)]  # Batch-Dim ergänzen

    def forward(self, model_input) -> torch.Tensor:
        self._features.clear()
        _ = self.model(model_input)
        pooled = self._features["pooled"]  # [1, D, 1, 1, 1] o.ä.
        return pooled.flatten()
