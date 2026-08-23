"""
V-JEPA(2) Wrapper.

WICHTIG - höherer Anpassungsaufwand als DINOv2/SlowFast:
V-JEPA ist kein torch.hub-Modell mit fester API, sondern wird über das
Repo https://github.com/facebookresearch/vjepa2 geladen. Die Repo-API
kann sich zwischen Commits ändern. Dieser Wrapper zeigt das GRUNDPRINZIP
(Checkpoint laden, Encoder-Only-Forward, Mean-Pool über Patch-Tokens),
du musst vor dem ersten Lauf:
  1. das Repo klonen und in den Pfad aufnehmen (sys.path o.ä.)
  2. den exakten Import-Pfad für den Encoder-Builder prüfen
     (z.B. `from src.models.vision_transformer import vit_large` o.ä. -
     Name variiert je nach Repo-Version, siehe deren README/Beispielscript)
  3. den Checkpoint-Pfad in `checkpoint_path` setzen (Download laut
     Repo-README, Link von deinem Betreuer: ai.meta.com/research/vjepa)

Falls das zeitlich zu aufwändig wird, ist ein sauberer Fallback, V-JEPA
über die HuggingFace-Integration zu laden (`transformers`, sofern für
die verwendete V-JEPA2-Checkpointversion verfügbar) - dann vereinfacht
sich preprocess()/forward() stark, analog zu models/dinov2.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from .base import VideoFeatureExtractor

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
_CROP_SIZE = 224
_NUM_FRAMES = 16  # typische V-JEPA Clip-Länge, ggf. an Checkpoint anpassen


class VJEPAExtractor(VideoFeatureExtractor):
    name = "vjepa"

    def __init__(
        self,
        device: str | None = None,
        repo_path: str = "/path/to/vjepa2",       # <-- anpassen
        checkpoint_path: str = "/path/to/vjepa2_ckpt.pt",  # <-- anpassen
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        sys.path.insert(0, repo_path)
        try:
            # TODO: exakten Importpfad je nach Repo-Version verifizieren
            from src.models.vision_transformer import vit_large  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Konnte V-JEPA-Encoder nicht importieren. Prüfe repo_path "
                "und den Importpfad in models/vjepa.py gegen das aktuelle "
                "facebookresearch/vjepa2 Repo (README / evals-Skripte "
                "zeigen den korrekten Builder-Aufruf)."
            ) from e

        self.model = vit_large(img_size=_CROP_SIZE, num_frames=_NUM_FRAMES)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("encoder", ckpt.get("model", ckpt))
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device).eval()

        self.mean = torch.tensor(_MEAN).view(1, 3, 1, 1, 1).to(self.device)
        self.std = torch.tensor(_STD).view(1, 3, 1, 1, 1).to(self.device)

    def preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [T, C, H, W] -> [1, C, T, H, W]
        x = frames.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
        x = F.interpolate(
            x, size=(_NUM_FRAMES, _CROP_SIZE, _CROP_SIZE),
            mode="trilinear", align_corners=False,
        )
        x = (x - self.mean) / self.std
        return x

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        # Encoder-Only-Forward -> Patch-Token-Sequenz [1, N, D]
        tokens = self.model(model_input)
        return tokens.mean(dim=1).squeeze(0)  # Mean-Pool über alle Tokens -> [D]
