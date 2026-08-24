"""
V-JEPA2 Wrapper - über die HuggingFace-Integration (transformers), NICHT
über torch.hub.

Hintergrund des Strategiewechsels: Der offizielle torch.hub-Endpunkt
(facebookresearch/vjepa2) hat aktuell zwei Probleme, die nicht an unserem
Code liegen: (1) der optionale Preprocessor braucht `cv2`, das wir nicht
in den Requirements haben, und (2) der Checkpoint-Download zeigt auf eine
offensichtlich interne Platzhalter-URL (`localhost:8300`) statt der
echten Download-Adresse - ein Bug im Repo selbst, nicht behebbar von
unserer Seite. Die HuggingFace-Integration hostet dieselben Gewichte
direkt selbst und umgeht dieses kaputte hubconf.py komplett.

Benötigt: `pip install -U transformers`.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import VideoFeatureExtractor

_HF_REPO = "facebook/vjepa2-vitl-fpc64-256"  # ViT-L, 256px, kleinste sinnvolle Variante


class VJEPAExtractor(VideoFeatureExtractor):
    name = "vjepa2-vitl-fpc64-256 (HuggingFace)"

    def __init__(self, device: str | None = None, hf_repo: str = _HF_REPO):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        from transformers import AutoModel, AutoVideoProcessor

        self.model = AutoModel.from_pretrained(hf_repo).to(self.device).eval()
        self.processor = AutoVideoProcessor.from_pretrained(hf_repo)

    def preprocess(self, frames: torch.Tensor):
        # frames: [T, C, H, W] float in [0, 1] -> HF-Prozessor erwartet
        # rohe Frame-Arrays (uint8, HWC), Normalisierung macht er selbst
        frames_np = (frames.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        video_list = [frames_np[t] for t in range(frames_np.shape[0])]

        inputs = self.processor([video_list], return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def forward(self, model_input: dict) -> torch.Tensor:
        with torch.no_grad():
            output = self.model(**model_input)
        return output.last_hidden_state.mean(dim=1).squeeze(0)