"""
Perturbationen für die Sensitivitätsanalyse.

Alle Funktionen: Input/Output [T, C, H, W] float Tensor in [0, 1].
Deterministisch über `seed`, damit dieselbe Störung reproduzierbar auf
denselben Clip angewendet wird (wichtig für Embedding-Shift-Vergleiche).
"""

from __future__ import annotations

import torch


def perturb_clean(frames: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Kontrollbedingung / Referenz."""
    return frames


def perturb_frame_repeat(frames: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Härteste temporale Störung: ein einzelnes (mittleres) Frame wird
    T-mal wiederholt. Entfernt JEDE Bewegungsinformation, Appearance
    bleibt exakt erhalten -> unterer Anker 'kein Zeitbezug mehr vorhanden'."""
    t = frames.shape[0]
    repeat_idx = t // 2
    single = frames[repeat_idx : repeat_idx + 1]
    return single.repeat(t, 1, 1, 1)


def perturb_shuffle(frames: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Zerstört die Reihenfolge vollständig, behält alle Einzelbilder."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(frames.shape[0], generator=g)
    return frames[perm]


def perturb_reverse(frames: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Invertiert die Abspielrichtung. Lokale Nachbarschaften bleiben
    erhalten -> schwächere Störung als Shuffle, testet primär Kausalität/Richtung."""
    return frames.flip(0)


def perturb_blur(frames: torch.Tensor, seed: int = 0, kernel_size: int = 15,
                  sigma: float = 5.0) -> torch.Tensor:
    """Starker Gaussian Blur: zerstört Feindetails/Textur, grobe
    Bewegungskontur bleibt erhalten."""
    import torchvision.transforms.functional as TF

    return torch.stack([TF.gaussian_blur(f, kernel_size, sigma) for f in frames])


PERTURBATIONS = {
    "clean": perturb_clean,
    "frame_repeat": perturb_frame_repeat,
    "shuffle": perturb_shuffle,
    "reverse": perturb_reverse,
    "blur": perturb_blur,
}
