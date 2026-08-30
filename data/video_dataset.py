"""
Dataset-Loading für lokal gespeicherte Video-Subsets.

Erwartetes Layout (Standard nach Download der offiziellen Splits):
    <root>/<split>/<class_name>/<video_id>.mp4

WICHTIGER SPEICHER-FIX: VideoSample speichert nur noch den Dateipfad,
KEINE vorab decodierten Frames mehr. Vorher wurden bei build_dataset_subset()
alle Videos eines Subsets sofort vollständig decodiert und im Speicher
gesammelt - bei größeren Subsets (z.B. 100 Klassen x 50 Videos, ~5000
Videos) führte das zu einem Speicherbedarf von >90 GB (bei 32 Frames x
224x224x3 float32 pro Video ≈ 19 MB/Video) und damit zu einem Absturz auf
Systemen mit weniger RAM. Jetzt werden nur Pfade gesammelt; das eigentliche
Decodieren passiert individuell und speicherschonend in build_bank()
(siehe retrieval.py), direkt vor der Embedding-Berechnung, sodass jeweils
nur EIN Video gleichzeitig als decodierter Tensor im Speicher liegt.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class VideoSample:
    video_id: str
    label: str
    path: Path  # Pfad statt vorab geladener Frames (siehe Speicher-Fix oben)


def _uniform_frame_indices(total_frames: int, n_frames: int) -> list[int]:
    """Gleichmäßiges Sampling von n_frames Indizes über die gesamte Cliplänge.
    Bei Clips mit weniger Frames als n_frames wird mit Wiederholung aufgefüllt."""
    if total_frames >= n_frames:
        return np.linspace(0, total_frames - 1, n_frames).round().astype(int).tolist()
    # kurzer Clip: alle Frames nehmen + letztes Frame auffüllen
    idx = list(range(total_frames))
    idx += [total_frames - 1] * (n_frames - total_frames)
    return idx


def load_video_frames(path: Path, n_frames: int, frame_size: int) -> torch.Tensor:
    """Lädt n_frames gleichmäßig gesampelte Frames aus einem Video.
    Nutzt decord (schnell, GPU-fähig); Fallback auf torchvision.io bei
    fehlendem Modul ODER bei Laufzeit-Dekodierfehlern."""
    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(str(path))
        indices = _uniform_frame_indices(len(vr), n_frames)
        frames = vr.get_batch(indices)
        frames = frames.permute(0, 3, 1, 2).float() / 255.0
    except ImportError:
        import torchvision

        video, _, _ = torchvision.io.read_video(str(path), output_format="TCHW")
        indices = _uniform_frame_indices(video.shape[0], n_frames)
        frames = video[indices].float() / 255.0
    except Exception as e:
        try:
            import torchvision

            video, _, _ = torchvision.io.read_video(str(path), output_format="TCHW")
            indices = _uniform_frame_indices(video.shape[0], n_frames)
            frames = video[indices].float() / 255.0
        except Exception:
            raise RuntimeError(
                f"Video konnte weder mit decord noch mit torchvision.io gelesen "
                f"werden: {path} (decord-Fehler: {e})"
            ) from e

    frames = torch.nn.functional.interpolate(
        frames, size=(frame_size, frame_size), mode="bilinear", align_corners=False
    )
    return frames


def build_flat_subset(
    root: Path,
    n_videos: int,
    n_frames: int,
    frame_size: int,
    seed: int = 42,
) -> list[VideoSample]:
    """Für Ordner OHNE Klassenstruktur (flach, keine Label-Info lokal
    vorhanden) - z.B. für einen ersten technischen Smoke-Test, bevor die
    echte annotierte Teilmenge vorliegt.

    Jedem Video wird eine EINDEUTIGE Pseudo-Klasse (seine eigene ID)
    zugewiesen. Das macht klassenbasierte Metriken (reference_retrieval,
    cross_condition_retrieval) bedeutungslos - dafür bleiben
    identity_retrieval und embedding_shift voll aussagekräftig, weil
    die auf Video-Identität statt Klassenzugehörigkeit beruhen. Genau
    diese beiden reichen aus, um zu prüfen, ob die Pipeline technisch
    korrekt läuft.

    Hinweis: n_frames/frame_size werden hier nicht mehr direkt zum Laden
    verwendet (siehe Speicher-Fix oben), bleiben aber in der Signatur für
    Abwärtskompatibilität mit smoke_test.py.
    """
    rng = random.Random(seed)
    if not root.exists():
        raise FileNotFoundError(f"{root} existiert nicht.")

    all_videos = sorted(
        [p for p in root.iterdir() if p.suffix.lower() in (".mp4", ".avi", ".webm")]
    )
    if not all_videos:
        raise FileNotFoundError(f"Keine Videodateien (.mp4/.avi/.webm) in {root} gefunden.")

    chosen = rng.sample(all_videos, min(n_videos, len(all_videos)))

    samples: list[VideoSample] = [
        VideoSample(video_id=vpath.stem, label=vpath.stem, path=vpath) for vpath in chosen
    ]

    print(f"[flat, ohne Labels] {len(samples)} Video-Pfade gesammelt aus {root}. "
          f"Nur identity_retrieval/embedding_shift sind hier aussagekräftig.")
    return samples


def build_dataset_subset(
    root: Path,
    split: str,
    n_classes: int,
    n_videos_per_class: int,
    n_frames: int,
    frame_size: int,
    seed: int = 42,
) -> list[VideoSample]:
    """Sampelt ein kontrolliertes Subset: n_classes Klassen, davon je
    n_videos_per_class Videos. Deterministisch über `seed`, damit über
    alle Modelle/Perturbationen exakt dieselben Videos verwendet werden
    (wichtig für Vergleichbarkeit der Matrix!).

    WICHTIG (Speicher-Fix): Sammelt nur Dateipfade, KEINE decodierten
    Frames. Das eigentliche Decodieren (inkl. Fehlerbehandlung für
    defekte Videos) passiert individuell in build_bank() (retrieval.py).
    n_frames/frame_size werden hier nicht mehr direkt genutzt, bleiben
    aber Teil der Signatur, damit run_experiment.py unverändert bleibt.
    """
    rng = random.Random(seed)
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"{split_dir} existiert nicht. Bitte config.py DATASETS[...]['root'] "
            f"auf den tatsächlichen Speicherort anpassen."
        )

    all_classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    chosen_classes = rng.sample(all_classes, min(n_classes, len(all_classes)))

    samples: list[VideoSample] = []
    for cls in chosen_classes:
        cls_dir = split_dir / cls
        videos = sorted(
            [p for p in cls_dir.iterdir() if p.suffix.lower() in (".mp4", ".avi", ".webm")]
        )
        chosen_videos = rng.sample(videos, min(n_videos_per_class, len(videos)))
        for vpath in chosen_videos:
            samples.append(VideoSample(video_id=vpath.stem, label=cls, path=vpath))

    print(f"[{root.name}] Subset zusammengestellt: {len(chosen_classes)} Klassen, "
          f"{len(samples)} Video-Pfade (Decodierung erfolgt individuell pro Lauf, "
          f"siehe build_bank() in retrieval.py).")
    return samples