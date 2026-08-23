"""
Vorbereitungsskript für Kinetics-400 (cvdfoundation-Mirror).

cvdfoundation liefert Videos FLACH pro Split (kein Klassenordner), plus eine
separate Annotations-CSV mit Spalten: label, youtube_id, time_start, time_end,
split. Der Dateiname folgt der Standard-Konvention:
    <youtube_id>_<time_start:06d>_<time_end:06d>.mp4

video_dataset.py (build_dataset_subset) erwartet dagegen das Layout
<root>/<split>/<class_name>/<video>.mp4.

Zwei Punkte, die dieses Skript automatisch handhabt:
  1. Windows-Kompatibilität: Symlinks brauchen unter Windows Admin-Rechte
     oder aktivierten Entwicklermodus. Falls das fehlschlägt, wird
     automatisch auf echtes Kopieren zurückgefallen (mit Warnung).
  2. Es wird NICHT der komplette Split verlinkt/kopiert, sondern direkt nur
     ein Subset von --n_classes Klassen x --n_videos_per_class Videos
     (deterministisch über --seed) - spart Zeit und v.a. bei Kopieren
     unnötigen Speicherplatz.

Erwartete Eingabestruktur (nach dem Download + Entpacken):
    <kinetics_raw_root>/
        val/                          # alle .mp4 Dateien flach
        annotations/val.csv           # label,youtube_id,time_start,time_end,split

Aufruf:
    python data/prepare_kinetics.py \
        --raw_root /pfad/zu/deinem/kinetics-dataset/k400 \
        --output_root /data/kinetics400_subset \
        --split val --n_classes 25 --n_videos_per_class 20
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

_WARNED_ABOUT_COPY = False


def sanitize(label: str) -> str:
    """'eating watermelon' -> 'eating_watermelon' (konsistente, dateisystemsichere
    Klassennamen)."""
    t = label.lower().strip()
    t = re.sub(r"[^a-z0-9]+", "_", t)
    return t.strip("_")


def expected_filename(youtube_id: str, time_start: str, time_end: str) -> str:
    return f"{youtube_id}_{int(time_start):06d}_{int(time_end):06d}.mp4"


def link_or_copy(src: Path, dst: Path) -> None:
    """Versucht einen Symlink, fällt bei Fehlern (typischerweise fehlende
    Windows-Rechte) automatisch auf eine echte Kopie zurück."""
    global _WARNED_ABOUT_COPY
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        if not _WARNED_ABOUT_COPY:
            print("Hinweis: Symlinks nicht möglich (unter Windows braucht das "
                  "Admin-Rechte oder aktivierten Entwicklermodus) - kopiere "
                  "Dateien stattdessen. Das ist langsamer, funktioniert aber "
                  "ohne Sonderrechte.")
            _WARNED_ABOUT_COPY = True
        shutil.copy2(src, dst)


def build_subset(raw_root: Path, output_root: Path, split: str,
                  n_classes: int, n_videos_per_class: int, seed: int) -> None:
    ann_path = raw_root / "annotations" / f"{split}.csv"
    if not ann_path.exists():
        ann_path = raw_root / f"{split}.csv"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"Keine Annotations-CSV gefunden (gesucht: {raw_root / 'annotations' / f'{split}.csv'} "
            f"und {raw_root / f'{split}.csv'}). Pruefe --raw_root."
        )

    video_dir = raw_root / split
    if not video_dir.exists():
        raise FileNotFoundError(f"{video_dir} existiert nicht. Pruefe --raw_root.")

    # Annotation einlesen und nach vorhandenen Dateien filtern, gruppiert nach Klasse
    videos_by_class: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    with open(ann_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls_name = sanitize(row["label"])
            fname = expected_filename(row["youtube_id"], row["time_start"], row["time_end"])
            src = video_dir / fname
            if src.exists():
                videos_by_class[cls_name].append((fname, src))

    if not videos_by_class:
        raise RuntimeError(
            "Keine der in der Annotation referenzierten Videos wurde lokal "
            f"gefunden in {video_dir}. Pruefe, ob der Download/die Extraktion "
            "vollstaendig lief."
        )

    rng = random.Random(seed)
    available_classes = [c for c, vids in videos_by_class.items() if len(vids) > 0]
    chosen_classes = rng.sample(available_classes, min(n_classes, len(available_classes)))

    out_split_dir = output_root / split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    n_linked = 0
    for cls_name in chosen_classes:
        vids = videos_by_class[cls_name]
        chosen = rng.sample(vids, min(n_videos_per_class, len(vids)))
        cls_dir = out_split_dir / cls_name
        cls_dir.mkdir(exist_ok=True)
        for fname, src in chosen:
            link_or_copy(src, cls_dir / fname)
            n_linked += 1

    print(f"Fertig: {n_linked} Videos in {len(chosen_classes)} Klassen bereitgestellt "
          f"unter {out_split_dir}")
    print(f"-> config.py: DATASETS['kinetics400_subset']['root'] = Path('{output_root}')")
    print(f"   (n_classes={n_classes}, n_videos_per_class={n_videos_per_class} bereits hier "
          f"angewendet - video_dataset.py sampelt aus genau diesem Subset, ohne weitere Reduktion)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=Path, required=True,
                         help="Wurzelordner des entpackten cvdfoundation-Downloads (enthaelt val/ und annotations/)")
    parser.add_argument("--output_root", type=Path, required=True,
                         help="Zielordner fuer die class_name/-Struktur")
    parser.add_argument("--split", default="val", choices=["val", "train", "test"])
    parser.add_argument("--n_classes", type=int, default=25)
    parser.add_argument("--n_videos_per_class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_subset(args.raw_root, args.output_root, args.split,
                 args.n_classes, args.n_videos_per_class, args.seed)
