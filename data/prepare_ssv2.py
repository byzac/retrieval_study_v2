"""
Vorbereitungsskript fuer Something-Something V2.

SSv2 liefert Videos FLACH nummeriert (1.webm, 2.webm, ...) plus separate
JSON-Annotationsdateien - anders als Kinetics-artiges class_name/-Layout.

Zwei Punkte, die dieses Skript automatisch handhabt (siehe auch
prepare_kinetics.py, identisches Prinzip):
  1. Windows-Kompatibilitaet: Symlinks brauchen unter Windows Admin-Rechte
     oder aktivierten Entwicklermodus. Falls das fehlschlaegt, wird
     automatisch auf echtes Kopieren zurueckgefallen.
  2. Es wird direkt nur ein Subset von --n_classes Klassen x
     --n_videos_per_class Videos bereitgestellt statt aller ~220k Videos.

Erwartete Eingabestruktur:
    <ssv2_raw_root>/
        20bn-something-something-v2/          # alle .webm Dateien flach
        something-something-v2-train.json
        something-something-v2-validation.json
        something-something-v2-labels.json    # Template -> Klassen-ID (ungenutzt hier)

Aufruf:
    python data/prepare_ssv2.py \
        --raw_root /pfad/zu/deinem/ssv2/download \
        --output_root /data/ssv2_subset \
        --split val --n_classes 25 --n_videos_per_class 20
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

_WARNED_ABOUT_COPY = False


def sanitize(template: str) -> str:
    """'Pushing [something] from left to right' -> 'pushing_something_from_left_to_right'."""
    t = template.lower()
    t = re.sub(r"[\[\]]", "", t)
    t = re.sub(r"[^a-z0-9]+", "_", t)
    return t.strip("_")


def link_or_copy(src: Path, dst: Path) -> None:
    """Versucht einen Symlink, faellt bei Fehlern (typischerweise fehlende
    Windows-Rechte) automatisch auf eine echte Kopie zurueck."""
    global _WARNED_ABOUT_COPY
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        if not _WARNED_ABOUT_COPY:
            print("Hinweis: Symlinks nicht moeglich (unter Windows braucht das "
                  "Admin-Rechte oder aktivierten Entwicklermodus) - kopiere "
                  "Dateien stattdessen. Das ist langsamer, funktioniert aber "
                  "ohne Sonderrechte.")
            _WARNED_ABOUT_COPY = True
        shutil.copy2(src, dst)


def build_subset(raw_root: Path, output_root: Path, split: str,
                  n_classes: int, n_videos_per_class: int, seed: int) -> None:
    split_map = {"val": "validation", "train": "train"}
    ann_split = split_map.get(split, split)

    ann_path = raw_root / f"something-something-v2-{ann_split}.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"{ann_path} nicht gefunden. Pruefe --raw_root und ob die "
            f"Annotationsdateien wie im Download entpackt vorliegen."
        )

    video_dir_candidates = [
        raw_root / "20bn-something-something-v2",
        raw_root / "videos",
        raw_root,
    ]
    video_dir = next((d for d in video_dir_candidates if d.exists() and any(d.glob("*.webm"))), None)
    if video_dir is None:
        raise FileNotFoundError(
            "Konnte den Ordner mit den .webm-Dateien nicht finden. Bitte "
            "video_dir_candidates in diesem Skript um deinen tatsaechlichen "
            "Pfad ergaenzen."
        )

    with open(ann_path) as f:
        annotations = json.load(f)

    videos_by_class: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for entry in annotations:
        video_id = entry["id"]
        cls_name = sanitize(entry["template"])
        src = video_dir / f"{video_id}.webm"
        if src.exists():
            videos_by_class[cls_name].append((f"{video_id}.webm", src))

    if not videos_by_class:
        raise RuntimeError(
            f"Keine der in der Annotation referenzierten Videos wurde lokal in "
            f"{video_dir} gefunden. Pruefe, ob der Download vollstaendig entpackt wurde."
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
    print(f"-> config.py: DATASETS['ssv2_subset']['root'] = Path('{output_root}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=Path, required=True,
                         help="Ordner mit den heruntergeladenen SSv2-Rohdaten (Videos + JSONs)")
    parser.add_argument("--output_root", type=Path, required=True,
                         help="Zielordner fuer die class_name/-Struktur")
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument("--n_classes", type=int, default=25)
    parser.add_argument("--n_videos_per_class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_subset(args.raw_root, args.output_root, args.split,
                 args.n_classes, args.n_videos_per_class, args.seed)
