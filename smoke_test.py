"""
Smoke-Test: prüft, ob die Pipeline technisch korrekt läuft - BEVOR die
echte annotierte Datenbasis vorliegt und BEVOR du auf GPU umsteigst.

Bewusst klein gehalten für CPU:
  - nur DINOv2 in der kleinen Variante (dinov2_vits14) - schnellstes Modell
  - nur wenige Videos (Default: 6)
  - nur 2 von 4 Perturbationen als Stichprobe (frame_repeat, shuffle) -
    reicht, um Perturbationslogik + Modellintegration zu prüfen; blur/
    reverse sind vom Code-Pfad her identisch und daher fürs Smoke-Testing
    redundant
  - nur identity_retrieval + embedding_shift (siehe build_flat_subset,
    diese zwei brauchen keine echten Klassenlabels)

Aufruf:
    python smoke_test.py --video_dir /pfad/zu/deinem/kinetics/subset

Was ein erfolgreicher Lauf bestätigt:
  1. Videos lassen sich laden und in Frames zerlegen (decord/torchvision)
  2. Perturbationen laufen ohne Shape-Fehler durch
  3. Das Modell lässt sich laden und liefert Embeddings der erwarteten Form
  4. Das Retrieval-/Metrik-Rechenwerk läuft fehlerfrei durch

Was es NICHT bestätigt: ob die inhaltlichen Ergebnisse "gut" sind - das
kann es auf 6 zufälligen, unlabeled Videos gar nicht leisten. Das ist
ausschließlich ein technischer Funktionstest.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from data.video_dataset import build_flat_subset
from models.dinov2 import DINOv2Extractor
from perturbations import PERTURBATIONS
from retrieval import build_bank, embedding_shift, identity_retrieval

SMOKE_TEST_PERTURBATIONS = ["frame_repeat", "shuffle"]  # Stichprobe reicht für den Funktionstest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=Path, required=True)
    parser.add_argument("--n_videos", type=int, default=6)
    parser.add_argument("--n_frames", type=int, default=16,  # weniger als die 32 der Hauptstudie -> schneller
                         help="Weniger Frames als in config.N_FRAMES_RAW (32) für schnelleren CPU-Test")
    parser.add_argument("--frame_size", type=int, default=224)
    args = parser.parse_args()

    print(f"=== Smoke-Test: lade {args.n_videos} Videos aus {args.video_dir} ===")
    t0 = time.time()
    samples = build_flat_subset(
        root=args.video_dir,
        n_videos=args.n_videos,
        n_frames=args.n_frames,
        frame_size=args.frame_size,
    )
    print(f"  Laden abgeschlossen ({time.time() - t0:.1f}s)")

    print("\n=== Lade DINOv2 (kleine Variante, CPU) ===")
    t0 = time.time()
    extractor = DINOv2Extractor(variant="dinov2_vits14")
    print(f"  Modell geladen ({time.time() - t0:.1f}s)")

    print("\n=== Baue Clean-Embedding-Bank ===")
    t0 = time.time()
    clean_bank = build_bank(extractor, samples, PERTURBATIONS["clean"])
    print(f"  {len(clean_bank.embeddings)} Embeddings, Dim={clean_bank.embeddings[0].shape[0]} "
          f"({time.time() - t0:.1f}s)")

    all_ok = True
    for pert_name in SMOKE_TEST_PERTURBATIONS:
        print(f"\n=== Perturbation: {pert_name} ===")
        t0 = time.time()
        try:
            perturbed_bank = build_bank(extractor, samples, PERTURBATIONS[pert_name])
            id_metrics = identity_retrieval(clean_bank, perturbed_bank)
            shift_metrics = embedding_shift(clean_bank, perturbed_bank)

            print(f"  identity_retrieval: {id_metrics}")
            print(f"  embedding_shift: mean={shift_metrics['mean_cosine_shift']:.4f}, "
                  f"std={shift_metrics['std_cosine_shift']:.4f}")
            print(f"  ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"  !! FEHLER bei {pert_name}: {e}")
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("SMOKE TEST BESTANDEN - Pipeline läuft technisch korrekt.")
        print("Nächste Schritte: echte Klassenlabels besorgen (Kinetics-CSV),")
        print("dann mit `python run_experiment.py --models dinov2` die echte")
        print("Konfiguration testen, danach schrittweise weitere Modelle/GPU.")
    else:
        print("SMOKE TEST FEHLGESCHLAGEN - siehe Fehler oben.")
    print("=" * 60)


if __name__ == "__main__":
    main()
