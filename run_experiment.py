"""
Haupt-Runner der Studie.

Läuft: für jedes Modell x für jedes Dataset x für jede Perturbation
       -> alle vier Retrieval-/Sensitivitäts-Metriken (A-D aus retrieval.py)
       -> Ergebnisse als CSV pro Lauf (results/<model>_<dataset>.csv)
       -> zusätzlich eine gesammelte Master-CSV für die Analyse/Plots

Aufruf:
    python run_experiment.py                      # komplette Kernmatrix
    python run_experiment.py --models dinov2       # nur ein Modell (Debug)
    python run_experiment.py --datasets kinetics400_subset
    python run_experiment.py --models dismo --perturbations grayscale  # gezielter Zusatztest
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

import config
from data.video_dataset import build_dataset_subset
from models.registry import get_extractor
from perturbations import PERTURBATIONS
from retrieval import (
    build_bank,
    cross_condition_retrieval,
    embedding_shift,
    identity_retrieval,
    reference_retrieval,
)


def flatten(prefix: str, d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            continue
        out[f"{prefix}_{k}"] = v
    return out


def run_one(model_name: str, dataset_name: str, perturbation_names: list[str]) -> list[dict]:
    ds_cfg = config.DATASETS[dataset_name]
    samples = build_dataset_subset(
        root=ds_cfg["root"],
        split=ds_cfg["split"],
        n_classes=ds_cfg["n_classes"],
        n_videos_per_class=ds_cfg["n_videos_per_class"],
        n_frames=config.N_FRAMES_RAW,
        frame_size=config.FRAME_SIZE,
        seed=ds_cfg["seed"],
    )

    print(f"\n=== Modell: {model_name} | Dataset: {dataset_name} ===")
    extractor = get_extractor(model_name)

    clean_bank = build_bank(extractor, samples, PERTURBATIONS["clean"], seed=config.RANDOM_SEED)
    ref_metrics = reference_retrieval(clean_bank, k_values=config.RECALL_K)
    print(f"  Referenz-Retrieval (ungestörte Query vs ungestörte Gallery): {ref_metrics}")

    rows = []
    base_row = {"model": model_name, "dataset": dataset_name, "n_videos": len(samples)}

    ref_row = dict(base_row, perturbation="clean")
    ref_row.update(flatten("class", ref_metrics))
    rows.append(ref_row)

    for pert_name in perturbation_names:
        t0 = time.time()
        pert_fn = PERTURBATIONS[pert_name]
        perturbed_bank = build_bank(extractor, samples, pert_fn, seed=config.RANDOM_SEED)

        cross_metrics = cross_condition_retrieval(clean_bank, perturbed_bank, k_values=config.RECALL_K)
        identity_metrics = identity_retrieval(clean_bank, perturbed_bank, k_values=config.RECALL_K)
        shift_metrics = embedding_shift(clean_bank, perturbed_bank)

        row = dict(base_row, perturbation=pert_name)
        row.update(flatten("class", cross_metrics))
        row.update(flatten("identity", identity_metrics))
        row.update(flatten("shift", shift_metrics))
        rows.append(row)

        print(f"  [{pert_name}] class_mAP={cross_metrics['mAP']:.3f}  "
              f"identity_recall@1={identity_metrics['recall@1']:.3f}  "
              f"mean_shift={shift_metrics['mean_cosine_shift']:.3f}  "
              f"({time.time() - t0:.1f}s)")

    # Bei gezielten Zusatztests (abweichend von der Standard-Hauptmatrix aus
    # config.PERTURBATION_NAMES) einen Suffix anhaengen, damit NICHT versehentlich
    # die regulaeren Hauptmatrix-Ergebnisse ueberschrieben werden.
    is_custom_run = set(perturbation_names) != set(config.PERTURBATION_NAMES)
    suffix = "_extra-" + "-".join(perturbation_names) if is_custom_run else ""
    out_path = config.RESULTS_DIR / f"{model_name}_{dataset_name}{suffix}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> gespeichert: {out_path}")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=config.CORE_MODELS)
    parser.add_argument("--datasets", nargs="+", default=list(config.DATASETS.keys()))
    parser.add_argument(
        "--perturbations", nargs="+", default=None,
        help="Fuer gezielte Zusatztests (z.B. nur 'grayscale' fuer ein Kontrollexperiment), "
             "ohne config.PERTURBATION_NAMES (= die Hauptmatrix) zu veraendern. "
             "Standard: alle in config.PERTURBATION_NAMES definierten Perturbationen.",
    )
    args = parser.parse_args()
    perturbation_names = args.perturbations or config.PERTURBATION_NAMES

    all_rows = []
    for model_name in args.models:
        for dataset_name in args.datasets:
            try:
                all_rows.extend(run_one(model_name, dataset_name, perturbation_names))
            except Exception as e:
                print(f"!! Fehler bei {model_name} x {dataset_name}: {e}")
                continue

    is_custom_run = perturbation_names != config.PERTURBATION_NAMES
    if is_custom_run:
        print(f"\nHinweis: Gezielter Zusatztest (Perturbationen: {perturbation_names}) - "
              f"master_results.csv wird NICHT ueberschrieben, Ergebnisse liegen nur in "
              f"den einzelnen '..._extra-...'-Dateien in {config.RESULTS_DIR}.")
        return

    master_path = config.RESULTS_DIR / "master_results.csv"
    with open(master_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in all_rows for k in r}))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nGesamtergebnisse: {master_path}")


if __name__ == "__main__":
    main()