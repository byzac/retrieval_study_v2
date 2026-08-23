"""
Zentrale Konfiguration der Studie.
Alle anderen Skripte importieren aus dieser Datei -> EIN Ort, um die
Matrix (Modelle x Datasets x Perturbationen) anzupassen.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Erwartete Ordnerstruktur pro Dataset (klassisches "ImageFolder"-Layout,
# wie es nach dem Download von Kinetics-400 / SSv2-Subsets üblich ist):
#   DATASET_ROOT / <split> / <class_name> / <video_id>.mp4
DATASETS = {
    "kinetics400_subset": {
        "root": Path("/data/kinetics400_subset"),   # <-- anpassen
        "split": "val",
        "n_classes": 25,          # Subset-Größe: 25 Klassen
        "n_videos_per_class": 20, # x 20 Videos = 500 Videos gesamt
        "seed": 42,
    },
    "ssv2_subset": {
        "root": Path("/data/ssv2_subset"),           # <-- anpassen
        "split": "val",
        "n_classes": 25,
        "n_videos_per_class": 20,
        "seed": 42,
    },
}

# Modelle: VideoMAE bewusst ausgeschlossen (Absprache mit Betreuer, wird
# ggf. nur kurz im Related-Work-Kapitel erwähnt).
# DisMo ist laut Betreuer zentral für die Arbeit -> Teil der Kernmatrix,
# kein Stretch-Goal mehr (siehe models/dismo.py, nutzt torch.hub direkt).
CORE_MODELS = ["dinov2", "slowfast", "vjepa", "dismo"]
STRETCH_MODELS = []  # aktuell keine offenen Stretch-Modelle

# Frame-Sampling: wie viele Frames pro Clip werden geladen, bevor
# Perturbationen angewendet werden. Modelle mit anderen Anforderungen
# (z.B. SlowFast braucht 32 Frames für die Fast-Pathway) resamplen
# intern in ihrem preprocess().
N_FRAMES_RAW = 32
FRAME_SIZE = 224  # Rohauflösung beim Laden, Modelle resizen ggf. weiter

# Perturbationen, die tatsächlich in der Matrix laufen
# (grayscale bewusst ausgelassen -> zu schwache Störung, siehe Methodikkapitel)
PERTURBATION_NAMES = ["frame_repeat", "shuffle", "reverse", "blur"]

# Retrieval-Protokoll-Parameter
RECALL_K = (1, 5, 10)
RANDOM_SEED = 42
