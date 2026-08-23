# Retrieval-Studie: Temporale Struktur in Video-Repräsentationen

## Setup

```bash
pip install -r requirements.txt
```

1. `config.py` → `DATASETS[...]["root"]` auf die tatsächlichen Pfade deiner
   lokal gespeicherten Kinetics-400- bzw. SSv2-Subsets setzen. Erwartetes
   Layout: `<root>/<split>/<class_name>/<video_id>.mp4`
2. `models/vjepa.py` → `repo_path`/`checkpoint_path` anpassen (V-JEPA2-Repo
   klonen, Checkpoint laden laut deren README).
3. Testlauf mit einem Modell/Dataset zur Kontrolle:
   ```bash
   python run_experiment.py --models dinov2 --datasets kinetics400_subset
   ```
4. Volle Kernmatrix:
   ```bash
   python run_experiment.py
   ```
5. Heatmaps für die Ergebniskapitel:
   ```bash
   python analyze_results.py --dataset kinetics400_subset
   python analyze_results.py --dataset ssv2_subset
   ```

## Retrieval-Protokoll (für Kap. 4.4 der Arbeit)

Vier Bausteine, siehe Docstring in `retrieval.py` für Details:

| Baustein | Frage, die er beantwortet |
|---|---|
| **Reference Retrieval** (A) | Wie gut funktioniert Retrieval ohne jede Störung? (Obere Schranke) |
| **Cross-Condition Retrieval** (B) | Findet eine saubere Query noch Videos derselben Klasse in einer gestörten Gallery? |
| **Identity Retrieval** (C) | Bleibt ein Video nach der Störung als *dasselbe* Video erkennbar? |
| **Embedding-Shift** (D) | Wie stark verschiebt sich der Embedding-Vektor rein numerisch? |

Für die Forschungsfrage sind **C** und **D** die direktesten Sensitivitätsmaße,
**A/B** die klassenbasierte, praxisnähere Perspektive. Alle vier werden pro
Modell × Dataset × Perturbation berechnet und in `results/master_results.csv`
gesammelt.

## Bekannte offene Punkte (siehe Kommentare im Code)

- `models/slowfast.py`: Layer-Name für den Feature-Hook ggf. gegen aktuelle
  PyTorchVideo-Version prüfen (`print(model)`).
- `models/vjepa.py`: Import-Pfad und Checkpoint müssen gegen die tatsächlich
  verwendete vjepa2-Repo-Version verifiziert werden.
- `models/dismo.py`: lädt den offiziellen Motion-Extractor direkt über
  `torch.hub.load("CompVis/DisMo", "motion_extractor_large")` (kein
  manueller Checkpoint nötig). Vor dem ersten echten Lauf prüfen, ob die
  erwartete Auflösung 224x224 (unser Standard) oder 256x256 (Repo-Beispiel)
  ist - siehe Kommentar in der Datei.
- Dataset-Subset-Sampling ist deterministisch (`seed` in `config.py`) –
  bei Bedarf mehrere Seeds laufen lassen, um Ergebnisse gegen
  Subset-Zufälligkeit abzusichern (optional, falls Zeit reicht).

## Kernmatrix (Stand jetzt)

4 Modelle (DINOv2, SlowFast, V-JEPA, DisMo) x 2 Datasets (Kinetics-400-Subset,
SSv2-Subset) x 4 Perturbationen (frame_repeat, shuffle, reverse, blur) = 32
Zellen, je mit 4 Metrik-Bausteinen (A-D). Diving-48 bewusst zurückgestellt
(siehe Diskussion im Chat-Verlauf) - kann bei Zeitreserven als optionale
Erweiterung ergänzt werden, ohne dass sich am Code etwas ändern muss außer
einem neuen Eintrag in `config.DATASETS`.
