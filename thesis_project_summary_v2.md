# Projekt-Zusammenfassung (Update 2): Retrieval-Studie - Stand nach Modell-Debugging

Dies ist ein Update zum vorherigen Zusammenfassungsdokument. Seit damals wurden
beide Datensätze vollständig beschafft, alle vier Modelle technisch zum Laufen
gebracht (mit erheblichem Debugging-Aufwand), und mehrere Methodik-Fixes
vorgenommen. SSv2 wurde noch mit KEINEM Modell getestet.

## Setup-Umgebung (Nutzer-Rechner)

- Windows, PowerShell, conda `base`-Umgebung (venv-Umzug auf `retrieval_study`
  bewusst auf später verschoben, siehe unten)
- GPU: NVIDIA GeForce RTX 5070 Ti, 16 GB VRAM, funktioniert (CUDA 13.0 Wheels)
- Python 3.14 (sehr neu, potenzielle Kompatibilitätsquelle für künftige Fehler)
- Projekt wird über GitHub geklont/synchronisiert, NICHT über die ZIP-Datei

## Datenstand (beide Datensätze fertig vorbereitet)

- **Kinetics-400**: Val-Split vollständig heruntergeladen (cvdfoundation-Mirror,
  19.881 mp4s, 28,46 GB), Subset via `data/prepare_kinetics.py` erstellt:
  `C:\data\kinetics400_subset\val\` (25 Klassen x 20 Videos, deterministisch,
  seed=42)
- **SSv2**: Vollständig heruntergeladen (Qualcomm-Portal, 220.847 Videos,
  18,14 GB), Subset via `data/prepare_ssv2.py` erstellt:
  `C:\data\ssv2_subset\val\` (25 Klassen x 20 Videos)
  **WICHTIG: Noch mit KEINEM der vier Modelle getestet.** Nächster
  logischer Schritt nach Abschluss der Kinetics-Läufe.
- Beide Prepare-Skripte wurden um Windows-Symlink-Fallback (automatisches
  Kopieren statt Symlink bei fehlenden Admin-Rechten) und direktes
  Subset-Sampling (`--n_classes`/`--n_videos_per_class`) erweitert.

## Modell-Status: alle vier technisch lauffähig auf Kinetics

Jedes Modell brauchte mindestens eine Debugging-Runde. Zusammenfassung der
tatsächlich vorgenommenen Code-Fixes (alle bereits in den lokalen Dateien
des Nutzers umgesetzt):

### DINOv2 - lief von Anfang an fehlerfrei
Keine Fixes nötig.

### SlowFast - zwei Fixes nötig
1. **Fehlende Abhängigkeiten**: `fvcore`, `iopath` (torch.hub-Abhängigkeiten
   von pytorchvideo, nicht in requirements.txt vorhergesehen)
2. **Hook-Layer-Bug**: `self.model.blocks[-1].pool` existiert in der
   installierten PyTorchVideo-Version nicht (ResNetBasicHead hat dort kein
   `.pool`-Submodul mehr). Fix: Hook stattdessen auf `self.model.blocks[5]`
   (PoolConcatPathway) - liefert das gepoolte, konkatenierte 2304-dim
   Feature vor Dropout/Klassifikationskopf.
3. **Device-Mismatch**: `slow_idx` in `_PackPathway.forward()` wurde auf CPU
   erzeugt, Frames lagen aber auf GPU. Fix: `.to(frames.device)` ergänzt.

### V-JEPA - kompletter Strategiewechsel nötig
Ursprünglicher Ansatz (torch.hub, `facebookresearch/vjepa2`) hatte zwei
Bugs im Repo selbst: (1) optionaler Preprocessor braucht `cv2`, (2) der
Checkpoint-Download zeigte auf eine kaputte Platzhalter-URL
(`localhost:8300`). Nicht von unserer Seite behebbar.

**Lösung: Umstieg auf HuggingFace-Integration** (`transformers`,
`AutoModel`/`AutoVideoProcessor`, Repo `facebook/vjepa2-vitl-fpc64-256`).
Funktioniert zuverlässig, lief danach ohne weitere Fehler durch (nur sehr
lange Laufzeit: ~220s pro Perturbation für 500 Videos).

`models/vjepa.py` ist komplett neu geschrieben, alter Code (Repo-Klon,
`repo_path`/`checkpoint_path`) vollständig ersetzt.

### DisMo - mehrere Fixes, WICHTIGSTER BEFUND: Auflösungsbug
1. **Fehlende Abhängigkeiten**: `jaxtyping`, `diffusers` (DisMo-Repo ist an
   die CogVideoX-Pipeline gekoppelt, zieht mehr Abhängigkeiten als nötig)
2. **torch.hub Sicherheitsabfrage**: `trust_repo=True` ergänzt, damit die
   interaktive y/N-Bestätigung nicht bei jedem Lauf blockiert
3. **KRITISCH - Auflösungs-Bug, betrifft die Validität aller bisherigen
   DisMo-Ergebnisse**: `preprocess()` hatte KEINEN Resize-Schritt und lief
   daher mit `config.FRAME_SIZE=224`, obwohl das offizielle Repo-Beispiel
   explizit 256x256 vorschreibt
   (`torch.rand((B, T, 256, 256, 3))` im README). Fix: expliziter
   `F.interpolate(..., size=(256,256))` am Anfang von `preprocess()`
   ergänzt, UNABHÄNGIG von `config.FRAME_SIZE`.

   **Der erste vollständige DisMo-Kinetics-Lauf (25x20, vor dem Fix) ist
   UNGÜLTIG und darf nicht in die Auswertung übernommen werden.** Ein
   kleiner Verifikationslauf (3 Klassen x 3 Videos) nach dem Fix zeigt
   ähnliches, teils noch stärker ausgeprägtes Muster (frame_repeat-Shift
   sogar höher: 1,0 statt 0,704) - der volle 25x20-Lauf mit korrigiertem
   Code muss noch durchgeführt werden.

**Rechenzeit-Hinweis**: DisMo ist mit Abstand am langsamsten (~20-21 Minuten
PRO PERTURBATION bei 500 Videos, da `forward_sliding()` viele
Forward-Passes durch ein 1,13-GB-DINOv2-ViT-L-Backbone pro Video braucht).
Kompletter Lauf (Referenz + 4 Perturbationen) dauert ca. 85-100 Minuten.

## Methodik-Fixes in retrieval.py (nach Diskussion der Ergebnisse)

1. **Self-Exclusion-Bug behoben**: `cross_condition_retrieval()` (Baustein
   B) schloss das eigene Video ursprünglich NICHT aus der Gallery aus
   (anders als `reference_retrieval()`, Baustein A). Das führte dazu, dass
   bei permutationsinvarianten Perturbationen (shuffle/reverse bei DINOv2,
   wo das gestörte Embedding exakt identisch zum sauberen ist) der mAP
   künstlich über den Referenzwert stieg. Fix: `exclude_identical_id=True`
   auch in `cross_condition_retrieval()`.
2. **Rohe Cosine-Similarity zusätzlich zur Distanz**: `embedding_shift()`
   gibt jetzt sowohl `mean_cosine_similarity`/`std_cosine_similarity` als
   auch `mean_cosine_shift`/`std_cosine_shift` zurück (vorher nur Shift).
3. **Terminologie-Änderung**: "Referenz (clean vs clean)" umbenannt zu
   "Referenz-Retrieval (ungestörte Query vs. ungestörte Gallery)" - in
   `retrieval.py` (Docstring von `reference_retrieval()`) und
   `run_experiment.py` (Konsolenausgabe). Für Konsistenz mit der Thesis
   diesen Begriff durchgängig verwenden.

## requirements.txt - aktueller vollständiger Stand

```
torch>=2.1
torchvision>=0.16
decord
numpy
pandas
matplotlib
seaborn
tqdm
fvcore         # SlowFast (torch.hub pytorchvideo)
iopath         # dito
transformers   # V-JEPA2 über HuggingFace (models/vjepa.py)
jaxtyping      # DisMo (torch.hub CompVis/DisMo)
diffusers      # dito (CogVideoX-Pipeline-Kopplung)
```
(`timm`/`einops` waren ein Zwischenschritt für den verworfenen
torch.hub-V-JEPA-Ansatz und wurden wieder entfernt.)

## Bisherige Ergebnisse auf Kinetics-400 (25x20 Subset, sofern nicht anders vermerkt)

| Störung | DINOv2 shift | SlowFast shift | V-JEPA shift | DisMo shift (ALT/fehlerhaft) |
|---|---|---|---|---|
| frame_repeat | 0,119 | 0,451 | 0,272 | 0,704 (ungültig, s.o.) |
| shuffle | 0,000 | 0,169 | 0,047 | 0,418 (ungültig) |
| reverse | 0,000 | 0,028 | 0,023 | 0,195 (ungültig) |
| blur | 0,360 | 0,499 | 0,360 | 0,262 (ungültig) |

Referenz-mAP (ungestört): DINOv2=0,514, SlowFast=0,491, V-JEPA=0,180,
DisMo=0,130 (ungültig, muss neu berechnet werden).

**Interessante inhaltliche Befunde für die Diskussion:**
- DINOv2 ist exakt permutationsinvariant (Mean-Pooling über Frames) -
  mathematisch beweisbar, nicht nur empirisch: shift=0,000 bei
  shuffle/reverse.
- V-JEPA zeigt ÜBERRASCHEND schwächere Zeitsensitivität als SlowFast,
  obwohl die Architektur-Hypothese das Gegenteil nahelegt. Mögliche
  Erklärung: wir nutzen nur den rohen Encoder-Output, das eigentliche
  "Zeitverständnis" von V-JEPA steckt im (hier ungenutzten)
  Predictor-Netzwerk.
- Bei DisMo (auch im vorläufigen fehlerhaften Lauf sichtbar):
  frame_repeat > shuffle > reverse, aber reverse < blur - deutet auf
  lokale Bewegungs-MAGNITUDE-Kodierung statt reiner Richtungskodierung
  hin. Zu verifizieren mit dem korrigierten 25x20-Lauf.
- Empfehlung: für die Forschungsfrage eine abgeleitete Kennzahl
  "mean_shift(temporal) - mean_shift(visuell)" pro Modell berechnen, um
  SELEKTIVITÄT (nicht nur Reaktionsstärke) zu vergleichen - DisMo reagiert
  insgesamt am stärksten auf ALLES, nicht nur auf temporale Störungen.

## Offener TODO (aus vorherigem Dokument, weiterhin unerledigt)

Pretraining-Daten-Kontamination pro Modell dokumentieren (V-JEPA/DisMo noch
zu verifizieren, DINOv2/SlowFast schon geklärt) - siehe README.md im
Projekt für Details und Formulierungsvorschlag für die Limitations-Sektion.

## Nächste Schritte (in empfohlener Reihenfolge)

1. `config.py` zurück auf `n_classes=25, n_videos_per_class=20` stellen
   (war für den DisMo-Verifikationslauf temporär auf 3x3 reduziert)
2. DisMo mit korrigiertem Code auf vollem Kinetics-Subset neu laufen lassen
   (~85-100 Min) - überschreibt die ungültige alte CSV
3. Alle vier Modelle auf SSv2 testen (voraussichtlich neue,
   dataset-spezifische Bugs möglich, z.B. bei der .webm-Dekodierung über
   decord) - Reihenfolge: DINOv2 zuerst (schnellster Sanity-Check)
4. Pretraining-Kontamination-TODO abarbeiten
5. Ggf. Subset-Größe für finalen Lauf überdenken (25x20 ist am unteren
   Ende des Vertretbaren für die klassenbasierten Metriken)
6. Kompletter Matrixlauf ohne --models/--datasets Flags
7. `analyze_results.py` für Heatmaps (Kap. 6)
8. Am Thesistext weiterarbeiten: Kapitel 2 grob fertig diskutiert
   (Tiefenempfehlungen pro Unterkapitel liegen vor), Kapitel 3 (Related
   Work), 4/5 (Methodik/Setup - jetzt mit den echten Code-Entscheidungen
   auszuformulieren), 6/7 (Ergebnisse/Diskussion/Fazit) stehen noch aus

## Hinweis zur venv-Migration

Nutzer hat bereits eine leere `retrieval_study`-Conda-Umgebung angelegt,
Migration von `base` dorthin ist bewusst auf NACH Abschluss aller
Testläufe verschoben (um nicht mitten im Debugging alle Installationen
wiederholen zu müssen). Empfohlenes Vorgehen bei der Migration:
`pip freeze > working_versions.txt` in der funktionierenden `base`-Umgebung,
dann `pip install -r working_versions.txt` in der neuen Umgebung.
