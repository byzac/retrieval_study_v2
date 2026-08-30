"""
Retrieval-Protokoll (siehe Gliederung Kap. 4.4).
Drei komplementäre Bausteine, die zusammen die Forschungsfrage beantworten:
  A) reference_retrieval()
     Clean-Query gegen Clean-Gallery, Klassenebene.
     -> Obere Schranke / Sanity-Check: wie gut funktioniert Retrieval
        überhaupt, ohne jede Störung? (Kalibrierung für B)
  B) cross_condition_retrieval()
     Clean-Query gegen GESTÖRTE Gallery, Klassenebene (mAP, Recall@K).
     -> Praxisnahe Frage: "Wenn ich mit einem sauberen Video suche,
        finde ich noch Videos derselben Klasse, wenn die Gallery
        gestört wurde?" Abfall relativ zu (A) = Klassen-Sensitivität.
  C) identity_retrieval()
     Für jedes Video: Rang seines EIGENEN gestörten Gegenstücks unter
     allen gestörten Embeddings, wenn mit dem CLEAN-Embedding gesucht
     wird. Recall@1 hier beantwortet direkt: "Bleibt das Video nach der
     Störung noch als DASSELBE Video erkennbar?" - unabhängig von
     Klassengröße/-verteilung im Dataset.
  D) embedding_shift()
     Reine Cosine-Distanz zwischen Clean- und gestörtem Embedding
     desselben Videos. Direkteste, modellunabhängige Sensitivitätsmetrik,
     keine Gallery/Ranking involviert.
Für die Forschungsfrage ("wie sensibel reagieren Embeddings auf Verlust
von temporaler vs. visueller Information") sind C) und D) die zentralen
Metriken; A) und B) liefern die praxisnähere, klassenbasierte Perspektive
und dienen als Kalibrierung/Plausibilitätscheck.

WICHTIGER SPEICHER-FIX in build_bank(): Frames werden nicht mehr vorab
für das gesamte Subset gesammelt (das führte bei größeren Subsets zu
Speicherbedarf >90 GB und Systemabstürzen), sondern individuell pro
Video geladen, direkt vor der Embedding-Berechnung, und danach
automatisch wieder freigegeben.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import torch
import config
from data.video_dataset import load_video_frames


@dataclass
class EmbeddingBank:
    ids: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    embeddings: list[torch.Tensor] = field(default_factory=list)
    def stack(self) -> torch.Tensor:
        return torch.stack(self.embeddings)  # [N, D]


def build_bank(extractor, samples, perturb_fn, seed: int = 0) -> EmbeddingBank:
    """Baut eine EmbeddingBank auf.

    Speicher-Fix: Frames werden pro Video JETZT einzeln geladen (nicht
    vorab für das gesamte Subset gesammelt) und direkt nach der
    Embedding-Berechnung automatisch wieder freigegeben (der lokale
    `frames`/`perturbed`-Tensor verliert am Ende jeder Schleifeniteration
    seine letzte Referenz und wird vom Garbage Collector eingesammelt).
    Das hält den Speicherbedarf konstant klein (ein Video gleichzeitig),
    unabhängig von der Subset-Größe.

    Defekte Videos werden wie zuvor sauber übersprungen; da dieselbe
    Datei bei jedem der (typischerweise fünf) Aufrufe von build_bank()
    pro Modell/Dataset-Kombination deterministisch gleich fehlschlägt,
    bleiben die IDs zwischen clean_bank und allen perturbed_banks
    weiterhin konsistent (wichtig für die Asserts in identity_retrieval
    und embedding_shift weiter unten)."""
    bank = EmbeddingBank()
    for s in samples:
        try:
            frames = load_video_frames(
                s.path, n_frames=config.N_FRAMES_RAW, frame_size=config.FRAME_SIZE
            )
        except RuntimeError as e:
            print(f"  Warnung: Video übersprungen (nicht lesbar): {s.path.name} ({e})")
            continue
        perturbed = perturb_fn(frames, seed=seed)
        emb = extractor.embed(perturbed).cpu()
        bank.ids.append(s.video_id)
        bank.labels.append(s.label)
        bank.embeddings.append(emb)
    return bank


def _class_level_metrics(query_bank: EmbeddingBank, gallery_bank: EmbeddingBank,
                          k_values, exclude_identical_id: bool) -> dict[str, float]:
    Q = query_bank.stack()
    G = gallery_bank.stack()
    sims = Q @ G.T  # [Nq, Ng], Cosine (Embeddings sind L2-normalisiert)
    if exclude_identical_id:
        for i, qid in enumerate(query_bank.ids):
            for j, gid in enumerate(gallery_bank.ids):
                if qid == gid:
                    sims[i, j] = -float("inf")
    order = sims.argsort(dim=1, descending=True)
    recalls = {k: 0.0 for k in k_values}
    average_precisions = []
    for i, q_label in enumerate(query_bank.labels):
        ranked_labels = [gallery_bank.labels[j] for j in order[i].tolist()]
        relevant = [lbl == q_label for lbl in ranked_labels]
        for k in k_values:
            if any(relevant[:k]):
                recalls[k] += 1
        num_relevant_seen, precisions = 0, []
        for rank, is_rel in enumerate(relevant, start=1):
            if is_rel:
                num_relevant_seen += 1
                precisions.append(num_relevant_seen / rank)
        average_precisions.append(float(np.mean(precisions)) if precisions else 0.0)
    n = len(query_bank.labels)
    metrics = {f"recall@{k}": recalls[k] / n for k in k_values}
    metrics["mAP"] = float(np.mean(average_precisions))
    return metrics


def reference_retrieval(clean_bank: EmbeddingBank, k_values=(1, 5, 10)) -> dict[str, float]:
    """(A) Referenz-Retrieval (ungestörte Query vs ungestörte Gallery):
    obere Schranke, eigenes Video aus der Gallery ausgeschlossen."""
    return _class_level_metrics(clean_bank, clean_bank, k_values, exclude_identical_id=True)


def cross_condition_retrieval(clean_bank: EmbeddingBank, perturbed_bank: EmbeddingBank,
                                k_values=(1, 5, 10)) -> dict[str, float]:
    """(B) Clean-Query gegen gestörte Gallery, Klassenebene."""
    return _class_level_metrics(clean_bank, perturbed_bank, k_values, exclude_identical_id=True)


def identity_retrieval(clean_bank: EmbeddingBank, perturbed_bank: EmbeddingBank,
                         k_values=(1, 5, 10)) -> dict[str, float]:
    """(C) Findet die Query ihr EIGENES gestörtes Gegenstück wieder?
    Nutzt video_id statt label als Relevanzkriterium (genau 1 relevantes
    Item pro Query)."""
    assert clean_bank.ids == perturbed_bank.ids, "Reihenfolge/IDs müssen übereinstimmen"
    Q = clean_bank.stack()
    G = perturbed_bank.stack()
    sims = Q @ G.T
    order = sims.argsort(dim=1, descending=True)
    ranks = []
    for i, qid in enumerate(clean_bank.ids):
        ranked_ids = [perturbed_bank.ids[j] for j in order[i].tolist()]
        rank = ranked_ids.index(qid) + 1  # 1-indexiert
        ranks.append(rank)
    ranks_arr = np.array(ranks)
    metrics = {f"recall@{k}": float((ranks_arr <= k).mean()) for k in k_values}
    metrics["mean_rank"] = float(ranks_arr.mean())
    metrics["median_rank"] = float(np.median(ranks_arr))
    return metrics


def embedding_shift(clean_bank: EmbeddingBank, perturbed_bank: EmbeddingBank) -> dict:
    """(D) Cosine-Distanz zwischen Clean- und gestörtem Embedding desselben Videos."""
    assert clean_bank.ids == perturbed_bank.ids, "Reihenfolge/IDs müssen übereinstimmen"
    C = clean_bank.stack()
    P = perturbed_bank.stack()
    cos_sim = (C * P).sum(dim=1)
    cos_dist = 1.0 - cos_sim
    return {
        "mean_cosine_similarity": float(cos_sim.mean()),
        "std_cosine_similarity": float(cos_sim.std()),
        "mean_cosine_shift": float(cos_dist.mean()),
        "std_cosine_shift": float(cos_dist.std()),
        "per_video_shift": {vid: float(d) for vid, d in zip(clean_bank.ids, cos_dist.tolist())},
    }