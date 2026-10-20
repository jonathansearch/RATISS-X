#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛰️ GPS DES AMORCES — Sprint final scientifique.

Compare trois centroïdes de routage, construits avec le même plongement moyen
de tokens et le même modèle :
  A) questions d'entraînement seules ;
  B) nouveau corpus d'amorces d'entraînement seul ;
  C) questions + nouveau corpus d'amorces.

Évaluation figée : 36 amorces de test jamais utilisées pour les centroïdes B/C,
et 60 questions hors paraphrases. Aucun transformeur n'est exécuté : seule la
table d'embedding est lue, ligne par ligne.
"""
import json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gps_routeur import FAMILLES
from moteur_natif import Coffre

RACINE = Path(__file__).resolve().parent.parent


def construire(textes_par_id, labels):
    C = []
    for fam in FAMILLES:
        C.append(np.mean([textes_par_id[t] for t, y in labels if y == fam], axis=0))
    C = np.stack(C)
    return C / np.linalg.norm(C, axis=1, keepdims=True)


def predire(C, v):
    v = v / (np.linalg.norm(v) + 1e-12)
    s = C @ v
    return FAMILLES[int(np.argmax(s))], float(np.max(s))


def evaluer(C, items):
    details, ok = [], 0
    for attendu, texte, v in items:
        pred, score = predire(C, v)
        bon = pred == attendu
        ok += bon
        details.append({"texte": texte, "attendue": attendu, "predite": pred,
                        "confiance": round(score, 4), "correct": bool(bon)})
    return ok / len(items), details


def main():
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer
    snap = Path(snapshot_download("HuggingFaceTB/SmolLM2-360M-Instruct"))
    tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
    coffre = Coffre(str(snap / "model.safetensors"))

    questions = [s for s in json.loads((RACINE / "data" / "questionnaire.json").read_text())
                 if s["famille"] != "PARAPHRASE"]
    train_amorces = json.loads((RACINE / "data" / "amorces_entrainement.json").read_text())["amorces"]
    test_amorces = json.loads((RACINE / "data" / "amorces_test.json").read_text())["amorces"]
    assert not ({t for _, t in train_amorces} & {t for _, t in test_amorces})

    tous = ([s["question"] for s in questions] + [t for _, t in train_amorces]
            + [t for _, t in test_amorces])
    ids = {t: tok.encode(t).ids for t in tous}
    uniques = sorted({i for v in ids.values() for i in v})
    table = {i: coffre.lignes("model.embed_tokens.weight", i, i + 1, 960)[0]
             for i in uniques}
    E = {t: np.mean([table[i] for i in v], axis=0) for t, v in ids.items()}

    q_items = [(s["famille"], s["question"], E[s["question"]]) for s in questions]
    tr_items = [(f, t, E[t]) for f, t in train_amorces]
    te_items = [(f, t, E[t]) for f, t in test_amorces]

    variantes = {
        "A_questions_seules": construire(E, [(t, y) for y, t, _ in q_items]),
        "B_amorces_seules": construire(E, [(t, y) for y, t, _ in tr_items]),
        "C_questions_plus_amorces": construire(
            E, [(t, y) for y, t, _ in q_items] + [(t, y) for y, t, _ in tr_items]),
    }
    resultats = {}
    for nom, C in variantes.items():
        aq, dq = evaluer(C, q_items)
        aa, da = evaluer(C, te_items)
        resultats[nom] = {"exactitude_questions_60": round(aq, 4),
                          "exactitude_amorces_figees_36": round(aa, 4),
                          "details_questions": dq, "details_amorces": da}
        print(f"🛰️ {nom:26s} | questions {aq:.1%} | amorces figées {aa:.1%}", flush=True)

    A, C = resultats["A_questions_seules"], resultats["C_questions_plus_amorces"]
    retenu = bool(C["exactitude_amorces_figees_36"] > A["exactitude_amorces_figees_36"]
                  and C["exactitude_questions_60"] >= A["exactitude_questions_60"] - 0.05)
    np.savez(RACINE / "artifacts" / "gps_centroides_hybrides.npz",
             centroides=variantes["C_questions_plus_amorces"],
             familles=np.array(FAMILLES))
    bilan = {"marche": "GPS des amorces", "corpus_entrainement_amorces": len(tr_items),
             "test_amorces_figees": len(te_items), "test_questions": len(q_items),
             "variantes": resultats, "routeur_hybride_retenu": retenu,
             "critere": "C retenu seulement si C bat A sur les amorces et perd au plus 5 points sur les questions"}
    (RACINE / "results" / "RESULTATS_GPS_AMORCES.json").write_text(
        json.dumps(bilan, ensure_ascii=False, indent=1))
    print(f"🏁 routeur hybride C retenu selon le critère : {retenu}")
    print("💾 results/RESULTATS_GPS_AMORCES.json")


if __name__ == "__main__":
    main()
