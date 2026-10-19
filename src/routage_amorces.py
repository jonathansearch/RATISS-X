#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧭 ROUTAGE SUR AMORCES LIBRES — F2 (protocole docs/PROTOCOLE_DUEL_ET_ROUTAGE_ET_MOTEUR.md).
Le GPS (centroïdes figés, entraîné sur questions) route mal les amorces libres.
Candidat testé : embedding « contenu seul » (tokens hors mots-outils/ponctuation),
mêmes centroïdes, aucun ré-entraînement. Évals : amorces figées (36) + questions
de contrôle (questionnaire 78 + QA d'entraînement). CPU, déterministe.
"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gps_routeur import GPS, charger_table_embedding, embedding_question

RACINE = Path(__file__).resolve().parent.parent
MODELE = "HuggingFaceTB/SmolLM2-360M-Instruct"

MOTS_OUTILS = set("""le la les de des du un une et est sont en au aux pour par dans sur qui que quoi dont où
ce cette ces il elle ils elles on nous vous je tu se sa son ses leur leurs à a y ne pas plus très
the a an and is are was were of to in on for with as by at from or be been it its this that these those
i you he she we they me him her us them my his their our your what which who how when where why do does did
can could will would shall should may might must have has had not no yes if then than so such""".split())


def embedding_contenu(texte, tok, table):
    """Moyenne des embeddings des seuls tokens de contenu (figé : liste MOTS_OUTILS)."""
    ids = tok(texte, add_special_tokens=False)["input_ids"]
    gardes = []
    for i in ids:
        mot = tok.decode([i]).strip().lower().strip(".,;:!?\"'()")
        if any(c.isdigit() for c in mot) or (mot.isalpha() and mot not in MOTS_OUTILS and len(mot) > 1):
            gardes.append(i)
    if not gardes:                      # repli honnête : aucun token de contenu
        gardes = ids
    return table[gardes].mean(axis=0)


def exactitude(gps, items, fabrique, tok, table):
    bons, detail = 0, []
    for attendu, texte in items:
        fam, conf, _ = gps.predire(fabrique(texte, tok, table))
        bons += (fam == attendu)
        detail.append({"texte": texte, "attendu": attendu, "predit": fam,
                       "correct": fam == attendu, "cosinus": round(float(conf), 3)})
    return bons / len(items), detail


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELE)
    gps = GPS.charger(RACINE / "artifacts" / "gps_modele_360M.npz")
    table = charger_table_embedding(MODELE)

    amorces = [(f, t) for f, t in json.loads((RACINE / "data" / "amorces_test.json").read_text())["amorces"]]
    questionnaire = [(q["famille"], q["question"])
                     for q in json.loads((RACINE / "data" / "questionnaire.json").read_text())]
    from bistouri_backprop import QA
    questions_qa = [(fam, q) for fam, paires in QA.items() for q, _ in paires]

    bilan = {}
    for nom, items in [("amorces_libres", amorces), ("questions_questionnaire", questionnaire),
                       ("questions_entrainement", questions_qa)]:
        acc_std, det_std = exactitude(gps, items, embedding_question, tok, table)
        acc_cont, det_cont = exactitude(gps, items, embedding_contenu, tok, table)
        bilan[nom] = {"n": len(items), "gps_standard": round(acc_std, 4),
                      "gps_contenu_seul": round(acc_cont, 4)}
        print(f"{nom:26s} (n={len(items):2d}) : standard {acc_std:.1%} → contenu seul {acc_cont:.1%}")
        if nom == "amorces_libres":
            bilan["detail_amorces_standard"] = det_std
            bilan["detail_amorces_contenu"] = det_cont
    (RACINE / "results" / "RESULTATS_ROUTAGE_AMORCES.json").write_text(
        json.dumps(bilan, ensure_ascii=False, indent=1))
    print("💾 results/RESULTATS_ROUTAGE_AMORCES.json sauvegardé.")


if __name__ == "__main__":
    main()
