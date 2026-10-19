#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🍏 TRADUCTEUR SÉMANTIQUE RATISS — Étape 3 : résonance question → intention → concepts.
La question active l'intention la plus proche (banque de formulations connues),
qui diffuse vers ses mots-concepts canoniques ; ce vecteur-contexte enrichit
l'embedding AVANT le GPS. Contre le bug des paraphrases. 🍎
RATISS Labs — novembre 2026 — CPU uniquement.
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gps_routeur import GPS, charger_table_embedding, embedding_question, RACINE

ALPHA = 0.8   # figé dans le protocole, jamais réglé sur le test

CONCEPTS = {   # les concepts canoniques vers lesquels l'intention résonne
    "capitale_france": "géographie capitale pays ville France",
    "deux_plus_deux": "calcul addition nombre total résultat",
    "formule_eau": "science chimie molécule eau hydrogène oxygène",
    "bonjour_anglais": "langage anglais salutation bonjour traduction",
    "animal_miaule": "animal chat miaule souris domestique",
    "ingredients_omelette": "cuisine omelette oeufs ingrédients recette",
}
INTENTION_VERS_FAMILLE = {
    "capitale_france": "GEOGRAPHIE", "deux_plus_deux": "CALCUL",
    "formule_eau": "SCIENCE", "bonjour_anglais": "LANGAGE",
    "animal_miaule": "ANIMAUX", "ingredients_omelette": "CUISINE",
}

def normaliser(v):
    return v / (np.linalg.norm(v) + 1e-12)

class Traducteur:
    """Banque d'intentions (prototypes) + diffusion vers les concepts."""
    def __init__(self, tok, table):
        self.tok, self.table = tok, table
        questionnaire = json.loads((RACINE / "data" / "questionnaire.json").read_text())
        self.prototypes, self.vecteurs_concepts = {}, {}
        for intention, concepts in CONCEPTS.items():
            formulations = [s["question"] for s in questionnaire
                            if s.get("groupe") == intention]
            emb = [embedding_question(q, tok, table) for q in formulations]
            self.prototypes[intention] = normaliser(np.mean(emb, axis=0))
            self.vecteurs_concepts[intention] = embedding_question(concepts, tok, table)

    def traduire(self, question):
        v = embedding_question(question, self.tok, self.table)
        vn = normaliser(v)
        intention = max(self.prototypes, key=lambda i: float(self.prototypes[i] @ vn))
        v_enrichi = v + ALPHA * self.vecteurs_concepts[intention]
        return v_enrichi, intention

def main():
    from transformers import AutoTokenizer
    gps = GPS.charger(RACINE / "artifacts" / "gps_modele_360M.npz")
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    table = charger_table_embedding("HuggingFaceTB/SmolLM2-360M-Instruct")
    traducteur = Traducteur(tok, table)

    test = json.loads((RACINE / "data" / "paraphrases_test_v2.json").read_text())
    ok_seul = ok_trad = 0
    details = []
    for item in test:
        q, attendue = item["question"], item["famille"]
        fam_seule, _, _ = gps.predire(embedding_question(q, tok, table))
        v_enrichi, intention = traducteur.traduire(q)
        fam_trad, _, _ = gps.predire(v_enrichi)
        ok_seul += fam_seule == attendue; ok_trad += fam_trad == attendue
        details.append({"question": q, "attendue": attendue, "gps_seul": fam_seule,
                        "intention_activee": intention, "gps_traduit": fam_trad,
                        "ok_seul": fam_seule == attendue, "ok_traduit": fam_trad == attendue})
        print(f"{'✅' if fam_trad == attendue else '❌'} {q[:58]:60s} seul={fam_seule:11s} traduit={fam_trad}")

    n = len(test)
    bilan = {"alpha_fige": ALPHA, "n_test": n,
             "gps_seul": ok_seul / n, "gps_plus_traducteur": ok_trad / n,
             "verdict_E3": "CONFIRMEE" if ok_trad > ok_seul else "REFUTEE",
             "details": details}
    (RACINE / "results" / "RESULTATS_TRADUCTEUR.json").write_text(
        json.dumps(bilan, ensure_ascii=False, indent=1))
    print(f"\n🍏 GPS seul : {ok_seul}/{n} ({ok_seul/n:.1%}) | GPS + Traducteur : "
          f"{ok_trad}/{n} ({ok_trad/n:.1%}) → E3 {bilan['verdict_E3']}")

if __name__ == "__main__":
    main()
