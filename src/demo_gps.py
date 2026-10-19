#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""🗺️ Démo GPS : route des questions JAMAIS vues, en direct, sans transformeur."""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from gps_routeur import GPS, charger_table_embedding, embedding_question, RACINE

QUESTIONS = [
    "Quelle est la capitale du Japon ?",
    "Combien font 4 fois 5 ?",
    "Pourquoi les feuilles des arbres sont-elles vertes ?",
    "Comment dit-on chat en anglais ?",
    "Quel animal aboie et garde la maison ?",
    "Comment prépare-t-on une sauce tomate ?",
    "Dis-moi le nom de la capitale française.",
    "Si j'ai deux pommes et qu'on m'en donne deux autres, combien en ai-je ?",
]

def main():
    from transformers import AutoTokenizer
    gps = GPS.charger(RACINE / "artifacts" / "gps_modele_360M.npz")
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    table = charger_table_embedding("HuggingFaceTB/SmolLM2-360M-Instruct")
    print("🗺️ GPS en ligne — aucune couche transformeur ne tournera. Démonstration :\n")
    latences = []
    for passage, q in enumerate(QUESTIONS * 30):   # 30 passages pour une latence stable
        t0 = time.perf_counter()
        v = embedding_question(q, tok, table)
        fam, confiance, _ = gps.predire(v)
        latences.append((time.perf_counter() - t0) * 1000)
        if passage < len(QUESTIONS):               # on n'affiche que le premier passage
            taille = int(np.mean([len(m) for m in gps.masques[fam]]))
            print(f"❓ {q}\n   🧭 accord prédit : {fam} (confiance cosinus {confiance:.3f}) "
                  f"→ masque Mk ≈ {taille} neurones/couche prêts à s'allumer\n")
    print(f"⚡ Latence routage complet : médiane {np.median(latences):.3f} ms | "
          f"moyenne {np.mean(latences):.3f} ms (n={len(latences)}, CPU 2 cœurs)")

if __name__ == "__main__":
    main()
