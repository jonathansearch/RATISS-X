#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎢 G1 — SEUIL EN GÉNÉRATION LIBRE (protocole docs/PROTOCOLE_SEUIL_GENERATION_G1.md).
Pour un fond élite donné (--fond), le moteur V1 génère sur 6 amorces du jeu figé
(1 par famille, famille attendue servie : routage oracle) et publie les phrases brutes.
Usage : python3 src/seuil_generation.py --fond 0.70
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from moteur_v1 import MoteurV1, rss, LOCAL, RACINE


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fond", type=float, required=True)
    ap.add_argument("--selection", default="magnitude")
    ap.add_argument("--tranches", default="")
    ap.add_argument("--classement", action="store_true")
    ap.add_argument("--suffixe", default="")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    from huggingface_hub import snapshot_download
    snap = snapshot_download("HuggingFaceTB/SmolLM2-360M-Instruct")
    tok = Tokenizer.from_file(str(Path(snap) / "tokenizer.json"))
    moteur = MoteurV1(snap, Path(a.tranches) if a.tranches else
                      LOCAL / "tranches_entrainees_e4.npz", fond=a.fond,
                      selection=a.selection,
                      classements_npz=LOCAL / "classements_activation.npz"
                      if a.classement else None)
    amorces = [("GEOGRAPHIE", "The capital of Spain is"),
               ("CALCUL", "Two plus two equals"),
               ("SCIENCE", "Water is made of hydrogen and"),
               ("LANGAGE", "The French word for cat is"),
               ("ANIMAUX", "The animal that meows is"),
               ("CUISINE", "To bake a cake, mix flour and")]
    bilan = []
    for fam, prompt in amorces:
        t0 = time.perf_counter(); moteur.construire(fam); t_jeu = time.perf_counter() - t0
        texte, s_tok = moteur.generer(tok, prompt, fam)
        mots = texte.split()
        boucle = len(mots) >= 6 and len(set(mots)) <= len(mots) // 3
        bilan.append({"famille": fam, "prompt": prompt, "texte": texte,
                      "boucle_detectee": boucle, "secondes_par_token": round(s_tok, 3),
                      "construction_jeu_s": round(t_jeu, 2)})
        print(f"🎢 fond {a.fond:.0%} | {fam:11s} | {prompt} → {texte!r} "
              f"({s_tok:.2f}s/token, boucle={boucle})", flush=True)
    nom = f"RESULTATS_SEUIL_GENERATION_{int(round(a.fond*100))}{a.suffixe}.json"
    (RACINE / "results" / nom).write_text(json.dumps(
        {"marche": "G1 seuil en generation libre", "fond": a.fond,
         "rss_pic_mo": round(rss(), 1), "resultats": bilan}, ensure_ascii=False, indent=1))
    print(f"📏 RSS pic : {rss():.0f} Mo | 💾 results/{nom}")


if __name__ == "__main__":
    main()
