#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🗣️ GÉNÉRATION TRANCHE SEULE — D4 Option 2 (ordre du chef).
Dans les 2 dernières couches du 360M, le MLP dense est REMPLACÉ par la seule
tranche de l'accord routé par le GPS pour le prompt : aucun fallback dense,
aucune contribution inactive. Les 30 autres couches restent intactes.
Comparaison honnête : mêmes prompts, même décodage glouton, modèle intact en face.
Critère du chef : compléter de façon cohérente, sans boucler à l'infini.
CPU uniquement, BF16 (RAM oblige). RATISS Labs — novembre 2026.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
sys.path.insert(0, str(Path(__file__).parent))
from gps_routeur import GPS, charger_table_embedding, embedding_question
from bistouri_physique import TrancheMLP, forward_tranche

RACINE = Path(__file__).resolve().parent.parent
MODELE = "HuggingFaceTB/SmolLM2-360M-Instruct"
PROMPTS = [
    "La capitale de la France est",
    "Le ciel est",
    "The capital of France is",
    "Two plus two equals",
    "The animal that meows is",
]
MAX_NOUVEAUX = 14


class MLPTrancheSeule(nn.Module):
    """Ne calcule QUE la tranche de la famille courante. Rien d'autre. 🔪"""
    def __init__(self, mlp, masques):
        super().__init__()
        self.tranches = {}
        for fam, idx in masques.items():
            i = torch.tensor(idx)
            self.register_buffer(f"Wg_{fam}", mlp.gate_proj.weight.detach()[i].clone())
            self.register_buffer(f"Wu_{fam}", mlp.up_proj.weight.detach()[i].clone())
            self.register_buffer(f"Wd_{fam}", mlp.down_proj.weight.detach()[:, i].clone())
            self.tranches[fam] = i
        self.famille = "GEOGRAPHIE"

    def forward(self, x):
        fam = self.famille
        tr = TrancheMLP(getattr(self, f"Wg_{fam}"), getattr(self, f"Wu_{fam}"),
                        getattr(self, f"Wd_{fam}"), self.tranches[fam])
        return forward_tranche(x, tr)


def boucle_detectee(texte):
    mots = texte.split()
    return len(mots) >= 6 and len(set(mots)) <= len(mots) // 3


def generer(modele, tok, prompt):
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = modele.generate(ids, max_new_tokens=MAX_NOUVEAUX, do_sample=False,
                              pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).replace("\n", " ").strip()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELE)
    gps = GPS.charger(RACINE / "artifacts" / "gps_modele_360M.npz")
    table = charger_table_embedding(MODELE)
    familles = list(gps.masques.keys())

    print("🔬 Chargement du 360M (BF16, CPU)...", flush=True)
    modele = AutoModelForCausalLM.from_pretrained(MODELE, dtype=torch.bfloat16)
    modele.eval()
    n = modele.config.num_hidden_layers

    # routage GPS de chaque prompt (avant toute chirurgie)
    routes = {}
    for p in PROMPTS:
        fam, conf, _ = gps.predire(embedding_question(p, tok, table))
        routes[p] = fam
        print(f"🧭 « {p} » → accord {fam} (cosinus {conf:.3f})")

    print("\n🐘 Génération du modèle INTACT (référence)...", flush=True)
    ref = {}
    for p in PROMPTS:
        t0 = time.time(); ref[p] = generer(modele, tok, p)
        print(f"   {p} → {ref[p]!r} ({time.time()-t0:.1f}s)")

    # chirurgie : tranches seules sur les 2 dernières couches, masque à leur couche
    for li, couche_idx in enumerate((n - 2, n - 1)):
        masques = {fam: sorted(int(v) for v in gps.masques[fam][couche_idx]) for fam in familles}
        couche = modele.model.layers[couche_idx]
        couche.mlp = MLPTrancheSeule(couche.mlp, masques)

    print("\n🔪 Génération TRANCHE SEULE (zéro fallback dense)...", flush=True)
    resultats = []
    for p in PROMPTS:
        for couche_idx in (n - 2, n - 1):
            modele.model.layers[couche_idx].mlp.famille = routes[p]
        t0 = time.time(); texte = generer(modele, tok, p); duree = time.time() - t0
        boucle = boucle_detectee(texte)
        print(f"   {p} → {texte!r} ({duree:.1f}s, boucle={boucle})")
        resultats.append({"prompt": p, "accord_gps": routes[p], "modele_intact": ref[p],
                          "tranche_seule": texte, "boucle_detectee": boucle,
                          "secondes": round(duree, 1)})

    bilan = {"modele": MODELE, "precision": "BF16 CPU",
             "chirurgie": "MLP des 2 dernières couches remplacés par la tranche de l'accord "
                          "routé par le GPS (propre à chaque couche), aucun fallback dense",
             "max_nouveaux_tokens": MAX_NOUVEAUX, "decodage": "glouton",
             "resultats": resultats}
    (RACINE / "results" / "RESULTATS_GENERATION_TRANCHE.json").write_text(
        json.dumps(bilan, ensure_ascii=False, indent=1))
    print("\n💾 results/RESULTATS_GENERATION_TRANCHE.json sauvegardé.")


if __name__ == "__main__":
    main()
