#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📦 EXPORT DES TRANCHES ENTRAÎNÉES E4 — pour le moteur natif F3.
Ré-entraîne le modèle entier (moteur M, identique à E4) puis exporte les
tranches entraînées en BF16 (bits uint16) dans un .npz lisible par NumPy seul.
Fichier lourd local (artifacts/local/), jamais versionné. CPU uniquement.
"""
import gc, random, sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)
sys.path.insert(0, str(Path(__file__).parent))
from bistouri_backprop import FAMILLES
import extension_tranches as ext
from extension_tranches import SegmentN, evaluer, perte, AdamEtatsLegers, LOCAL

RACINE = Path(__file__).resolve().parent.parent


def main():
    torch.manual_seed(20261102); random.seed(20261102)
    data = torch.load(LOCAL / "segment_extension_32.pt", weights_only=False, mmap=True)
    exemples, etat = data["exemples"], data["etat"]
    train = [e for e in exemples if not e["eval"]]; dev = [e for e in exemples if e["eval"]]
    debut = etat["couche_debut"]
    gps = np.load(RACINE / "artifacts" / "gps_modele_360M.npz", allow_pickle=True)
    masques = [{fam: sorted(int(n) for n in gps["masques"][i][debut + li])
                for i, fam in enumerate(FAMILLES)} for li in range(len(etat["couches"]))]
    segment = SegmentN(etat, masques, leger=True, dtype_socle=torch.bfloat16)
    del etat, data; gc.collect()
    avant = evaluer(segment, dev)
    params = [p for c in segment.couches for p in c.mlp_p.parameters() if p.requires_grad]
    opt = AdamEtatsLegers(params, lr=ext.LR)
    par_famille = {f: [e for e in train if e["famille"] == f] for f in FAMILLES}
    for ep in range(ext.EPOQUES):
        for fam in FAMILLES:
            melange = par_famille[fam][:]
            random.Random(1000 + ep).shuffle(melange)
            for ex in melange:
                segment.train(); opt.zero_grad()
                perte(segment, ex).backward(); opt.step()
    apres = evaluer(segment, dev)
    print(f"📉 ré-entraînement contrôle : {avant:.4f} → {apres:.4f}", flush=True)

    sortie = {}
    for li, couche in enumerate(segment.couches):
        for fam in FAMILLES:
            for nom, p in [("Wg", couche.mlp_p.Wg[fam]), ("Wu", couche.mlp_p.Wu[fam]),
                           ("Wd", couche.mlp_p.Wd[fam])]:
                bits = p.data.detach().to(torch.bfloat16).view(torch.int16).numpy()
                sortie[f"c{li}_{fam}_{nom}"] = bits
    cible = LOCAL / "tranches_entrainees_e4.npz"
    np.savez(cible, **sortie)
    print(f"💾 {cible} : {cible.stat().st_size/1e6:.0f} Mo, {len(sortie)} tenseurs exportés")


if __name__ == "__main__":
    main()
