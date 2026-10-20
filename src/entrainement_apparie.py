#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 ENTRAÎNEMENT APPARIÉ AU RÉGIME D'INFÉRENCE — Sprint final scientifique.

Avant l'entraînement, ce script :
1) mesure, sur les exemples d'entraînement, la contribution moyenne de chaque
   neurone inactif (`E|activation| × ||W_down||`) et exporte ces classements ;
2) restreint ensuite la branche inactive au même fond élite magnitude que celui
   du moteur d'inférence (75 % par défaut) ;
3) entraîne uniquement les tranches dans ce régime, comme E4 ;
4) exporte les tranches appariées pour le moteur V1.

Même tâche, graine, LR, époques et optimiseur léger qu'E4. CPU uniquement.
"""
import argparse, gc, json, random, sys, types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(2)
sys.path.insert(0, str(Path(__file__).parent))
from bistouri_backprop import FAMILLES
from bistouri_physique import TrancheMLP, forward_tranche
import extension_tranches as ext
from extension_tranches import SegmentN, evaluer, perte, AdamEtatsLegers, LOCAL

RACINE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fond", type=float, default=0.75)
    a = ap.parse_args()
    pct = int(round(a.fond * 100))
    torch.manual_seed(20261102); random.seed(20261102)

    data = torch.load(LOCAL / "segment_extension_32.pt", weights_only=False, mmap=True)
    exemples, etat = data["exemples"], data["etat"]
    train = [e for e in exemples if not e["eval"]]
    dev = [e for e in exemples if e["eval"]]
    debut = etat["couche_debut"]
    gps = np.load(RACINE / "artifacts" / "gps_modele_360M.npz", allow_pickle=True)
    masques = [{fam: sorted(int(n) for n in gps["masques"][i][debut + li])
                for i, fam in enumerate(FAMILLES)} for li in range(len(etat["couches"]))]
    segment = SegmentN(etat, masques, leger=True, dtype_socle=torch.bfloat16)
    del etat, data; gc.collect()

    # --- 1) Contributions observées des neurones (segment complet, avant patch)
    stats, comptes = [], {fam: 0 for fam in FAMILLES}
    originaux = []
    for li, couche in enumerate(segment.couches):
        mlp = couche.mlp_p
        stats.append({fam: torch.zeros(mlp.Wg0.shape[0]) for fam in FAMILLES})
        originaux.append(mlp.forward)

        def fabriquer_stats(mlp, li):
            def collecte(x, fam):
                with torch.no_grad():
                    xf = x.detach().float()
                    h = F.silu(F.linear(xf, mlp.Wg0.float())) * F.linear(xf, mlp.Wu0.float())
                    contrib = h.abs().mean(dim=tuple(range(h.dim() - 1))) * mlp.Wd0.float().norm(dim=0)
                    stats[li][fam] += contrib.cpu()
                return originaux[li](x, fam)
            return collecte
        mlp.forward = fabriquer_stats(mlp, li)
    segment.eval()
    with torch.no_grad():
        for ex in train:
            comptes[ex["famille"]] += 1
            segment(ex["h_debut"], ex["famille"])
    classements = {}
    for li, couche in enumerate(segment.couches):
        mlp = couche.mlp_p
        del mlp.forward
        for fam in FAMILLES:
            score = (stats[li][fam] / max(1, comptes[fam])).numpy()
            tranche = {int(v) for v in mlp.indices[fam].tolist()}
            inactifs = np.array([i for i in range(len(score)) if i not in tranche], dtype=np.int32)
            classements[f"act_{li}_{fam}"] = inactifs[np.argsort(-score[inactifs])]
    chemin_classements = LOCAL / "classements_activation.npz"
    np.savez(chemin_classements, **classements)
    print(f"📊 classements d'activation exportés : {chemin_classements.name}", flush=True)
    del stats; gc.collect()

    # --- 2) Patch apparié : inactive déjà restreinte au fond élite magnitude
    for li, couche in enumerate(segment.couches):
        mlp = couche.mlp_p
        mlp.elites = {}
        for fam in FAMILLES:
            mag = (mlp.Wg0.float().norm(dim=1) + mlp.Wu0.float().norm(dim=1)
                   + mlp.Wd0.float().norm(dim=0))
            tranche = {int(v) for v in mlp.indices[fam].tolist()}
            inactifs = torch.tensor([i for i in range(mag.numel()) if i not in tranche])
            elite = inactifs[torch.argsort(mag[inactifs], descending=True)]
            mlp.elites[fam] = elite[:int(round(a.fond * len(elite)))]

        def fabriquer_apparie(mlp):
            def forward_apparie(x, fam):
                elite = mlp.elites[fam]
                with torch.no_grad():
                    h = (F.silu(F.linear(x.detach(), mlp.Wg0[elite]))
                         * F.linear(x.detach(), mlp.Wu0[elite]))
                    y_fond = F.linear(h, mlp.Wd0[:, elite])
                tranche = TrancheMLP(mlp.Wg[fam], mlp.Wu[fam], mlp.Wd[fam], mlp.indices[fam])
                return (y_fond.float() + forward_tranche(x.float(), tranche)).to(x.dtype)
            return forward_apparie
        mlp.forward = fabriquer_apparie(mlp)

    avant = evaluer(segment, dev)
    params = [p for c in segment.couches for p in c.mlp_p.parameters() if p.requires_grad]
    opt = AdamEtatsLegers(params, lr=ext.LR)
    par_famille = {f: [e for e in train if e["famille"] == f] for f in FAMILLES}
    pas = 0
    for ep in range(ext.EPOQUES):
        for fam in FAMILLES:
            melange = par_famille[fam][:]
            random.Random(1000 + ep).shuffle(melange)
            for ex in melange:
                segment.train(); opt.zero_grad()
                perte(segment, ex).backward(); opt.step(); pas += 1
    apres = evaluer(segment, dev)
    print(f"🎯 entraînement apparié {pct}% : {avant:.4f} → {apres:.4f} "
          f"({(avant-apres)/avant:+.1%}, {pas} pas)", flush=True)

    sortie = {}
    for li, couche in enumerate(segment.couches):
        for fam in FAMILLES:
            for nom, p in [("Wg", couche.mlp_p.Wg[fam]), ("Wu", couche.mlp_p.Wu[fam]),
                           ("Wd", couche.mlp_p.Wd[fam])]:
                sortie[f"c{li}_{fam}_{nom}"] = p.data.detach().to(torch.bfloat16).view(torch.int16).numpy()
    cible = LOCAL / f"tranches_appariees_{pct}.npz"
    np.savez(cible, **sortie)
    bilan = {"marche": "entrainement apparie au regime d'inference", "fond": a.fond,
             "perte_eval_avant_regime": round(avant, 4),
             "perte_eval_apres": round(apres, 4),
             "amelioration_relative": round((avant - apres) / avant, 4),
             "pas": pas, "tranches": cible.name,
             "classements_activation": chemin_classements.name,
             "reference_E4_non_apparie": {"perte_eval_apres": 1.2449}}
    (RACINE / "results" / f"RESULTATS_ENTRAINEMENT_APPARIE_{pct}.json").write_text(
        json.dumps(bilan, ensure_ascii=False, indent=1))
    print(f"💾 {cible.name} + results/RESULTATS_ENTRAINEMENT_APPARIE_{pct}.json", flush=True)


if __name__ == "__main__":
    main()
