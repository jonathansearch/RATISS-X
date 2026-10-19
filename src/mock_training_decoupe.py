#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⏱️ MOCK-TRAINING D2 — banc de mécanique du découpage physique (PAS un entraînement de LLM).
Proposition validée : docs/PROPOSITION_DECOUPAGE_PHYSIQUE.md, marche D2.

Paramètres FIGÉS avant exécution (ne pas retoucher après coup) :
  bloc       : MLP réel de la couche 15 de SmolLM2-360M (gate/up/down, BF16→FP32)
  accord     : masque GPS GEOGRAPHIE de la couche 15 (artifacts/gps_modele_360M.npz)
  entrée     : lot fixe de 128 vecteurs, graine 20261104 (synthétique, étiqueté)
  tâche      : régression vers la sortie du bloc maître (dense : sortie complète ;
               tranche : composante active du maître — tâches étiquetées séparément)
  étudiant   : poids maître + bruit graine figée (amplitude 0,01) — sinon perte nulle
  optimiseur : Adam, lr 1e-3 · échauffement 10 pas · mesure 50 pas · CPU, 2 fils
Mesures : temps/pas (médiane), RAM pic du processus (RSS), éléments de gradient
réellement calculés (premier pas), mémoire des états Adam (mesurée sur les tenseurs).
Le coût d'extraction de la tranche est chronométré À PART et rapporté, jamais caché.
"""
import argparse, json, resource, statistics, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(2)
sys.path.insert(0, str(Path(__file__).parent))
from bistouri_physique import extraire_tranche, forward_tranche, forward_dense

RACINE = Path(__file__).resolve().parent.parent
COUCHE, FAMILLE = 15, "GEOGRAPHIE"
GRAINE, TOKENS, LR, ECHAUFFEMENT, PAS = 20261104, 128, 1e-3, 10, 50


def charger_bloc():
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    racine = Path(snapshot_download("HuggingFaceTB/SmolLM2-360M-Instruct"))
    with safe_open(str(racine / "model.safetensors"), framework="pt") as f:
        base = f"model.layers.{COUCHE}.mlp"
        return tuple(f.get_tensor(f"{base}.{n}.weight").float()
                     for n in ("gate_proj", "up_proj", "down_proj"))


def indices_accord():
    gps = np.load(RACINE / "artifacts" / "gps_modele_360M.npz", allow_pickle=True)
    familles = list(gps["familles"])
    return sorted(int(n) for n in gps["masques"][familles.index(FAMILLE)][COUCHE])


def mesurer_extraction(Wg, Wu, Wd, idx):
    temps = []
    for _ in range(30):
        t0 = time.perf_counter(); extraire_tranche(Wg, Wu, Wd, idx)
        temps.append((time.perf_counter() - t0) * 1000)
    return float(statistics.median(temps))


def scenario(nom):
    torch.manual_seed(GRAINE)
    Wg0, Wu0, Wd0 = charger_bloc()
    idx = indices_accord()
    extraction_ms = mesurer_extraction(Wg0, Wu0, Wd0, idx) if nom == "tranche" else 0.0

    x = torch.randn(TOKENS, Wg0.shape[1])                     # lot fixe, synthétique
    with torch.no_grad():
        if nom == "dense":
            cible = forward_dense(x, Wg0, Wu0, Wd0)
        else:
            cible = forward_tranche(x, extraire_tranche(Wg0, Wu0, Wd0, idx))

    bruit = lambda W: W + 0.01 * torch.randn_like(W)          # étudiant ≠ maître
    if nom == "dense":
        params = [bruit(W).requires_grad_(True) for W in (Wg0, Wu0, Wd0)]
        def avant(): return forward_dense(x, *params)
    else:
        tr = extraire_tranche(bruit(Wg0), bruit(Wu0), bruit(Wd0), idx)
        params = [tr.W_gate.requires_grad_(True), tr.W_up.requires_grad_(True),
                  tr.W_down.requires_grad_(True)]
        tranche = type(tr)(params[0], params[1], params[2], tr.indices)
        def avant(): return forward_tranche(x, tranche)

    opt = torch.optim.Adam(params, lr=LR)
    pertes, temps, grads_premier_pas = [], [], None
    for pas in range(ECHAUFFEMENT + PAS):
        t0 = time.perf_counter()
        opt.zero_grad()
        perte = F.mse_loss(avant(), cible)
        perte.backward()
        if grads_premier_pas is None:
            grads_premier_pas = int(sum(int((p.grad != 0).sum()) for p in params))
        opt.step()
        if pas >= ECHAUFFEMENT:
            temps.append(time.perf_counter() - t0); pertes.append(float(perte))
    etats_adam = int(sum(v.numel() for p in params for v in opt.state[p].values()
                         if isinstance(v, torch.Tensor))) * 4   # octets FP32

    bilan = {"scenario": nom, "couche": COUCHE, "famille_accord": FAMILLE,
             "n_actifs": len(idx) if nom == "tranche" else int(Wg0.shape[0]),
             "parametres_entraines": int(sum(p.numel() for p in params)),
             "elements_gradient_premier_pas": grads_premier_pas,
             "etats_adam_octets": etats_adam,
             "temps_par_pas_ms_median": float(statistics.median(temps)) * 1000,
             "extraction_tranche_ms_median": extraction_ms,
             "perte_debut_mesure": pertes[0], "perte_fin": pertes[-1],
             "ram_pic_mo": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
             "pas_mesures": PAS, "tokens_par_pas": TOKENS}
    sortie = RACINE / "results" / f"RESULTATS_DECOUPAGE_D2_{nom}.json"
    sortie.write_text(json.dumps(bilan, ensure_ascii=False, indent=1))
    print(f"⏱️ {nom}: {bilan['temps_par_pas_ms_median']:.1f} ms/pas | RAM {bilan['ram_pic_mo']} Mo | "
          f"grads {grads_premier_pas:,} | perte {pertes[0]:.5f} → {pertes[-1]:.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["dense", "tranche"], required=True)
    scenario(ap.parse_args().scenario)
