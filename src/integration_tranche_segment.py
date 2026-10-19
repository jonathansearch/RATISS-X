#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔪💉 INTÉGRATION D3 — la tranche physique branchée sur le segment réel (2 dernières couches, 360M).
Proposition : docs/PROPOSITION_DECOUPAGE_PHYSIQUE.md, marche D3 (Go du chef).

Variante honnête annoncée :
  y = contribution_inactive_gelée(x_détaché, sans graphe) + forward_tranche_active(x)
- Au départ, tranche = extraction exacte des poids d'origine → le segment est
  STRICTEMENT identique au modèle de base (contrôlé par la perte initiale).
- Pendant l'entraînement, seuls les paramètres des tranches bougent ; l'erreur
  ne redescend vers la couche précédente QUE par le chemin actif (le raccourci).
- Masques : ceux du GPS, pris à LEUR couche (30 pour la première couche du
  segment, 31 pour la seconde) — plus de transfert de masque comme en Étape 2.
Mêmes données, même split, mêmes hyperparamètres que le Bistouri (Adam 5e-4,
3 époques, batch 1 groupé par famille, graine 20261102). CPU uniquement.
"""
import json, random, resource, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
sys.path.insert(0, str(Path(__file__).parent))
from bistouri_backprop import QA, FAMILLES, LOCAL          # mêmes données que l'Étape 2
from bistouri_physique import TrancheMLP, forward_tranche

RACINE = Path(__file__).resolve().parent.parent
EPOQUES, LR = 3, 5e-4


class MLPPhysique(nn.Module):
    """MLP dont seule la tranche de l'accord est un paramètre entraînable."""
    def __init__(self, mlp, masques):
        super().__init__()
        self.register_buffer("Wg0", mlp.gate_proj.weight.detach().clone())
        self.register_buffer("Wu0", mlp.up_proj.weight.detach().clone())
        self.register_buffer("Wd0", mlp.down_proj.weight.detach().clone())
        inter = self.Wg0.shape[0]
        self.register_buffer("masque_inactif", torch.ones(len(FAMILLES), inter))
        self.Wg, self.Wu, self.Wd = nn.ParameterDict(), nn.ParameterDict(), nn.ParameterDict()
        self.indices = {}
        for i, fam in enumerate(FAMILLES):
            idx = torch.tensor(masques[fam])
            self.indices[fam] = idx
            self.masque_inactif[i, idx] = 0.0
            self.Wg[fam] = nn.Parameter(self.Wg0[idx].clone())
            self.Wu[fam] = nn.Parameter(self.Wu0[idx].clone())
            self.Wd[fam] = nn.Parameter(self.Wd0[:, idx].clone())

    def forward(self, x, fam):
        i = FAMILLES.index(fam)
        with torch.no_grad():   # inactive gelée, calculée sur x détaché : aucun graphe
            h = F.silu(F.linear(x.detach(), self.Wg0)) * F.linear(x.detach(), self.Wu0)
            y_inactive = F.linear(h * self.masque_inactif[i], self.Wd0)
        tranche = TrancheMLP(self.Wg[fam], self.Wu[fam], self.Wd[fam], self.indices[fam])
        return y_inactive + forward_tranche(x, tranche)


class CoucheP(nn.Module):
    def __init__(self, couche, mlp_physique):
        super().__init__(); self.couche = couche; self.mlp_p = mlp_physique
    def forward(self, h, pos_emb, fam):
        reste = h
        x = self.couche.input_layernorm(h)
        att, _ = self.couche.self_attn(hidden_states=x, attention_mask=None,
                                       position_embeddings=pos_emb)
        h = reste + att
        reste = h
        x = self.couche.post_attention_layernorm(h)
        return reste + self.mlp_p(x, fam)


class SegmentP(nn.Module):
    def __init__(self, etat, masques_par_couche):
        super().__init__()
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import (
            LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding)
        cfg = etat["config"]
        conf = LlamaConfig(hidden_size=cfg["hidden_size"],
                           intermediate_size=cfg["intermediate_size"],
                           num_hidden_layers=cfg["num_hidden_layers"],
                           num_attention_heads=cfg["num_attention_heads"],
                           num_key_value_heads=cfg["num_key_value_heads"],
                           rms_norm_eps=cfg["rms_norm_eps"],
                           rope_theta=cfg["rope_parameters"]["rope_theta"],
                           max_position_embeddings=cfg["max_position_embeddings"],
                           vocab_size=cfg["vocab_size"])
        self.rotary = LlamaRotaryEmbedding(config=conf)
        self.rotary.load_state_dict(etat["rotary"])
        self.couches = nn.ModuleList()
        for li, nom in enumerate(("couche28", "couche29")):
            c = LlamaDecoderLayer(config=conf, layer_idx=28)
            c.load_state_dict(etat[nom])
            self.couches.append(CoucheP(c, MLPPhysique(c.mlp, masques_par_couche[li])))
        self.norme = LlamaRMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.norme.load_state_dict(etat["norme"])
        self.tete = nn.Linear(cfg["hidden_size"], etat["tete"].shape[0], bias=False)
        self.tete.weight = nn.Parameter(etat["tete"], requires_grad=False)
        for p in self.parameters(): p.requires_grad = False
        self.float()
        for couche in self.couches:                    # seules les tranches s'entraînent
            for p in couche.mlp_p.parameters(): p.requires_grad = True

    def forward(self, h, fam):
        h = h.unsqueeze(0)
        pos = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
        pos_emb = self.rotary(h, pos)
        for couche in self.couches: h = couche(h, pos_emb, fam)
        return self.tete(self.norme(h))[0]


def perte(segment, ex):
    logits = segment(ex["h28"], ex["famille"])
    cible = ex["ids"][ex["debut_reponse"]:]
    return F.cross_entropy(logits[ex["debut_reponse"] - 1:-1].float(), cible)


@torch.no_grad()
def evaluer(segment, exemples):
    segment.eval()
    return float(np.mean([perte(segment, ex).item() for ex in exemples]))


def main():
    torch.manual_seed(20261102); random.seed(20261102)
    data = torch.load(LOCAL / "segment.pt", weights_only=False)
    exemples, etat = data["exemples"], data["etat"]
    train = [e for e in exemples if not e["eval"]]; dev = [e for e in exemples if e["eval"]]
    n_couches = etat["config"]["num_hidden_layers"]

    gps = np.load(RACINE / "artifacts" / "gps_modele_360M.npz", allow_pickle=True)
    masques_par_couche = []
    for li, couche_modele in enumerate((n_couches - 2, n_couches - 1)):
        masques_par_couche.append({fam: sorted(int(n) for n in gps["masques"][i][couche_modele])
                                   for i, fam in enumerate(FAMILLES)})

    segment = SegmentP(etat, masques_par_couche)
    perte_avant = evaluer(segment, dev)
    base = json.loads((RACINE / "results" / "RESULTATS_BISTOURI_base.json").read_text())
    print(f"🩺 perte initiale segment-tranche : {perte_avant:.4f} "
          f"(base gelée de référence : {base['perte_eval_avant']:.4f})")

    params = [p for c in segment.couches for p in c.mlp_p.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    par_famille = {f: [e for e in train if e["famille"] == f] for f in FAMILLES}
    maj_mesuree, t0, pas = None, time.time(), 0
    for ep in range(EPOQUES):
        for fam in FAMILLES:
            melange = par_famille[fam][:]
            random.Random(1000 + ep).shuffle(melange)
            for ex in melange:
                segment.train(); opt.zero_grad()
                l = perte(segment, ex); l.backward()
                if maj_mesuree is None:
                    maj_mesuree = int(sum(int((p.grad != 0).sum()) for p in params
                                          if p.grad is not None))
                opt.step(); pas += 1
    duree = time.time() - t0
    perte_apres = evaluer(segment, dev)

    ref_dense = json.loads((RACINE / "results" / "RESULTATS_BISTOURI_dense.json").read_text())
    ref_corr = json.loads((RACINE / "results" / "RESULTATS_BISTOURI_sparse.json").read_text())
    bilan = {"methode": "tranche physique (inactive gelée sans graphe + tranche entraînable)",
             "perte_eval_avant": round(perte_avant, 4), "perte_eval_apres": round(perte_apres, 4),
             "amelioration_relative": round((perte_avant - perte_apres) / perte_avant, 4),
             "ecart_initial_vs_base": round(abs(perte_avant - base["perte_eval_avant"]), 4),
             "params_entrainables_total": int(sum(p.numel() for p in params)),
             "params_mis_a_jour_par_pas": maj_mesuree,
             "secondes_par_pas": round(duree / pas, 4), "pas_total": pas,
             "ram_pic_mo": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
             "references_meme_protocole": {
                 "dense": {"apres": ref_dense["perte_eval_apres"],
                           "maj_par_pas": ref_dense["params_mis_a_jour_par_pas"],
                           "s_par_pas": ref_dense["secondes_par_pas"], "ram_mo": ref_dense["ram_pic_mo"]},
                 "correction_sparse_etape2": {"apres": ref_corr["perte_eval_apres"],
                           "maj_par_pas": ref_corr["params_mis_a_jour_par_pas"],
                           "s_par_pas": ref_corr["secondes_par_pas"], "ram_mo": ref_corr["ram_pic_mo"]}}}
    (RACINE / "results" / "RESULTATS_DECOUPAGE_D3.json").write_text(
        json.dumps(bilan, ensure_ascii=False, indent=1))
    print(f"🔪 tranche physique : perte {perte_avant:.4f} → {perte_apres:.4f} "
          f"({bilan['amelioration_relative']:+.1%}) | maj/pas={maj_mesuree:,} | "
          f"{bilan['secondes_par_pas']}s/pas | RAM {bilan['ram_pic_mo']} Mo")


if __name__ == "__main__":
    main()
