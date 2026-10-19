#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚖️ DUEL DENSE — F1 (protocole docs/PROTOCOLE_DUEL_ET_ROUTAGE_ET_MOTEUR.md).
Face à la tranche (E2) : le même segment de 8 couches, mais MLP DENSE FP32
entièrement entraînable (59,0 M de paramètres), attention gelée BF16, Adam FP32
classique, même tâche/graine/époques/masque causal. CPU uniquement.
"""
import gc, json, random, resource, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)
sys.path.insert(0, str(Path(__file__).parent))
from bistouri_backprop import FAMILLES
import extension_tranches as ext
from extension_tranches import SegmentN, evaluer, perte, LOCAL

RACINE = Path(__file__).resolve().parent.parent
N_COUCHES = 8


class SegmentDense(SegmentN):
    """SegmentN dont le MLP reste dense et entraînable (l'adversaire du duel)."""
    def __init__(self, etat, n):
        nn.Module.__init__(self)
        self.leger = True; self.dtype_socle = torch.bfloat16
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
        from integration_tranche_segment import CoucheP

        class MLPDense(nn.Module):
            def __init__(self, mlp): super().__init__(); self.mlp = mlp
            def forward(self, x, fam=None):
                return self.mlp(x.float()).to(x.dtype)   # MLP FP32, flux BF16

        self.couches = nn.ModuleList()
        for etat_couche in etat["couches"]:
            with torch.device("meta"):
                c = LlamaDecoderLayer(config=conf, layer_idx=28)
            c.load_state_dict(etat_couche, assign=True)
            self.couches.append(CoucheP(c, MLPDense(c.mlp)))
        self.norme = LlamaRMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.norme.load_state_dict(etat["norme"])
        self.tete = nn.Linear(cfg["hidden_size"], etat["tete"].shape[0], bias=False)
        self.tete.weight = nn.Parameter(etat["tete"], requires_grad=False)
        for p in self.parameters(): p.requires_grad = False
        self.to(torch.bfloat16)
        for couche in self.couches:      # le MLP dense, seul, passe FP32 entraînable
            for p in couche.mlp_p.mlp.parameters():
                p.data = p.data.float(); p.requires_grad = True


def main():
    torch.manual_seed(20261102); random.seed(20261102)
    data = torch.load(LOCAL / f"segment_extension_{N_COUCHES}.pt", weights_only=False, mmap=True)
    exemples, etat = data["exemples"], data["etat"]
    train = [e for e in exemples if not e["eval"]]; dev = [e for e in exemples if e["eval"]]
    perte_pleine = data["perte_pleine_dev"]
    segment = SegmentDense(etat, N_COUCHES)
    del etat, data; gc.collect()
    avant = evaluer(segment, dev)
    print(f"🩺 identité dense : {avant:.4f} contre plein {perte_pleine:.4f} "
          f"→ écart {abs(avant-perte_pleine):.4f}", flush=True)
    params = [p for c in segment.couches for p in c.mlp_p.mlp.parameters() if p.requires_grad]
    print(f"⚖️ paramètres entraînables (dense) : {sum(p.numel() for p in params):,}", flush=True)
    opt = torch.optim.Adam(params, lr=ext.LR)
    par_famille = {f: [e for e in train if e["famille"] == f] for f in FAMILLES}
    maj, t0, pas = None, time.time(), 0
    for ep in range(ext.EPOQUES):
        for fam in FAMILLES:
            melange = par_famille[fam][:]
            random.Random(1000 + ep).shuffle(melange)
            for ex in melange:
                segment.train(); opt.zero_grad()
                perte(segment, ex).backward()
                if maj is None:
                    maj = int(sum(int((p.grad != 0).sum()) for p in params if p.grad is not None))
                opt.step(); pas += 1
    duree = time.time() - t0
    apres = evaluer(segment, dev)
    bilan = {"duel": "dense MLP FP32 vs tranche (E2), segment 8 couches, même protocole",
             "perte_eval_avant": round(avant, 4), "perte_eval_apres": round(apres, 4),
             "amelioration_relative": round((avant - apres) / avant, 4),
             "params_entrainables_total": int(sum(p.numel() for p in params)),
             "params_mis_a_jour_par_pas": maj,
             "secondes_par_pas": round(duree / pas, 4), "pas_total": pas,
             "ram_pic_mo": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
             "tranche_E2": {"perte_eval_apres": 2.1216, "maj_par_pas": 2874240,
                            "secondes_par_pas": 0.0975, "ram_pic_mo": 1546.7}}
    (RACINE / "results" / "RESULTATS_DUEL_DENSE_8.json").write_text(
        json.dumps(bilan, ensure_ascii=False, indent=1))
    print(f"⚖️ DENSE 8 couches : {avant:.4f} → {apres:.4f} ({(avant-apres)/avant:+.1%}) | "
          f"maj/pas={maj:,} | {bilan['secondes_par_pas']}s/pas | RAM {bilan['ram_pic_mo']} Mo")


if __name__ == "__main__":
    main()
