#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🪜 EXTENSION DES TRANCHES — Option 1, marche E1 (protocole docs/PROTOCOLE_EXTENSION_E1.md).
Généralise la chirurgie D3 à N dernières couches (défaut figé : 4).
Usage :
  python3 src/extension_tranches.py --preparer
  python3 src/extension_tranches.py --entrainer
CPU uniquement. Fichiers lourds locaux (artifacts/local/), jamais versionnés.
"""
import argparse, gc, json, random, resource, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
sys.path.insert(0, str(Path(__file__).parent))
from bistouri_backprop import QA, FAMILLES
from integration_tranche_segment import MLPPhysique, CoucheP

RACINE = Path(__file__).resolve().parent.parent
LOCAL = RACINE / "artifacts" / "local"
MODELE = "HuggingFaceTB/SmolLM2-360M-Instruct"
N_COUCHES_SEGMENT = 4          # défaut = marche E1 ; E2 passe --couches 8
EPOQUES, LR = 3, 5e-4          # identiques à D3


def preparer(n_couches):
    global N_COUCHES_SEGMENT
    N_COUCHES_SEGMENT = n_couches
    from transformers import AutoModelForCausalLM, AutoTokenizer
    LOCAL.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODELE)
    modele = AutoModelForCausalLM.from_pretrained(MODELE, dtype=torch.bfloat16)
    modele.eval()
    total = modele.config.num_hidden_layers
    debut = total - N_COUCHES_SEGMENT
    captures = []
    crochet = modele.model.layers[debut].register_forward_pre_hook(
        lambda m, args: captures.append(args[0].detach().clone()))
    exemples, pertes_pleines = [], []
    for fam in FAMILLES:
        for pos, (q, rep) in enumerate(QA[fam]):
            prompt = f"Question: {q}\nAnswer:"
            ids_p = tok(prompt, add_special_tokens=False)["input_ids"]
            ids_r = tok(" " + rep, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            ids = torch.tensor([ids_p + ids_r])
            captures.clear()
            with torch.no_grad():
                logits = modele(input_ids=ids).logits[0]
            cible = ids[0][len(ids_p):]
            perte = F.cross_entropy(logits[len(ids_p) - 1:-1].float(), cible).item()
            exemples.append({"famille": fam, "eval": pos in {4, 9}, "ids": ids[0],
                             "debut_reponse": len(ids_p), "h_debut": captures[0][0].float()})
            if pos in {4, 9}: pertes_pleines.append(perte)
    crochet.remove()
    etat = {"couches": [modele.model.layers[debut + i].state_dict()
                        for i in range(N_COUCHES_SEGMENT)],
            "norme": modele.model.norm.state_dict(),
            "rotary": modele.model.rotary_emb.state_dict(),
            "tete": modele.get_output_embeddings().weight.detach().clone(),
            "config": modele.config.to_dict(), "couche_debut": debut}
    torch.save({"exemples": exemples, "etat": etat,
                "perte_pleine_dev": float(np.mean(pertes_pleines))},
               LOCAL / f"segment_extension_{N_COUCHES_SEGMENT}.pt")
    print(f"✅ segment {N_COUCHES_SEGMENT} couches préparé (début couche {debut}) | "
          f"perte modèle plein sur l'éval : {np.mean(pertes_pleines):.4f}")
    (LOCAL / f"segment_extension_{N_COUCHES_SEGMENT}.meta").write_text(str(debut))


class SegmentN(nn.Module):
    def __init__(self, etat, masques_par_couche, leger=False, dtype_socle=torch.bfloat16):
        super().__init__()
        self.leger = leger
        self.dtype_socle = dtype_socle
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
        for li, etat_couche in enumerate(etat["couches"]):
            c = LlamaDecoderLayer(config=conf, layer_idx=28)
            c.load_state_dict(etat_couche)
            physique = MLPPhysique(c.mlp, masques_par_couche[li], leger=leger)
            if leger:
                c.mlp = nn.Identity()   # M1 : on libère l'ancienne résidence MLP
            self.couches.append(CoucheP(c, physique))
        self.norme = LlamaRMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.norme.load_state_dict(etat["norme"])
        self.tete = nn.Linear(cfg["hidden_size"], etat["tete"].shape[0], bias=False)
        self.tete.weight = nn.Parameter(etat["tete"], requires_grad=False)
        for p in self.parameters(): p.requires_grad = False
        if leger:
            self.to(self.dtype_socle)   # tout le socle gelé descend (BF16 ou FP16)
            for couche in self.couches:  # ... sauf les tranches, remises en FP32
                for p in couche.mlp_p.parameters():
                    p.data = p.data.float(); p.requires_grad = True
        else:
            self.float()
            for couche in self.couches:
                for p in couche.mlp_p.parameters(): p.requires_grad = True

    def forward(self, h, fam):
        if self.leger: h = h.to(self.dtype_socle)
        h = h.unsqueeze(0)
        seq = h.shape[1]
        pos = torch.arange(seq, device=h.device).unsqueeze(0)
        pos_emb = self.rotary(h, pos)
        # Masque causal explicite : aucun token ne voit le futur (piège détecté
        # au contrôle d'identité E1 — voir rapport). Sans lui, la perte ment.
        masque = torch.full((seq, seq), float("-inf"), dtype=h.dtype, device=h.device)
        masque = torch.triu(masque, diagonal=1).view(1, 1, seq, seq)
        for couche in self.couches: h = couche(h, pos_emb, fam, masque)
        return self.tete(self.norme(h))[0]


def perte(segment, ex):
    logits = segment(ex["h_debut"], ex["famille"])
    cible = ex["ids"][ex["debut_reponse"]:]
    return F.cross_entropy(logits[ex["debut_reponse"] - 1:-1].float(), cible)


@torch.no_grad()
def evaluer(segment, exemples):
    segment.eval()
    return float(np.mean([perte(segment, ex).item() for ex in exemples]))


def entrainer(n_couches, marche, leger=False, dtype_socle=torch.bfloat16):
    torch.manual_seed(20261102); random.seed(20261102)
    data = torch.load(LOCAL / f"segment_extension_{n_couches}.pt", weights_only=False)
    exemples, etat = data["exemples"], data["etat"]
    train = [e for e in exemples if not e["eval"]]; dev = [e for e in exemples if e["eval"]]
    debut, total = etat["couche_debut"], etat["config"]["num_hidden_layers"]

    gps = np.load(RACINE / "artifacts" / "gps_modele_360M.npz", allow_pickle=True)
    masques = [{fam: sorted(int(n) for n in gps["masques"][i][debut + li])
                for i, fam in enumerate(FAMILLES)} for li in range(len(etat["couches"]))]

    perte_pleine = data["perte_pleine_dev"]
    segment = SegmentN(etat, masques, leger=leger, dtype_socle=dtype_socle)
    del etat, data; gc.collect()   # M1 : libérer AUSSI le conteneur (copies d'état ~250 Mo)
    perte_avant = evaluer(segment, dev)
    print(f"🩺 identité : segment opéré (avant) {perte_avant:.4f} contre modèle plein "
          f"{perte_pleine:.4f} → écart {abs(perte_avant - perte_pleine):.4f}")

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
                    maj_mesuree = int(sum(int((p.grad != 0).sum()) for p in params if p.grad is not None))
                opt.step(); pas += 1
    duree = time.time() - t0
    perte_apres = evaluer(segment, dev)
    bilan = {"marche": marche, "mode_memoire_legere_bf16": leger,
             "couches_operees": [debut, total - 1],
             "perte_modele_plein_dev": round(perte_pleine, 4),
             "perte_eval_avant": round(perte_avant, 4), "perte_eval_apres": round(perte_apres, 4),
             "amelioration_relative": round((perte_avant - perte_apres) / perte_avant, 4),
             "params_entrainables_total": int(sum(p.numel() for p in params)),
             "params_mis_a_jour_par_pas": maj_mesuree,
             "secondes_par_pas": round(duree / pas, 4), "pas_total": pas,
             "ram_pic_mo": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
             "reference_D3_2_couches": {"perte_apres": 2.389, "amelioration_relative": 0.6096,
                                        "maj_par_pas": 725760, "s_par_pas": 0.0497}}
    (RACINE / "results" / f"RESULTATS_EXTENSION_{marche}.json").write_text(
        json.dumps(bilan, ensure_ascii=False, indent=1))
    print(f"🪜 {marche} ({n_couches} couches) : perte {perte_avant:.4f} → {perte_apres:.4f} "
          f"({bilan['amelioration_relative']:+.1%}) | maj/pas={maj_mesuree:,} | "
          f"{bilan['secondes_par_pas']}s/pas | RAM {bilan['ram_pic_mo']} Mo")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preparer", action="store_true")
    ap.add_argument("--entrainer", action="store_true")
    ap.add_argument("--couches", type=int, default=4)
    ap.add_argument("--marche", default="E1")
    ap.add_argument("--memoire_leger", action="store_true", help="M1 : socle BF16, tranches FP32")
    ap.add_argument("--dtype_socle", choices=["bf16", "fp16"], default="bf16")
    a = ap.parse_args()
    if a.preparer: preparer(a.couches)
    elif a.entrainer: entrainer(a.couches, a.marche, leger=a.memoire_leger,
                                dtype_socle=torch.bfloat16 if a.dtype_socle == 'bf16' else torch.float16)
