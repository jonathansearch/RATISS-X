#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
💉 BISTOURI RATISS — Étape 2 : backprop par raccourci, mesurée sur CPU.
Compare, sur les 2 dernières couches du cobaye SmolLM2-135M :
  base (gelé) | dense (tous les MLP entraînables) | raccourci (delta sur le masque Mk seul).
Aucun GPU. RAM = RSS du processus. RATISS Labs — novembre 2026.
"""
import argparse, copy, json, random, resource, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
RACINE = Path(__file__).resolve().parent.parent
LOCAL = RACINE / "artifacts" / "bistouri_local"   # fichiers lourds, JAMAIS versionnés
MODELE = "HuggingFaceTB/SmolLM2-360M-Instruct"  # le grand frère, ordre du chef
FAMILLES = ["GEOGRAPHIE", "CALCUL", "SCIENCE", "LANGAGE", "ANIMAUX", "CUISINE"]
DEV_POS = {4, 9}          # positions figées d'évaluation dans chaque famille
EPOQUES, LR = 3, 5e-4     # figés dans le protocole

QA = {
 "GEOGRAPHIE": [("What is the capital of France?", "Paris"), ("What is the capital of Spain?", "Madrid"),
   ("What is the capital of Italy?", "Rome"), ("What is the capital of Canada?", "Ottawa"),
   ("What is the capital of Japan?", "Tokyo"), ("What is the largest ocean on Earth?", "The Pacific Ocean"),
   ("On which continent is the Sahara desert?", "Africa"), ("Which river flows through Paris?", "The Seine"),
   ("What is the capital of Germany?", "Berlin"), ("In which country is Mount Everest?", "Nepal")],
 "CALCUL": [("How much is 2 plus 2?", "4"), ("How much is 7 times 6?", "42"),
   ("How much is 100 divided by 4?", "25"), ("How much is 15 minus 9?", "6"),
   ("How much is 3 times 3?", "9"), ("What is half of 50?", "25"),
   ("How much is 12 plus 18?", "30"), ("How much is 9 times 9?", "81"),
   ("How much is 100 minus 1?", "99"), ("What is double 21?", "42")],
 "SCIENCE": [("Is the Sun a star?", "Yes"), ("What is the chemical formula of water?", "H2O"),
   ("Why is the sky blue?", "Because air scatters blue light"), ("What is gravity?", "A force that attracts masses"),
   ("At what temperature does water boil?", "100 degrees Celsius"), ("Which planet is closest to the Sun?", "Mercury"),
   ("What do plants need to grow?", "Light, water and air"), ("What happens when ice is heated?", "It melts into water"),
   ("Why does the Moon shine at night?", "It reflects sunlight"), ("What is a black hole?", "A region where gravity traps light")],
 "LANGAGE": [("How do you say bonjour in English?", "Hello"), ("What is the plural of horse in French?", "Chevaux"),
   ("What is the opposite of big?", "Small"), ("How do you say thank you in Spanish?", "Gracias"),
   ("How many letters are in the French alphabet?", "26"), ("What does the word house mean in French?", "Maison"),
   ("How do you say cat in English?", "Cat"), ("What is the opposite of hot?", "Cold"),
   ("How do you say merci in English?", "Thank you"), ("What is the plural of child?", "Children")],
 "ANIMAUX": [("Which animal meows and hunts mice?", "A cat"), ("How many legs does a spider have?", "Eight"),
   ("What is the largest land animal?", "The African elephant"), ("Which animal makes honey?", "The bee"),
   ("What does a giant panda eat?", "Bamboo"), ("Which animal barks and guards the house?", "A dog"),
   ("Which bird is a symbol of wisdom?", "The owl"), ("Which fish has sharp teeth and lives in the ocean?", "A shark"),
   ("Which animal lays eggs and flies?", "A bird"), ("How many legs does an insect have?", "Six")],
 "CUISINE": [("What ingredients do you need for an omelette?", "Eggs, butter and salt"),
   ("How long does a hard-boiled egg take?", "About ten minutes"), ("How do you melt chocolate gently?", "In a bain-marie"),
   ("Which spice gives curry its yellow color?", "Turmeric"), ("What makes bread rise?", "Yeast"),
   ("What do you put in a fruit salad?", "Fresh fruits"), ("How do you cook simple white rice?", "Boil it in water"),
   ("What utensil stirs a soup?", "A spoon"), ("How do you make a simple coffee?", "Pour hot water over ground coffee"),
   ("At what temperature do you bake a cake?", "About 180 degrees Celsius")],
}

# ---------------------------------------------------------------- préparation
def preparer():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    LOCAL.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODELE)
    modele = AutoModelForCausalLM.from_pretrained(MODELE, dtype=torch.bfloat16); modele.eval()
    exemples, captures = [], []
    n_couches = modele.config.num_hidden_layers
    crochet = modele.model.layers[n_couches - 2].register_forward_pre_hook(
        lambda m, args: captures.append(args[0].detach().clone()))
    for fam in FAMILLES:
        for pos, (q, rep) in enumerate(QA[fam]):
            prompt = f"Question: {q}\nAnswer:"
            ids_p = tok(prompt, add_special_tokens=False)["input_ids"]
            ids_r = tok(" " + rep, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            ids = torch.tensor([ids_p + ids_r])
            captures.clear()
            with torch.no_grad(): modele(input_ids=ids)
            exemples.append({"famille": fam, "eval": pos in DEV_POS, "ids": ids[0],
                             "debut_reponse": len(ids_p)})
            captures and None
            exemples[-1]["h28"] = captures[0][0].clone()
            exemples[-1]["h28"] = exemples[-1]["h28"].float()
    crochet.remove()
    etat = {"couche28": modele.model.layers[n_couches - 2].state_dict(),
            "couche29": modele.model.layers[n_couches - 1].state_dict(),
            "norme": modele.model.norm.state_dict(),
            "rotary": modele.model.rotary_emb.state_dict(),
            "tete": modele.get_output_embeddings().weight.detach().clone(),
            "config": modele.config.to_dict()}
    torch.save({"exemples": exemples, "etat": etat}, LOCAL / "segment.pt")
    print(f"✅ segment préparé : {len(exemples)} exemples, états cachés cachés localement "
          f"({(LOCAL/'segment.pt').stat().st_size/1e6:.0f} Mo, non versionné)")

# ---------------------------------------------------------------- segment
class DeltaFamille(nn.Module):
    """Correction calculée UNIQUEMENT sur les neurones du masque de la famille. ⚡"""
    def __init__(self, masques, dim=960):
        super().__init__()
        self.Wg, self.Wu, self.Wd = nn.ParameterDict(), nn.ParameterDict(), nn.ParameterDict()
        for fam, idx in masques.items():
            self.register_buffer(f"idx_{fam}", idx)
            n = len(idx)
            self.Wg[fam] = nn.Parameter(torch.randn(n, dim) * 0.02)
            self.Wu[fam] = nn.Parameter(torch.randn(n, dim) * 0.02)
            self.Wd[fam] = nn.Parameter(torch.zeros(dim, n))   # zéro : départ = modèle intact
    def forward(self, x, fam):
        return F.linear(F.silu(F.linear(x, self.Wg[fam])) * F.linear(x, self.Wu[fam]), self.Wd[fam])

class Couche(nn.Module):
    def __init__(self, couche, delta=None):
        super().__init__(); self.couche = couche; self.delta = delta
    def forward(self, h, pos_emb, fam, attention_mask=None):
        reste = h
        x = self.couche.input_layernorm(h)
        att, _ = self.couche.self_attn(hidden_states=x, attention_mask=attention_mask,
                                       position_embeddings=pos_emb)
        h = reste + att
        reste = h
        x = self.couche.post_attention_layernorm(h)
        sortie = self.couche.mlp(x)
        if self.delta is not None: sortie = sortie + self.delta(x, fam)
        return reste + sortie

class Segment(nn.Module):
    def __init__(self, etat, masques=None):
        super().__init__()
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import (
            LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding)
        cfg = etat["config"]
        conf = LlamaConfig(
            hidden_size=cfg["hidden_size"], intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            rms_norm_eps=cfg["rms_norm_eps"], rope_theta=cfg["rope_parameters"]["rope_theta"],
            max_position_embeddings=cfg["max_position_embeddings"],
            vocab_size=cfg["vocab_size"])
        self.rotary = LlamaRotaryEmbedding(config=conf)
        self.rotary.load_state_dict(etat["rotary"])
        couches = []
        for nom in ("couche28", "couche29"):
            c = LlamaDecoderLayer(config=conf, layer_idx=28)
            c.load_state_dict(etat[nom])
            delta = DeltaFamille(masques, dim=cfg["hidden_size"]) if masques else None
            couches.append(Couche(c, delta))
        self.couches = nn.ModuleList(couches)
        self.norme = LlamaRMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.norme.load_state_dict(etat["norme"])
        self.tete = nn.Linear(cfg["hidden_size"], etat["tete"].shape[0], bias=False)
        self.tete.weight = nn.Parameter(etat["tete"], requires_grad=False)
        for p in self.parameters(): p.requires_grad = False
        self.float()   # segment d'entraînement en FP32, quel que soit le cobaye chargé

    def forward(self, h, fam=None):
        h = h.unsqueeze(0)                                   # (1, seq, dim)
        seq = h.shape[1]
        pos = torch.arange(seq, device=h.device).unsqueeze(0)
        pos_emb = self.rotary(h, pos)
        # Masque causal explicite (revalidation) : aucun token ne voit le futur.
        masque = torch.full((seq, seq), float("-inf"), dtype=h.dtype, device=h.device)
        masque = torch.triu(masque, diagonal=1).view(1, 1, seq, seq)
        for couche in self.couches: h = couche(h, pos_emb, fam, masque)
        return self.tete(self.norme(h))[0]                   # (seq, vocab)

# ---------------------------------------------------------------- entraînement
def perte(segment, ex):
    logits = segment(ex["h28"], ex["famille"])
    ids = ex["ids"]
    cible = ids[ex["debut_reponse"]:]
    pred = logits[ex["debut_reponse"] - 1:-1]
    return F.cross_entropy(pred.float(), cible)

@torch.no_grad()
def evaluer(segment, exemples):
    segment.eval()
    vals = [perte(segment, ex).item() for ex in exemples]
    return float(np.mean(vals))

def scenario(nom):
    torch.manual_seed(20261102); random.seed(20261102)
    data = torch.load(LOCAL / "segment.pt", weights_only=False)
    exemples, etat = data["exemples"], data["etat"]
    train = [e for e in exemples if not e["eval"]]; dev = [e for e in exemples if e["eval"]]

    masques = None
    if nom == "sparse":
        gps = np.load(RACINE / "artifacts" / "gps_modele_360M.npz", allow_pickle=True)
        masques = {}
        avant_derniere = etat["config"]["num_hidden_layers"] - 2
        for i, fam in enumerate(FAMILLES):
            # masque de la couche avant-dernière, appliqué aux deux couches (transfert documenté)
            m28 = torch.tensor(sorted(int(n) for n in gps["masques"][i][avant_derniere]))
            masques[fam] = m28
    segment = Segment(etat, masques)
    perte_avant = evaluer(segment, dev)

    if nom == "base":
        json.dump({"scenario": nom, "perte_eval_avant": perte_avant, "perte_eval_apres": perte_avant,
                   "params_mis_a_jour_par_pas": 0, "secondes_par_pas": 0.0,
                   "ram_pic_mo": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024},
                  open(RACINE / "results" / "RESULTATS_BISTOURI_base.json", "w"), indent=1)
        print(f"🩺 base (gelé) : perte éval = {perte_avant:.4f}"); return

    if nom == "dense":
        params = [p for c in segment.couches for p in c.couche.mlp.parameters()]
    else:
        params = [p for c in segment.couches for p in c.delta.parameters()]
    for p in params: p.requires_grad = True
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
    bilan = {"scenario": nom, "perte_eval_avant": round(perte_avant, 4),
             "perte_eval_apres": round(perte_apres, 4),
             "amelioration_relative": round((perte_avant - perte_apres) / perte_avant, 4),
             "params_entrainables_total": int(sum(p.numel() for p in params)),
             "params_mis_a_jour_par_pas": maj_mesuree,
             "secondes_par_pas": round(duree / pas, 4), "pas_total": pas,
             "ram_pic_mo": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    json.dump(bilan, open(RACINE / "results" / f"RESULTATS_BISTOURI_{nom}.json", "w"), indent=1)
    print(f"💉 {nom} : perte {perte_avant:.4f} → {perte_apres:.4f} | "
          f"maj/pas={maj_mesuree:,} | {bilan['secondes_par_pas']}s/pas | RAM pic {bilan['ram_pic_mo']} Mo")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preparer", action="store_true")
    ap.add_argument("--scenario", choices=["base", "dense", "sparse"])
    a = ap.parse_args()
    if a.preparer: preparer()
    elif a.scenario: scenario(a.scenario)
