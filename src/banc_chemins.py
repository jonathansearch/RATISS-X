#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛤️ BANC DES CHEMINS (générique) — lit les itinéraires d'activation d'un cobaye,
sans le modifier. Utilisé pour le 135M (banc n°1) puis refait sur le grand frère 360M.
RATISS Labs — CPU uniquement — les chemins bruts restent locaux (jamais versionnés).
"""
import argparse, json, random, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(2)
RACINE = Path(__file__).resolve().parent.parent
TOP_FRAC = 0.05
random.seed(20261018)

def jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--etiquette", default="360M")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    stimuli = json.loads((RACINE / "data" / "questionnaire.json").read_text())
    print(f"🔬 Chargement du cobaye {a.modele} (BF16, CPU)...", flush=True)
    tok = AutoTokenizer.from_pretrained(a.modele)
    modele = AutoModelForCausalLM.from_pretrained(a.modele, dtype=torch.bfloat16)
    modele.eval()
    couches = modele.model.layers
    captures = {}
    crochets = []
    for i, couche in enumerate(couches):
        def fabrique(idx):
            def pre_hook(module, inputs): captures[idx] = inputs[0].detach()
            return pre_hook
        crochets.append(couche.mlp.down_proj.register_forward_pre_hook(fabrique(i)))

    chemins, t0 = [], time.time()
    for n, s in enumerate(stimuli):
        texte = f"Question : {s['question']}\nRéponse courte :"
        ids = tok(texte, return_tensors="pt").input_ids
        captures.clear()
        with torch.no_grad(): modele(input_ids=ids)
        chemin = []
        for i in range(len(couches)):
            force = captures[i][0].float().abs().mean(dim=0)
            k = max(1, int(TOP_FRAC * force.numel()))
            chemin.append(set(torch.topk(force, k).indices.tolist()))
        chemins.append(chemin)
        if (n + 1) % 26 == 0: print(f"   ... {n+1}/{len(stimuli)} ({time.time()-t0:.0f}s)", flush=True)
    for c in crochets: c.remove()
    duree = time.time() - t0

    # --- contrôle génération : le cobaye répond-il au questionnaire basique ?
    exemples_q = ["What is the capital of France?", "How much is 7 times 6?",
                  "What is the chemical formula of water?", "Quelle est la capitale de la France ?",
                  "Combien font 7 fois 6 ?", "Quel animal miaule et chasse les souris ?"]
    reponses = []
    for q in exemples_q:
        texte = tok.apply_chat_template([{"role": "user", "content": q}],
                                        add_generation_prompt=True, tokenize=False)
        ids = tok(texte, return_tensors="pt").input_ids
        with torch.no_grad():
            out = modele.generate(ids, max_new_tokens=20, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        reponses.append({"question": q, "reponse": tok.decode(
            out[0][ids.shape[1]:], skip_special_tokens=True).replace("\n", " ")[:140]})
    del modele

    idx_fam = [i for i, s in enumerate(stimuli) if s["famille"] != "PARAPHRASE"]
    idx_para = [i for i, s in enumerate(stimuli) if s["famille"] == "PARAPHRASE"]
    etiquettes = [stimuli[i]["famille"] for i in idx_fam]

    def sim(i, j): return float(np.mean([jaccard(chemins[i][L], chemins[j][L])
                                         for L in range(len(chemins[0]))]))
    def ecart(etq, indices):
        intra, inter = [], []
        for x in range(len(indices)):
            for y in range(x + 1, len(indices)):
                (intra if etq[x] == etq[y] else inter).append(sim(indices[x], indices[y]))
        return float(np.mean(intra)), float(np.mean(inter)), float(np.mean(intra) - np.mean(inter))

    intra, inter, obs = ecart(etiquettes, idx_fam)
    mel = etiquettes[:]; compte = 0
    for _ in range(2000):
        random.shuffle(mel)
        if ecart(mel, idx_fam)[2] >= obs: compte += 1
    p_h1 = (compte + 1) / 2001

    def mots(i): return set(stimuli[i]["question"].lower().replace("?", " ").replace("'", " ").split())
    intra_l = float(np.mean([jaccard(mots(idx_fam[x]), mots(idx_fam[y]))
                             for x in range(len(idx_fam)) for y in range(x + 1, len(idx_fam))
                             if etiquettes[x] == etiquettes[y]]))
    inter_l = float(np.mean([jaccard(mots(idx_fam[x]), mots(idx_fam[y]))
                             for x in range(len(idx_fam)) for y in range(x + 1, len(idx_fam))
                             if etiquettes[x] != etiquettes[y]]))

    sim_para = [sim(idx_para[x], idx_para[y]) for x in range(len(idx_para))
                for y in range(x + 1, len(idx_para))
                if stimuli[idx_para[x]]["groupe"] == stimuli[idx_para[y]]["groupe"]]
    correct = 0
    for i in idx_fam:
        j = max((j for j in idx_fam if j != i), key=lambda j: sim(i, j))
        correct += stimuli[j]["famille"] == stimuli[i]["famille"]

    par_couche = []
    for L in range(len(chemins[0])):
        iL = [jaccard(chemins[idx_fam[x]][L], chemins[idx_fam[y]][L])
              for x in range(len(idx_fam)) for y in range(x + 1, len(idx_fam)) if etiquettes[x] == etiquettes[y]]
        oL = [jaccard(chemins[idx_fam[x]][L], chemins[idx_fam[y]][L])
              for x in range(len(idx_fam)) for y in range(x + 1, len(idx_fam)) if etiquettes[x] != etiquettes[y]]
        par_couche.append({"couche": L, "ecart": round(float(np.mean(iL) - np.mean(oL)), 4)})

    freq = [Counter() for _ in range(len(chemins[0]))]
    for i in idx_fam:
        for L, ens in enumerate(chemins[i]):
            for neurone in ens: freq[L][neurone] += 1
    hubs = [sum(1 for _, c in f.items() if c >= 0.8 * len(idx_fam)) for f in freq]

    local = RACINE / "artifacts" / "local"; local.mkdir(parents=True, exist_ok=True)
    (local / f"chemins_{a.etiquette}.json").write_text(json.dumps(
        [[sorted(c) for c in ch] for ch in chemins]))
    bilan = {"modele": a.modele, "etiquette": a.etiquette, "precision": "BF16 (CPU, RAM oblige)",
             "n_questions": len(stimuli), "duree_lecture_s": round(duree, 1),
             "H1": {"intra": intra, "inter": inter, "ecart": obs, "p_permutation": p_h1,
                    "verdict": "CONFIRMEE" if (p_h1 < 0.05 and obs > 0) else "REFUTEE"},
             "lexical_mots": {"intra": intra_l, "inter": inter_l, "ecart": intra_l - inter_l},
             "H2_paraphrases": {"similarite": float(np.mean(sim_para)), "intra_famille": intra,
                    "verdict": "CONFIRMEE" if float(np.mean(sim_para)) > intra else "REFUTEE"},
             "exactitude_1NN": correct / len(idx_fam), "par_couche": par_couche,
             "hubs_total": int(sum(hubs)),
             "hubs_positions_total": int(len(chemins[0]) * len(chemins[0][0])),
             "generation_controle": reponses}
    (RACINE / "results" / f"RESULTATS_BANC_{a.etiquette}.json").write_text(
        json.dumps(bilan, ensure_ascii=False, indent=1))
    print(f"🎯 H1 intra={intra:.4f} inter={inter:.4f} écart={obs:+.4f} p={p_h1:.4f} | "
          f"1NN={correct/len(idx_fam):.1%} | para={np.mean(sim_para):.4f}")
    print(f"💾 chemins locaux + RESULTATS_BANC_{a.etiquette}.json sauvegardés.")

if __name__ == "__main__":
    main()
