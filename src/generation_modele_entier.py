#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🗣️ GÉNÉRATION DU MODÈLE ENTIER OPÉRÉ — test de feu E4 (protocole docs/PROTOCOLE_GENERATION_E4.md).
Le 360M entier (32 couches) est entraîné par tranches (moteur M, comme E4), puis génère
SEUL : embeddings → 32 couches opérées → tête. Aucun MLP dense, aucun fallback.
Référence honnête : modèle intact, mêmes prompts, même décodage glouton (18 tokens).
CPU uniquement. RATISS Labs — octobre 2026.
"""
import gc, json, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(2)
sys.path.insert(0, str(Path(__file__).parent))
from gps_routeur import GPS, charger_table_embedding, embedding_question
from bistouri_backprop import FAMILLES
import extension_tranches as ext
from extension_tranches import SegmentN, evaluer, perte, AdamEtatsLegers, LOCAL

RACINE = Path(__file__).resolve().parent.parent
MODELE = "HuggingFaceTB/SmolLM2-360M-Instruct"
PROMPTS = ["La capitale de la France est", "Le ciel est", "The capital of France is",
           "Two plus two equals", "The animal that meows is"]
MAX_NOUVEAUX = 18


def boucle_detectee(texte):
    mots = texte.split()
    return len(mots) >= 6 and len(set(mots)) <= len(mots) // 3


def generer_intact(modele, tok, prompt):
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = modele.generate(ids, max_new_tokens=MAX_NOUVEAUX, do_sample=False,
                              pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).replace("\n", " ").strip()


def generer_tranches(segment, embed, tok, prompt, famille):
    ids = tok(prompt, return_tensors="pt").input_ids[0]
    nouveaux = []
    with torch.no_grad():
        for _ in range(MAX_NOUVEAUX):
            h = F.embedding(ids.unsqueeze(0), embed)[0]          # (seq, caché) BF16
            logits = segment(h, famille)                          # 32 couches opérées + tête
            nxt = int(torch.argmax(logits[-1]).item())   # forward rend (seq, vocab), sans batch
            nouveaux.append(nxt)
            ids = torch.cat([ids, torch.tensor([nxt])])
            if nxt == tok.eos_token_id:
                break
    return tok.decode(nouveaux, skip_special_tokens=True).replace("\n", " ").strip()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELE)
    gps = GPS.charger(RACINE / "artifacts" / "gps_modele_360M.npz")
    table = charger_table_embedding(MODELE)

    print("🛠️ Préparation du segment 32 couches (identique à E4)...", flush=True)
    ext.preparer(32)

    print("🐘 Chargement du modèle intact (référence + embeddings)...", flush=True)
    modele = AutoModelForCausalLM.from_pretrained(MODELE, dtype=torch.bfloat16)
    modele.eval()
    routes, refs = {}, {}
    for p in PROMPTS:
        fam, conf, _ = gps.predire(embedding_question(p, tok, table))
        routes[p] = fam
        print(f"🧭 « {p} » → accord {fam} (cosinus {conf:.3f})", flush=True)
    for p in PROMPTS:
        t0 = time.time(); refs[p] = generer_intact(modele, tok, p)
        print(f"   intact | {p} → {refs[p]!r} ({time.time()-t0:.1f}s)", flush=True)
    del modele, table, gps; gc.collect()   # RIEN ne doit résider pendant l'entraînement

    print("🎹 Entraînement E4 (moteur M) dans le processus...", flush=True)
    torch.manual_seed(20261102); random.seed(20261102)
    data = torch.load(LOCAL / "segment_extension_32.pt", weights_only=False, mmap=True)
    exemples, etat = data["exemples"], data["etat"]
    train = [e for e in exemples if not e["eval"]]; dev = [e for e in exemples if e["eval"]]
    debut = etat["couche_debut"]
    gps_npz = np.load(RACINE / "artifacts" / "gps_modele_360M.npz", allow_pickle=True)
    masques = [{fam: sorted(int(n) for n in gps_npz["masques"][i][debut + li])
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
    print(f"📉 entraînement : {avant:.4f} → {apres:.4f} ({(avant-apres)/avant:+.1%})", flush=True)

    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    racine_m = snapshot_download(MODELE)
    with safe_open(str(Path(racine_m) / "model.safetensors"), framework="pt") as f:
        embed = f.get_tensor("model.embed_tokens.weight")   # BF16, chargé après l'entraînement
    print("\n🔪 Génération MODÈLE ENTIER OPÉRÉ (zéro fallback dense)...", flush=True)
    resultats = []
    for p in PROMPTS:
        t0 = time.time(); texte = generer_tranches(segment, embed, tok, p, routes[p]); duree = time.time() - t0
        boucle = boucle_detectee(texte)
        print(f"   tranches | {p} → {texte!r} ({duree:.1f}s, boucle={boucle})", flush=True)
        resultats.append({"prompt": p, "accord_gps": routes[p], "modele_intact": refs[p],
                          "modele_entier_opere": texte, "boucle_detectee": boucle,
                          "secondes": round(duree, 1)})
    bilan = {"modele": MODELE, "chirurgie": "modèle entier 32 couches opéré en tranches "
             "(moteur M), génération embeddings → segment opéré → tête, aucun MLP dense",
             "entrainement_dans_le_processus": {"perte_avant": round(avant, 4), "perte_apres": round(apres, 4)},
             "max_nouveaux_tokens": MAX_NOUVEAUX, "decodage": "glouton", "resultats": resultats}
    (RACINE / "results" / "RESULTATS_GENERATION_E4.json").write_text(
        json.dumps(bilan, ensure_ascii=False, indent=1))
    print("\n💾 results/RESULTATS_GENERATION_E4.json sauvegardé.")


if __name__ == "__main__":
    main()
