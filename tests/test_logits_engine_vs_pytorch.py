#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 L1 — CONTRÔLE DES LOGITS : moteur natif (NumPy) contre segment PyTorch.
Prompt figé : « La capitale de la France est » (accord GEOGRAPHIE, routage GPS E4).
Trois exécutions comparées sur les états cachés après les couches 2, 8, 16, 32
et sur les logits finaux :
  A) moteur natif, tranche seule            (le suspect)
  B) PyTorch, tranche seule (même mode)     → si A≠B dès le début : BUG du moteur
  C) PyTorch, inactive + tranche (mode E4)  → si B≈A mais B≠C en profondeur : PHÉNOMÈNE
Le segment PyTorch est ré-entraîné exactement comme E4 dans ce banc (contrôle 1,2449),
puis ses tranches sont exportées pour le moteur. CPU uniquement, fichiers locaux rincés après.
"""
import gc, json, random, subprocess, sys, types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(2)
RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))
from bistouri_backprop import FAMILLES
from bistouri_physique import TrancheMLP, forward_tranche
import extension_tranches as ext
from extension_tranches import SegmentN, evaluer, perte, AdamEtatsLegers, LOCAL

PROMPT, FAMILLE = "La capitale de la France est", "GEOGRAPHIE"
ETAPES = (2, 8, 16, 32)


def cosinus(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def forward_capture(segment, h0, tranche_seule):
    """Forward manuel du segment en capturant les états après les couches-étapes."""
    if tranche_seule:   # on neutralise la branche inactive de chaque MLP physique
        for couche in segment.couches:
            def fabrique(mlp_p):
                def avant_seule(self, x, fam):
                    tr = TrancheMLP(self.Wg[fam], self.Wu[fam], self.Wd[fam], self.indices[fam])
                    return forward_tranche(x.float(), tr).to(x.dtype)
                return types.MethodType(avant_seule, mlp_p)
            couche.mlp_p.forward = fabrique(couche.mlp_p)
    h = h0.unsqueeze(0)
    seq = h.shape[1]
    pos = torch.arange(seq).unsqueeze(0)
    pos_emb = segment.rotary(h, pos)
    masque = torch.triu(torch.full((seq, seq), float("-inf"), dtype=h.dtype), diagonal=1).view(1, 1, seq, seq)
    etapes = {}
    with torch.no_grad():
        for li, couche in enumerate(segment.couches):
            h = couche(h, pos_emb, FAMILLE, masque)
            if (li + 1) in ETAPES: etapes[li + 1] = h[0].float().numpy().copy()
        logits = segment.tete(segment.norme(h))[0, -1].float().numpy()
    return etapes, logits


def main_torch():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ext.MODELE if hasattr(ext, "MODELE") else "HuggingFaceTB/SmolLM2-360M-Instruct")

    print("🛠️ Préparation + entraînement E4 (contrôle du banc)...", flush=True)
    ext.preparer(32)
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
    print(f"📉 contrôle entraînement : {evaluer(segment, dev):.4f} (attendu ≈ 1,2449)", flush=True)

    # export des tranches entraînées pour le moteur (même format que F3)
    sortie = {}
    for li, couche in enumerate(segment.couches):
        for fam in FAMILLES:
            for nom, p in [("Wg", couche.mlp_p.Wg[fam]), ("Wu", couche.mlp_p.Wu[fam]),
                           ("Wd", couche.mlp_p.Wd[fam])]:
                sortie[f"c{li}_{fam}_{nom}"] = p.data.detach().to(torch.bfloat16).view(torch.int16).numpy()
    np.savez(LOCAL / "tranches_entrainees_e4.npz", **sortie)
    def rss():
        for l in open("/proc/self/status"):
            if l.startswith("VmRSS"): return int(l.split()[1]) // 1024
    print(f"   RSS après export : {rss()} Mo", flush=True)
    del sortie, opt, params, par_famille, train, dev, exemples; gc.collect()  # purge avant capture
    print(f"   RSS après purge : {rss()} Mo", flush=True)

    # embeddings du prompt (safetensors en cache local, sans appel réseau)
    from safetensors import safe_open
    candidats = sorted(Path.home().glob(
        ".cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/*/model.safetensors"))
    assert candidats, "safetensors introuvable dans le cache local"
    with safe_open(str(candidats[-1]), framework="pt") as f:
        embed = f.get_tensor("model.embed_tokens.weight")
    ids = tok(PROMPT, return_tensors="pt").input_ids[0]
    h0 = F.embedding(ids.unsqueeze(0), embed)[0]
    print(f"   RSS après embeddings : {rss()} Mo", flush=True)

    print(f"   capture C (complet)… RSS {rss()} Mo", flush=True)
    etapes_C, logits_C = forward_capture(segment, h0, tranche_seule=False)
    print(f"🎹 PyTorch complet : argmax={int(np.argmax(logits_C))} "
          f"({tok.decode([int(np.argmax(logits_C))])!r})", flush=True)
    # NOTE : forward_capture en mode tranche seule modifie les forwards — on reconstruit
    # les méthodes d'origine en réassignant la méthode de classe via nouvelle instance logique :
    print(f"   capture B (tranche seule)… RSS {rss()} Mo", flush=True)
    etapes_B, logits_B = forward_capture(segment, h0, tranche_seule=True)
    print(f"🔪 PyTorch tranche seule : argmax={int(np.argmax(logits_B))} "
          f"({tok.decode([int(np.argmax(logits_B))])!r})", flush=True)
    np.savez(LOCAL / "controle_torch.npz",
             **{f"B{n}": etapes_B[n] for n in ETAPES},
             **{f"C{n}": etapes_C[n] for n in ETAPES},
             logits_B=logits_B, logits_C=logits_C)
    print("💾 captures PyTorch sauvegardées — le processus torch peut mourir en paix.")


def main_compare():
    """Phase légère (NumPy) : moteur en sous-processus, puis comparaisons."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    r = subprocess.run([sys.executable, str(RACINE / "src" / "moteur_natif.py"), "--controle"],
                       capture_output=True, text=True, cwd=RACINE)
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:])
    z = np.load(LOCAL / "controle_moteur.npz")
    w = np.load(LOCAL / "controle_torch.npz")
    print(f"\n{'couche':>7} | {'cos A(moteur) vs B(torch seul)':>32} | {'cos B vs C(complet)':>20}")
    bilan = {"prompt": PROMPT, "famille": FAMILLE, "etapes": {}}
    for n in ETAPES:
        cab = cosinus(z[f"h{n}"], w[f"B{n}"]); cbc = cosinus(w[f"B{n}"], w[f"C{n}"])
        bilan["etapes"][str(n)] = {"cos_moteur_vs_torch_tranche_seule": round(cab, 4),
                                   "cos_tranche_seule_vs_complet": round(cbc, 4)}
        print(f"{n:>7} | {cab:>32.4f} | {cbc:>20.4f}")
    bilan["logits"] = {
        "cos_moteur_vs_torch_tranche_seule": round(cosinus(z["logits"], w["logits_B"]), 4),
        "cos_tranche_seule_vs_complet": round(cosinus(w["logits_B"], w["logits_C"]), 4),
        "argmax_moteur": int(np.argmax(z["logits"])),
        "argmax_torch_tranche_seule": int(np.argmax(w["logits_B"])),
        "argmax_torch_complet": int(np.argmax(w["logits_C"])),
        "tokens": {"moteur": tok.decode([int(np.argmax(z["logits"]))]),
                   "torch_tranche_seule": tok.decode([int(np.argmax(w["logits_B"]))]),
                   "torch_complet": tok.decode([int(np.argmax(w["logits_C"]))])}}
    print(f"logits  | cos A/B={bilan['logits']['cos_moteur_vs_torch_tranche_seule']:.4f} | "
          f"cos B/C={bilan['logits']['cos_tranche_seule_vs_complet']:.4f}")
    print(f"argmax  : moteur={bilan['logits']['tokens']['moteur']!r} | "
          f"torch seul={bilan['logits']['tokens']['torch_tranche_seule']!r} | "
          f"torch complet={bilan['logits']['tokens']['torch_complet']!r}")
    (RACINE / "results" / "RESULTATS_CONTROLE_LOGITS_L1.json").write_text(
        json.dumps(bilan, ensure_ascii=False, indent=1))
    print("💾 results/RESULTATS_CONTROLE_LOGITS_L1.json sauvegardé.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "torch":
        main_torch()
    elif len(sys.argv) > 1 and sys.argv[1] == "compare":
        main_compare()
    else:   # orchestration : deux processus séparés (la sandbox a ~2 Go de RAM)
        for phase in ("torch", "compare"):
            r = subprocess.run([sys.executable, __file__, phase], cwd=RACINE)
            if r.returncode != 0:
                sys.exit(f"phase {phase} en échec (code {r.returncode})")
