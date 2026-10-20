#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏗️ MOTEUR NATIF v0 — F3 (protocole docs/PROTOCOLE_DUEL_ET_ROUTAGE_ET_MOTEUR.md).
Inférence SANS PyTorch (NumPy + tokenizers seuls) du modèle entier opéré :
poids gelés lus à la demande dans le safetensors d'origine (read(), pas de mmap),
MLP en TRANCHE SEULE entraînée (les neurones inactifs ne sont ni chargés ni calculés),
génération gloutonne. Le routage (famille par prompt) vient du GPS, en amont.
Mesure : RSS pic du processus, temps par token. Prototype de faisabilité.
"""
import json, resource, struct, sys, time
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parent.parent
LOCAL = RACINE / "artifacts" / "local"
FAMILLES = ["GEOGRAPHIE", "CALCUL", "SCIENCE", "LANGAGE", "ANIMAUX", "CUISINE"]


def dec_bf16(octets):
    """Décode des octets BF16 (little-endian) en float32, sans PyTorch."""
    u = np.frombuffer(octets, dtype="<u2").astype(np.uint32)
    return (u << 16).view(np.float32)


class Coffre:
    """Lecture de tenseurs safetensors par read() à offset (page cache ≠ RSS)."""
    def __init__(self, chemin):
        self.f = open(chemin, "rb")
        (taille,) = struct.unpack("<Q", self.f.read(8))
        self.meta = json.loads(self.f.read(taille))
        self.base = 8 + taille

    def lire(self, nom):
        info = self.meta[nom]
        debut, fin = info["data_offsets"]
        self.f.seek(self.base + debut)
        brut = self.f.read(fin - debut)
        return dec_bf16(brut).reshape(info["shape"]).copy()

    def lignes(self, nom, r0, r1, largeur):
        info = self.meta[nom]
        debut = info["data_offsets"][0] + r0 * largeur * 2
        self.f.seek(self.base + debut)
        return dec_bf16(self.f.read((r1 - r0) * largeur * 2)).reshape(r1 - r0, largeur).copy()


def rmsnorm(x, w, eps):
    return x * (1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


class MoteurNatif:
    def __init__(self, snapshot, tranches_npz):
        cfg = json.loads((Path(snapshot) / "config.json").read_text())
        self.H = cfg["hidden_size"]; self.eps = cfg.get("rms_norm_eps", 1e-5)
        self.n_tetes = cfg["num_attention_heads"]; self.n_kv = cfg["num_key_value_heads"]
        self.dim_tete = self.H // self.n_tetes
        self.theta = cfg.get("rope_theta", 10000.0)
        self.n_couches = cfg["num_hidden_layers"]
        self.coffre = Coffre(str(Path(snapshot) / "model.safetensors"))
        self.z = np.load(tranches_npz)
        self.tranches = {}   # (couche, famille) -> (Wg, Wu, Wd) FP32, chargées à la demande
        self.fond = 0.0      # S1 : fraction de la branche inactive conservée (0..1)
        self.renormaliser = False   # S2 : multiplier la sortie du fond partiel par 1/p
        self.inter = cfg["intermediate_size"]
        gps = np.load(RACINE / "artifacts" / "gps_modele_360M.npz", allow_pickle=True)
        self.masques = gps["masques"]   # [famille][couche] -> indices de neurones de la tranche

    def tranche(self, li, fam):
        if (li, fam) not in self.tranches:
            def d(nom):
                bits = self.z[f"c{li}_{fam}_{nom}"]
                return (bits.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(bits.shape)
            self.tranches[(li, fam)] = (d("Wg"), d("Wu"), d("Wd"))
        return self.tranches[(li, fam)]

    def avant(self, ids, fam, etapes_voulues=()):
        etapes = {}
        H, dt = self.H, self.dim_tete
        h = np.stack([self.coffre.lignes("model.embed_tokens.weight", i, i + 1, H)[0] for i in ids])
        seq = len(ids)
        pos = np.arange(seq)
        inv_freq = 1.0 / (self.theta ** (np.arange(0, dt, 2) / dt))
        demi = np.outer(pos, inv_freq)                       # (seq, dt/2)
        cos = np.concatenate([np.cos(demi)] * 2, axis=-1)    # convention HF : cos/sin dupliqués
        sin = np.concatenate([np.sin(demi)] * 2, axis=-1)
        masque = np.triu(np.full((seq, seq), -1e30), k=1)
        for li in range(self.n_couches):
            p = f"model.layers.{li}."
            x = rmsnorm(h, self.coffre.lire(p + "input_layernorm.weight"), self.eps)
            q = (x @ self.coffre.lire(p + "self_attn.q_proj.weight").T).reshape(seq, self.n_tetes, dt)
            k = (x @ self.coffre.lire(p + "self_attn.k_proj.weight").T).reshape(seq, self.n_kv, dt)
            v = (x @ self.coffre.lire(p + "self_attn.v_proj.weight").T).reshape(seq, self.n_kv, dt)

            def rope(t):   # rotate_half, comme LlamaRotaryEmbedding
                moitie = dt // 2
                rot = np.concatenate([-t[..., moitie:], t[..., :moitie]], axis=-1)
                return t * cos[:, None, :] + rot * sin[:, None, :]
            q, k = rope(q), rope(k)
            rep = self.n_tetes // self.n_kv
            k = np.repeat(k, rep, axis=1); v = np.repeat(v, rep, axis=1)
            scores = np.einsum("shd,thd->hst", q, k) / np.sqrt(dt) + masque
            scores = scores - scores.max(axis=-1, keepdims=True)
            att = np.exp(scores); att /= att.sum(axis=-1, keepdims=True)
            ctx = np.einsum("hst,thd->shd", att, v).reshape(seq, H)
            h = h + ctx @ self.coffre.lire(p + "self_attn.o_proj.weight").T
            x = rmsnorm(h, self.coffre.lire(p + "post_attention_layernorm.weight"), self.eps)
            Wg, Wu, Wd = self.tranche(li, fam)          # la tranche entraînée, toujours là
            y = (silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T
            if self.fond > 0:   # S1 : fond dense = fraction structurée des neurones inactifs
                actifs = {int(v) for v in self.masques[FAMILLES.index(fam)][li]}
                inactifs = np.array([i for i in range(self.inter) if i not in actifs])
                n = int(round(self.fond * len(inactifs)))
                if n > 0:
                    sel = inactifs[np.linspace(0, len(inactifs) - 1, n).astype(int)] if n < len(inactifs) else inactifs
                    Wgf = self.coffre.lire(p + "mlp.gate_proj.weight")[sel]
                    Wuf = self.coffre.lire(p + "mlp.up_proj.weight")[sel]
                    Wdf = self.coffre.lire(p + "mlp.down_proj.weight")[:, sel]
                    y_fond = (silu(x @ Wgf.T) * (x @ Wuf.T)) @ Wdf.T
                    if self.renormaliser: y_fond = y_fond / self.fond   # S2 : boost ×1/p 🧮
                    y = y + y_fond
            h = h + y
            if (li + 1) in etapes_voulues: etapes[li + 1] = h.copy()
        h = rmsnorm(h, self.coffre.lire("model.norm.weight"), self.eps)
        return h[-1], etapes

    def logits(self, vecteur):
        V, H, pas = 49152, self.H, 8192
        morceaux = []
        for r0 in range(0, V, pas):
            W = self.coffre.lignes("model.embed_tokens.weight", r0, min(r0 + pas, V), H)
            morceaux.append(W @ vecteur)
        return np.concatenate(morceaux)

    def generer(self, tok, prompt, fam, max_nouveaux=18, eos=2):
        ids = tok.encode(prompt).ids
        sortis, temps = [], []
        for _ in range(max_nouveaux):
            t0 = time.perf_counter()
            nxt = int(np.argmax(self.logits(self.avant(ids + sortis, fam)[0])))
            temps.append(time.perf_counter() - t0)
            if nxt == eos: break
            sortis.append(nxt)
        return tok.decode(sortis), float(np.mean(temps))


def main():
    from tokenizers import Tokenizer
    from huggingface_hub import snapshot_download
    snap = snapshot_download("HuggingFaceTB/SmolLM2-360M-Instruct")
    if "--balayage_s2" in sys.argv:   # S2 : fond renormalisé ×1/p 🧮🔥
        tok = Tokenizer.from_file(str(Path(snap) / "tokenizer.json"))
        moteur = MoteurNatif(snap, LOCAL / "tranches_entrainees_e4.npz")
        moteur.renormaliser = True
        ref = np.load(LOCAL / "controle_torch.npz")
        prompt, fam = "La capitale de la France est", "GEOGRAPHIE"
        ids = tok.encode(prompt).ids
        def cos(a, b):
            a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        bilan = []
        for fond in (0.01, 0.05, 0.10, 1.0):
            moteur.fond = fond
            vecteur, etapes = moteur.avant(ids, fam, etapes_voulues=(32,))
            logits = moteur.logits(vecteur)
            texte, s_tok = moteur.generer(tok, prompt, fam)
            mots = texte.split()
            boucle = len(mots) >= 6 and len(set(mots)) <= len(mots) // 3
            ligne = {"fond": fond, "cos_etat_32": round(cos(etapes[32], ref["C32"]), 4),
                     "cos_logits": round(cos(logits, ref["logits_C"]), 4),
                     "argmax": tok.decode([int(np.argmax(logits))]), "generation": texte,
                     "boucle": boucle, "secondes_par_token": round(s_tok, 3)}
            bilan.append(ligne)
            print(f"🧮 fond boosté {fond:5.0%} (×{1/fond:.0f}) : cos état32={ligne['cos_etat_32']:.4f} | "
                  f"cos logits={ligne['cos_logits']:.4f} | argmax={ligne['argmax']!r} | "
                  f"gén={texte!r} (boucle={boucle})", flush=True)
        (RACINE / "results" / "RESULTATS_FOND_S2.json").write_text(json.dumps(
            {"marche": "S2 fond renormalise ×1/p", "prompt": prompt, "famille": fam,
             "reference": "PyTorch complet E4 (controle_torch.npz)", "balayage": bilan},
            ensure_ascii=False, indent=1))
        print("💾 results/RESULTATS_FOND_S2.json sauvegardé.")
        return
    if "--balayage" in sys.argv:   # S1 : seuil de stabilisation (fond dense minimal)
        tok = Tokenizer.from_file(str(Path(snap) / "tokenizer.json"))
        moteur = MoteurNatif(snap, LOCAL / "tranches_entrainees_e4.npz")
        ref = np.load(LOCAL / "controle_torch.npz")
        prompt, fam = "La capitale de la France est", "GEOGRAPHIE"
        ids = tok.encode(prompt).ids
        def cos(a, b):
            a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        bilan = []
        for fond in (0.0, 0.01, 0.05, 0.10, 1.0):
            moteur.fond = fond
            vecteur, etapes = moteur.avant(ids, fam, etapes_voulues=(32,))
            logits = moteur.logits(vecteur)
            texte, s_tok = moteur.generer(tok, prompt, fam)
            mots = texte.split()
            boucle = len(mots) >= 6 and len(set(mots)) <= len(mots) // 3
            ligne = {"fond": fond, "cos_etat_32": round(cos(etapes[32], ref["C32"]), 4),
                     "cos_logits": round(cos(logits, ref["logits_C"]), 4),
                     "argmax": tok.decode([int(np.argmax(logits))]), "generation": texte,
                     "boucle": boucle, "secondes_par_token": round(s_tok, 3)}
            bilan.append(ligne)
            print(f"🧱 fond {fond:5.0%} : cos état32={ligne['cos_etat_32']:.4f} | "
                  f"cos logits={ligne['cos_logits']:.4f} | argmax={ligne['argmax']!r} | "
                  f"gén={texte!r} (boucle={boucle})", flush=True)
        (RACINE / "results" / "RESULTATS_FOND_S1.json").write_text(json.dumps(
            {"marche": "S1 seuil de stabilisation", "prompt": prompt, "famille": fam,
             "reference": "PyTorch complet E4 (controle_torch.npz)", "balayage": bilan},
            ensure_ascii=False, indent=1))
        print("💾 results/RESULTATS_FOND_S1.json sauvegardé.")
        return
    if "--controle" in sys.argv:   # L1 : exporte états cachés + logits pour comparaison
        tok = Tokenizer.from_file(str(Path(snap) / "tokenizer.json"))
        moteur = MoteurNatif(snap, LOCAL / "tranches_entrainees_e4.npz")
        prompt, fam = "La capitale de la France est", "GEOGRAPHIE"
        ids = tok.encode(prompt).ids
        vecteur, etapes = moteur.avant(ids, fam, etapes_voulues=(2, 8, 16, 32))
        logits = moteur.logits(vecteur)
        np.savez(LOCAL / "controle_moteur.npz",
                 **{f"h{n}": etapes[n] for n in (2, 8, 16, 32)}, logits=logits)
        print(f"🔬 contrôle moteur exporté | argmax={int(np.argmax(logits))} "
              f"({tok.decode([int(np.argmax(logits))])!r})")
        return
    tok = Tokenizer.from_file(str(Path(snap) / "tokenizer.json"))
    gen = json.loads((RACINE / "results" / "RESULTATS_GENERATION_E4.json").read_text())
    routes = {r["prompt"]: r["accord_gps"] for r in gen["resultats"]}
    prompts = ["La capitale de la France est", "Le ciel est", "Two plus two equals"]
    moteur = MoteurNatif(snap, LOCAL / "tranches_entrainees_e4.npz")
    bilan = []
    for p in prompts:
        texte, s_par_token = moteur.generer(tok, p, routes[p])
        mots = texte.split()
        boucle = len(mots) >= 6 and len(set(mots)) <= len(mots) // 3
        print(f"🏗️ natif | {p} → {texte!r} ({s_par_token:.2f}s/token, boucle={boucle})", flush=True)
        bilan.append({"prompt": p, "famille_gps": routes[p], "texte": texte,
                      "boucle_detectee": boucle, "secondes_par_token": round(s_par_token, 3)})
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"📏 RSS pic du moteur natif : {rss:.0f} Mo (critère : < 300 Mo)")
    (RACINE / "results" / "RESULTATS_MOTEUR_NATIF.json").write_text(json.dumps(
        {"moteur": "NumPy + tokenizers, zéro PyTorch, MLP tranche seule entraînée (E4)",
         "rss_pic_mo": round(rss, 1), "resultats": bilan}, ensure_ascii=False, indent=1))
    print("💾 results/RESULTATS_MOTEUR_NATIF.json sauvegardé.")


if __name__ == "__main__":
    main()
