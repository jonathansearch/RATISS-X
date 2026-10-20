#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🪶 MOTEUR V2 — prototype int4 résident (Sprint final scientifique).

Tous les poids de calcul résidents (attention partagée et jeu MLP de la famille
servie) sont quantifiés int4 par ligne et empaquetés deux par octet. La table
d'embedding/logits reste lue par petits morceaux. Un seul jeu MLP résident.
La déquantification est refaite couche par couche, à la volée : le prototype
cherche d'abord la mémoire <300 Mo, pas encore la vitesse optimale.
"""
import gc, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from moteur_v1 import MoteurV1, depuis_bits, vers_bits, rss, LOCAL, RACINE
from moteur_natif import rmsnorm, silu, FAMILLES


def quantifier_lignes(W):
    """Quantification symétrique int4 par ligne, empaquetée sur l'axe 1."""
    W = np.asarray(W, dtype=np.float32)
    echelle = np.max(np.abs(W), axis=1) / 7.0
    echelle[echelle == 0] = 1.0
    q = np.clip(np.rint(W / echelle[:, None]), -8, 7).astype(np.int16)
    code = (q & 0xF).astype(np.uint8)
    if code.shape[1] % 2:
        code = np.pad(code, ((0, 0), (0, 1)))
    paquet = code[:, 0::2] | (code[:, 1::2] << 4)
    return paquet, echelle.astype(np.float32), W.shape[1]


def dequantifier(paquet, echelle, colonnes):
    code = np.empty((paquet.shape[0], paquet.shape[1] * 2), dtype=np.int16)
    code[:, 0::2] = paquet & 0xF
    code[:, 1::2] = paquet >> 4
    code[code > 7] -= 16
    return code[:, :colonnes].astype(np.float32) * echelle[:, None]


class MoteurV2(MoteurV1):
    def construire_partage(self):
        if self.partage is not None:
            return
        self.partage = []
        for li in range(self.n_couches):
            p = f"model.layers.{li}."
            bloc = {
                "norm_in": vers_bits(self.coffre.lire(p + "input_layernorm.weight")),
                "norm_post": vers_bits(self.coffre.lire(p + "post_attention_layernorm.weight")),
            }
            for nom, cle in [("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"), ("o_proj", "o")]:
                paquet, ech, cols = quantifier_lignes(
                    self.coffre.lire(p + f"self_attn.{nom}.weight"))
                bloc[cle] = (paquet, ech, cols)
            self.partage.append(bloc)
        self.norme_finale = vers_bits(self.coffre.lire("model.norm.weight"))

    def construire(self, fam):
        self.construire_partage()
        if self.famille_residente == fam:
            return
        self.mlp = None
        gc.collect()
        fi = FAMILLES.index(fam)
        jeu = []
        for li in range(self.n_couches):
            p = f"model.layers.{li}."
            Wg = self.coffre.lire(p + "mlp.gate_proj.weight")
            Wu = self.coffre.lire(p + "mlp.up_proj.weight")
            Wd = self.coffre.lire(p + "mlp.down_proj.weight")
            tranche_idx = np.array(sorted(int(v) for v in self.masques[fi][li]))
            if self.fond >= 1.0:
                utiles = np.arange(self.inter)
            else:
                actifs = {int(v) for v in tranche_idx}
                inactifs = np.array([i for i in range(self.inter) if i not in actifs])
                if self.classements is None:
                    mag = (np.linalg.norm(Wg, axis=1) + np.linalg.norm(Wu, axis=1)
                           + np.linalg.norm(Wd, axis=0))
                    elite = inactifs[np.argsort(-mag[inactifs])][:int(round(self.fond * len(inactifs)))]
                else:
                    elite = self.classements[f"act_{li}_{fam}"][:int(round(self.fond * len(inactifs)))]
                utiles = np.sort(np.concatenate([tranche_idx, elite]))
            Wg_u, Wu_u, Wd_u = Wg[utiles].copy(), Wu[utiles].copy(), Wd[:, utiles].copy()
            if self.fond < 1.0:
                def d(nom):
                    bits = self.z[f"c{li}_{fam}_{nom}"]
                    return (bits.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(bits.shape)
                pos = {int(n): j for j, n in enumerate(utiles)}
                cols = np.array([pos[int(n)] for n in tranche_idx])
                Wg_u[cols] = d("Wg"); Wu_u[cols] = d("Wu"); Wd_u[:, cols] = d("Wd")
            jeu.append({
                "Wg": quantifier_lignes(Wg_u),
                "Wu": quantifier_lignes(Wu_u),
                "Wd": quantifier_lignes(Wd_u),
            })
            del Wg, Wu, Wd, Wg_u, Wu_u, Wd_u
        self.mlp = jeu
        self.famille_residente = fam
        gc.collect()

    def avant(self, ids, fam):
        self.construire(fam)
        H, dt, seq = self.H, self.dim_tete, len(ids)
        h = np.stack([self.coffre.lignes("model.embed_tokens.weight", i, i + 1, H)[0]
                      for i in ids]).astype(np.float32)
        pos = np.arange(seq)
        demi = np.outer(pos, 1.0 / (self.theta ** (np.arange(0, dt, 2) / dt)))
        cos = np.concatenate([np.cos(demi)] * 2, axis=-1)
        sin = np.concatenate([np.sin(demi)] * 2, axis=-1)
        masque = np.triu(np.full((seq, seq), -1e30, dtype=np.float32), k=1)
        for li in range(self.n_couches):
            c, m = self.partage[li], self.mlp[li]
            x = rmsnorm(h, depuis_bits(c["norm_in"]), self.eps)
            q = (x @ dequantifier(*c["q"]).T).reshape(seq, self.n_tetes, dt)
            k = (x @ dequantifier(*c["k"]).T).reshape(seq, self.n_kv, dt)
            v = (x @ dequantifier(*c["v"]).T).reshape(seq, self.n_kv, dt)
            def rope(t):
                mo = dt // 2
                rot = np.concatenate([-t[..., mo:], t[..., :mo]], axis=-1)
                return t * cos[:, None, :] + rot * sin[:, None, :]
            q, k = rope(q), rope(k)
            rep = self.n_tetes // self.n_kv
            k = np.repeat(k, rep, axis=1); v = np.repeat(v, rep, axis=1)
            scores = np.einsum("shd,thd->hst", q, k) / np.sqrt(dt) + masque
            scores = scores - scores.max(axis=-1, keepdims=True)
            att = np.exp(scores); att /= att.sum(axis=-1, keepdims=True)
            ctx = np.einsum("hst,thd->shd", att, v).reshape(seq, H)
            h = h + ctx @ dequantifier(*c["o"]).T
            x = rmsnorm(h, depuis_bits(c["norm_post"]), self.eps)
            h = h + (silu(x @ dequantifier(*m["Wg"]).T)
                     * (x @ dequantifier(*m["Wu"]).T)) @ dequantifier(*m["Wd"]).T
        return rmsnorm(h, depuis_bits(self.norme_finale), self.eps)[-1]

    def logits(self, vecteur):
        morceaux = []
        for r0 in range(0, 49152, 1024):
            W = self.coffre.lignes("model.embed_tokens.weight", r0,
                                   min(r0 + 1024, 49152), self.H)
            morceaux.append(W @ vecteur)
        return np.concatenate(morceaux)


def main():
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer
    snap = snapshot_download("HuggingFaceTB/SmolLM2-360M-Instruct")
    tok = Tokenizer.from_file(str(Path(snap) / "tokenizer.json"))
    moteur = MoteurV2(snap, LOCAL / "tranches_entrainees_e4.npz", fond=0.75,
                      selection="activation",
                      classements_npz=LOCAL / "classements_activation.npz")
    amorces = [("GEOGRAPHIE", "The capital of Spain is"),
               ("GEOGRAPHIE", "The capital of Italy is"),
               ("GEOGRAPHIE", "La capitale de la France est")]
    bilan = []
    for fam, prompt in amorces:
        t0 = time.perf_counter(); moteur.construire(fam)
        t_jeu = time.perf_counter() - t0
        texte, s_tok = moteur.generer(tok, prompt, fam)
        bilan.append({"famille": fam, "prompt": prompt, "texte": texte,
                      "secondes_par_token": round(s_tok, 3),
                      "construction_jeu_s": round(t_jeu, 2),
                      "rss_apres_mo": round(rss(), 1)})
        print(f"🪶 V2 int4 | {prompt} → {texte!r} ({s_tok:.2f}s/token, RSS {rss():.0f} Mo)",
              flush=True)
    sortie = {"marche": "prototype moteur V2 int4 resident", "fond": 0.75,
              "selection": "activation", "rss_pic_mo": round(rss(), 1),
              "cible_mo": 300, "cible_atteinte": bool(rss() < 300),
              "resultats": bilan}
    (RACINE / "results" / "RESULTATS_MOTEUR_V2_INT4.json").write_text(
        json.dumps(sortie, ensure_ascii=False, indent=1))
    print(f"📏 RSS pic {rss():.0f} Mo | cible <300 Mo : {rss() < 300}")


if __name__ == "__main__":
    main()
