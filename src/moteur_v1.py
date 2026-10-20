#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚙️ MOTEUR V1 À POIDS RÉSIDENTS — (protocole docs/PROTOCOLE_MOTEUR_V1.md).
V1 charge UNE FOIS : (a) l'attention/normes partagées (BF16, indépendantes de la
famille), (b) les matrices MLP compactes de LA famille servie (tranche entraînée +
élite magnitude), en BF16 résident. Un seul jeu MLP résident à la fois : changer de
famille = reconstruire le jeu (le routeur décide, le moteur obéit). Pendant la
génération : aucune lecture de poids, aucun masquage, aucune indexation (seule la
table d'embedding des logits est lue par morceaux — décision du protocole).
NumPy + tokenizers, zéro PyTorch. Mesures : RSS repos/pic, démarrages, s/token.
"""
import gc, json, resource, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from moteur_natif import Coffre, rmsnorm, silu, FAMILLES

RACINE = Path(__file__).resolve().parent.parent
LOCAL = RACINE / "artifacts" / "local"


def vers_bits(x):
    return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def depuis_bits(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


class MoteurV1:
    def __init__(self, snapshot, tranches_npz, fond, selection="magnitude",
                 classements_npz=None):
        cfg = json.loads((Path(snapshot) / "config.json").read_text())
        self.H = cfg["hidden_size"]; self.eps = cfg.get("rms_norm_eps", 1e-5)
        self.n_tetes = cfg["num_attention_heads"]; self.n_kv = cfg["num_key_value_heads"]
        self.dim_tete = self.H // self.n_tetes
        self.theta = cfg.get("rope_theta", 10000.0)
        self.n_couches = cfg["num_hidden_layers"]; self.inter = cfg["intermediate_size"]
        self.fond = fond
        self.selection = selection
        self.classements = np.load(classements_npz) if classements_npz else None
        self.coffre = Coffre(str(Path(snapshot) / "model.safetensors"))
        self.z = np.load(tranches_npz)
        gps = np.load(RACINE / "artifacts" / "gps_modele_360M.npz", allow_pickle=True)
        self.masques = gps["masques"]
        self.partage = None        # attention + normes, chargées une fois
        self.mlp = None            # jeu MLP résident de la famille servie
        self.famille_residente = None

    def construire_partage(self):
        if self.partage is not None: return
        self.partage = []
        for li in range(self.n_couches):
            p = f"model.layers.{li}."
            self.partage.append({
                "norm_in": vers_bits(self.coffre.lire(p + "input_layernorm.weight")),
                "norm_post": vers_bits(self.coffre.lire(p + "post_attention_layernorm.weight")),
                "q": vers_bits(self.coffre.lire(p + "self_attn.q_proj.weight")),
                "k": vers_bits(self.coffre.lire(p + "self_attn.k_proj.weight")),
                "v": vers_bits(self.coffre.lire(p + "self_attn.v_proj.weight")),
                "o": vers_bits(self.coffre.lire(p + "self_attn.o_proj.weight"))})
        self.norme_finale = vers_bits(self.coffre.lire("model.norm.weight"))

    def construire(self, fam):
        """Installe le jeu MLP résident de `fam` (un seul à la fois)."""
        self.construire_partage()
        if self.famille_residente == fam: return
        self.mlp = None; gc.collect()
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
            if self.fond < 1.0:   # la tranche entraînée remplace ses lignes/colonnes
                def d(nom):
                    bits = self.z[f"c{li}_{fam}_{nom}"]
                    return (bits.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(bits.shape)
                pos = {int(n): j for j, n in enumerate(utiles)}
                cols = np.array([pos[int(n)] for n in tranche_idx])
                Wg_u[cols] = d("Wg"); Wu_u[cols] = d("Wu"); Wd_u[:, cols] = d("Wd")
            jeu.append({"Wg": vers_bits(Wg_u), "Wu": vers_bits(Wu_u), "Wd": vers_bits(Wd_u)})
            del Wg, Wu, Wd, Wg_u, Wu_u, Wd_u
        self.mlp = jeu; self.famille_residente = fam; gc.collect()

    def avant(self, ids, fam):
        self.construire(fam)
        H, dt, seq = self.H, self.dim_tete, len(ids)
        h = np.stack([self.coffre.lignes("model.embed_tokens.weight", i, i + 1, H)[0] for i in ids])
        pos = np.arange(seq)
        demi = np.outer(pos, 1.0 / (self.theta ** (np.arange(0, dt, 2) / dt)))
        cos = np.concatenate([np.cos(demi)] * 2, axis=-1)
        sin = np.concatenate([np.sin(demi)] * 2, axis=-1)
        masque = np.triu(np.full((seq, seq), -1e30), k=1)
        for li in range(self.n_couches):
            c, m = self.partage[li], self.mlp[li]
            x = rmsnorm(h, depuis_bits(c["norm_in"]), self.eps)
            q = (x @ depuis_bits(c["q"]).T).reshape(seq, self.n_tetes, dt)
            k = (x @ depuis_bits(c["k"]).T).reshape(seq, self.n_kv, dt)
            v = (x @ depuis_bits(c["v"]).T).reshape(seq, self.n_kv, dt)
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
            h = h + ctx @ depuis_bits(c["o"]).T
            x = rmsnorm(h, depuis_bits(c["norm_post"]), self.eps)
            h = h + (silu(x @ depuis_bits(m["Wg"]).T) * (x @ depuis_bits(m["Wu"]).T)) @ depuis_bits(m["Wd"]).T
        return rmsnorm(h, depuis_bits(self.norme_finale), self.eps)[-1]

    def logits(self, vecteur):
        morceaux = []
        for r0 in range(0, 49152, 8192):
            W = self.coffre.lignes("model.embed_tokens.weight", r0, min(r0 + 8192, 49152), self.H)
            morceaux.append(W @ vecteur)
        return np.concatenate(morceaux)

    def generer(self, tok, prompt, fam, max_nouveaux=18, eos=2):
        ids, sortis, temps = tok.encode(prompt).ids, [], []
        for _ in range(max_nouveaux):
            t0 = time.perf_counter()
            nxt = int(np.argmax(self.logits(self.avant(ids + sortis, fam))))
            temps.append(time.perf_counter() - t0)
            if nxt == eos: break
            sortis.append(nxt)
        return tok.decode(sortis), float(np.mean(temps))


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    from tokenizers import Tokenizer
    from huggingface_hub import snapshot_download
    mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "hybride"
    assert mode in ("hybride", "dense")
    snap = snapshot_download("HuggingFaceTB/SmolLM2-360M-Instruct")
    tok = Tokenizer.from_file(str(Path(snap) / "tokenizer.json"))
    moteur = MoteurV1(snap, LOCAL / "tranches_entrainees_e4.npz",
                      fond=0.70 if mode == "hybride" else 1.0)
    gen = json.loads((RACINE / "results" / "RESULTATS_GENERATION_E4.json").read_text())
    routes = {r["prompt"]: r["accord_gps"] for r in gen["resultats"]}
    prompts = ["La capitale de la France est", "Le ciel est", "Two plus two equals"]
    bilan, constructions = [], []
    for prompt in prompts:
        fam = routes[prompt]
        if moteur.famille_residente != fam:
            t0 = time.perf_counter(); moteur.construire(fam)
            constructions.append({"famille": fam, "secondes": round(time.perf_counter() - t0, 2),
                                  "rss_apres_mo": round(rss(), 1)})
            print(f"🏗️ jeu résident {fam} construit en {constructions[-1]['secondes']}s "
                  f"| RSS ~{rss():.0f} Mo", flush=True)
        texte, s_tok = moteur.generer(tok, prompt, fam)
        mots = texte.split()
        boucle = len(mots) >= 6 and len(set(mots)) <= len(mots) // 3
        bilan.append({"prompt": prompt, "famille_gps": fam, "texte": texte,
                      "boucle_detectee": boucle, "secondes_par_token": round(s_tok, 3)})
        print(f"⚙️ v1-{mode} | {prompt} → {texte!r} ({s_tok:.2f}s/token, boucle={boucle})", flush=True)
    print(f"📏 RSS pic (v1-{mode}) : {rss():.0f} Mo")
    (RACINE / "results" / f"RESULTATS_MOTEUR_V1_{mode}.json").write_text(json.dumps(
        {"marche": "Moteur V1 a poids residents", "mode": mode,
         "constructions": constructions, "rss_pic_mo": round(rss(), 1),
         "resultats": bilan}, ensure_ascii=False, indent=1))
    print(f"💾 results/RESULTATS_MOTEUR_V1_{mode}.json sauvegardé.")


if __name__ == "__main__":
    main()
