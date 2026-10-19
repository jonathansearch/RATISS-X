#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🗺️ GPS DU MIXER — Routeur d'accords RATISS-X (Étape 1)
Prédit, en < 1 ms et SANS transformeur, quel accord (famille de chemin + masque Mk)
activer pour une question entrante. RATISS Labs — Yaoundé — novembre 2026.
Le cobaye reste SmolLM2-135M jusqu'à validation complète. 🐜
"""
import argparse, json, time
from pathlib import Path
import numpy as np

RACINE = Path(__file__).resolve().parent.parent
FAMILLES = ["GEOGRAPHIE", "CALCUL", "SCIENCE", "LANGAGE", "ANIMAUX", "CUISINE"]
GROUPE_VERS_FAMILLE = {
    "capitale_france": "GEOGRAPHIE", "deux_plus_deux": "CALCUL",
    "formule_eau": "SCIENCE", "bonjour_anglais": "LANGAGE",
    "animal_miaule": "ANIMAUX", "ingredients_omelette": "CUISINE",
}

# ---------------------------------------------------------------- utilitaires
def jaccard(a: set, b: set) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0

def chemin_vers_masques(chemins_famille, seuil=0.5):
    """Prototype d'accord : neurones présents dans >= seuil des chemins (par couche)."""
    n_couches = len(chemins_famille[0])
    masques = []
    for L in range(n_couches):
        compte = {}
        for ch in chemins_famille:
            for neurone in ch[L]:
                compte[neurone] = compte.get(neurone, 0) + 1
        masques.append({n for n, c in compte.items() if c >= seuil * len(chemins_famille)})
    return masques

def similarite_masque_chemin(masques, chemin):
    return float(np.mean([jaccard(m, set(c)) for m, c in zip(masques, chemin)]))

# ---------------------------------------------------------------- embeddings
def charger_table_embedding(modele="HuggingFaceTB/SmolLM2-360M-Instruct"):
    """Charge UNIQUEMENT la table d'embedding du cobaye (aucune couche transformeur)."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    racine = Path(snapshot_download(modele))
    with safe_open(str(racine / "model.safetensors"), framework="pt") as f:
        table = f.get_tensor("model.embed_tokens.weight").float().numpy()
    return table

def embedding_question(texte, tok, table):
    ids = tok(texte, add_special_tokens=False)["input_ids"]
    return table[ids].mean(axis=0)

# ---------------------------------------------------------------- le GPS
class GPS:
    """Routeur par centroïdes d'accords : 6 familles x 576 nombres. Minuscule. ⚡"""
    def __init__(self):
        self.centroides = None      # (n_familles, dim)
        self.masques = None         # dict famille -> liste d'ensembles (par couche)

    def entrainer(self, X, y, masques_par_famille):
        self.centroides = np.stack([X[y == f].mean(axis=0) for f in FAMILLES])
        n = np.linalg.norm(self.centroides, axis=1, keepdims=True)
        self.centroides = self.centroides / n
        self.masques = masques_par_famille
        return self

    def predire(self, vecteur):
        v = vecteur / (np.linalg.norm(vecteur) + 1e-12)
        scores = self.centroides @ v
        k = int(np.argmax(scores))
        return FAMILLES[k], float(scores[k]), scores

    def sauvegarder(self, chemin):
        np.savez_compressed(
            chemin, centroides=self.centroides, familles=np.array(FAMILLES),
            masques=np.array([[sorted(m) for m in self.masques[f]] for f in FAMILLES], dtype=object))

    @classmethod
    def charger(cls, chemin):
        d = np.load(chemin, allow_pickle=True)
        g = cls(); g.centroides = d["centroides"]
        g.masques = {f: [set(m) for m in d["masques"][i]] for i, f in enumerate(FAMILLES)}
        return g

# ---------------------------------------------------------------- expérience
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--etiquette", default="360M")
    ap.add_argument("--chemins", default=str(RACINE / "artifacts" / "local" / "chemins_360M.json"),
                    help="chemins bruts du banc (locaux, non versionnés)")
    args = ap.parse_args()
    ETIQUETTE = args.etiquette

    from transformers import AutoTokenizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import StratifiedKFold

    stimuli = json.loads((RACINE / "data" / "questionnaire.json").read_text())
    chemins = [[set(c) for c in ch] for ch in json.loads(Path(args.chemins).read_text())]
    idx_fam = [i for i, s in enumerate(stimuli) if s["famille"] != "PARAPHRASE"]
    idx_para = [i for i, s in enumerate(stimuli) if s["famille"] == "PARAPHRASE"]

    print("🛰️  Table d'embedding du cobaye (consultation seule, 0 couche calculée)...")
    tok = AutoTokenizer.from_pretrained(args.modele)
    table = charger_table_embedding(args.modele)
    X = np.stack([embedding_question(s["question"], tok, table) for s in stimuli])
    y_fam = np.array([stimuli[i]["famille"] for i in idx_fam])
    X_fam, X_para = X[idx_fam], X[idx_para]
    print(f"   embeddings : {X.shape} (questions x dim)")

    # masques prototypes par famille (tous les chemins de la famille)
    masques = {f: chemin_vers_masques([chemins[i] for i in idx_fam if stimuli[i]["famille"] == f])
               for f in FAMILLES}
    tailles = {f: int(np.mean([len(m) for m in masques[f]])) for f in FAMILLES}

    # --- C1 : validation croisée leave-one-out (centroïdes + logistique), 5-fold (MLP)
    def pred_centroide(Xtr, ytr, x):
        c = np.stack([Xtr[ytr == f].mean(axis=0) for f in FAMILLES])
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        v = x / (np.linalg.norm(x) + 1e-12)
        return FAMILLES[int(np.argmax(c @ v))]

    ok_c = ok_l = 0
    for a, i in enumerate(idx_fam):
        reste = [j for j in range(len(idx_fam)) if j != a]
        if pred_centroide(X_fam[reste], y_fam[reste], X_fam[a]) == y_fam[a]: ok_c += 1
        lr = LogisticRegression(max_iter=1000, C=1.0).fit(X_fam[reste], y_fam[reste])
        if lr.predict(X_fam[a].reshape(1, -1))[0] == y_fam[a]: ok_l += 1
    acc_centroide, acc_logit = ok_c / len(idx_fam), ok_l / len(idx_fam)

    skf = StratifiedKFold(5, shuffle=True, random_state=20261102)
    ok_m = 0
    for tr, te in skf.split(X_fam, y_fam):
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=20261102)
        mlp.fit(X_fam[tr], y_fam[tr]); ok_m += int((mlp.predict(X_fam[te]) == y_fam[te]).sum())
    acc_mlp = ok_m / len(idx_fam)
    print(f"🎯 C1 exactitude LOO centroïdes={acc_centroide:.1%} | logistique={acc_logit:.1%} | MLP(5-fold)={acc_mlp:.1%} (hasard 16.7%)")

    # --- C3 : masque prédit vs vrai chemin, contre baseline autoroute (leave-one-out propre)
    sco_pred, sco_auto = [], []
    for a, i in enumerate(idx_fam):
        fam_vraie = stimuli[i]["famille"]
        mem = [chemins[j] for j in idx_fam if stimuli[j]["famille"] == fam_vraie and j != i]
        autres = [chemins[j] for j in idx_fam if stimuli[j]["famille"] != fam_vraie]
        m_pred = chemin_vers_masques(mem)
        m_auto = chemin_vers_masques(autres + mem, seuil=0.8)
        sco_pred.append(similarite_masque_chemin(m_pred, chemins[i]))
        sco_auto.append(similarite_masque_chemin(m_auto, chemins[i]))
    c3 = {"masque_famille": float(np.mean(sco_pred)), "autoroute_seule": float(np.mean(sco_auto))}
    print(f"✂️  C3 Jaccard masque/chemin : famille={c3['masque_famille']:.4f} vs autoroute={c3['autoroute_seule']:.4f}")

    # --- GPS final entraîné sur les 60 questions, testé sur paraphrases (C4)
    gps = GPS().entrainer(X_fam, y_fam, masques)
    ok_p = 0; details = []
    for a, i in enumerate(idx_para):
        fam, score, _ = gps.predire(X_para[a])
        attendue = GROUPE_VERS_FAMILLE[stimuli[i]["groupe"]]
        bon = fam == attendue; ok_p += bon
        details.append({"question": stimuli[i]["question"], "attendue": attendue,
                        "predite": fam, "confiance": round(score, 3), "correct": bool(bon)})
    acc_para = ok_p / len(idx_para)
    print(f"🍎 C4 paraphrases : {acc_para:.1%}")

    # --- C2 : latence (embedding + routage), tokenisation rapportée à part
    questions_test = [s["question"] for s in stimuli[:12]] * 20
    t_tok = t_route = 0.0; reps = 3
    for _ in range(reps):
        for q in questions_test:
            t0 = time.perf_counter(); v = embedding_question(q, tok, table); t1 = time.perf_counter()
            gps.predire(v); t2 = time.perf_counter()
            t_tok += t1 - t0; t_route += t2 - t1
    n = reps * len(questions_test)
    lat = {"embedding_ms": t_tok / n * 1000, "routage_ms": t_route / n * 1000}
    lat["total_ms"] = lat["embedding_ms"] + lat["routage_ms"]
    print(f"⚡ C2 latence moyenne : embedding={lat['embedding_ms']:.3f} ms | routage seul={lat['routage_ms']:.3f} ms | total={lat['total_ms']:.3f} ms")

    gps.sauvegarder(RACINE / "artifacts" / f"gps_modele_{ETIQUETTE}.npz")
    bilan = {"cobaye": args.modele,
             "C1_exactitude": {"centroides_LOO": acc_centroide, "logistique_LOO": acc_logit,
                               "MLP_2couches_5fold": acc_mlp, "hasard": 1/6},
             "C2_latence_ms": lat, "C3_masque": c3, "C4_paraphrases": acc_para,
             "tailles_masques_par_couche": tailles, "paraphrases_detail": details,
             "gps_retenu": "centroïdes (le plus petit, le plus rapide, entraîné sur 60 questions)"}
    (RACINE / "results" / f"RESULTATS_GPS_{ETIQUETTE}.json").write_text(json.dumps(bilan, ensure_ascii=False, indent=1))
    print(f"💾 artifacts/gps_modele_{ETIQUETTE}.npz + results/RESULTATS_GPS_{ETIQUETTE}.json sauvegardés. GPS en ligne ! 🗺️")

if __name__ == "__main__":
    main()
