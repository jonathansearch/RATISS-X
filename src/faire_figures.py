#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""📸 Génère les 4 figures officielles (docs/images/) depuis les résultats mesurés.
Valeurs 135M : rapports docs/RAPPORT_ETAPE1_GPS.md et docs/RAPPORT_ETAPES_2_3.md (historique git).
Valeurs 360M : results/*.json de ce repo. Aucun chiffre inventé."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent.parent
SORTIE = R / "docs" / "images"; SORTIE.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .3})

def lire(nom): return json.loads((R / "results" / nom).read_text())

# ---------- Figure 1 : chemins + GPS, petit frère contre grand frère
fig, ax = plt.subplots(figsize=(7.2, 4.2))
cats = ["Famille devinée\nau chemin (1-NN)", "GPS\ncentroïdes", "GPS\nMLP 2 couches", "GPS\nparaphrases C4"]
v135 = [63.3, 56.7, 73.3, 50.0]
g360 = lire("RESULTATS_GPS_360M.json"); b360 = lire("RESULTATS_BANC_360M.json")
v360 = [b360["exactitude_1NN"] * 100, g360["C1_exactitude"]["centroides_LOO"] * 100,
        g360["C1_exactitude"]["MLP_2couches_5fold"] * 100, g360["C4_paraphrases"] * 100]
x = np.arange(len(cats)); w = .38
ax.bar(x - w / 2, v135, w, label="SmolLM2-135M", color="#8da9c4")
ax.bar(x + w / 2, v360, w, label="SmolLM2-360M (grand frère)", color="#ee6c4d")
ax.axhline(100 / 6, ls="--", c="grey", lw=1); ax.text(3.45, 18, "hasard 16,7 %", ha="right", c="grey", fontsize=8)
ax.set_ylabel("Exactitude (%)"); ax.set_xticks(x, cats); ax.set_ylim(0, 88); ax.legend()
ax.set_title("Figure 1 — Lire les chemins et router : le grand frère progresse partout")
fig.tight_layout(); fig.savefig(SORTIE / "fig1_chemins_gps.png", dpi=140); plt.close(fig)

# ---------- Figure 2 : le Bistouri, paramètres et amélioration
fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.0))
ax = axes[0]
modeles = ["135M", "360M"]; dense = [5308416, 14745600]; sparse = [87552, 241920]
x = np.arange(2)
ax.bar(x - .2, dense, .4, label="Dense", color="#8da9c4"); ax.bar(x + .2, sparse, .4, label="Raccourci RATISS", color="#ee6c4d")
ax.set_yscale("log"); ax.set_xticks(x, modeles); ax.set_ylabel("Paramètres mis à jour / pas (log)")
ax.set_title("a) ÷60 de mises à jour"); ax.legend(fontsize=8)
ax = axes[1]
amel_dense = [50.1, 59.3]; amel_sparse = [6.0, 10.8]
ax.bar(x - .2, amel_dense, .4, label="Dense", color="#8da9c4"); ax.bar(x + .2, amel_sparse, .4, label="Raccourci", color="#ee6c4d")
ax.set_xticks(x, modeles); ax.set_ylabel("Baisse de la perte d'éval. (%)")
ax.set_title("b) Le dense apprend encore mieux"); ax.legend(fontsize=8)
fig.suptitle("Figure 2 — Bistouri CPU : l'économie est massive, l'apprentissage reste à rattraper")
fig.tight_layout(); fig.savefig(SORTIE / "fig2_bistouri.png", dpi=140); plt.close(fig)

# ---------- Figure 3 : le Traducteur face aux paraphrases scellées
fig, ax = plt.subplots(figsize=(6.8, 4.0))
t360 = lire("RESULTATS_TRADUCTEUR.json")
cats = ["135M\nGPS seul", "135M\n+ Traducteur", "360M\nGPS seul", "360M\n+ Traducteur"]
vals = [22.2, 33.3, t360["gps_seul"] * 100, t360["gps_plus_traducteur"] * 100]
ax.bar(cats, vals, color=["#8da9c4", "#5e8c61", "#8da9c4", "#ee6c4d"])
ax.set_ylabel("Exactitude (%)"); ax.set_ylim(0, 50)
ax.set_title("Figure 3 — Paraphrases inédites scellées (n=18) : le Traducteur aide, le plafond persiste")
fig.tight_layout(); fig.savefig(SORTIE / "fig3_traducteur.png", dpi=140); plt.close(fig)

# ---------- Figure 4 : efficacité CPU du grand frère
fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.0))
bd = lire("RESULTATS_BISTOURI_dense.json"); bs = lire("RESULTATS_BISTOURI_sparse.json")
ax = axes[0]
ax.bar(["Dense", "Raccourci"], [bd["secondes_par_pas"], bs["secondes_par_pas"]], color=["#8da9c4", "#ee6c4d"])
ax.set_ylabel("Secondes / pas"); ax.set_title("a) Temps par pas (360M)")
ax = axes[1]
ax.bar(["Dense", "Raccourci"], [bd["ram_pic_mo"], bs["ram_pic_mo"]], color=["#8da9c4", "#ee6c4d"])
ax.set_ylabel("RAM pic (Mo)"); ax.set_title("b) RAM CPU (360M, zéro GPU)")
fig.suptitle("Figure 4 — Efficacité CPU du Raccourci sur le grand frère")
fig.tight_layout(); fig.savefig(SORTIE / "fig4_efficacite_cpu.png", dpi=140); plt.close(fig)

print("📸 4 figures écrites dans docs/images/")
