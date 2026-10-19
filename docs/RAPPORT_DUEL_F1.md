# ⚖️ MICRO-RAPPORT F1 — LE DUEL DENSE, À ARMES MESURÉES
**Protocole :** `PROTOCOLE_DUEL_ET_ROUTAGE_ET_MOTEUR.md` (8 couches, même tâche/graine/
époques/masque causal ; dense = MLP FP32 entièrement entraînable, Adam FP32 classique).

## 📏 Le duel, côte à côte (segment de 8 couches)
| Mesure | Tranche (E2) | **Dense (F1)** | Lecture |
|---|---|---|---|
| Perte finale | 2,122 (−65,5 %) | **1,829 (−70,2 %)** | le dense gagne la perte pure |
| Mises à jour / pas | 2 874 240 | 58 982 400 | **÷20,5 pour la tranche** |
| Secondes / pas | **0,098** | 0,399 | **tranche 4,1× plus rapide/pas** |
| RAM pic | 1547 Mo (FP32) | 1558 Mo | comparable à cette profondeur |
| Identité (contrôle) | écart 0,021 | écart 0,027 | instruments sains des deux côtés |

## ⚖️ Verdict honnête
À profondeur et protocole égaux, **le dense obtient une meilleure perte finale** (1,83
contre 2,12) : il entraîne vingt fois plus de paramètres, ce serait suspect qu'il en
soit autrement. La tranche, elle, livre **84 % de l'amélioration relative du dense**
(65,5/70,2) avec **5 % des mises à jour** et un pas **4 fois plus rapide**. Le duel ne
couronne pas un vainqueur absolu : il prixe exactement le compromis RATISS X —
un peu de perte finale contre un ordre de grandeur de coût en moins. 📉

## 🧱 Le mur du modèle entier (résultat en soi)
Le même duel à 32 couches est **impossible dans cette sandbox** : MLP FP32 dense
(~944 Mo) + gradients (~944 Mo) + états Adam FP32 (~1,9 Go) dépassent 3 Go, quand la
tranche a entraîné le modèle entier en **1580 Mo** (E4). À profondeur complète, la
question n'est plus « qui perd le moins » mais « qui peut jouer le match ici ». 🐘

## 📦 Fichiers
`src/duel_dense.py` · `results/RESULTATS_DUEL_DENSE_8.json` · ce rapport. Commit local.

*Micro-rapport scellé — RATISS Labs, Yaoundé. Un duel publié avec le score des deux côtés. 😜🥳*
