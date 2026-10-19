# 🔬 MICRO-RAPPORT L1 — VERDICT : PAS DE BUG, UN PHÉNOMÈNE ARCHITECTURAL
**Banc :** `tests/test_logits_engine_vs_pytorch.py` (deux processus séparés, la sandbox
a ~2 Go). Prompt figé « La capitale de la France est », accord GEOGRAPHIE. Entraînement
de contrôle reproduit **1,2449** avant toute comparaison. Trois exécutions :
**A** = moteur natif tranche seule · **B** = PyTorch tranche seule · **C** = PyTorch complet (E4).

## 📏 Cosinus des états cachés (après chaque couche-étape) et des logits
| Étape | **A moteur vs B PyTorch-seul** (bug ?) | **B seul vs C complet** (phénomène ?) |
|---|---|---|
| Couche 2 | **1,0000** | 0,9227 |
| Couche 8 | **1,0000** | 0,9999 |
| Couche 16 | **1,0000** | 0,9999 |
| Couche 32 | **0,9987** | **0,4615** |
| Logits finaux | **0,9909** | **0,3229** |
| Argmax | « Gibraltar » = « Gibraltar » | C prédit « le », B « Gibraltar » |

## ⚖️ Verdict (selon le protocole du chef)
- **Le moteur natif est innocenté.** À mode égal (tranche seule), NumPy et PyTorch
  produisent les mêmes états à 4 décimales sur presque tout le réseau et le même
  token final. L'implémentation (RoPE convention HF, GQA, RMSNorm, décodage BF16)
  est fidèle. Le déraillement de F3 n'est **pas** un bug. ✅
- **Le phénomène est réel et localisé.** Sans branche inactive, l'état caché dérive
  dès la couche 2 (0,9227), reste masqué par le flux résiduel aux couches moyennes
  (0,9999 — le résidu domine en magnitude), puis **s'effondre entre les couches 16
  et 32 (0,4615)** jusqu'à changer complètement la distribution de sortie (0,3229).
  Les tranches E4 ont été entraînées **avec** l'inactive : la couper à l'inférence
  profonde détruit l'équilibre appris. À 2 couches (D4), la dérive n'avait pas la
  profondeur pour s'accumuler — voilà pourquoi D4 « marchait ». 📉

## 🧭 Conséquence pour l'architecture (loi de banc, à confirmer à d'autres profondeurs)
**L'inférence « tranche seule pure » ne passe pas l'échelle en profondeur.** Le moteur
natif devra conserver un **fond dense minimal** : calculer la branche inactive en BF16
à la demande (poids lus par `read()`, sans résidence), ou une fraction dense réduite —
question quantitative ouverte (quel pourcentage de fond suffit à restabiliser la
couche 32 ?), protocolisable en une marche. Le moteur v0 reste validé comme **moteur**
(fidélité prouvée, 0,55 s/token) ; c'est le **mode d'inférence** qui est réfuté à 32 couches.

## 🪦 Incidents de banc publiés
Le banc a été tué trois fois par la RAM (processus torch ~1,6 Go + moteur en parallèle
impossible dans 2 Go) avant d'être scindé en deux processus séquentiels qui se
succèdent sans coexister. Publié, comme toujours.

## 📦 Fichiers
`tests/test_logits_engine_vs_pytorch.py` (phases `torch` puis `compare`, à lancer
séquentiellement) · `src/moteur_natif.py` (mode `--controle`) ·
`results/RESULTATS_CONTROLE_LOGITS_L1.json` · ce rapport. Lourds rincés. Push ordonné
par le chef pour ce cas de figure (phénomène documenté).

*Micro-rapport scellé — RATISS Labs, Yaoundé. On cherchait un bug ; on a trouvé une loi. 😜🥳*
