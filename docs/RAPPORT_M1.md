# 🧹 MICRO-RAPPORT M1 — LE PASSEPORT MÉMOIRE N'EST PAS (ENCORE) OBTENU, ET ON SAIT POURQUOI
**Marche :** M1 (`PROTOCOLE_M1.md`, ordre du chef). Relance du seul test 8 couches, mode mémoire légère.
**Verdict global : ❌ passeport refusé à ce stade — aucun E3, aucun push, comme ordonné.** 🫡

## 📏 Critères du chef, jugés sans retouche
| Critère figé | Cible | Mesuré (BF16, 8 couches) | Verdict |
|---|---|---|---|
| RAM pic | < 900 Mo | **1262 Mo** | ❌ |
| Perte finale | 2,122 ± 0,05 | **2,117** (−65,6 %, identité 6,149/6,170) | ✅ impeccable |
| Temps par pas | ≤ 0,0975 s (E2) | **0,177 s** | ❌ presque 2× plus lent |

La chirurgie mémoire **ne dégrade absolument pas l'apprentissage** (2,117 contre 2,122 en FP32) :
le downcasting est innocenté sur la qualité. Ce sont la RAM et la vitesse qui refusent le passeport.

## 🔍 Analyse du goulot (diagnostics mesurés, pas devinés)
**Pesée du processus par étapes (RSS)** : imports 219 Mo → chargement du fichier segment 464 Mo →
construction **1020 Mo** → après libération + `gc.collect()` **880 Mo** → pic entraînement **1262 Mo**.

**Inventaire du segment vivant (8 couches)** : socle attention/normes ~39 Mo (BF16) ·
buffers MLP gelés ~118 Mo · tête de sortie ~94 Mo · tranches FP32 ~67 Mo.
Le reste du pic vient de trois couches de « graisse d'instrument » :
1. **Le pic de construction** : le fichier d'état et le segment construit coexistent
   (~250 Mo de copies) et l'allocateur Python/PyTorch **garde** ces pages ensuite.
   Notre premier correctif (libérer le conteneur) n'a rien changé au pic : il était déjà marqué.
2. **L'état d'entraînement** : Adam (2 états × 16,8 M paramètres de tranches ≈ **134 Mo**)
   + gradients conservés (**~67 Mo**) = ~200 Mo qui s'ajoutent au segment vivant.
3. **La lenteur BF16 sur ce CPU** : le calcul BF16 y est émulé/converti — **0,174 s/pas**
   contre 0,098 en FP32. Le FP16, testé en diagnostic, est **catastrophique : 3,64 s/pas**.
   Sur cette machine, « plus léger » ne veut pas dire « plus rapide » : c'est mesuré.

## ⚖️ Conclusion d'ingénierie
Avec PyTorch sur cette sandbox (socle d'import ~219 Mo, allocateur qui retient le pic,
Adam FP32), la barre des 900 Mo pour 8 couches entraînées n'est pas atteignable par le
seul downcasting. Ce n'est pas un échec de l'architecture RATISS X — le futur moteur natif
(chargement de tranches à la demande, sans interpréteur Python) vise précisément ce socle —
mais **le passeport M1, dans ses critères écrits, n'est pas délivré**. On ne maquille pas.

## 🛤️ Ajustements proposés pour une marche M2 (sur Go séparé du chef)
1. **Chargement sans duplication** : construire le segment par `load_state_dict(assign=True)`
   / mmap, couche par couche, pour tuer le pic de construction (gain espéré ~250 Mo, à mesurer).
2. **Attaquer l'état d'optimiseur** : quantifier les états Adam ou changer d'optimiseur —
   **changement de protocole** (hyperparamètres), donc décision explicite du chef requise.
3. **Accepter le FP32 rapide et chasser ailleurs** : libérer l'ancien MLP en FP32 (fait en M1
   pour le mode léger, à porter en mode normal) + chargement sans duplication ; la RAM visée
   serait alors à re-mesurer, sans promesse.

## 📦 Fichiers de la marche
Mode léger (`--memoire_leger`, `--dtype_socle`) dans `src/extension_tranches.py` et
`src/integration_tranche_segment.py` · `results/RESULTATS_EXTENSION_M1.json` · ce rapport ·
`MANIFESTE.json` re-scellé. Fichiers lourds évacués. Commit local, **rien de poussé**.

*Micro-rapport scellé — RATISS Labs, Yaoundé. Un passeport refusé et expliqué vaut mieux
qu'un passeport maquillé. Étape par étape. 😜🥳*
