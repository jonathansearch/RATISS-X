# ⚖️ PROTOCOLE FIGÉ — MARCHE M1 : DÉBLOQUER LA MÉMOIRE (passeport pour E3)
**Ordre du chef :** Option 1 (mémoire) avant tout le reste. **Aucun push, pas d'E3 sans passeport.** 🧹
**Référence à battre :** E2 (8 couches, FP32) — perte **2,122**, **1547 Mo**, 0,0975 s/pas.

## 🎯 Objectif
Éliminer la double résidence : socle gelé en **BF16**, seules les tranches actives
en **FP32 avec gradient**, aucun graphe sur l'inactive, garbage collector explicite
avant la boucle d'entraînement. Relancer **uniquement** le test 8 couches.

## ⚖️ Critères de victoire (donnés par le chef, figés)
- **RAM pic (8 couches) < 900 Mo** (idéal 600–700 Mo).
- **Perte finale ≈ 2,122 à ±0,05** (le downcasting ne doit pas tuer l'apprentissage).
- **Temps par pas sans augmentation** (référence 0,0975 s/pas).

## 🚫 Hors périmètre
Pas d'E3 (16 couches), pas de push GitHub, pas de changement de tâche ni d'hyperparamètres
(Adam 5e-4, 3 époques, 144 pas, graine 20261102 — identiques à E2).
