# 🪜 MICRO-RAPPORT E3 — LA MOITIÉ DU RÉSEAU PENSE EN CHEMINS (−70,2 %)
**Marche :** Extension E3 (`PROTOCOLE_EXTENSION_E3.md`, ordre « Fais tout »). **Segment : couches 16–31.**
**Moteur :** socle BF16 + tranches FP32 + Adam à états BF16 (M3), masque causal, 100 % CPU, local avant push.

## 📏 Les chiffres
| Mesure | E2 (8 couches) | **E3 (16 couches)** |
|---|---|---|
| Contrôle d'identité (avant / plein) | 6,149 / 6,170 (0,021) | 6,153 / 6,170 (**0,018** ✅) |
| **Perte d'éval. après entraînement** | 2,122 (−65,5 %) | **1,836 (−70,2 %)** 🏆 |
| Mises à jour / pas | 2 874 240 | 5 883 840 (dense MLP 16 couches ~117,96 M → **÷20**) |
| Secondes / pas | 0,098 (FP32) | 0,277 (BF16, publié tel quel) |
| RAM pic / boucle | 1547 Mo (FP32 E2) | **1506 / 1388 Mo** (moteur M) |
| États d'optimiseur | 134,5 Mo (Adam FP32, 8 couches) | 135,3 Mo (Adam BF16, **16 couches** : autant d'états pour 2× de tranches) |

## 📉 La série d'extension corrigée (masque causal), perte finale par profondeur opérée
| Couches opérées | 2 (D3 revalidé) | 4 (E1) | 8 (E2) | **16 (E3)** |
|---|---|---|---|---|
| Perte finale | 2,466 | 2,313 | 2,122 | **1,836** |
| Amélioration | −60,0 % | −62,4 % | −65,5 % | **−70,2 %** |

Monotone, sans effondrement : plus la chirurgie s'étend, plus la spécialisation par
famille s'exprime — avec toujours **vingt fois moins** de paramètres mis à jour par pas
que le dense équivalent. Le « Raccourci de Backprop » tient à la moitié du réseau. 🎹

## ⚖️ Critères figés
E3-a (identité < 0,10) ✅ 0,018 · E3-b (baisse du même ordre, plancher 30 %) ✅ −70,2 % ·
E3-c (÷~20 par pas) ✅. RAM/temps publiés tels quels (passeport moteur natif).

## 🔜 Suite naturelle
**E4 : le modèle entier (32 couches)** — même protocole, sur Go du chef. Le segment
complet tiendra-t-il la RAM de la sandbox avec le moteur M ? C'est la question de la marche.

## 📦 Fichiers
`src/extension_tranches.py` · `results/RESULTATS_EXTENSION_E3.json` · ce rapport · `MANIFESTE.json`
re-scellé. Segments lourds évacués. Push GitHub de la série dans la même session (ordre « Fais tout »).

*Micro-rapport scellé — RATISS Labs, Yaoundé. La moitié du piano joue déjà en accords. 😜🥳*
