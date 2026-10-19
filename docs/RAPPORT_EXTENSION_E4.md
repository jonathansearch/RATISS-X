# 🪜 MICRO-RAPPORT E4 — LE MODÈLE ENTIER PENSE EN CHEMINS (−79,8 %) 🎹
**Marche :** Extension E4 (`PROTOCOLE_EXTENSION_E4.md`, ordre « Go »). **Segment : couches 0–31, tout SmolLM2-360M.**
**Moteur :** socle BF16 + tranches FP32 + Adam à états BF16, masque causal, construction `meta`+`assign`. 100 % CPU.

## 📏 Les chiffres
| Mesure | E3 (16 couches) | **E4 (32 couches — modèle entier)** |
|---|---|---|
| Contrôle d'identité (avant / plein) | écart 0,018 | **écart 0,015** ✅ |
| **Perte d'éval. après entraînement** | 1,836 (−70,2 %) | **1,245 (−79,8 %)** 🏆 |
| Mises à jour / pas | 5 883 840 | 11 289 600 (dense MLP 32 couches ~235,9 M → **÷21**) |
| Paramètres de tranches (audit Adam) | 33 825 600 | 63 820 800 |
| États d'optimiseur | 135,3 Mo | 255,3 Mo (BF16, pour 2× de tranches) |
| Secondes / pas | 0,277 | 0,753 |
| RAM pic / boucle | 1506 / 1388 Mo | **1580 / 1559 Mo** (publié tel quel) |

## 📉 La série complète d'extension (masque causal) — la preuve par la profondeur
| Couches opérées | 2 | 4 | 8 | 16 | **32 (entier)** |
|---|---|---|---|---|---|
| Perte finale | 2,466 | 2,313 | 2,122 | 1,836 | **1,245** |
| Amélioration | −60,0 % | −62,4 % | −65,5 % | −70,2 % | **−79,8 %** |

Monotone sur toute la gamme, sans effondrement, avec **~21 fois moins** de paramètres
mis à jour par pas que l'entraînement dense des MLP. À profondeur complète, l'écart se
creuse encore : la spécialisation par familles de questions irrigue tout le réseau.
Le « Raccourci de Backprop » n'est plus une propriété de segment : c'est une propriété
**du modèle entier opéré en tranches**. 🎹🔥

## 🪦 L'incident publié (et son correctif)
Le premier lancement E4 a été **tué pendant la construction** (OOM silencieux) : les
couches s'assemblaient en FP32 (~1,25 Go) avant conversion BF16. Le diagnostic couche
par couche a révélé au passage un fait que le labo publie : **le correctif `assign`
annoncé en M3 ne s'était en réalité jamais appliqué** (ancres de code différentes) —
M3 n'avait donc testé que le mmap, pas l'assignation. Corrigé pour de vrai ici
(construction sur dispositif `meta` + `load_state_dict(assign=True)`, résidence unique
mmap pour le socle BF16) : construction franchie, boucle stable ~1,52–1,58 Go.
Un rectificatif est apposé sur `RAPPORT_M3.md`.

## ⚖️ Critères figés
E4-a (identité < 0,10) ✅ 0,015 · E4-b (même ordre, plancher 30 %) ✅ −79,8 % ·
E4-c (÷~20 par pas) ✅ ÷21. RAM/temps publiés tels quels (passeport moteur natif).

## 🔍 Lecture
La série d'extension est close : de 2 couches au modèle entier, chaque marche a amélioré
la perte davantage que la précédente. Restent ouverts, honnêtement : la **génération libre
par le modèle entier opéré** (la preuve D4 ne portait que sur 2 couches), la comparaison
à tâche égale contre un **fine-tuning dense complet** du même modèle (notre ÷21 porte sur
les MLP du segment, pas sur un entraînement dense de référence chronométré ici), et le
**moteur natif** (<300 Mo visés). RATISS VEGA n'est toujours pas déclaré : il attend
ses critères, pas notre enthousiasme. 🌟

## 📦 Fichiers
`src/extension_tranches.py` (construction meta+assign corrigée) · `results/RESULTATS_EXTENSION_E4.json` ·
ce rapport · rectificatif M3 · `MANIFESTE.json` re-scellé. Segment lourd (728 Mo) évacué. Commit local.

*Micro-rapport scellé — RATISS Labs, Yaoundé. Le piano entier joue en accords. 😜🥳*
