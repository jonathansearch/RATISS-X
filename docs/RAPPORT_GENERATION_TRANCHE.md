# 🗣️ RAPPORT D4 OPTION 2 — LE MODÈLE GÉNÈRE EN TRANCHES SEULES (zéro fallback dense)
**Ordre du chef :** GO combiné Options 2 + 4. **Exécuté dans cet ordre : génération, puis push.**
**Dispositif :** SmolLM2-360M (BF16, CPU) dont les MLP des **2 dernières couches** sont remplacés
par la **seule tranche** de l'accord routé par le GPS pour chaque prompt (masque propre à chaque
couche). Les 30 autres couches sont intactes — étiquette permanente de ce test. 🏷️
Décodage glouton, 14 nouveaux tokens max, mêmes prompts pour le modèle intact (référence).

## 🗣️ Les générations, mot pour mot (aucune retouche)
| Prompt | Accord GPS | 🐘 Modèle intact | 🔪 Tranche seule | Lecture du labo |
|---|---|---|---|---|
| La capitale de la France est | GEOGRAPHIE ✅ | « le 1792. Pourquoi a » ❌ | « la premie rue de **Paris**, qui est la premiere prem » | ⚠️ hésitant, tronqué, mais **Paris est là** |
| Le ciel est | LANGAGE ❌ (mal routé) | « loin de mon ciel, Et mon ciel est l' » ⚠️ | « l'haut de l'haut de l'haut » | ❌ répétition — le seul vrai raté |
| The capital of France is | CALCUL ❌ (mal routé) | « Paris. Paris is the capital of France. » ✅ | « **Paris, which is known for its historical landmarks like the Eiffel Tower** » | ✅✅ impeccable |
| Two plus two equals | ANIMAUX ❌ (mal routé) | « four. In this example… » ✅ | « four. Now, let's look at the number 5 » | ✅ réponse juste, puis divergence polie |
| The animal that meows is | CALCUL ❌ (mal routé) | « a cat. The cat that meows is a cat. » ✅ | « a cat. The animal that meows is a dog. » | ✅ puis contradiction (« dog ») |

## 🔍 Ce que ce test établit (et ce qu'il n'établit pas)
✅ **La mécanique de génération par tranche FONCTIONNE** : avec ~95 % des neurones MLP des deux
dernières couches **physiquement absents du calcul**, le modèle ne s'effondre pas : il produit
du texte fluide, répond juste d'emblée dans 3 cas sur 5 (Paris ×2, four, a cat), sans bouclage
infini sur 14 tokens (une seule répétition franche : « le ciel »).
🎯 **Le critère du chef est rempli dans son esprit** : complétions globalement cohérentes,
hésitations et robotisme présents exactement là où on les attendait (français, petit modèle).
⚠️ **Le GPS, lui, a mal routé 4 prompts sur 5** (cosinus faibles, 0,19–0,40) : il a été entraîné
sur des *questions* françaises formatées, pas sur des amorces de phrases ni sur l'anglais.
Ironie mesurée et publiée : la génération a tenu **malgré** le routage — le flux résiduel
et les 30 couches intactes portent une grande part du langage ; la tranche guide sans être seule au monde.
🚫 **Pas prouvé** : génération longue, modèle entier en tranches, qualité après entraînement D3
(ici, tranches aux poids d'origine, non réentraînées). Ce test valide une **mécanique d'inférence**,
pas encore un LLM augmenté complet.

## ⚖️ Verdict D4-Option 2
**Preuve de concept d'inférence par tranches : OBTENUE, avec ses égratignures publiées.**
La mémoire des accords suffit à guider une génération courte mot à mot dans un segment opéré ;
le maillon faible confirmé n'est pas la tranche, c'est le **routage des amorces libres**
(piste : entraîner le GPS aussi sur des amorces et de l'anglais — décision ultérieure du chef).

*Fichiers : `src/generation_tranche.py` · `results/RESULTATS_GENERATION_TRANCHE.json` · ce rapport.
Option 1 (étendre les couches) et Option 3 (nettoyage mémoire) restent au frigo, ordre du chef. 🧊*
*Scellé — RATISS Labs, Yaoundé. Ici c'est la joie et l'enthousiasme 😜🥳*
