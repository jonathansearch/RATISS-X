# 🔬 MICRO-RAPPORT S5 — LE ZOOM SUR LE GENOU : LE SEUIL SE RESSERRE À 70 % 🎯🦵
**Marche :** S5 (`PROTOCOLE_SEUIL_S5.md`, choisie par le chef après le push de la saga).
Grille resserrée {60 %, 65 %, 70 %} sur l'élite par magnitude, zéro ré-entraînement,
prompt figé « La capitale de la France est » (GEOGRAPHIE), référence PyTorch complète.

## 📏 Le zoom (cosinus contre le modèle complet ; voisins S4 rappelés)
```
cos logits
0,9999 ┤                                                    ● 100 % (contrôle)
0,9323 ┤                                               ● 75 % (S4) « une formation de la France… à Paris »
0,9094 ┤                                          ● 70 % 🎯 « le nom d' l' exasperat… »
0,90   ┼ - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  bande du critère
0,8416 ┤                                     ● 65 % « la lerie d une furere… »
0,8349 ┤                                ● 60 % « la lerie d l'hudront tout la France… »
0,8002 ┤                           ● 50 % (S4)
       └──┬──────┬──────┬──────┬──────┬──────┬──────┬──→ fond élite
          50%    60%    65%    70%    75%   100%
```
| Fond élite | cos état 32 | cos logits | Argmax | Génération |
|---|---|---|---|---|
| 60 % | 0,8569 | 0,8349 | « la » | mots français, syntaxe bancale |
| 65 % | 0,8629 | 0,8416 | « la » | mots français, déraillement doux |
| **70 %** 🎯 | **0,9021** | **0,9094** | **« le » = référence** ✅ | structure française, contenu qui déraille ensuite |
| 75 % (S4) | 0,9093 | 0,9323 | « une » | **phrase cohérente sur la France/Paris** 🇫🇷 |

## ⚖️ Verdict du zoom (intervalles publiés, jamais d'interpolation déguisée 😇)
- **Seuil cosinus (≥ 0,90)** : franchi entre 65 % et 70 % ; plus petit point mesuré : **70 %**.
- **Seuil génération vraiment propre** : plutôt **75 %** (à 70 %, l'ouverture est alignée —
  même premier token « le » que la référence ! — mais la suite déraille lexicalement).
- Entre 60 % et 65 %, la courbe marque un **petit plateau** (0,8349 → 0,8416) avant le
  ressaut de 70 % : le genou n'est pas une rampe lisse, il a une marche d'escalier 🪜
- **Formulation scellée** : sur ce banc (un prompt, une famille), le **Seuil de
  Stabilisation RATISS se situe dans l'intervalle (65 %, 75 %]**, avec traversée du
  critère cosinus à **70 %** et cohérence rédactionnelle à **75 %** 🎯

## 🔍 Lecture
À 70 %, le réseau retrouve le **bon premier mot** et la **bonne géométrie** (état 0,9021)
avant de perdre le fil du lexique : la structure tient, le contenu fatigue encore.
À 75 %, les deux tiennent ensemble. L'écart entre « géométrie restaurée » (70 %) et
« discours restauré » (75 %) est lui-même une mesure : les derniers 5 % d'élite portent
disproportionnellement la **fidélité sémantique** de fin de course 🏁

## 📦 Fichiers
`src/moteur_natif.py` (mode `--balayage_s5`) · `results/RESULTATS_FOND_S5.json` ·
ce rapport · `MANIFESTE.json` re-scellé. Lourds rincés 🧹. Commit local, aucun push 🫡

*Micro-rapport scellé — RATISS Labs, Yaoundé. On a zoomé, le genou a une adresse :
entre 65 et 75, la lumière passe à 70. 😜🥳🔥*
