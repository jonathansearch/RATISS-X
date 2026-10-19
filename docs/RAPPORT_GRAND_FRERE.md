# 🐘 RAPPORT GRAND FRÈRE — TOUS LES TESTS REFAITS SUR SmolLM2-360M

> 🩺 **RECTIFICATIF (revalidation au masque causal)** : les pertes absolues de segment de ce
> rapport ont été mesurées avec un harnais sans masque causal (fuite du futur, découverte en E1).
> Le tableau corrigé qui les supersède est dans `docs/RAPPORT_REVALIDATION_D3.md`
> (tranche 2,466 contre dense 2,699 : la conclusion tient). Historique conservé tel quel.
**Date :** 2 novembre 2026 · **Ordre du chef :** « on appelle le grand frère » · **Repo local, NON poussé** 🚫
**Protocole :** `PROTOCOLE_GRAND_FRERE.md` — mêmes stimuli, mêmes critères, mêmes hyperparamètres.
**Étiquette d'honnêteté :** le 360M tourne en BF16 sur CPU (RAM oblige), le 135M tournait en FP32. 🏷️

## ⚡ VERDICT EN 10 SECONDES
Le grand frère fait progresser **presque tout** : famille devinée au chemin **63 % → 68 %**,
GPS **57 % → 65 %** (MLP **73 % → 77 %**), paraphrases du GPS **50 % → 56 %**, Traducteur
**33 % → 39 %**, et il répond enfin juste en français basique (« La capitale de la France
est Paris » ✅). Mais les deux faiblesses de fond **persistent en plus doux** : le Bistouri
sparse reste loin du dense (−11 % contre −59 %), et les paraphrases libres restent
sous les 40 %. **L'architecture RATISS est confortée ; aucun des deux cobayes n'est encore VEGA.** 🌟

## 🔁 1. BANC DES CHEMINS — la carte du grand frère 🛤️
| Mesure | 🐜 135M | 🐘 360M |
|---|---|---|
| H1 intra / inter / écart | 0,575 / 0,520 / +0,055 (p=0,0005) | 0,550 / 0,491 / **+0,058** (p=0,0005) ✅ |
| Famille devinée au seul chemin (1-NN) | 63,3 % | **68,3 %** 📈 |
| Autoroute (part des positions-hubs) | 55 % | 52 % (même paysage) |
| Entonnoir final (dernière couche) | écart +0,014 (effondré) | **+0,040** : les familles restent séparées plus tard 📈 |
| Paraphrases (similarité d'intention) | 0,534 | 0,503 (toujours < intra-famille : le bug de surface persiste) |
| Génération FR contrôle | « La capitale de la France est la France » ❌ | **« La capitale de la France est Paris »** ✅ · « 7 fois 6 » boucle encore en FR ⚠️ |

Lecture : chez le grand frère, la séparation des familles **persiste jusqu'au bout du réseau**
au lieu de s'écraser dans l'entonnoir final — ses chemins portent l'intention plus loin. 🎯

## 🗺️ 2. GPS — le routeur sur la nouvelle carte
| Critère | Cible | 🐜 135M | 🐘 360M | Verdict |
|---|---|---|---|---|
| C1 centroïdes / logistique / MLP | ≥ 50 % | 56,7 / 51,7 / 73,3 % | **65,0 / 66,7 / 76,7 %** | ✅📈 |
| C2 latence totale | < 1 ms | 0,062 ms (médiane démo) | **0,047 ms** (médiane démo) | ✅ |
| C3 masque vs autoroute | > autoroute | 0,649 / 0,506 | 0,628 / 0,482 | ✅ |
| C4 paraphrases Étape 1 | ≥ 50 % | 50,0 % | **55,6 %** | ✅📈 |
| Taille des accords | sparse | ~72/1536 (4,7 %) | ~113/2560 (4,4 %) | même sparsité 🎵 |

## 💉 3. BISTOURI CPU — mêmes gestes, plus gros patient (ZÉRO GPU)
| Mesure CPU | 🐜 135M dense → sparse | 🐘 360M dense → sparse |
|---|---|---|
| Perte éval de départ (segment) | 4,18 | 6,12 ⚠️ (voir note) |
| Amélioration dense / sparse | −50 % / −6 % | **−59 % / −11 %** 📈 |
| Paramètres mis à jour / pas | 5,31 M / 87 552 (÷60) | 14,75 M / 241 920 (**÷61**) |
| Temps par pas dense / sparse | 0,036 / 0,024 s | 0,104 / **0,046 s (2,25× plus vite)** ⚡ |
| RAM pic dense / sparse | 712 / 718 Mo (rien) | 1056 / **908 Mo (−14 %)** 📉 |

⚠️ *Note honnête : la perte de départ du segment 360M (6,12) est plus haute que celle du 135M (4,18).*
C'est la perte **du segment reconstruit** (2 couches alimentées par états cachés, précision BF16
à l'extraction), pas un classement officiel des deux modèles — la génération de contrôle montre
d'ailleurs le 360M meilleur en français. On publie le chiffre tel quel, avec son étiquette. 😇

Lecture : le Raccourci **s'améliore avec la taille** (−11 % contre −6 %, vrai petit gain RAM
cette fois, pas de pas 2,25× plus rapide que le dense) — mais E2-b reste en échec partiel :
le dense garde une large avance d'apprentissage. Le découpage physique des matrices
reste LE chantier pour transformer l'essai. 🔪

## 🍏 4. TRADUCTEUR — même test scellé, jamais réécrit (18 paraphrases)
| | 🐜 135M | 🐘 360M |
|---|---|---|
| GPS seul | 22,2 % | 22,2 % (la surface perd toujours) |
| GPS + Traducteur | 33,3 % (+11 pts) | **38,9 % (+17 pts)** 📈 |
| Bonne intention activée | 13/18 | 13/18 (même oreille) |
| Routage si intention bonne | 6/13 | **7/13** |

Le Traducteur aide **davantage** le grand frère, mais le plafond (~39 %) dit la même chose
des deux côtés : tant que le vecteur-concept est fabriqué avec les embeddings du cobaye
lui-même, le souffle sémantique reste faible. Le tissu `.ratiss` externe (Étape 3 complète,
pas encore branché) reste la piste sérieuse. 🍏

## ⚖️ 5. CONCLUSION DU LABO
1. ✅ **L'architecture RATISS passe l'épreuve du changement d'échelle** : carte, GPS,
   Bistouri et Traducteur fonctionnent sur le 360M sans réécriture, et presque tous
   les indicateurs montent. Ce n'était pas un artefact du petit modèle.
2. 🐘 **Le grand frère devient le cobaye de référence** pour la suite (le 135M part
   en réserve honorable : ses résultats restent dans l'historique git).
3. 🔪 **Prochain chantier technique** (décision du chef) : le découpage physique des
   sous-matrices — charger et calculer la seule tranche $M_k$, pour que le gain RAM/temps
   du Raccourci devienne massif au lieu d'être partiel.
4. 🌟 **RATISS VEGA** : pas encore. Le jour où sparse ≈ dense en apprentissage avec
   60× moins de mises à jour, on construira le premier modèle augmenté.

## 🧹 Hygiène de l'espace (ordre du chef)
Modèles et caches évacués après mesures ; fichiers lourds temporaires (chemins bruts,
segment d'entraînement ~138 Mo) supprimés. Le workspace persisté repasse **sous les 50 Mo**
— vérification chiffrée dans le compte rendu de session. 🪶

*Scellé — RATISS Labs, Yaoundé. 100 % CPU. Ici c'est la joie et l'enthousiasme 😜🥳*
