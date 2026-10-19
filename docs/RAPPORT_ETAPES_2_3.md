# 💉🍏 RAPPORT ÉTAPES 2 & 3 — LE BISTOURI COUPE, LE TRADUCTEUR SOUFFLE… MAIS LE COBAYE TOUSSE

> 🩺 **RECTIFICATIF (revalidation au masque causal)** : les pertes absolues de segment de ce
> rapport ont été mesurées avec un harnais sans masque causal (fuite du futur, découverte en E1).
> Le tableau corrigé qui les supersède est dans `docs/RAPPORT_REVALIDATION_D3.md`
> (tranche 2,466 contre dense 2,699 : la conclusion tient). Historique conservé tel quel.
**Date :** 2 novembre 2026 · **Repo local RATISS-X (NON poussé, ordre du chef 🚫)** · **100 % CPU, zéro GPU**
**Cobaye :** SmolLM2-135M — petit, peu de contexte, faible en français : condition assumée et mesurée.
**Protocole :** `PROTOCOLE_ETAPES_2_3.md`, critères figés avant exécution ⚖️

---

## ⚡ VERDICT EN 10 SECONDES
- 💉 **Étape 2 (Bistouri)** : le Raccourci apprend (perte 4,18 → **3,93**) en mettant à jour
  **60× moins de paramètres par pas** que le dense (87 552 contre 5 308 416) et en allant
  **33 % plus vite par pas** — mais l'amélioration reste faible face au dense (4,18 → **2,09**)
  et la **RAM ne baisse pas** (712 → 718 Mo). Réussite **partielle**, pas triomphe.
- 🍏 **Étape 3 (Traducteur)** : sur 18 paraphrases inédites et scellées, le GPS seul tombe à
  **22 %** (la surface le perd), le Traducteur le remonte à **33 %** : critère formel rempli,
  niveau absolu **insuffisant**.
- ⚖️ **Règle de décision du chef** : E2 et E3 ne réussissent pas franchement tous les deux →
  **la porte « fortifier ou changer de cobaye » est officiellement ouverte.** Détail et conseil ci-dessous.

---

## 💉 1. ÉTAPE 2 — CE QUI A ÉTÉ MESURÉ (CPU, RAM, rien d'autre)

Dispositif : les 2 dernières couches du modèle (là où vivent les chemins), alimentées par
les états du modèle gelé ; tâche de 60 questions-réponses basiques en anglais (langue compétente
du cobaye), 48 entraînement / 12 évaluation ; même optimiseur, même taux, même nombre de pas.

| Mesure CPU | 🩺 Gelé (base) | 🐘 Dense | 💉 Raccourci RATISS |
|---|---|---|---|
| Perte d'évaluation (avant → après) | 4,185 | 4,185 → **2,087** (−50 %) | 4,185 → **3,934** (−6 %) |
| Paramètres mis à jour / pas | 0 | 5 308 416 | **87 552 (÷ 60 !)** |
| Temps par pas | — | 0,0363 s | **0,0243 s (−33 %)** |
| RAM pic du processus | 712 Mo | 712 Mo | 718 Mo (**aucun gain**) |

### 🔍 Lecture sans maquillage, en 3 points
1. **Le principe du raccourci est VIVANT** : en ne touchant que les ~72 neurones du masque
   familial par couche, le modèle s'améliore quand même sur des questions jamais entraînées.
   La backprop « le long du chemin » n'est pas une métaphore : elle descend, elle corrige. 🎯
2. **Mais la dose était trop faible pour rivaliser avec le dense** : correction ajoutée sur
   2 couches seulement, masque de la couche 28 transféré à la 29, initialisation à zéro
   prudente. Le dense, lui, réécrit librement 5,3 M de paramètres : −50 % contre −6 %.
   E2-b (rester dans le même ordre de grandeur d'amélioration) est donc jugé **en échec partiel**.
3. **La RAM n'a pas bougé, et on sait pourquoi** : notre v1 *ajoute* une correction sparse
   **par-dessus** un segment dense entièrement chargé (la tête de vocabulaire pèse à elle
   seule ~113 Mo). Le vrai gain RAM exigera le **découpage physique** des sous-matrices
   (ne charger en mémoire que la tranche active) — pas encore implémenté, pas revendiqué. 😇

## 🍏 2. ÉTAPE 3 — LE TRADUCTEUR FACE AUX PARAPHRASES INÉDITES

Le Traducteur laisse la question **résonner** : elle active l'intention la plus proche dans
la banque des 18 formulations connues (Étape 1), qui diffuse vers ses mots-concepts
(« calcul, addition, nombre… »). Le vecteur de la question est enrichi de ce contexte
($\alpha = 0{,}8$, figé), puis le GPS route. Test : 18 paraphrases **jamais vues**, scellées avant.

| | Exactitude sur le test scellé |
|---|---|
| GPS seul | 4/18 = **22,2 %** |
| GPS + Traducteur | 6/18 = **33,3 %** ✅ critère E3 formellement rempli |

### 🔬 Le diagnostic fin (là où ça devient intéressant, chef)
- 🧭 **L'oreille du Traducteur est bonne** : l'intention correcte est activée dans **13/18** cas (72 %).
- 💨 **Mais son souffle est mal dirigé** : quand l'intention est bonne, le routage final ne suit
  que 6 fois sur 13. Le vecteur-concept, construit avec les embeddings **du même petit modèle
  faible en français**, pousse parfois la question vers la mauvaise famille (CUISINE attire
  toujours trop de monde, notre attracteur-maison 👀).
- 📉 Le GPS seul s'effondre de 50 % (Étape 1) à 22 % sur des formulations plus libres :
  plus la paraphrase s'éloigne des mots appris, plus le cobaye perd le sens. C'est mesuré,
  c'est exactement la maladie diagnostiquée par le chef : **trop petit, trop peu de contexte.**

## ⚖️ 3. APPLICATION DE LA RÈGLE DE DÉCISION (ordre du chef)
> « Si par l'architecture RATISS le modèle arrive à s'améliorer, on garde et on continue ;
> sinon, il faudra soit fortifier ce petit modèle, soit prendre un autre un peu plus gros. »

**État des lieux honnête :** le modèle **s'améliore** (E2 : −6 % en sparse ; E3 : +11 points),
donc l'architecture RATISS n'est **pas réfutée**. Mais les deux améliorations sont **fragiles**,
et le plafond vient visiblement du cobaye lui-même (français faible, géométrie d'embedding pauvre),
pas seulement de nos outils. La condition « on garde et on continue **tel quel** » n'est pas remplie.

### 🛤️ Les deux options officielles, pour décision du chef
- **(a) FORTIFIER le 135M** 💪 : entraînement renforcé ciblé français + paraphrases, raccourci
  appliqué à plus de couches, découpage physique des matrices pour enfin gagner la RAM.
  *Pour :* souveraineté maximale, le petit reste notre mascotte des PC 4 Go.
  *Contre :* on entraîne contre ses faiblesses de naissance (langue, capacité).
- **(b) MONTER D'UN CRAN** 🐘 : prendre le grand frère **SmolLM2-360M** (× 2,7 paramètres)
  ou **Qwen2.5-0,5B** (déjà au catalogue du labo, bien meilleur en français), toujours en CPU,
  et re-passer les 3 étapes dessus. La carte, le GPS et le Bistouri sont **déjà codés et portables**.
  *Pour :* on teste l'architecture sur un cerveau qui comprend la langue des questions.
  *Contre :* RAM et temps CPU montent (à mesurer, jamais à deviner).

**Conseil de l'équipage (le chef décide 🫡)** : l'option (b) avec le 360M en premier —
un seul cran, pas dix — parce que nos deux échecs partiels pointent le même coupable :
la pauvreté des représentations du 135M en français. Si le 360M fait s'envoler E2/E3,
l'architecture est validée et RATISS VEGA aura son cerveau de départ. Sinon, on fortifiera.

## 🚫 Ce qui n'a PAS été fait (pour mémoire)
Aucun push GitHub. Aucune mesure GPU. Aucun changement de cobaye. Aucun critère retouché après coup.

*Scellé — RATISS Labs, Yaoundé. Ici c'est la joie et l'enthousiasme 😜🥳, et la méthode par-dessus tout.*
*Fichiers : `src/bistouri_backprop.py` · `src/traducteur_semantique.py` · `results/RESULTATS_BISTOURI_*.json` · `results/RESULTATS_TRADUCTEUR.json` · test scellé `data/paraphrases_test_v2.json` (fichiers lourds du segment : locaux, non versionnés)*
