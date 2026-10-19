# 🔪 PROPOSITION — DÉCOUPAGE PHYSIQUE DES SOUS-MATRICES ($W_{actif}$)
## Document à valider par le chef AVANT tout Go 🚦
**Statut :** 📝 PROPOSITION — aucune expérience lancée, aucun code d'entraînement écrit, rien de poussé.
**Repo :** RATISS-X (local) · **Cobaye de référence :** SmolLM2-360M 🐘 · **100 % CPU, zéro GPU, zéro « VRAM »** (on mesure la **RAM** et les secondes, comme toujours chez nous).

---

## 🎯 1. CE QU'ON VEUT PROUVER (et ce qu'on ne prétend pas)
Aujourd'hui, notre Bistouri *ajoute* une petite correction sparse **par-dessus** un modèle dense
entièrement chargé : on économise les mises à jour (÷61 paramètres/pas), mais la matrice complète
$W$ reste en mémoire et traverse toujours le calcul. Le découpage physique veut passer au niveau supérieur :

> **Avant même le forward, extraire physiquement la seule sous-matrice utile**
> $W_{actif} = W \odot M_k$ — puis ne calculer le forward, le backward et l'optimiseur
> **que sur cette tranche**, et ne remettre à jour que cette tranche dans $W$.

✅ **Ce que cette étape peut prouver :** l'économie réelle (RAM, temps, états d'optimiseur)
d'une rétropropagation qui ne touche physiquement que ~4,4 % des neurones MLP.
🚫 **Ce qu'elle ne prouvera pas encore :** que le modèle complet « pense » aussi bien en ne
calculant que la tranche (ça, ce sera l'étape d'intégration, plus tard, sur ton ordre séparé).

---

## 🧠 2. L'IDÉE EN MOTS SIMPLES (le scalpel, enfin)
Dans chaque MLP du cobaye, trois matrices travaillent ensemble (par couche, 360M) :
$W_{gate}$ et $W_{up}$ ($2560 \times 960$) qui « ouvrent » 2560 neurones intermédiaires,
et $W_{down}$ ($960 \times 2560$) qui referme. Notre GPS sait déjà quels ~113 neurones
(sur 2560) forment l'accord d'une famille. Le découpage consiste à **découper aux ciseaux** :

$$W^{actif}_{gate} = W_{gate}[k,\ :] \quad W^{actif}_{up} = W_{up}[k,\ :] \quad
  W^{actif}_{down} = W_{down}[:,\ k] \qquad (k = \text{indices de } M_k)$$

puis à ne faire voyager l'activation que dans ce couloir étroit :
$h_{actif} = \text{SiLU}(x\,W^{actif\top}_{gate}) \odot (x\,W^{actif\top}_{up})$,
$\;y = h_{actif}\,W^{actif\top}_{down}$.
Le reste de la matrice **n'est ni chargé dans le calcul, ni différencié, ni stocké par l'optimiseur**.

---

## 🪜 3. LE PLAN, MICRO-ÉTAPE PAR MICRO-ÉTAPE (un Go du chef par marche 🚦)
> Règle nouvelle du labo : **jamais tout d'un coup**. Chaque marche se termine par un
> micro-rapport ; la suivante ne démarre que sur ton « Go » explicite.

### 🟢 Marche D1 — L'extracteur de tranche (aucun entraînement)
- **Faire :** une fonction `extraire_tranche(W_gate, W_up, W_down, indices_k)` qui renvoie
  les trois sous-matrices, + un forward « tranche seule » pour un bloc MLP isolé.
- **Tester (unitaire, sans modèle complet) :** vérifier que la tranche reconstruit **exactement**
  la sortie du bloc dense quand le masque vaut « tout le monde » (k = tous les indices),
  et qu'avec k = accord d'une famille, la sortie tranche-seule est bien la composante active du dense.
- **Livrable :** `src/bistouri_physique.py` (partie extraction) + tests dans `tests/`.
- **Critère de passage :** égalité numérique au seuil flottant (tolérance 1e-5) sur le test « masque plein ».

### 🟢 Marche D2 — Le mock-training de 10 pas (le banc de mesure)
- **Faire :** sur **un seul bloc MLP réel** extrait du 360M (gelé par ailleurs), comparer :
  (a) backward **dense** sur le bloc complet, (b) backward **tranche seule** (indices = accord GPS).
  Entrée : un lot fixe de vecteurs (graine figée), perte de régression simple vers la sortie
  du bloc dense d'origine (le tranche-apprenti apprend à imiter la composante active du maître).
- **Mesurer (CPU uniquement) :** temps médian par pas (sur ≥ 50 pas après échauffement),
  **RAM pic du processus** (RSS), nombre d'éléments de gradient réellement calculés,
  mémoire des états Adam (théorique : 2 × paramètres entraînés × 4 octets, étiquetée « calculée »).
- **Livrable :** `results/RESULTATS_DECOUPAGE_D2.json` + micro-rapport dans ce dossier.
- **Aucun entraînement du LLM ici** : c'est un banc de mécanique, étiqueté comme tel. 🏷️

### 🟢 Marche D3 — Mini-intégration au segment (seulement si D2 tient ses promesses)
- **Faire :** brancher le forward/backward « tranche » sur les 2 dernières couches (le même
  segment que le Bistouri actuel), variante **B honnête** : contribution inactive gelée calculée
  sans graphe (`no_grad`) + tranche active entraînable physiquement découpée.
- **Mesurer :** perte d'évaluation avant/après (mêmes 60 Q/R anglaises, mêmes hyperparamètres
  que le protocole Étapes 2–3), temps/pas, RAM pic, face au dense et au sparse-correction actuels.
- **Livrable :** rapport comparatif à 3 colonnes (dense / correction sparse / tranche physique).

### 🟢 Marche D4 — Verdict et suite (décision du chef)
- Selon les chiffres : généraliser à plus de couches, ou s'arrêter et publier l'échec partiel.
- **RATISS VEGA** ne démarre que si, un jour, tranche physique ≈ dense en apprentissage
  avec une fraction des ressources. Pas avant. 🌟

---

## ⚖️ 4. CRITÈRES DE VICTOIRE PROPOSÉS POUR D2 (à figer par le chef avant le Go)
| Critère | Seuil proposé | Pourquoi |
|---|---|---|
| Éléments de gradient calculés | ≈ part du masque (~4–5 % du dense) | c'est la définition même du découpage |
| États Adam | même ratio (~4–5 %) | conséquence directe, calculée pas mesurée |
| Temps par pas (bloc isolé) | **au moins 2× plus rapide** que le dense | sinon l'indexation mange le gain |
| RAM pic du banc D2 | **inférieure** au dense, écart rapporté en Mo | le chiffre exact sera mesuré, jamais promis |
| Fidélité « masque plein » (D1) | écart ≤ 1e-5 | le scalpel doit d'abord prouver qu'il ne déforme rien |

⚠️ **Aucun seuil de RAM absolu type « < 300 Mo » n'est promis** : sur un banc Python/PyTorch,
le socle (interpréteur, tenseurs, tête de vocabulaire) domine souvent. On rapportera
l'écart dense ↔ tranche, en Mo, avec son étiquette. Honnêteté d'abord. 😇

---

## ⚠️ 5. RISQUES ET PIÈGES CONNUS (écrits avant, pas découverts après)
1. **Surcoût d'indexation** : découper (`index_select`) copie des données ; sur CPU et petites
   tranches, la copie peut coûter plus cher que le calcul économisé. D2 est fait pour le mesurer.
2. **Petites matrices, mauvais rendement BLAS** : un produit $x \times (960 \times 113)$ peut
   sous-utiliser les unités vectorielles face au gros produit dense. Le gain n'est pas garanti.
3. **Masque qui change à chaque question** : re-découper à chaque famille a un coût ; on mesurera
   le découpage **inclus** dans le temps de pas, pas caché hors chronomètre. ⏱️
4. **Forward « tranche seule » ≠ modèle complet** : en D2, la tranche apprend une composante ;
   toute phrase du type « le LLM marche avec 4 % » serait un mensonge à ce stade. 🏷️
5. **Réinjection dans $W$** : la remise à jour de la tranche dans la matrice globale
   (écriture par indices) devra être testée unitairement (D1) pour éviter toute corruption silencieuse.

## 🚫 6. CE QUE JE NE FERAI PAS SANS TON ORDRE EXPLICITE
- Lancer D2/D3 avant ton Go sur D1, ni enchaîner les marches dans la même session sans feu vert.
- Toucher au modèle complet, changer de cobaye, ou pousser quoi que ce soit sur GitHub.
- Présenter un chiffre de RAM/temps qui n'a pas été **mesuré sur ce banc, ce jour-là**.

## 🧾 7. FICHIERS QUI NAÎTRAIENT (si Go)
`src/bistouri_physique.py` · `tests/test_bistouri_physique.py` · `results/RESULTATS_DECOUPAGE_D2.json` ·
micro-rapports `docs/RAPPORT_DECOUPAGE_D1.md` puis `D2`… · mise à jour de `MANIFESTE.json` (SHA-256).
**Repo toujours léger** : aucun poids versionné, modèles évacués après mesure, rinçage final. 🧽

---
*Proposition rédigée le 2 novembre 2026 — en attente de lecture et de Go du chef.*
*RATISS Labs, Yaoundé. Ici c'est la joie et l'enthousiasme 😜🥳 — et le pas-à-pas par-dessus tout.*
