# 🔪 MICRO-RAPPORT D1 — L'EXTRACTEUR DE TRANCHE EST NÉ ET IL NE DÉFORME RIEN
**Marche :** D1 uniquement (proposition `PROPOSITION_DECOUPAGE_PHYSIQUE.md`, Go du chef reçu).
**Pas fait, exprès :** aucun entraînement, aucun modèle chargé, marche D2 non commencée. ⏸️

## 🪜 Ce qui a été fait, dans l'ordre
1. **09 h du labo — le code** : création de `src/bistouri_physique.py` avec quatre pièces :
   - `normaliser_indices()` : les indices d'un accord deviennent uniques, triés, reproductibles.
     (Permuter les neurones intermédiaires ne change pas la sortie si `W_down` suit la même
     permutation : on canonise donc l'ordre pour que la tranche soit toujours la même.)
   - `extraire_tranche(W_gate, W_up, W_down, k)` : découpe **physique** par `index_select`
     (vraie copie mémoire) → $W_{gate}[k,:]$, $W_{up}[k,:]$, $W_{down}[:,k]$, avec contrôle
     des formes et des indices (accord vide ou hors-limites = erreur propre, jamais silencieuse).
   - `forward_tranche(x, tranche)` : $y = [\text{SiLU}(xW^{actif\top}_{gate}) \odot (xW^{actif\top}_{up})]\,W^{actif\top}_{down}$,
     en 2D et 3D. `forward_dense()` sert de référence de test.
   - `reinjecter_tranche(...)` : remise de la tranche dans les matrices globales par
     `index_copy_`, **uniquement aux indices de l'accord**.
2. **Les tests** : `tests/test_bistouri_physique.py` — 8 tests neufs, graine `20261103`,
   aucun poids de modèle (matrices aléatoires contrôlées + un test aux dimensions réelles du 360M).
3. **La mesure d'écart exacte** : petit script de contrôle rejouable (voir § chiffres).

## 📏 Les chiffres mesurés (CPU, ce jour)
| Contrôle | Critère figé | Mesuré | Verdict |
|---|---|---|---|
| Masque plein = dense (bloc jouet 24×16) | écart ≤ 1e-5 | **0,000000** | ✅ |
| Masque plein = dense (dimensions 360M : 2560×960) | écart ≤ 1e-5 | **0,000000** | ✅ |
| Tranche d'accord aux dimensions 360M (113 neurones) | — | **325 440 paramètres contre 7 372 800 en dense = 4,41 %** | 🎯 la taille visée |
| Tests unitaires (8 neufs + 3 GPS) | tout vert | **11/11 passés en 1,1 s** | ✅ |

Détail des 8 tests : égalité masque plein (2D et 3D, indices mélangés), tranche = composante
active du dense masqué, copie physiquement indépendante (modifier la tranche ne touche pas `W`),
réinjection limitée à l'accord (le reste est bit-à-bit inchangé), gradients présents **uniquement**
dans la tranche (les matrices globales gelées n'ont aucun `.grad`), erreurs propres, formes 360M.

## 🔍 Ce que D1 prouve — et ne prouve pas (étiquettes honnêtes 🏷️)
✅ **Prouvé** : l'extraction physique est exacte, la tranche est un objet autonome calculable
et différenciable, la réinjection est chirurgicale. Le scalpel coupe droit.
🚫 **Pas prouvé** : aucune économie de temps ni de RAM en entraînement (c'est D2), aucune
qualité de modèle (c'est D3). Aucun chiffre d'entraînement n'existe encore pour le découpage.

## 📦 Fichiers de la marche
`src/bistouri_physique.py` · `tests/test_bistouri_physique.py` · ce rapport ·
`docs/PROPOSITION_DECOUPAGE_PHYSIQUE.md` (validée par le Go) · `MANIFESTE.json` re-scellé.
Commit local de scellement ; **rien de poussé sur GitHub** (en attente de ton ordre).

## 🚦 Suite
Marche **D2 — mock-training de mesure sur un bloc MLP isolé** (temps/pas, RAM pic, gradients,
états Adam ; critères proposés déjà écrits dans la proposition) : **en attente de ton Go, chef.**

*Micro-rapport scellé — RATISS Labs, Yaoundé. Étape par étape, détaillé, reproductible. 😜🥳*
