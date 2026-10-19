# ⏱️ MICRO-RAPPORT D2 — LE MOCK-TRAINING PARLE : 11× PLUS VITE, −270 Mo, 5 % DES GRADIENTS
**Marche :** D2 uniquement (`PROPOSITION_DECOUPAGE_PHYSIQUE.md`, Go du chef). **D3 non commencée.** ⏸️
**Nature exacte du banc :** mécanique d'entraînement sur **un bloc MLP isolé** — PAS un entraînement de LLM. 🏷️

## 🪜 Ce qui a été fait, dans l'ordre
1. **Le banc** (`src/mock_training_decoupe.py`, paramètres figés dans son en-tête avant exécution) :
   bloc MLP **réel** de la couche 15 du 360M (gate/up/down, converti FP32), accord = masque GPS
   **GEOGRAPHIE couche 15 (128 neurones sur 2560)** ; lot fixe de 128 vecteurs synthétiques
   (graine `20261104`, étiqueté) ; étudiant = poids du maître + bruit figé (sinon perte nulle) ;
   Adam lr 1e-3, 10 pas d'échauffement, **50 pas chronométrés**, CPU 2 fils.
2. **Deux processus séparés** (pour une RAM pic comparable) : `--scenario dense` puis `--scenario tranche`.
   Le dense apprend la sortie complète du maître ; la tranche apprend **sa composante active**
   (cibles différentes et étiquetées : leurs pertes ne se comparent pas entre elles).
3. **Consolidation** dans `results/RESULTATS_DECOUPAGE_D2.json` avec verdicts des critères figés.

## 📏 Les chiffres mesurés (CPU uniquement, ce jour)
| Mesure | 🐘 Dense (bloc complet) | 🔪 Tranche physique | Rapport |
|---|---|---|---|
| Paramètres entraînés | 7 372 800 | 368 640 (128 neurones) | — |
| **Éléments de gradient calculés (1ᵉʳ pas)** | 7 372 800 | 368 640 | **5,00 %** ✅ |
| **États Adam (mesurés sur tenseurs)** | 58,98 Mo | 2,95 Mo | **5,00 %** ✅ |
| **Temps par pas (médiane, 50 pas)** | 30,0 ms | **2,7 ms** | **11,3× plus rapide** ✅ |
| Coût d'extraction de la tranche (à part, médiane 30×) | — | 0,64 ms | même ajoutée à chaque pas : ~9× plus rapide |
| **RAM pic du processus** | 630,7 Mo | **360,3 Mo** | **−270,4 Mo (−43 %)** ✅ |
| Perte (tâche propre à chaque scénario) | 44,70 → 0,303 | 2,16 → 0,040 | les deux apprennent leur cible |

## ⚖️ Verdicts des critères figés (proposition §4, jugés sans retouche)
- Gradients ≈ 4–5 % du dense : ✅ **CONFIRMÉ (5,00 %)**
- États Adam au même ratio : ✅ **CONFIRMÉ (59,0 → 2,9 Mo)**
- Au moins 2× plus rapide par pas : ✅ **CONFIRMÉ, et largement (11,3×)**
- RAM inférieure au dense, écart rapporté : ✅ **CONFIRMÉ (−270 Mo)** — aucun seuil absolu n'avait été promis, l'écart est le résultat.

## 🔍 Lecture honnête (transdisciplinaire, comme toujours 🌈)
- **Pourquoi si rapide ?** Le produit matriciel dense dépense son temps sur 2560 neurones
  dont 95 % ne servent pas l'accord ; la tranche ne multiplie que le couloir utile.
  Le surcoût d'extraction (0,64 ms, payé une fois par accord) reste petit devant le gain.
- **Pourquoi la RAM ne tombe-t-elle pas à 5 % ?** Le **socle** (Python + PyTorch + tenseurs
  d'entrée) pèse ~300 Mo incompressibles sur ce banc : 360 Mo au total, c'est le socle
  + 5 % du variable. Le gain se lit sur la **part variable**, et il est massif.
- **Ce que D2 ne dit toujours pas** : si un LLM dont on ne calculerait que les tranches
  répondrait aussi bien (la tranche du banc imite une composante, elle ne « pense » pas).
  C'est exactement la question de D3, pas une conclusion de D2. 🏷️

## 📦 Fichiers de la marche
`src/mock_training_decoupe.py` · `results/RESULTATS_DECOUPAGE_D2.json` (+ les deux partiels par scénario) ·
ce micro-rapport · `MANIFESTE.json` re-scellé. Modèle téléchargé pour la mesure puis **évacué** (rinçage). 🧽
Commit local ; **rien de poussé** sans ton ordre.

## 🚦 Suite
Marche **D3 — mini-intégration au segment des 2 dernières couches** (variante honnête :
contribution inactive gelée sans graphe + tranche active entraînable ; perte réelle sur les
60 Q/R, face au dense et à la correction sparse actuelle) : **en attente de ton Go, chef.**

*Micro-rapport scellé — RATISS Labs, Yaoundé. Étape par étape, mesuré, reproductible. 😜🥳*
