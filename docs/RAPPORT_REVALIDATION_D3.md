# 🩺 RAPPORT DE REVALIDATION — LE TABLEAU D3 RÉÉCRIT AU MASQUE CAUSAL (et la tranche gagne toujours)
**Protocole :** `PROTOCOLE_REVALIDATION_D3.md` — seul changement d'instrument : **masque causal explicite**.
**Même segment (couches 30–31, 360M), mêmes 60 Q/R, mêmes hyperparamètres, même graine. CPU, local, non poussé.**

## 📏 Le nouveau tableau de référence (corrigé, il supersède l'ancien pour les pertes absolues)
| Mesure CPU | 🩺 Base | 🐘 Dense | 💉 Correction sparse | 🔪 **Tranche physique** |
|---|---|---|---|---|
| Perte d'éval. avant → après | 6,157 → 6,157 | 6,157 → **2,699** (−56,2 %) | 6,157 → 5,501 (−10,6 %) | 6,157 → **2,466 (−60,0 %)** 🏆 |
| Paramètres mis à jour / pas | 0 | 14 745 600 | 241 920 | 725 760 (**÷20**) |
| Secondes / pas | — | 0,1026 | 0,0394 | 0,0458 (**2,2× plus vite que dense**) |
| RAM pic | — | 1055 Mo | 906 Mo | 962 Mo |

## ⚖️ Critères du protocole, jugés
- **R1 (identité du harnais corrigé)** : ✅ perte de base **6,1566** contre 6,1702 pour le modèle
  plein entier mesurée en E1 sur la même évaluation → écart **0,0136**, le bruit de précision attendu.
  Le harnais ne ment plus.
- **R2 (classement rapporté tel quel)** : ✅ **tranche (2,466) < dense (2,699) < correction (5,501)**.
  La tranche physique reste devant le dense, avec une marge même un peu plus nette qu'historiquement
  (écart 0,234 contre 0,104 dans l'ancien tableau).

## 🕰️ Historique contre corrigé (ce qui change, ce qui ne change pas)
| Scénario | Ancien (sans masque) | Corrigé (masque causal) | Lecture |
|---|---|---|---|
| Dense | 4,185→2,087 (135M) · 6,119→2,493 (360M) | 6,157→2,699 | pertes absolues anciennes optimistes (fuite du futur) |
| Correction sparse | → 5,459 (−10,8 %) | → 5,501 (−10,6 %) | quasi inchangée en relatif |
| Tranche physique | → 2,389 (−61,0 %) | → **2,466 (−60,0 %)** | **la conclusion survit à la correction** 🎉 |

🧭 **Phrase-bilan** : le piège du masque gonflait les chiffres absolus de tout le monde ;
une fois l'instrument réparé, **le classement et le message scientifique restent debout** :
la rétropropagation par tranche spécialisée apprend aussi bien (ici : un peu mieux)
que le dense, avec 20× moins de mises à jour par pas et 2,2× plus vite sur ce segment.

## 📌 Rectificatifs apposés
Une note de rectification a été ajoutée en tête de `RAPPORT_ETAPES_2_3.md`,
`RAPPORT_GRAND_FRERE.md` et `RAPPORT_DECOUPAGE_D3.md` : leurs pertes absolues de segment
sont supersédées par ce tableau ; leurs comparaisons internes restent des comparaisons
appariées sous l'ancien instrument. Les versions publiées sur GitHub restent l'historique ;
ce corrigé partira au prochain push ordonné par le chef.

## 📦 Fichiers de la marche
Correctif masque causal : `src/bistouri_backprop.py`, `src/integration_tranche_segment.py` ·
Résultats rejoués : `results/RESULTATS_BISTOURI_{base,dense,sparse}.json`, `results/RESULTATS_DECOUPAGE_D3.json` ·
ce rapport · `MANIFESTE.json` re-scellé. Segment lourd et modèle évacués. Commit local, rien de poussé.

## 🚦 Suites proposées (une par Go, chef)
1. 🪜 **E2 : 8 couches** au masque causal (la montée continue, instrument propre).
2. 📤 Pousser la revalidation + les rectificatifs sur GitHub (la science corrigée en public, notre signature).

*Rapport scellé — RATISS Labs, Yaoundé. On mesure, on nomme, on corrige, on publie. 😜🥳*
