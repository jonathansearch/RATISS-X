# 🪜 MICRO-RAPPORT E1 — LES TRANCHES TIENNENT SUR 4 COUCHES (et un piège a été débusqué)
**Marche :** Option 1 / E1 (protocole `PROTOCOLE_EXTENSION_E1.md`, ordre du chef). **100 % CPU, local, non poussé.**
**Segment opéré : couches 28–31 du SmolLM2-360M** (au lieu de 30–31 en D3), même tâche, mêmes hyperparamètres.

## 🚨 D'ABORD, LA DÉCOUVERTE DU CONTRÔLE (la plus importante du rapport)
Notre contrôle d'identité renforcé comparait la perte initiale du segment opéré à celle du
**modèle plein entier** : premier essai **5,4476 contre 6,1702** (écart 0,72). Plutôt que de
célébrer un « −64 % » suspect, nous avons enquêté :
- test jumeau BF16/FP32 : l'écart persiste (5,4359) → **ce n'était pas la précision** ;
- cause trouvée : le harnais de segment appelait l'attention **sans masque causal** —
  chaque token pouvait regarder les tokens **futurs** de la réponse. La perte mentait. 🎭
- **Correction** : masque causal triangulaire explicite dans le segment. Nouvel écart
  d'identité : **0,0168** (6,1534 contre 6,1702) — le bruit de précision attendu. ✅

⚠️ **Conséquence honnête sur le passé** : les scénarios segment de l'Étape 2 et de D3 partageaient
le même harnais sans masque. Leurs comparaisons internes (dense contre sparse, même instrument,
même fuite des deux côtés) restent des comparaisons appariées, mais leurs **pertes absolues
étaient optimistes**. Une marche de revalidation « D3 au masque causal » est proposée au chef —
pas exécutée ici. C'est exactement pour cela que les contrôles existent. 🫡

## 📏 Les chiffres E1 corrigés (masque causal, CPU)
| Mesure | Valeur | Référence |
|---|---|---|
| Perte modèle plein (éval, 12 questions) | 6,1702 | mesurée à la préparation |
| Perte segment opéré avant entraînement | 6,1534 (écart 0,0168 ✅) | identité vérifiée |
| **Perte après entraînement tranche (4 couches)** | **2,3128 (−62,4 %)** | critère E1-a ✅ |
| Paramètres mis à jour / pas | 1 440 000 | dense MLP équivalent 4 couches ≈ 29,5 M → **÷20** (E1-c ✅) |
| Secondes / pas | 0,0653 | D3 (2 couches) : 0,0497 |
| RAM pic du processus | 1184 Mo | D3 : 969 Mo (plus de couches = plus de segment) |
| Pas d'entraînement | 144 (3 époques) | identique à D3 |

## ⚖️ Verdict des critères figés
- **E1-a** (la perte baisse) : ✅ 6,153 → 2,313.
- **E1-b** (même ordre d'amélioration que D3, −61 %) : ✅ **−62,4 %** — l'extension de 2 à 4 couches
  ne dégrade pas l'apprentissage par tranche ; le résultat final (2,31) est même légèrement
  meilleur que celui du segment 2 couches (2,39, harnais différent : comparaison indicative seulement).
- **E1-c** (très inférieur au dense équivalent) : ✅ ÷20 par pas.

## 🔍 Lecture simple
Doubler la profondeur opérée n'a pas cassé la mécanique : chaque couche ajoutée apporte
sa tranche d'accord, l'erreur continue de descendre **le long des chemins**, et le coût reste
proportionnel (2× plus de paramètres mis à jour pour 2× plus de couches — toujours 20× sous le dense).
Le prix se paie en RAM du segment et en temps par pas, pas en qualité d'apprentissage. 🛤️

## 📦 Fichiers de la marche
`src/extension_tranches.py` · correctif masque causal aussi appliqué à `src/integration_tranche_segment.py`
(partagé) · `results/RESULTATS_EXTENSION_E1.json` · ce rapport · `MANIFESTE.json` re-scellé.
Segment lourd local (169 Mo) et modèle **évacués** après mesure. Commit local, **rien de poussé**.

## 🚦 Suites proposées (une par Go, chef)
1. 🩺 **Revalider D3 au masque causal** (2 couches, même protocole corrigé) pour réécrire le tableau de référence proprement.
2. 🪜 **E2 : 8 couches** (si la revalidation tient), puis 16, puis 32 — la montée vers le modèle entier.
3. 📤 Pousser E1 (+ la découverte du masque) sur GitHub quand tu l'ordonneras — les échecs et pièges font partie de la science publiée.

*Micro-rapport scellé — RATISS Labs, Yaoundé. On mesure, on nomme, on publie — surtout les pièges. 😜🥳*
