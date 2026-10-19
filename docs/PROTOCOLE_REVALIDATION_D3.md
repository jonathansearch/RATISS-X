# ⚖️ PROTOCOLE FIGÉ — REVALIDATION D3 AU MASQUE CAUSAL
**Ordre du chef :** Option 1 après E1. **Repo local, aucun push sans ordre.** 🫡

## 🎯 Question
Le tableau de référence du segment 2 couches (base / dense / correction sparse /
tranche physique), mesuré avec un harnais d'attention **sans masque causal**
(fuite du futur découverte en E1), que vaut-il quand on refait **exactement le même
protocole** avec le masque causal corrigé ?

## 🎛️ Dispositif (figé)
- Même segment (couches 30–31 du 360M), mêmes 60 Q/R (48 train / 12 éval, split figé),
  mêmes hyperparamètres partout (Adam 5e-4, 3 époques, 144 pas, batch 1, graine 20261102).
- Seul changement d'instrument : **masque causal triangulaire explicite** dans les
  deux harnais (`bistouri_backprop.py`, `integration_tranche_segment.py`).
- Les 4 scénarios sont rejoués dans l'ordre : base → dense → correction → tranche physique.
- Les anciens chiffres restent dans l'historique git et sur GitHub ; ce protocole
  produit le **tableau corrigé** qui les supersède pour les pertes absolues.

## ⚖️ Critères
- **R1** : la perte initiale (base) doit se rapprocher de la perte du modèle plein
  mesurée en E1 sur la même éval (6,1702) — contrôle d'identité du harnais corrigé.
- **R2** : le classement dense / correction / tranche est rapporté tel qu'il sort,
  même s'il contredit le tableau historique. Aucun critère ne sera retouché après coup.
