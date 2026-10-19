# ⚖️ PROTOCOLE FIGÉ — OPTION 1, MARCHE E2 : ÉTENDRE LES TRANCHES (4 → 8 COUCHES)
**Ordre du chef :** Option 1 après revalidation D3. **Cobaye :** SmolLM2-360M · **100 % CPU** · local, aucun push. 🫡

## 🎯 Question
L'apprentissage par tranche physique tient-il quand on opère les **8 dernières couches
(24–31)** du modèle — le quart du réseau — avec le harnais **au masque causal corrigé** ?

## 🎛️ Dispositif (figé, hérité d'E1 à l'identique sauf la profondeur)
- Même tâche (60 Q/R anglaises, 48 train / 12 éval), même chirurgie par couche
  (inactive gelée sans graphe + tranche d'accord à sa couche), masques GPS par couche.
- Mêmes hyperparamètres : Adam 5e-4, 3 époques, 144 pas, batch 1 groupé par famille, graine 20261102.
- Contrôle d'identité : perte initiale du segment contre **perte du modèle plein**
  recalculée à la préparation sur la même évaluation (référence E1 : 6,1702).

## ⚖️ Critères (figés)
- **E2-a** : la perte d'évaluation **baisse** nettement après entraînement.
- **E2-b** : amélioration relative du même ordre que E1 (−62,4 %) et D3 revalidé (−60,0 %) ;
  une chute sous 30 % marquerait le début de la limite d'extension.
- **E2-c** : paramètres mis à jour par pas ≪ dense MLP équivalent 8 couches (~59,0 M).

## 🚫 Hors périmètre
Pas de 16/32 couches (marches suivantes, Go séparés). Pas de génération. Pas de push.
