# ⚖️ PROTOCOLE FIGÉ — MARCHE E5 : LE MOTEUR NATIF ÉCONOMIQUE ⚙️🔥
**Ordre du chef :** Priorité 2 du double GO (après le push de la saga, tête `d23c7df`).
**Recette issue de la saga des fonds** : inférence = tranche entraînée + **70 % de
l'élite inactive par magnitude** (seuil cosinus S5 : 0,9094 ; cohérence à 75 %).

## 🎯 Dispositif (figé avant mesure)
- Même moteur natif (NumPy, zéro PyTorch), mêmes tranches entraînées E4 exportées.
- **Deux configurations comparées à prompts identiques** (3 prompts de F3, routages
  GPS de la génération E4, glouton 18 tokens) :
  - `hybride` : tranche + fond élite **70 %** (sélection magnitude, sans boost) ;
  - `dense` : tranche + fond **100 %** (le moteur en mode complet, référence interne).
- Mesures par configuration, **dans des processus séparés** : RSS pic du processus,
  temps moyen par token, générations brutes + détecteur de boucle 📢

## ⚖️ Critères de lecture
- L'hybride **génère de façon cohérente** (pas de boucle, texte lisible) sur les prompts.
- Gains mesurés et publiés tels quels : ΔRSS et Δtemps/token hybride contre dense —
  **sans promesse chiffrée à l'avance** (la part MLP n'est qu'une fraction du coût :
  attention, embeddings et tête restent identiques entre les deux modes 😇).
- Aucun ré-entraînement, aucun push dans cette marche 🚫
