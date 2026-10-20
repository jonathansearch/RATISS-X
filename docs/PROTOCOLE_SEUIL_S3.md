# ⚖️ PROTOCOLE FIGÉ — MARCHE S3 : LE FOND PAR MAGNITUDE (l'élite des inactifs) 🎯👑
**Ordre du chef :** « GO IMMÉDIAT POUR S3 (LE FOND PAR MAGNITUDE) ». Héritage :
S1 (fond uniforme : échec ~0,43) → S2 (fond uniforme boosté ×1/p : pire, ~0,08).
Dernier candidat debout : **la sélection par contenu, pas par échantillonnage** 🏋️

## 🎯 Dispositif (figé avant mesure)
- **Aucun ré-entraînement** : poids E4, référence PyTorch complète régénérée (contrôle 1,2449).
- Pour chaque couche et la famille du prompt, les neurones **inactifs** (hors tranche)
  sont classés par **magnitude de poids** : `||ligne gate||₂ + ||ligne up||₂ + ||colonne down||₂`
  (poids d'origine, critère statique, indépendant de l'entrée — calculé une fois).
- Le fond = les **top-K** de ce classement, `K = p × (nombre d'inactifs)`, **sans boost**
  (la magnitude naturelle des champions fait le travail, dixit le chef 💪).
- Balayage `p ∈ {1 %, 5 %, 10 %, 100 %}` (100 % = contrôle : tout le fond, comme S1/S2).
- Prompt figé « La capitale de la France est » (GEOGRAPHIE). Métriques inchangées :
  cosinus état couche 32 + logits contre référence, argmax, génération 18 tokens + boucle.

## ⚖️ Critère de victoire
Plus petit `p` avec **cosinus logits > 0,95** ET génération cohérente → l'élite suffit.
Sinon : la carte des fonds est complète et publiée — la structure inactive serait
irréductible à toute sélection partielle statique testée. 🎯
