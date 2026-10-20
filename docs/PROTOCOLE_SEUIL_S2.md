# ⚖️ PROTOCOLE FIGÉ — MARCHE S2 : LE FOND RENORMALISÉ (×1/p) 🧮🔥
**Ordre du chef :** « GO IMMÉDIAT POUR S2 (FOND RENORMALISÉ ×1/p) ». Suite de S1
(aucun fond 1/5/10 % non renormalisé ne restabilise : cosinus logits ~0,43 ; 100 % = 0,9999).

## 🎯 L'idée testée (une seule variable change par rapport à S1)
Le fond partiel (fraction `p` des neurones inactifs, sélection équidistante, poids
d'origine — **rigoureusement comme S1**) voit sa sortie **multipliée par 1/p** avant
d'être ajoutée à la tranche entraînée : `sortie MLP = tranche + (1/p) × fond_partiel`.
Hypothèse du chef : la direction du fond est bonne, c'est sa **magnitude** (~p fois
trop faible) qui brise la géométrie — restaurer l'énergie doit restabiliser le réseau.

## 📏 Balayage et métriques (identiques à S1)
- `p ∈ {1 %, 5 %, 10 %, 100 %}` (à 100 %, ×1/p = ×1 : point de contrôle interne).
- Prompt figé « La capitale de la France est » (GEOGRAPHIE), référence PyTorch complète.
- Cosinus état couche 32 + logits, argmax, génération gloutonne 18 tokens, détecteur de boucle.

## ⚖️ Critère de victoire
Le plus petit `p` avec **cosinus logits > 0,95** et génération cohérente = le « ciment
mathématique » est trouvé. Sinon : publié tel quel (la renormalisation peut aussi
amplifier le bruit — ce sera écrit). Zéro ré-entraînement, hygiène RAM de L1/S1,
aucun push dans cette marche. 🫡
