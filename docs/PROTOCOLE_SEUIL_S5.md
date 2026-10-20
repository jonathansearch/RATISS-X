# ⚖️ PROTOCOLE FIGÉ — MARCHE S5 : LE ZOOM SUR LE GENOU (60/65/70 %) 🔬🦵
**Ordre du chef :** Option S5 choisie après le push de la saga (tête distante `6c46cb9`).
Suite de S4 : le seuil (cosinus logits ≥ 0,90 + génération cohérente) est atteint à
75 % (0,9323), le genou est entre 50 % (0,8002) et 75 %. On resserre la maille 🎯

## 🎯 Dispositif (rien ne change sauf la grille)
- Même moteur natif, même sélection **élite par magnitude**, même prompt figé
  « La capitale de la France est » (GEOGRAPHIE), même référence PyTorch complète
  (contrôle d'entraînement 1,2449), **zéro ré-entraînement** 🫡
- Grille du zoom : **p ∈ {60 %, 65 %, 70 %}** — points voisins déjà mesurés :
  50 % → 0,8002 · 75 % → 0,9323 (S4, non remesurés).

## ⚖️ Lecture attendue et critère
Le plus petit `p` du zoom avec **cosinus logits ≥ 0,90** affine le Seuil RATISS ;
la traversée exacte restera bornée entre deux points de grille voisins (on publie
l'intervalle, jamais une interpolation déguisée en mesure 😇). Générations publiées
brutes, comme toujours 📢 Aucun push dans cette marche 🚫
