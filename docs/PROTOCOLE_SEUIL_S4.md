# ⚖️ PROTOCOLE FIGÉ — MARCHE S4 : LE BALAYAGE ÉLARGI (le genou de la courbe) 📈🦵
**Ordre du chef :** « GO IMMÉDIAT POUR S4 ». **Aucun push** tant que la courbe n'est pas
complète (Option 2 en attente, ordre du chef). Héritage S3 : l'élite par magnitude
monte en pente régulière (0,46 → 0,51 → 0,63 à 10 %) sans atteindre le critère.

## 🎯 Dispositif (une seule chose change : la grille)
- Même moteur, même sélection **top-K par magnitude** (||gate||+||up||+||down||),
  même prompt figé « La capitale de la France est » (GEOGRAPHIE), même référence
  PyTorch complète (contrôle d'entraînement 1,2449), **zéro ré-entraînement** 🫡
- Grille élargie : **p ∈ {25 %, 50 %, 75 %}** — les points 1/5/10 % (S3) et 100 %
  (contrôles S1–S3, cosinus logits 0,9999) complètent la courbe sans être remesurés.

## ⚖️ Objectif et critère
Trouver le **pourcentage minimal** où le cosinus des logits dépasse **0,90–0,95**
avec génération cohérente : le **Seuil de Stabilisation RATISS** 🎯
Si la courbe plafonne sous le critère même à 75 % : publié tel quel, la frontière
finale sera entre 75 % et 100 % — ce sera écrit, pas extrapolé 😇
