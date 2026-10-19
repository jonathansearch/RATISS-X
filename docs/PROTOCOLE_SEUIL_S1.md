# ⚖️ PROTOCOLE FIGÉ — MARCHE S1 : LE SEUIL DE STABILISATION (le « fond dense minimal »)
**Ordre du chef :** sa « Marche M3 » — enregistrée **S1** au journal car M3 désigne déjà
la marche Adam-BF16 (aucune collision d'archive). **Suite directe de L1** : la tranche
seule s'effondre entre les couches 16 et 32 (cosinus 0,4615) ; on cherche le point de bascule.

## 🎯 Dispositif (figé avant mesure)
- **Aucun ré-entraînement** : poids E4 entraînés (référence PyTorch complète régénérée
  par la phase `torch` de L1, contrôle 1,2449), on ne modifie que l'**inférence** du moteur natif.
- Le moteur calcule : tranche entraînée + **fond** = fraction `p` des neurones inactifs
  (hors tranche, poids d'origine), sélection **structurée déterministe** (répartition
  équidistante dans l'ensemble inactif de chaque couche, même graine logique partout).
- Balayage : **p ∈ {0 %, 1 %, 5 %, 10 %, 100 %}** — 0 % et 100 % sont les références
  (échec et succès attendus). Prompt figé : « La capitale de la France est » (GEOGRAPHIE).
- Mesures par `p` : cosinus de l'état caché post-couche-32 et des **logits finaux**
  contre la référence complète PyTorch ; génération gloutonne 18 tokens + détecteur de boucle.

## ⚖️ Critère de victoire (du chef)
Le plus petit `p` tel que **cosinus des logits > 0,95** ET génération cohérente
(pas de boucle). Si même 10 % s'effondre : publié tel quel, la frontière est la donnée.

## 🚫 Hors périmètre
Pas de ré-entraînement, pas d'autre prompt pour le critère (les autres générations
sont des observations), pas de push avant validation du chef.
