# ⚖️ PROTOCOLE FIGÉ — OPTION 1, MARCHE E1 : ÉTENDRE LES TRANCHES (2 → 4 COUCHES)
**Ordre du chef :** « passe à la suite 1 » — Option 1 (plus de couches), **par marches**. 🪜
**Cobaye :** SmolLM2-360M · **100 % CPU** · repo local, aucun push sans ordre.

## 🎯 Question
L'apprentissage par tranche physique, spectaculaire sur les 2 dernières couches
(D3 : perte 6,119 → 2,389, mieux que le dense), **tient-il quand on double la profondeur
opérée** — segment des **4 dernières couches (28–31)** du modèle ?

## 🎛️ Dispositif (figé, hérité de D3 à l'identique sauf la profondeur)
- Même tâche : 60 Q/R anglaises du Bistouri (48 entraînement / 12 évaluation, split figé).
- Même chirurgie par couche : inactive gelée sans graphe + tranche de l'accord (à sa couche
  propre) entraînable, initialisée aux poids d'origine.
- Mêmes hyperparamètres : Adam 5e-4, 3 époques, batch 1 groupé par famille, graine 20261102.
- Entrée du segment : états cachés du modèle gelé capturés avant la couche 28.
- **Contrôle d'identité renforcé** : la préparation calcule aussi la perte d'évaluation
  du **modèle plein entier** ; la perte initiale du segment opéré devra l'égaler
  (écart rapporté, jamais caché).

## ⚖️ Critères de verdict (figés)
- **E1-a** : la perte d'évaluation du segment 4-couches-tranche **baisse** après entraînement.
- **E1-b** : l'amélioration relative est du même ordre que celle de D3 (−61 % sur 2 couches) ;
  un effondrement (amélioration < 20 %) marquerait la limite de l'extension.
- **E1-c** : paramètres mis à jour par pas très inférieurs au dense équivalent 4 couches
  (4 × 7 372 800 × … MLP denses des 4 couches ≈ 29,5 M en ordre de grandeur).

## 🚫 Hors périmètre de cette marche
Pas de 8/16/32 couches (marches suivantes, sur Go séparés). Pas de génération.
Pas de nettoyage mémoire. Pas de push GitHub. On mesure, on rapporte, le chef décide. 🫡
