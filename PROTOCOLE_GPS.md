# ⚖️ PROTOCOLE FIGÉ — ÉTAPE 1 : LE GPS DU MIXER (Routeur d'accords)
**Date :** 2 novembre 2026 (reprise en douceur 🛋️) · **Repo :** RATISS-X · **Auteur :** Jonathan Evina
**Règle :** critères écrits AVANT l'entraînement du routeur. Un test qui ne peut pas échouer est interdit. 🫡

## 🧭 Question
Quand une question arrive, peut-on prédire **quel accord activer** (famille de chemin + masque
de neurones $M_k$) en **moins de 1 ms sur CPU**, **sans faire tourner le transformeur** ?

## 🎛️ Instrument proposé (GPS v1)
1. **Entrée** : embedding moyen de la question — simple consultation de la table d'embedding
   du cobaye SmolLM2-135M (aucune couche transformeur calculée !).
2. **Cerveau du GPS** : classifieur minuscule (centroïdes de familles, comparé à une régression
   logistique et à un MLP 2 couches). Taille visée : quelques milliers de paramètres.
3. **Sortie** : famille prédite → masque $M_k$ = prototype d'accord de la famille
   (neurones présents dans ≥ 50 % des chemins de cette famille au banc d'essai n°1).

## ⚖️ Critères de verdict (figés)
- **C1 Exactitude** : validation croisée (leave-one-out) sur les 60 questions du banc n°1
  nettement au-dessus du hasard (16,7 %). Cible : ≥ 50 %.
- **C2 Vitesse** : latence médiane de routage (tokenisation incluse et hors tokenisation
  rapportées séparément, honnêteté oblige) < 1 ms sur ce CPU 2 cœurs. Cible rêvée : ~0,1 ms.
- **C3 Qualité du masque** : le masque prédit doit recouvrir le vrai chemin de la question
  (Jaccard) **mieux que le masque « autoroute seule »** (baseline paresseuse).
- **C4 Robustesse paraphrases** : sur les 18 paraphrases (jamais utilisées pour fabriquer
  les masques), le GPS doit retrouver la famille d'intention dans ≥ 50 % des cas.
  Sinon : le GPS hérite du bug de surface du cobaye, on le dira. 🍎

## 🚫 Garde-fous du repo (ordre du chef)
- Repo **léger** : aucun poids de modèle, aucune donnée lourde. Le cobaye SmolLM2-135M
  reste l'unique modèle de test jusqu'à validation complète de la Phase 2.
- Le futur modèle augmenté s'appellera **RATISS VEGA** — il ne naîtra qu'après validation. 🌟
