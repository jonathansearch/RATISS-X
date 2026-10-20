# ⚖️ PROTOCOLE FIGÉ — MOTEUR V1 À POIDS RÉSIDENTS ⚙️
**Ordre du chef :** Priorité 2 du double GO (push E5 fait : tête `82785a0`).
**Leçon d'E5 à corriger** : sans cache, la sparsité ne paie pas — le moteur v0 relisait
et découpait les poids à **chaque token**. V1 charge les poids utiles **une seule fois**.

## 🎯 Dispositif (figé avant mesure)
- Nouveau moteur `src/moteur_v1.py` (NumPy, zéro PyTorch) : au démarrage, pour la
  famille demandée, il construit des **matrices compactes résidentes** par couche :
  attention gelée (BF16) + MLP réduit aux neurones utiles, gardés en BF16 en RAM.
  Aucune lecture fichier, aucun masquage, aucune indexation pendant la génération
  (la table d'embedding des logits reste lue par morceaux — décision écrite ici).
- Deux modes comparés, processus séparés, 3 prompts de F3, mêmes routages :
  - `hybride` : neurones utiles = tranche entraînée + **top 70 % élite** par magnitude ;
  - `dense` : tous les neurones (référence, même code, mêmes résidents).
- Mesures : **RSS au repos** (après chargement), **RSS pic**, **temps/token**,
  temps de démarrage, générations brutes + boucle 📢

## ⚖️ Critères de lecture
- L'hybride doit être **plus léger et/ou plus rapide** que le dense à code égal,
  sinon la sparsité résidente est réfutée à son tour 😇
- Stabilité des générations publiée telle quelle (E5 a montré que 70 % ne garantit
  pas la génération libre multi-prompts : on revérifie, on ne promet rien).
- Tranches E4 ré-exportées pour l'occasion ; aucun ré-entraînement du modèle,
  aucun push dans cette marche 🚫
