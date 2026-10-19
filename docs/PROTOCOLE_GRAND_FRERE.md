# ⚖️ PROTOCOLE FIGÉ — REPASSER TOUS LES TESTS SUR LE GRAND FRÈRE
**Date :** 2 novembre 2026 · **Ordre du chef :** « on appelle le grand frère » 🐘
**Nouveau cobaye :** SmolLM2-360M-Instruct (360M, 32 couches, hidden 960, MLP 2560)
**Ancien cobaye :** SmolLM2-135M — ses résultats restent dans l'historique git et les rapports.

## 🔁 Les 4 tests à refaire, à l'identique (mêmes stimuli, mêmes critères)
1. **Banc n°1 — Carte des chemins** : 78 stimuli du questionnaire. H1 (intra > inter,
   permutation 2000, p < 0,05), contrôle lexical, H2 paraphrases, famille devinée au chemin (1-NN).
2. **Étape 1 — GPS** : mêmes critères C1 (≥ 50 %), C2 (< 1 ms), C3 (masque > autoroute),
   C4 (paraphrases ≥ 50 %). Masques reconstruits depuis la carte du 360M.
3. **Étape 2 — Bistouri CPU** : mêmes critères E2-a/b/c, segment = 2 dernières couches,
   mêmes 60 Q/R anglaises, mêmes hyperparamètres (Adam 5e-4, 3 époques, batch 1).
   **Zéro GPU, zéro VRAM : RAM CPU et secondes/pas uniquement.**
4. **Étape 3 — Traducteur** : même test scellé `paraphrases_test_v2.json` (18 questions,
   jamais réécrites), même α = 0,8 figé. Critère E3 : traduit > seul.

## 🧹 Règle d'hygiène de l'espace (ordre du chef)
- Workspace **rincé avant et après** : modèles, caches et fichiers lourds temporaires
  sont évacués dès qu'ils ont servi. **Plafond persisté : 50 Mo** (sur 128 Mo possibles).
- Seuls restent : code, protocoles, rapports, résultats JSON légers, artefacts du GPS (Ko).

## ⚖️ Lecture finale attendue
Comparaison honnête 135M contre 360M, test par test, avec la même grille :
si le grand frère fait s'envoler les critères fragiles (C4, E2-b, E3 absolu),
l'architecture RATISS est validée sur un cerveau plus capable — sinon, on le dira.
*Précision d'honnêteté : le 360M tourne ici en BF16 sur CPU (RAM oblige), le 135M tournait
en FP32 ; les comparaisons de vitesse/RAM doivent se lire avec cette étiquette.* 🏷️
