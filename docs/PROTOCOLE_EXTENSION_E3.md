# ⚖️ PROTOCOLE FIGÉ — EXTENSION E3 : 16 COUCHES (la moitié du réseau)
**Ordre du chef :** « Fais tout », à la suite de la déclaration passeport moteur natif.
**Segment : couches 16–31** de SmolLM2-360M. Même tâche, même graine 20261102, mêmes époques.

## 🔧 Moteur retenu (le meilleur mesuré de la série M)
Socle gelé **BF16** + tranches FP32 (`--memoire_leger`) + **Adam à états BF16**
(`--optimiseur adam_bf16`, validé en M3 : perte identique, états ÷2) + libération de
l'ancienne résidence MLP + nettoyage `gc`. Chargement mmap **non** retenu (M3).

## ⚖️ Critères de victoire (figés avant la mesure)
- **E3-a** : contrôle d'identité sain (écart segment avant entraînement / modèle plein < 0,10).
- **E3-b** : la perte d'évaluation baisse nettement, amélioration du même ordre que la
  série (plancher : 30 % ; série corrigée : −60,0 % à −65,6 %).
- **E3-c** : mises à jour par pas très inférieures au dense MLP équivalent 16 couches
  (~117,96 M de paramètres MLP) — facteur attendu ~÷20.
- RAM et temps par pas **publiés tels quels** (plus de passeport bloquant, cf. déclaration).

## 🚫 Hors périmètre
Modèle entier (32 couches) = marche E4 séparée. Aucun changement de tâche ni de graine.
