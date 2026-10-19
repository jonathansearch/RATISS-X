# ⚖️ PROTOCOLE FIGÉ — EXTENSION E4 : LE MODÈLE ENTIER (32 COUCHES)
**Ordre du chef :** « Go » — marche finale de la série d'extension. **Segment : couches 0–31**
(la totalité de SmolLM2-360M). Même tâche, même graine 20261102, mêmes époques, masque causal.

## 🔧 Moteur (le meilleur mesuré de la série M)
Socle gelé **BF16** + tranches FP32 (`--memoire_leger`) + **Adam à états BF16** (M3) +
libération de l'ancienne résidence MLP + chargement `mmap` (ici sans copie : le socle
reste en BF16, l'`assign` devient résidence unique) + nettoyage `gc`.

## ⚖️ Critères de victoire (figés avant la mesure)
- **E4-a** : contrôle d'identité sain (écart segment avant / modèle plein < 0,10).
- **E4-b** : perte d'évaluation en baisse nette, même ordre que la série
  (plancher 30 % ; série : −60,0 / −62,4 / −65,5 / −70,2 %).
- **E4-c** : mises à jour par pas très inférieures au dense MLP 32 couches
  (~235,9 M de paramètres MLP) — facteur attendu ~÷20.
- RAM et temps par pas **publiés tels quels** (passeport moteur natif).
- ⚠️ Risque connu et accepté : le pic RAM peut approcher le plafond de la sandbox
  (~2 Go). Si le processus est tué, on le publie et on ajuste — pas de maquillage.

## 🚫 Hors périmètre
Aucun changement de tâche, de graine ni d'hyperparamètres. RATISS VEGA reste non déclaré.
