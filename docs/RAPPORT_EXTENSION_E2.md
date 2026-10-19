# 🪜 MICRO-RAPPORT E2 — LES TRANCHES TIENNENT SUR 8 COUCHES (le quart du réseau)
**Marche :** Option 1 / E2 (`PROTOCOLE_EXTENSION_E2.md`, ordre du chef). **Segment : couches 24–31.**
**Harnais au masque causal corrigé** (hérité d'E1 et de la revalidation). 100 % CPU, local, non poussé.

## 📏 Les chiffres (même tâche, mêmes hyperparamètres, même graine)
| Mesure | E1 (4 couches) | **E2 (8 couches)** | Dense équivalent (ordre) |
|---|---|---|---|
| Contrôle d'identité (segment avant / modèle plein) | 6,153 / 6,170 (écart 0,017) | 6,149 / 6,170 (**écart 0,021** ✅) | — |
| **Perte d'éval. après entraînement** | 2,313 (−62,4 %) | **2,122 (−65,5 %)** 🏆 | — |
| Paramètres mis à jour / pas | 1 440 000 | 2 874 240 | ~59,0 M MLP (8 couches) → **÷20** |
| Secondes / pas | 0,0653 | 0,0975 | — |
| RAM pic du processus | 1184 Mo | **1547 Mo** ⚠️ | — |
| Paramètres de tranches stockés (6 familles) | — | 16 810 560 | — |

## ⚖️ Verdict des critères figés
- **E2-a** (la perte baisse nettement) : ✅ 6,149 → 2,122.
- **E2-b** (même ordre que E1/D3 revalidé, plancher 30 %) : ✅ **−65,5 %**, la meilleure
  amélioration relative de la série — plus on opère de couches, plus la spécialisation
  par famille a de profondeur pour s'exprimer, sans signe d'effondrement à ce stade.
- **E2-c** (très inférieur au dense équivalent) : ✅ ÷20 par pas, stable depuis 2 couches.

## 🧱 Le mur qui approche (découverte d'ingénierie, à ne pas cacher)
La RAM grimpe presque linéairement avec la profondeur opérée (969 → 1184 → **1547 Mo**) :
notre harnais garde le segment entier en FP32, **plus** la double résidence déjà signalée
(ancien MLP + buffers). À ce rythme, **16 couches dépasseraient la RAM de cette sandbox
(~2 Go)** avant même de parler du modèle entier. Conséquence méthodique, écrite avant
la prochaine marche : l'extension suivante exigera d'abord **soit** le segment gelé en BF16,
**soit** le nettoyage mémoire (Option 3, aujourd'hui au frigo) — sinon l'instrument casse
avant la science. On ne forcera pas. 🫡

## 🔍 Lecture simple
Le quart du réseau pense déjà « en chemins entraînés » : la perte finale descend à mesure
que la chirurgie s'étend (2 couches : 2,47 · 4 : 2,31 · 8 : 2,12 — harnais causal pour
les trois derniers chiffres de la série corrigée), pendant que le coût par pas reste
**vingt fois** sous le dense. La limite actuelle n'est plus l'apprentissage : c'est la
**mémoire de notre propre banc d'essai**. 🛤️

## 📦 Fichiers de la marche
`src/extension_tranches.py` (généralisé `--couches/--marche`) · `results/RESULTATS_EXTENSION_E2.json` ·
ce rapport · `MANIFESTE.json` re-scellé. Segment local (244 Mo) et modèle évacués. Commit local, rien de poussé.

## 🚦 Suites proposées (une par Go, chef)
1. 🧹 **Débloquer la mémoire** : segment gelé BF16 ou libération de l'ancien MLP (mini-marche
   d'ingénierie, mesurée) — le passeport pour les 16 couches.
2. 🪜 Puis **E3 : 16 couches**, et **E4 : modèle entier (32)** si le passeport tient.
3. 📤 Pousser la série corrigée (E1, revalidation, E2) sur GitHub.

*Micro-rapport scellé — RATISS Labs, Yaoundé. Étape par étape, mesuré, reproductible. 😜🥳*
