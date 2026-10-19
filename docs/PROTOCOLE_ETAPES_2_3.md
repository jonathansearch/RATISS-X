# ⚖️ PROTOCOLE FIGÉ — ÉTAPES 2 & 3 (CPU uniquement, ordre du chef)
**Date :** 2 novembre 2026 · **Repo local RATISS-X (NON poussé 🚫)** · Cobaye : SmolLM2-135M 🐜
**Contexte assumé :** le cobaye est petit, faible en français, peu de contexte. Ce n'est pas
une excuse : c'est la condition expérimentale. On verra si l'architecture RATISS l'améliore quand même.

## 💉 ÉTAPE 2 — Le Bistouri (backprop par raccourci, version CPU)
**Question :** en n'entraînant que les neurones du chemin actif $M_k$ (contre un entraînement
dense des mêmes couches), que gagne-t-on en paramètres mis à jour, RAM CPU et temps par pas —
et la perte (loss) s'améliore-t-elle quand même sur des questions jamais entraînées ?

**Dispositif (figé) :**
- Segment d'étude : les **2 dernières couches** du cobaye (28–29) + norme finale + tête de sortie,
  alimentées par les états cachés pré-calculés du modèle gelé. Les MLP sont la cible (les chemins y vivent).
- Tâche : 60 questions-réponses basiques **en anglais** (langue compétente du cobaye, sinon on mesure
  sa faiblesse en français, pas notre architecture) — 48 entraînement / 12 évaluation, tirage figé.
- **Dense** : tous les paramètres MLP des 2 couches sont entraînables (~5,3 M).
- **Raccourci RATISS** : MLP gelés + correction (delta) calculée **uniquement sur les neurones
  du masque de la famille** prédite par le GPS (~72/1536 par couche), initialisée à zéro
  (le modèle de départ est donc strictement identique).
- Même optimiseur (Adam), même taux (5e-4), même nombre de pas (3 époques, batch 1 groupé par famille).
- **Mesures CPU uniquement** : perte éval avant/après, paramètres réellement mis à jour par pas,
  RAM pic (RSS du processus), temps par pas. **Aucun chiffre GPU, aucun VRAM.** 🚫

**Critères (figés) :**
- E2-a : la perte d'évaluation du Raccourci doit **baisser** par rapport au modèle gelé.
- E2-b : elle doit rester dans le même ordre de grandeur que le Dense (pas d'effondrement).
- E2-c : les paramètres mis à jour par pas doivent chuter d'un facteur ≥ 5.

## 🍏 ÉTAPE 3 — Le Traducteur sémantique (contre le bug des paraphrases 🍎)
**Question :** en injectant un **vecteur de concept** avant le GPS (résonance : la question
active l'intention la plus proche dans une banque de formulations connues, qui diffuse vers
ses mots-concepts canoniques), le routage des paraphrases s'améliore-t-il ?

**Dispositif (figé) :**
- Banque d'intentions : les 18 paraphrases de l'Étape 1 (maintenant données d'appui, test clos).
- Vecteur enrichi : $v' = v + 0{,}8 \times$ embedding des mots-concepts de l'intention activée ($\alpha$ figé).
- **Nouveau test scellé** : 18 paraphrases inédites des 6 mêmes intentions, écrites dans
  `data/paraphrases_test_v2.json` AVANT l'exécution, jamais utilisées ailleurs.
- Comparaison : GPS seul contre GPS + Traducteur, même questions, même jour.

**Critère (figé) :** E3 : exactitude du GPS + Traducteur **> GPS seul** sur le test scellé.

## ⚖️ RÈGLE DE DÉCISION FINALE (ordre du chef)
- Si E2 et E3 réussissent → **on garde SmolLM2-135M** et on continue la Phase 2 avec lui. 🟢
- Si l'un des deux échoue franchement → deux options à trancher par le chef :
  **(a) fortifier** ce petit modèle (entraînement renforcé, plus de données) 💪,
  ou **(b) prendre un modèle un peu plus gros** (plus de paramètres/contexte) 🐘.
