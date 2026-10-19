# ⚖️ PROTOCOLE FIGÉ — MARCHE M2 : VITESSE FP32 + CHIRURGIE DE L'OPTIMISEUR
**Ordre du chef :** Piste 3 modifiée « RATISS ». **Protocole scientifique inchangé**
(Adam sur les tranches, même tâche, mêmes hyperparamètres). **Aucun push, pas d'E3 sans passeport.**
**Références :** E2 (FP32) perte 2,122 · 0,0975 s/pas · 1547 Mo ; D3 2 couches 0,0458 s/pas.

## 🎯 Objectif
1. Retour au **FP32** partout (le BF16/FP16 CPU est mesuré plus lent — M1).
2. **Isolation totale de l'optimiseur** : Adam ne reçoit QUE les paramètres des tranches
   (`W_gate/up/down` des accords de chaque couche opérée), jamais le socle gelé —
   audit chiffré à l'appui (nombre de paramètres et taille des états).
3. **Nettoyage pré-forward agressif** : tranches extraites, références intermédiaires
   supprimées, `gc.collect()` explicite, `no_grad` strict sur toute l'inactive.
4. RAM mesurée **pendant** la boucle d'entraînement (échantillonnage RSS + pic processus).

## ⚖️ Critères de victoire (donnés par le chef, figés)
- **RAM pic < 900 Mo** (visée ~700–800 Mo).
- **Temps par pas ≤ 0,060 s**.
- **Perte finale ≈ 2,12**.

## 🚫 Hors périmètre
Pas de mmap (Piste 1 en réserve), pas d'E3, pas de push, pas de changement d'optimiseur
ni d'hyperparamètres.
