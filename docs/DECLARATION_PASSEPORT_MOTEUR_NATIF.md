# 🏳️ DÉCLARATION — PASSEPORT MÉMOIRE TRANSFÉRÉ AU MOTEUR NATIF (Piste 3, ordre du chef)
**Date :** session du 19 octobre 2026. **Décideur :** Jonathan Evina. (chef).

## Ce qui a été prouvé (M1, M2, M3)
- Le downcasting BF16 du socle ne dégrade pas l'apprentissage (perte 2,117 contre 2,122 FP32).
- L'optimiseur ne voit que les tranches (16 810 560 params audités) ; ses états peuvent être
  divisés par 2 en BF16 sans aucun effet mesurable sur la perte (2,1216 identique).
- À protocole constant dans l'interpréteur Python/PyTorch de la sandbox, le plancher pour
  8 couches entraînées est ~1,16–1,25 Go (socle d'interpréteur ~219 Mo, allocateur qui
  retient les pics, tête de sortie 189 Mo en FP32, gradients et états d'optimiseur).
  La barre « < 900 Mo » y est donc structurellement hors d'atteinte — fait mesuré, publié.

## Déclaration
La cible mémoire de RATISS X (< 900 Mo, à terme < 300 Mo, « Agent Souverain 20 W »)
est un **critère du futur moteur natif** (chargement des tranches à la demande, pas de
calcul des neurones inactifs, aucun interpréteur Python résiduel) et **cesse d'être un
critère bloquant pour le harnais PyTorch**, dont le rôle est de valider la science :
pertes, contrôles d'identité, routage, facteur ÷20 de mises à jour, génération par tranches.
À partir de cette déclaration, chaque marche publie sa RAM telle quelle, sans passeport
binaire, et l'extension en profondeur (E3 : 16 couches, puis modèle entier) peut reprendre.

*Déclaration scellée — RATISS Labs, Yaoundé. On déplace la barre là où elle se joue vraiment :
dans le moteur, pas dans le thermomètre. 😜🥳*
