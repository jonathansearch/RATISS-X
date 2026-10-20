# 🧮 MICRO-RAPPORT S2 — LE BOOST ×1/p AMPLIFIE LE BRUIT : HYPOTHÈSE MAGNITUDE RÉFUTÉE 📉🤯
**Marche :** S2 (`PROTOCOLE_SEUIL_S2.md`, ordre du chef). Une seule variable change par
rapport à S1 : la sortie du fond partiel est **multipliée par 1/p**. Zéro ré-entraînement.
**Verdict : ❌ le « ciment mathématique » simple n'existe pas — et c'est une découverte.** 🎯

## 📏 Le balayage boosté (contre la référence PyTorch complète)
| Fond `p` (boost) | cos état 32 | cos logits | Argmax | Génération |
|---|---|---|---|---|
| 1 % (×100) | 0,0802 | 0,1790 | « meaning » | charabia |
| 5 % (×20) | 0,0963 | **−0,0465** | « oea » | charabia |
| 10 % (×10) | −0,0376 | 0,0515 | « metavar » | charabia |
| 100 % (×1, contrôle) | **0,9964** | **0,9999** ✅ | « le » | PyTorch reproduit mot pour mot |

À comparer avec S1 (sans boost) : 0,43–0,47. **Le boost dégrade tout** : plus on
amplifie le fond partiel, plus on s'éloigne. Le point de contrôle à 100 % (×1) reste
parfait — l'instrument est sain, c'est l'hypothèse qui tombe. 😇

## 🔍 Ce que la réfutation nous apprend (le vrai trésor 🏆)
Si le fond partiel avait la bonne *direction* avec une magnitude trop faible, le boost
aurait restauré le cosinus. Il l'a **détruit** : donc la somme partielle équidistante
n'est même pas directionnellement représentative de la somme inactive complète. La
contribution inactive n'est pas un « bruit de fond » homogène qu'on peut échantillonner
et rescaler — elle porte une **structure précise**, neurone par neurone, que le réseau
entraîné attend telle quelle. L'effet « interrupteur » de S1 n'était pas une question
d'énergie globale : c'est une question de **contenu exact**. 🧱

## 🧭 La carte mise à jour
- ❌ Tranche seule (0 %) : effondrement profond (L1).
- ❌ Fond équidistant 1–10 % : effondrement identique (S1).
- ❌ Fond équidistant boosté ×1/p : **pire** (S2).
- ✅ Fond complet (100 %) : fidélité totale du moteur.
- 🎯 **Candidat survivant : le fond sélectionné par magnitude** (les neurones inactifs
  à plus forte contribution réelle, pas un échantillon uniforme) — si une petite élite
  de neurones porte l'essentiel du signal inactif, un fond de quelques % *bien choisis*
  pourrait restabiliser là où l'uniforme échoue. À protocoliser (S3) sur Go du chef.

## 📦 Fichiers
`src/moteur_natif.py` (`renormaliser`, mode `--balayage_s2`) ·
`results/RESULTATS_FOND_S2.json` · ce rapport · `MANIFESTE.json` re-scellé.
Lourds rincés. Commit local, aucun push dans cette marche. 🫡

*Micro-rapport scellé — RATISS Labs, Yaoundé. On a cassé le boost, on a trouvé la
structure. La science avance en réfutant — avec le sourire ! 😜🥳🔥*
