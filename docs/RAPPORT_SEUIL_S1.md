# 🧱 MICRO-RAPPORT S1 — LE SEUIL DE STABILISATION N'EST PAS UN POURCENTAGE DOUX
**Marche :** S1 (`PROTOCOLE_SEUIL_S1.md`, ordonnée « M3 » par le chef, archivée S1).
Aucun ré-entraînement : moteur natif + fraction `p` de neurones inactifs (sélection
équidistante déterministe, poids d'origine) ajoutée à la tranche entraînée.
Référence : PyTorch complet E4 (contrôle 1,2449, argmax « le »). Prompt figé, GEOGRAPHIE.

## 📏 Le balayage (état caché post-couche-32 et logits, contre la référence complète)
| Fond `p` | cos état 32 | cos logits | Argmax | Génération (18 tokens) |
|---|---|---|---|---|
| 0 % | 0,4458 | 0,3246 | « Gibraltar » | charabia répété (référence d'échec L1) |
| 1 % | 0,4581 | 0,4053 | « en » | charabia |
| 5 % | 0,4150 | 0,4152 | « es » | charabia |
| 10 % | 0,4684 | 0,4281 | « delays » | charabia |
| **100 %** | **0,9964** | **0,9999** | **« le »** ✅ | **« le 17 septembre 1793… »** = PyTorch, mot pour mot |

## ⚖️ Critère du chef, jugé
Plus petit `p` avec cosinus logits > 0,95 et génération cohérente : **aucun dans
{1 %, 5 %, 10 %}**. Le critère n'est satisfait qu'à **100 %**. Publié tel quel. 😇

## 🔍 Lecture — la frontière n'est pas là où on l'attendait
Le « fond » n'agit pas comme un ciment proportionnel au nombre de neurones : à sélection
équidistante, 10 % des neurones inactifs (~245/couche) ne restaurent **rien** (0,43),
puis le système bascule d'un coup à 100 % (0,9999). Deux lectures, non départagées :
1. **Question de magnitude** : la somme inactive partielle vaut ~`p` fois le vecteur
   attendu par le réseau entraîné — la direction est peut-être là, mais dix fois trop
   faible pour contrebalancer la dérive. Un fond **renormalisé (×1/p)** est le candidat
   suivant, à protocoliser séparément (ce n'était pas le protocole figé ici).
2. **Question de structure** : la stabilité pourrait dépendre de neurones inactifs
   précis (forte magnitude de sortie), pas d'un échantillon uniforme — un fond
   **sélectionné par magnitude** serait alors le bon candidat.
Le moteur, lui, confirme encore sa fidélité : à fond 100 %, il reproduit la génération
PyTorch à l'identique, cette fois sur 18 tokens. ✅

## 📦 Fichiers
`src/moteur_natif.py` (paramètre `fond`, mode `--balayage`) ·
`results/RESULTATS_FOND_S1.json` · ce rapport · `MANIFESTE.json` re-scellé.
Lourds rincés. Commit local — **pas de push avant validation du chef** (ordre S1).

*Micro-rapport scellé — RATISS Labs, Yaoundé. La frontière est nette : ce n'est pas
un robinet, c'est un interrupteur — reste à trouver le fusible exact. 😜🥳*
