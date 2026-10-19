# 🧭 MICRO-RAPPORT F2 — ROUTAGE SUR AMORCES : LE CANDIDAT EST REJETÉ, LE PROBLÈME RESTE OUVERT
**Protocole :** `PROTOCOLE_DUEL_ET_ROUTAGE_ET_MOTEUR.md`. Candidat : embedding « contenu
seul » (tokens hors mots-outils figés), mêmes centroïdes GPS, zéro ré-entraînement.
Jeux figés avant mesure : `data/amorces_test.json` (36 amorces) + questions de contrôle.

## 📏 Exactitudes du routeur (centroïdes inchangés)
| Jeu d'éval | GPS standard | **Contenu seul** | Verdict |
|---|---|---|---|
| Amorces libres (n=36) | 30,6 % | 36,1 % | +5,5 pts, mais faible en absolu |
| Questions questionnaire (n=78) | **71,8 %** | 52,6 % | ❌ **−19,2 pts : dégradation nette** |
| Questions d'entraînement QA (n=60) | 30,0 % | 53,3 % | +23,3 pts (style différent des centroïdes) |

## ⚖️ Verdict selon le critère figé
Le critère exigeait : amorces améliorées **sans dégrader** les questions. Il n'est pas
rempli → **le candidat « contenu seul » n'est pas adopté**. Le labo publie le rejet. 😇
L'enseignement, lui, est net : retirer les mots-outils aide les textes courts de style
télégraphique (amorces, QA d'entraînement) mais **casse la signature des vraies questions**
dont les centroïdes ont été construits — le routage dépend du style autant que du sujet.

## 🔍 Le vrai problème, mieux cerné
Même le meilleur des deux reste à **36 % sur amorces libres** : le routeur actuel, pensé
pour des questions, ne sait pas router des débuts de phrase. La piste propre n'est pas
un filtre, c'est un **corpus d'entraînement d'amorces** (distinct du jeu de test figé)
pour construire des centroïdes — ou un double routage question/amorce — à protocoliser
séparément. En génération, la robustesse observée en E4 (réponses justes malgré 3/5 de
mauvais routages à 32 couches) rend ce chantier moins critique qu'à 2 couches, mais il
reste le maillon faible nommé du système. 🧭

## 📦 Fichiers
`src/routage_amorces.py` · `data/amorces_test.json` (scellé) ·
`results/RESULTATS_ROUTAGE_AMORCES.json` (détails par amorce) · ce rapport. Commit local.

*Micro-rapport scellé — RATISS Labs, Yaoundé. Un candidat rejeté proprement vaut une piste comprise. 😜🥳*
