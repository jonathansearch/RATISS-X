# 🏁 RAPPORT — SPRINT FINAL SCIENTIFIQUE RATISS X

**Ordre du chef : « tout les reste de pb scientifique fais ça en 1 même si ça te fait 30min fais tout ».**
Protocole figé avant mesures : `docs/PROTOCOLE_SPRINT_FINAL_SCIENTIFIQUE.md`.
Cobaye : SmolLM2-360M, CPU uniquement, génération gloutonne de 18 tokens, routage
oracle par famille attendue pour les tests de génération. Textes publiés bruts,
sans embellissement.

## 🌟 Résultat central : l'élite par activation change le verdict à 75 %

À fond égal de 75 %, avec les mêmes tranches E4 non appariées, remplacer le
classement des neurones inactifs par magnitude des poids par leur **contribution
observée** `E|activation_i| × ||W_down[:, i]||` donne **5/6 générations stables**,
contre un effondrement du régime magnitude. C'est le principal résultat du sprint.

| Amorce | Magnitude 75 % | Activation 75 % |
|---|---|---|
| The capital of Spain is | ` Madridquerque Madrid Madrid Spain,,,,...` | ` Madrid.",` ✅ |
| Two plus two equals | ` 66` | ` four. 4 minus 3 equals 1` ✅ |
| Water is made of hydrogen and | ` being in into the air." It20...` | ` oxygen."` ✅ |
| The French word for cat is | ` Cat » Cat Cat Cat...` | ` Maison.",` ❌ faux |
| The animal that meows is | ` business is a customer buys...` | ` a cat` ✅ |
| To bake a cake, mix flour and | ` in eggs, and ' A?'.2333444663"` | ` sugar in a bowl, add eggs and butter in a pan, heat butter in a pan` ✅ |

**Verdict : 5/6 stable pour l'activation, contre 0/6 stable pour la magnitude**
sur cette grille stricte (une réponse courte arrêtée par EOS compte comme stable
si elle est juste et propre). Limite nette : routage encore oracle, six amorces
anglaises seulement ; ce n'est pas encore une validation générale.

## 1. 🎢 Seuil de génération en magnitude : aucun fond partiel ne passe

| Fond magnitude | Générations stables (lecture stricte) | Verdict |
|---:|---:|---|
| 70 % | 0/6 | déraillements publiés en G1 |
| 75 % | 0/6 | boucles et charabia |
| 80 % | 1/6 | seule `Madrid` tient |
| 85 % | 0/6 | répétitions et réponses fausses |
| 90 % | 1/6 | seule `Madrid` tient nettement |
| 95 % | 2/6 | `and H2O."` et `a cat` tiennent |
| 100 % | 6/6 | référence propre de G1 |

Le critère pré-déclaré demandait au moins 4/6. **Aucun fond magnitude entre 70 et
95 % ne borne donc un seuil de génération.** Les débuts souvent justes continuent
de se composer en erreurs sur 18 tokens.

## 2. 🎯 Entraînement apparié à 75 % : amélioration réelle, échec au critère

Les tranches des 32 couches ont été réentraînées pendant que la branche inactive
était déjà restreinte au fond magnitude 75 % utilisé à l'inférence.

- Perte d'évaluation dans le régime apparié : **8,8704 → 3,8156 (−57,0 %)**, 144 pas.
- Référence E4 entraîné à fond complet : **1,2449** (régime différent, comparaison indicative).
- Génération V1 à 75 % : **2/6 stables** (`Madrid`, `cat`), contre 4/6 exigées.

Textes bruts : `42` pour le calcul, répétitions pseudo-optiques pour la science,
`Cat Assistant` pour le langage, fragment de cuisine. **L'appariement aide mais ne
sauve pas la sélection magnitude.** Il ne doit pas être combiné rétroactivement à
l'activation dans ce sprint : cette combinaison devient une prochaine hypothèse,
à protocoliser séparément.

## 3. 🛰️ GPS des amorces : pas de routeur universel retenu

Nouveau corpus d'entraînement : 48 amorces, disjoint des 36 amorces figées.

| Centroïdes entraînés sur | Questions (60) | Amorces figées (36) |
|---|---:|---:|
| A. Questions seules | **93,3 %** | 30,6 % |
| B. Amorces seules | 20,0 % | **72,2 %** |
| C. Questions + amorces | 81,7 % | 47,2 % |

Le candidat C améliore les amorces mais perd 11,6 points sur les questions, au-delà
des 5 points autorisés. **Aucun routeur unique n'est retenu.** En revanche, le
résultat dessine proprement deux spécialistes : A pour les questions, B pour les
amorces. Choisir automatiquement entre ces deux modes reste une hypothèse future,
non mesurée ici.

## 4. 🪶 Moteur V2 int4 : cible mémoire atteinte, vitesse et qualité non validées

Prototype : attention partagée et MLP d'une seule famille en int4 par ligne,
deux valeurs par octet, déquantification à la volée, embedding lu par morceaux de
1024 lignes, sélection activation à 75 %.

- **RSS pic : 286 Mo** — cible historique <300 Mo atteinte ✅
- Vitesse : **3,63 à 4,01 s/token** — beaucoup plus lent que V1 ❌
- Générations brutes sur 3 amorces GEOGRAPHIE :
  - ` Madrid.",` ✅
  - ` Rome!", Rome!", Rome!"...` — bon fait, boucle de ponctuation ⚠️
  - ` ce lieu de la Nou de Rouba que la N...` ❌

**Verdict : passeport mémoire seulement.** L'int4 prouve qu'un moteur résident
peut descendre sous 300 Mo, mais cette implémentation n'est ni le moteur rapide
ni le moteur de qualité de RATISS.

## 🧭 État scientifique à la fin du sprint

| Question | Réponse mesurée |
|---|---|
| Un fond magnitude partiel suffit-il en génération ? | **Non**, de 70 à 95 % |
| Entraîner au régime magnitude 75 % suffit-il ? | **Non**, 2/6 malgré −57 % de perte |
| Quels neurones du fond garder ? | **Ceux qui contribuent**, pas seulement les gros poids : 5/6 à 75 % |
| Un seul centroïde route-t-il questions et amorces ? | **Non** ; deux spécialistes se dégagent |
| Peut-on passer sous 300 Mo ? | **Oui**, 286 Mo en int4, mais lent et qualité mixte |
| Le régime RATISS est-il choisi ? | **Non : décision souveraine du chef** |

## 📦 Artefacts

`src/entrainement_apparie.py` · `src/gps_amorces.py` · `src/moteur_v2.py` ·
`data/amorces_entrainement.json` · résultats JSON du sprint dans `results/` ·
classements et tranches lourdes uniquement en `artifacts/local/`, rincés après usage.

*Sprint scellé sans déplacer les critères — RATISS Labs, Yaoundé. Ici c'est la joie
et l'enthousiasme 😜🥳, et les échecs restent publiés à côté de la percée.*
