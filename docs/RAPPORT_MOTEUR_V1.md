# ⚙️ MICRO-RAPPORT MOTEUR V1 — LA SPARSITÉ RÉSIDENTE PAIE (VITESSE), LA STABILITÉ RESTE À CONQUÉRIR 🏁
**Marche :** Moteur V1 (`PROTOCOLE_MOTEUR_V1.md`, ordre du chef après le push `82785a0`).
Poids **résidents** : attention partagée BF16 chargée une fois + un seul jeu MLP compact
par famille servie (tranche entraînée + élite 70 %), aucune lecture/indexation en génération.
**Verdict : ✅ gains d'ingénierie réels — ❌ stabilité inchangée. Les deux publiés.** 😇

## 📏 Le face-à-face à code égal (processus séparés, 3 prompts)
| Mesure | Hybride (tranche + élite 70 %) | Dense V1 (tous neurones) |
|---|---|---|
| **Temps / token** | **0,94 – 0,96 s** ⚡ | 1,28 – 1,30 s |
| RSS après 1ʳᵉ famille | ~669 Mo | (pic ci-dessous) |
| **RSS pic (3 familles servies à tour de rôle)** | **848 Mo** | 1030 Mo |
| Construction d'un jeu de famille | 1,5 – 2,1 s | idem (même code) |
| Générations | identiques à E5 : 1 ⚠️, 2 ❌ | modèle d'origine* |

*Note d'honnêteté : en mode dense, V1 n'injecte pas les tranches entraînées (fond 100 %
= poids d'origine) ; ses générations sont celles du modèle intact (« four. » ✅ inclus).
La comparaison RAM/vitesse est à code égal ; la comparaison de texte, elle, oppose
l'hybride entraîné au modèle d'origine — les générations hybrides sont strictement
celles déjà publiées en E5, ni meilleures ni pires 📢

## ⚖️ Critères du protocole, jugés
- **Plus léger et/ou plus rapide à code égal : ✅ les deux.** −18 % de RSS pic et
  **1,36× plus rapide par token** : dès que les poids utiles sont résidents et que
  la génération ne relit ni ne découpe plus rien, retirer 30 % des neurones retire
  bien ~30 % du travail MLP — la leçon d'E5 est corrigée par l'architecture mémoire 🧠
- **Stabilité des générations : ❌ inchangée** (mêmes sorties qu'E5, token pour token) :
  V1 change *où vivent les poids*, pas *les mathématiques* du seuil. Le chantier
  « seuil en génération libre multi-prompts » reste le verrou scientifique 🎢
- Le pic RSS hybride (848 Mo) est gonflé par les **changements de famille successifs**
  (l'allocateur retient les jeux précédents) : au repos mono-famille, le moteur tient
  en ~669 Mo — la marge vers la cible <300 Mo du moteur natif final reste un vrai
  chantier (BF16 → int8 ? élite plus petite ?), chiffré ici sans promesse 📏

## 🔍 Lecture
E5 avait réfuté « la sparsité par masquage à la volée » ; V1 valide « la sparsité par
**résidence préparée** ». La recette d'un moteur RATISS économique se précise :
sélectionner une fois (tranche + élite), compacter, résider, puis ne plus jamais
toucher au reste du modèle pendant l'inférence ⚙️ Reste à rendre l'hybride *stable*
en génération libre — ce n'est plus un problème de moteur, c'est le problème du seuil
multi-prompts, déjà nommé au journal 🎯

## 🪦 Incident publié
La première construction gardait les jeux des 3 familles + attention dupliquée :
1601 Mo au repos 💥 Refonte immédiate (attention partagée, un seul jeu résident),
c'est la version mesurée ci-dessus. La course d'écriture qui a fait tourner
l'ancienne version au premier essai est aussi consignée au journal de bord 😅

## 📦 Fichiers
`src/moteur_v1.py` · `results/RESULTATS_MOTEUR_V1_hybride.json` ·
`results/RESULTATS_MOTEUR_V1_dense.json` · ce rapport · `MANIFESTE.json` re-scellé.
Lourds rincés 🧹. Commit local, aucun push dans cette marche 🫡

*Micro-rapport scellé — RATISS Labs, Yaoundé. La sparsité ne paie que préparée :
on charge l'accord, on joue, on ne fouille plus dans le piano pendant le concert. 😜🥳🔥*
