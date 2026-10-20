# ⚙️ MICRO-RAPPORT E5 — LE MOTEUR « ÉCONOMIQUE » 70 % N'EST PAS (ENCORE) ÉCONOMIQUE 💥
**Marche :** E5 (`PROTOCOLE_MOTEUR_E5.md`, ordre du chef après le push `d23c7df`).
Moteur natif comparé à lui-même : **hybride** (tranche + fond élite 70 %) contre
**dense** (tranche + fond 100 %), 3 prompts, processus séparés, générations brutes 📢
**Verdict : ❌ les trois critères tombent — et chacun enseigne quelque chose.** 🎓

## 📏 Le face-à-face (processus séparés, mêmes prompts, mêmes routages)
| Mesure | Hybride 70 % | **Dense 100 %** |
|---|---|---|
| RSS pic | 526 Mo | **387 Mo** |
| Temps / token | 2,01 – 2,21 s | **1,83 – 2,07 s** |
| « La capitale de la France est » (GEOGRAPHIE) | « le nom d' l' exasperat dans l'… » ⚠️ | **« le 17 septembre 1793… »** ✅ |
| « Le ciel est » (LANGAGE) | « atmospherique atmospherique atmospérée… » ❌ | « verte. ») ✅ (court) |
| « Two plus two equals » (ANIMAUX*) | « an library is note arr… problem: let arr… » ❌ | **« four. »** ✅ |

*routages GPS hérités de la génération E4, publiés tels quels.

## ⚖️ Critères du protocole, jugés sans fard 😇
- **Génération cohérente de l'hybride : ❌** 1 prompt sur 3 tient à peu près ; les deux
  autres déraillent. Le seuil cosinus mesuré en **un pas** sur **un prompt** (S5 : 0,9094)
  ne garantit pas la génération libre **multi-tokens** sur d'autres prompts : chaque
  token généré devient l'entrée du suivant, et l'écart se compose 🎢
- **Gain mémoire : ❌ inversé** — le classement par magnitude fait lire les matrices
  complètes et l'indexation dispersée coûte plus cher que le dense régulier dans ce
  moteur v0 (allocations transitoires retenues : 526 contre 387 Mo) 🧱
- **Gain vitesse : ❌ absent** — le coût du moteur est dominé par la **lecture et la
  conversion des poids** à chaque token (aucun cache), pas par le nombre de neurones
  calculés : retirer 30 % des neurones ne retire presque rien du temps 📏

## 🔍 Les trois lois que ce refus nous offre 🏆
1. **Un seuil mono-prompt n'est pas une recette générale** : le Seuil RATISS devra être
   re-mesuré en génération libre, multi-prompts, avant de dimensionner un moteur.
2. **Sans cache de poids, la sparsité ne paie pas** : un vrai moteur économique devra
   garder en mémoire les seuls poids utiles (tranche + élite sélectionnée **une fois**),
   au lieu de relire le modèle entier à chaque token — c'est un problème d'architecture
   mémoire, pas de mathématiques 🧠
3. **Le dense du moteur reste fidèle** (générations = PyTorch) : l'instrument de mesure
   est sain, c'est l'hypothèse « 70 % suffit partout » qui est réfutée en génération 📐

## 🚦 Suite propre (à protocoliser sur Go)
Re-mesurer le seuil **en génération libre multi-prompts** (le cosinus un-pas reste un
outil de diagnostic, pas le critère final), puis un moteur v1 avec **poids utiles
résidents** (chargés une fois, mesurés en RSS) plutôt que relus à chaque token ⚙️

## 📦 Fichiers
`src/moteur_natif.py` (mode `--e5 hybride|dense`) · `results/RESULTATS_MOTEUR_E5_hybride.json`
· `results/RESULTATS_MOTEUR_E5_dense.json` · ce rapport · `MANIFESTE.json` re-scellé.
Lourds rincés 🧹. Commit local, aucun push dans cette marche 🫡

*Micro-rapport scellé — RATISS Labs, Yaoundé. Le moteur « économique » coûte plus cher
que le dense : on publie, on comprend, on reconstruit mieux. 😜🥳🔥*
