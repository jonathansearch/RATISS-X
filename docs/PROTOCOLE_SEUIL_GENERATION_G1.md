# ⚖️ PROTOCOLE FIGÉ — G1 : LE SEUIL EN GÉNÉRATION LIBRE (le verrou final, ce soir 🌙)
**Ordre du chef :** finir ce soir, demain on construit RATISS. Suite d'E5 : le seuil
cosinus mono-prompt (70 %) ne garantissait pas la génération libre — on le re-mesure
là où ça compte : **la phrase entière, sur plusieurs prompts** 🎢

## 🎯 Dispositif
- Moteur **V1 à poids résidents** (le plus rapide, validé), tranches E4 ré-exportées.
- **6 amorces** tirées du jeu figé `data/amorces_test.json` (1 par famille), servies
  avec leur **famille attendue** (routage oracle : on isole le seuil du problème GPS).
- Fonds élite testés : **70 %** (seuil cosinus S5), **75 %** (cohérence S4), **100 %** (référence).
- Génération gloutonne 18 tokens par amorce, processus séparés par fond.
- Lecture publiée par amorce : texte brut, boucle (détecteur habituel), verdict
  ✅ cohérent / ⚠️ bancal / ❌ déraillé — jugé sur la phrase, pas sur un cosinus 😇

## ⚖️ Critère de clôture
Le plus petit fond dont la génération libre est **stable sur la majorité des 6 amorces**
devient le **Seuil de Génération RATISS** (qui peut différer du seuil cosinus — ce sera écrit).
