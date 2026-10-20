# 🎢 MICRO-RAPPORT G1 — LE SEUIL DE GÉNÉRATION N'EST PAS LE SEUIL COSINUS 📉
**Marche :** G1 (`PROTOCOLE_SEUIL_GENERATION_G1.md`, ordre de clôture du chef, veille
de la construction RATISS). Moteur V1, 6 amorces figées (1/famille, famille attendue
servie), glouton 18 tokens, fonds {70 %, 75 %, 100 %}, textes **bruts** 📢

## 📢 Les phrases, sans filtre
| Amorce | 70 % | 75 % | **100 %** |
|---|---|---|---|
| The capital of Spain is | « Madrid Spain. » puis charabia ⚠️ | « Madrid Madrid Madrid,,,, » ❌ | **« Madrid. The capital of Spain is Madrid. »** ✅ |
| Two plus two equals | « 66666 » ❌ | « 66 » ❌ | **« four. This is a fundamental property… »** ✅ |
| Water is made of hydrogen and | « two using water… air air air » ❌ | « being in into the air… » ❌ | **« oxygen atoms. Hydrogen is the lightest… »** ✅ |
| The French word for cat is | **« Cat. »** ✅ puis stop | « Cat Cat Cat Cat… » ❌ boucle | **« "chat," which is also the name… »** ✅ |
| The animal that meows is | « is its convenient to… a cat » ❌ | « business is a customer… cat » ❌ | **« a cat. The animal that meows is a cat. »** ✅ |
| To bake a cake, mix flour and | « water, it bake, one,0… » ❌ | « in eggs, and ' A?'… » ⚠️ | **« sugar in a bowl. Add eggs and milk… »** ✅ |

## ⚖️ Critère de clôture, jugé
Plus petit fond stable sur la majorité des 6 amorces en génération libre :
**aucun des deux testés** (70 % : ~1,5/6 · 75 % : ~1/6). Seul **100 % fait 6/6**.
**Le Seuil de Génération RATISS est donc > 75 % sur ce banc** (non mesuré entre 75 et
100 — ce sera un réglage de la construction, pas une extrapolation) 🎯

## 🔍 La loi de la veille (à graver avant de construire 🏛️)
En **un pas**, le modèle à 70 % prédit presque comme le complet (cosinus 0,9094, S5).
En **génération libre**, chaque token produit devient l'entrée suivante : l'écart se
compose et la phrase casse en quelques tokens — souvent **après un début juste**
(« Madrid », « Cat »). Le critère qui dimensionne un moteur de génération n'est donc
ni le cosinus ni l'argmax initial : c'est la **stabilité de la trajectoire complète**.
Corollaire pour demain : nos tranches ont été **entraînées avec le fond complet** puis
testées avec un fond réduit — désaccord entraînement/inférence. La construction RATISS
devra choisir en connaissance de cause : fond complet pour la fiabilité immédiate, ou
**entraînement au même régime que l'inférence** (la voie propre, à protocoliser) 🧭

## 📦 Fichiers
`src/seuil_generation.py` · `results/RESULTATS_SEUIL_GENERATION_{70,75,100}.json` ·
ce rapport · dossier de veille `docs/PRET_POUR_RATISS.md` · `MANIFESTE.json` re-scellé.
Lourds rincés 🧹. Push de clôture ce soir (ordre du chef) 🌍

*Micro-rapport scellé — RATISS Labs, Yaoundé, veille de construction. On sait
exactement sur quel sol on bâtit demain. 😜🥳🔥*
