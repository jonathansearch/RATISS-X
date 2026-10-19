# 🏃 MICRO-RAPPORT M2 — LA VITESSE FP32 EST LÀ, L'OPTIMISEUR EST INNOCENTÉ, LA RAM RÉSISTE
**Marche :** M2 (`PROTOCOLE_M2.md`, ordre du chef, Piste 3 modifiée « RATISS »).
**Verdict : ❌ passeport toujours refusé — mais chaque coupable est désormais pesé au gramme.** 🫡

## 📏 Critères du chef, jugés sans retouche
| Critère figé | Cible | Mesuré (FP32, 8 couches) | Verdict |
|---|---|---|---|
| RAM pic | < 900 Mo | **1250 Mo** (boucle : max 1160 Mo) | ❌ |
| Temps par pas | ≤ 0,060 s | **0,104 s** | ❌ |
| Perte finale | ≈ 2,12 | **2,1216** (−65,5 %, identité 6,149/6,170) | ✅ identique à E2 |

Gain réel par rapport à E2 (même mode FP32) : **−297 Mo** grâce à la libération de
l'ancienne résidence MLP portée en FP32 + conteneur d'état libéré. La science est intacte
au millième près. Mais la barre des 900 Mo tient toujours.

## 🔐 L'audit de l'optimiseur — hypothèse du chef testée et tranchée
L'audit intégré au harnais le prouve : **Adam reçoit exactement 16 810 560 paramètres,
soit les tranches seules**, et ses états pèsent **134,5 Mo** — exactement 2 états FP32
(moyenne, variance) pour ces seuls paramètres. Le socle gelé n'a **jamais** été transmis
à l'optimiseur (vérifié dans le code et compté à l'exécution) : il n'y a donc **aucun état
fantôme à tuer**. Ces 134 Mo sont le prix incompressible d'Adam sur les tranches stockées,
sauf à changer d'optimiseur — ce que le protocole interdit, à juste titre.

## 🧱 Le plancher structurel, pesé pièce par pièce
Le segment FP32 vivant contient à lui seul : buffers MLP gelés ~236 Mo · tête de sortie
~189 Mo · attention/normes ~79 Mo · tranches ~67 Mo = **~571 Mo de tenseurs**, auxquels
s'ajoutent l'interpréteur + PyTorch (~219 Mo), Adam (134 Mo), les gradients (~67 Mo) et
les activations. La boucle seule culmine à **1160 Mo** : même une construction parfaite
(Piste 1, mmap/`assign`, gain espéré ~90–250 Mo sur le pic) ne ferait pas passer la barre
des 900 Mo **à protocole constant** dans cet interpréteur. C'est désormais un fait mesuré,
plus une intuition.

## ⏱️ Pourquoi 0,060 s était hors de portée ici
Le forward de la branche **inactive reste dense** (calculée entièrement, puis masquée) :
le coût croît avec la profondeur (2 couches : 0,046 s → 8 couches : 0,104 s). Le vrai
accélérateur RATISS — **ne pas calculer du tout les neurones inactifs** (indexation des
seules lignes/colonnes de l'accord) — relève du moteur natif, pas de ce harnais PyTorch.
Publication honnête : le temps par pas actuel valide la faisabilité, pas la vitesse finale.

## ⚖️ Conclusion et suite proposée
À protocole scientifique constant et dans cette sandbox PyTorch, le passeport « < 900 Mo »
pour 8 couches est **structurellement hors d'atteinte** : M1 l'a montré par le type,
M2 par l'optimiseur innocenté et le plancher pesé. Les options restantes, au choix du chef :
1. **Piste 1 (mmap/`assign`)** : rogner le pic de construction — honnêtement insuffisant seul ici.
2. **Protocole d'optimiseur allégé** (états quantifiés ou autre) : vraie piste RAM, mais
   **changement de protocole scientifique** — décision explicite requise.
3. **Déclarer le passeport « moteur natif »** : la barre des 900 Mo (et celle des 300 Mo du
   manifeste RATISS X) devient un critère du futur moteur sans interpréteur, pendant que
   le harnais PyTorch continue de valider la **science** (pertes, routage, ÷20) à chaque marche.
   Dans ce cas, E3 (16 couches) reste faisable ici (pic estimé ~1,6 Go, proche de la limite
   sandbox ~2 Go) avec la RAM publiée telle quelle.

## 📦 Fichiers de la marche
Chirurgie dans `src/extension_tranches.py` (libération MLP tous modes, audit Adam, RSS en
boucle) · `results/RESULTATS_EXTENSION_M2.json` · ce rapport · `MANIFESTE.json` re-scellé.
Lourds évacués. Commit local, **rien de poussé, pas d'E3**.

*Micro-rapport scellé — RATISS Labs, Yaoundé. Innocenter un suspect avec des chiffres,
c'est aussi faire avancer l'enquête. Étape par étape. 😜🥳*
