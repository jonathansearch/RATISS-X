# 🧪 MICRO-RAPPORT M3 — ADAM BF16 VALIDÉ, MMAP ÉCARTÉ, PASSEPORT TRANSFÉRÉ AU MOTEUR NATIF
**Marche :** M3 (`PROTOCOLE_M3.md`, ordre « Fais tout »). 8 couches, calcul FP32, deux pistes testées ensemble.

## 📏 Résultats (référence M2 : 2,1216 · 0,104 s · pic 1250 Mo · boucle 1160 Mo · états 134,5 Mo)
| Mesure | M3 mesuré | Lecture |
|---|---|---|
| Perte finale | **2,1216** (−65,5 %) | ✅ **identique à Adam FP32 au millième** |
| États d'optimiseur | **67,2 Mo** | ✅ **divisés par 2**, comme calculé |
| Temps par pas | 0,106 s | ✅ sans explosion |
| RAM pic / boucle | 1348 / 1348 Mo | ❌ **pire que M2** (pages mmap résidentes comptées en RSS) |

## ⚖️ Verdicts par piste
- **Piste 2 (Adam à états BF16) : ✅ adoptée.** Même mathématique Adam (bêtas/eps/lr inchangés),
  états quantifiés : la perte finale ne bouge pas d'un millième sur 144 pas. C'est un
  changement de protocole d'ingénierie validé par la mesure, désormais disponible via
  `--optimiseur adam_bf16`.
- **Piste 1 (mmap/`assign`) : ❌ écartée dans cette sandbox.** Le chargement sans copie fait
  résider les pages du fichier mappé dans le RSS pendant que les copies FP32 du calcul
  existent aussi : le pic monte au lieu de descendre. Publié, pas maquillé.
- **Piste 3 (passeport moteur natif) : ✅ déclarée** dans `DECLARATION_PASSEPORT_MOTEUR_NATIF.md` :
  la barre mémoire (< 900 Mo, et à terme < 300 Mo) devient un critère du futur moteur sans
  interpréteur Python ; le harnais actuel continue de valider la science (pertes, routage, ÷20)
  avec sa RAM publiée telle quelle à chaque marche.

## 📦 Fichiers
`src/extension_tranches.py` (Adam BF16, chargement mmap testé mais non retenu pour la RAM) ·
`results/RESULTATS_EXTENSION_M3.json` · ce rapport · déclaration séparée. Commit local à ce stade.

*Micro-rapport scellé — RATISS Labs, Yaoundé. Une piste adoptée, une écartée, les deux publiées. 😜🥳*
