# ⚖️ PROTOCOLE FIGÉ — MARCHE M3 : MMAP/ASSIGN + ADAM À ÉTATS BF16 (ordre « Fais tout »)
**Contexte :** M1 (BF16) et M2 (FP32 + audit) ont refusé le passeport 900 Mo et pesé chaque
coupable. Le chef ordonne les trois pistes : (1) chargement sans duplication, (2) optimiseur
allégé — **changement de protocole assumé et autorisé** : même mise à jour Adam (bêtas, eps,
lr 5e-4 inchangés), mais états (moyenne/variance) stockés en **BF16** ; (3) passeport moteur natif.

## 🎯 Ce qui est testé ici (8 couches, FP32 pour le calcul, comme M2)
- Chargement du segment par `torch.load(mmap=True)` + construction des couches sur
  dispositif `meta` avec `load_state_dict(assign=True)` : aucune copie du socle à la construction.
- Optimiseur `AdamEtatsLegers` : états en BF16 (134,5 Mo → ~67 Mo attendus), audit intégré.
- Le reste est identique à M2/E2 (tâche, graine, époques, familles, nettoyage, libération MLP).

## ⚖️ Critères de victoire M3 (figés)
- **Perte finale ≈ 2,12 (±0,05)** : la quantification des états ne doit pas casser Adam.
- **RAM pic et RAM de boucle en baisse nette** par rapport à M2 (1250 / 1160 Mo).
- **Temps par pas sans explosion** (référence M2 : 0,104 s).
- La barre historique des 900 Mo est jugée en connaissance de cause : si elle tient encore,
  le **passeport moteur natif** (document séparé) prend le relais, comme ordonné.

## 🚦 Enchaînement ordonné par le chef (« Fais tout »)
M3 (8 couches) → déclaration passeport moteur natif → **E3 : 16 couches** avec le meilleur
moteur mesuré → **push GitHub** de toute la série corrigée (D3 revalidé, E1, E2, M1, M2, M3, E3).
