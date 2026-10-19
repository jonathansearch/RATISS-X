# ⚖️ PROTOCOLES FIGÉS — TRIPLE ORDRE DU CHEF (F1 duel dense · F2 routage · F3 moteur natif)

## F1 ⚖️ Duel dense (même tâche, mêmes hyperparamètres, masque causal)
Le dense complet du modèle entier (235,9 M de paramètres MLP FP32 + gradients + états)
exige >3 Go : **impossible dans cette sandbox — ce sera publié comme un résultat en soi**
(la tranche, elle, entraîne le modèle entier ici en 1580 Mo). Le duel mesuré se fait donc
à **8 couches** ( dense MLP FP32 entraînable, attention gelée BF16, Adam FP32 classique,
144 pas, graine 20261102), face à E2 tranche (2,122 · 2,87 M maj/pas · 0,098 s/pas FP32).
Critères : perte finale, secondes/pas, maj/pas, RAM — publiés côte à côte, sans extrapoler
au-delà de ce qui est mesuré (le facteur d'échelle 8→32 couches sera dit, pas maquillé).

## F2 🧭 Routage GPS sur amorces libres
Constat (génération E4) : le GPS, entraîné sur des **questions**, route mal les **amorces
libres** (3/5 mal routées). Correctif candidat figé : **embedding « contenu seul »** —
moyenne des embeddings des tokens hors mots-outils (liste figée FR/EN) et hors ponctuation,
mêmes centroïdes GPS, aucun ré-entraînement. Jeu d'éval figé AVANT mesure :
`data/amorces_test.json` (36 amorces, 6 par famille, FR/EN mélangées, jamais utilisées
pour régler quoi que ce soit) + contrôle sur les questions du questionnaire d'origine.
Critère : l'exactitude sur amorces s'améliore **sans dégrader** celle sur questions.

## F3 🏗️ Moteur natif v0 (prototype, critère < 300 Mo de RSS)
Moteur d'inférence **sans PyTorch** (Python + NumPy seuls) : poids gelés lus en `memmap`
dans le safetensors d'origine (BF16 décodé à la volée), tranches **entraînées en E4**
exportées dans un fichier binaire compact, MLP en **tranche seule** (les neurones
inactifs ne sont ni chargés ni calculés — le vrai geste RATISS), génération gloutonne
sur 3 prompts. Critères figés : **RSS du processus < 300 Mo**, génération stable
(pas de boucle), temps/token publié. Portée honnête : prototype de **faisabilité du
moteur**, pas le moteur final (ni cache KV, ni noyau C).
