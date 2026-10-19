# ⚖️ PROTOCOLE FIGÉ — GÉNÉRATION LIBRE DU MODÈLE ENTIER OPÉRÉ (test de feu E4)
**Ordre du chef :** priorité 1 du GO combiné, avant le push. **Question :** le modèle entier
(32 couches) entraîné uniquement par tranches (E4) génère-t-il de façon stable, **zéro
fallback dense** (embeddings → 32 couches opérées → tête, aucun MLP dense calculé) ?

## 🔧 Dispositif
- Même entraînement E4 (moteur M, mêmes hyperparamètres/graine) rejoué dans le processus,
  perte avant/après affichée pour contrôle (attendue ≈ 1,245).
- Génération **gloutonne, 18 tokens**, famille GPS routée une fois par prompt (statique).
- Prompts : les 5 de D4 (comparabilité directe), dont les 3 demandés par le chef
  (« La capitale de la France est », « Le ciel est », « Two plus two equals »).
- Référence honnête : mêmes prompts, même décodage, **modèle intact** en face.
- Aucun cache KV : séquence recalculée à chaque token (correct, juste plus lent).

## ⚖️ Critères de lecture (figés, qualitatifs et publiés bruts)
- ✅ **Stable** : texte lisible, cohérence locale, pas de boucle détectée.
- ⚠️ **Hésitant** : lisible mais robotique/partiellement faux — noté tel quel.
- ❌ **Échec** : boucle (détecteur de D4) ou charabia complet — publié tel quel.
- Le succès de la marche = la mécanique d'inférence 32 couches **fonctionne** ;
  la perfection littéraire n'est PAS le critère (ordre du chef).
