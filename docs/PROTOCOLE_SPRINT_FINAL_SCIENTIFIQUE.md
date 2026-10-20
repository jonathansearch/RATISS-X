# ⚖️ PROTOCOLE FIGÉ — SPRINT FINAL SCIENTIFIQUE (ordre du chef : « fais tout »)

Objectif : fermer en une seule campagne les problèmes scientifiques encore ouverts
avant la construction RATISS. Aucun critère n'est déplacé après coup.

## 1. Seuil de génération entre 75 % et 100 %
Moteur V1, 6 amorces G1 en routage oracle, génération gloutonne 18 tokens.
Fonds supplémentaires : 80 %, 85 %, 90 %, 95 %. Une génération est comptée stable
si elle reste lisible et dans le sujet sans boucle sur l'essentiel des 18 tokens.
Le premier fond stable sur au moins 4/6 amorces borne le seuil de génération.

## 2. Entraînement apparié au régime d'inférence
Les tranches sont réentraînées sur les 32 couches pendant que la branche inactive
est déjà restreinte au même fond élite magnitude de 75 % que l'inférence.
Même tâche, graine, Adam léger, LR et époques qu'E4. Les tranches appariées sont
exportées puis testées par le moteur V1 à fond 75 % sur les 6 amorces G1.
Succès : au moins 4/6 générations stables, contre le régime non apparié.

## 3. Élite par activation contre élite par magnitude
Pour chaque couche/famille, les neurones inactifs sont aussi classés par
contribution observée sur les exemples d'entraînement :
`E|activation_i| × ||W_down[:, i]||`. À fond 75 % et tranches E4 non appariées,
la génération des 6 amorces est comparée à la sélection magnitude.

## 4. GPS des amorces
Un corpus d'entraînement d'amorces nouveau et disjoint du test figé est créé avant
mesure. Trois routeurs centroïdes sont comparés : questions seules, amorces seules,
questions + amorces. Évaluations : 36 amorces figées et 60 questions du questionnaire
non-paraphrases. Le routeur retenu doit améliorer les amorces sans effondrer les questions.

## 5. Descente mémoire du moteur
Prototype V2 : poids de calcul résidents quantifiés en int4 par ligne, un seul
accord résident, table d'embedding lue par morceaux. Mesures sur 3 amorces de la
même famille : RSS pic, temps/token, générations. Cible historique : < 300 Mo.

## 6. Synthèse de construction
Le dossier de veille est mis à jour avec les régimes réellement validés, leurs coûts
et les critères proposés pour écrire la naissance de RATISS VEGA. Aucun nom ni
critère de VEGA n'est inventé à la place du chef.

*Sprint figé avant exécution — RATISS Labs, Yaoundé.*
