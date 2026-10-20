# 🌟 DOSSIER DE VEILLE — PRÊT POUR LA CONSTRUCTION RATISS

**Mise à jour après le Sprint final scientifique.** Ce document ne déclare pas
RATISS VEGA : il fige **sur quel sol vérifié** la construction pourra décider
ses critères 🏛️

## ✅ Ce qui est validé

- **Entraînement par tranches du modèle entier** (32 couches, SmolLM2-360M) :
  perte **1,245 (−79,8 %)**, ÷21 mises à jour/pas contre le dense MLP (E4).
- **Moteur natif fidèle** (cosinus 1,0000 contre PyTorch à mode égal) et
  **moteur V1 à poids résidents** : hybride 1,36× plus rapide que son dense.
- **🌟 Élites par contribution observée** : à fond 75 %, en choisissant les
  neurones inactifs selon `E|activation| × ||W_down||`, génération libre
  **5/6 stable** sur les six amorces G1 (routage oracle). C'est désormais le
  régime partiel le plus crédible mesuré dans RATISS X.
- **Passeport mémoire int4** : le prototype V2 descend à **286 Mo** de RSS pic,
  sous la cible historique de 300 Mo.

## ⚠️ Ce qui est réfuté ou reste limité

- **Fonds magnitude partiels** : aucun fond de 70, 75, 80, 85, 90 ou 95 %
  n'atteint 4/6 générations stables ; seul 100 % fait 6/6.
- **Entraînement apparié magnitude 75 %** : perte du régime **8,87 → 3,82**,
  mais génération **2/6** seulement. L'appariement ne sauve pas la magnitude.
- **GPS universel** : questions seules = 93,3 % questions / 30,6 % amorces ;
  amorces seules = 20,0 % / 72,2 % ; mélange = 81,7 % / 47,2 %. Aucun routeur
  unique ne passe le critère. Deux spécialistes se dégagent, sans sélecteur validé.
- **V2 int4** : mémoire validée, mais **3,6–4,0 s/token** et qualité mixte
  (1 propre, 1 bonne mais bouclée, 1 fausse sur le petit test GEOGRAPHIE).
- **Limites de la percée activation** : 6 amorces anglaises, routage oracle,
  tranches E4 non appariées. À confirmer sur prompts nouveaux, autres familles
  en routage réel et éventuellement entraînement apparié à l'activation.

## 🧭 Décisions qui appartiennent au chef

1. **Régime de démarrage proposé par les preuves, à trancher par le chef :**
   fond activation 75 % (rapide à construire sur V1, 5/6 sur le banc actuel),
   fond complet (6/6, plus lourd), ou nouvelle campagne appariée à l'activation.
2. **Critères de naissance de RATISS VEGA — proposition à écrire avant entraînement :**
   génération stable sur un jeu nouveau et plus large que les 6 amorces G1 ;
   routage non-oracle explicite (questions, amorces ou sélecteur de mode validé) ;
   budget mémoire et vitesse annoncés avant mesure ; textes bruts publiés.
3. Le nom, le périmètre et le moment de naissance de **RATISS VEGA** restent
   souverains : ce dossier ne les préjuge pas.

*Dossier de veille mis à jour — RATISS Labs, Yaoundé. Le sol est maintenant
beaucoup plus net : la contribution observée bat la magnitude, la mémoire
<300 Mo existe, et les deux échecs restants sont nommés sans fard. 😜🥳🇨🇲*
