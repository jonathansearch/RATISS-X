# 🌟 DOSSIER DE VEILLE — PRÊT POUR LA CONSTRUCTION RATISS (décision du chef : demain)
**État scellé la veille au soir.** Ce document ne déclare pas RATISS VEGA : il fige
**sur quel sol vérifié** la construction de demain pourra décider ses critères 🏛️

## ✅ Ce qui est validé et public (GitHub, tête à jour ce soir)
- **Entraînement par tranches du modèle entier** (32 couches, SmolLM2-360M) :
  perte **1,245 (−79,8 %)**, ÷21 mises à jour/pas contre le dense MLP, identité 0,015 (E4).
- **Duel dense** : le dense perd moins (1,83 contre 2,12 à 8 couches) pour 20× de
  mises à jour et 4,1× plus lent/pas ; à 32 couches le dense ne tient pas dans 2 Go (F1).
- **Génération du modèle entier opéré** (avec fond complet) : 5/5 stable, 3/5 juste (E4-gen).
- **Moteur natif fidèle** (NumPy, cosinus 1,0000 contre PyTorch à mode égal, L1) et
  **moteur V1 à poids résidents** : hybride 1,36× plus rapide que son dense (V1).
- **La courbe du Seuil de Stabilisation** (un pas, mono-prompt) : genou (65 %, 75 %] (S1–S5).

## ⚠️ Ce qui est réfuté ou ouvert (à ne PAS supposer demain)
- **Seuil de génération ≠ seuil cosinus** : en génération libre multi-prompts, 70 % et
  75 % de fond élite **déraillent** ; seul 100 % fait 6/6 (G1, ce soir). Le seuil de
  génération est > 75 % sur ce banc (non mesuré entre 75 et 100).
- **Désaccord entraînement/inférence** : les tranches E4 sont entraînées avec fond
  complet. Deux voies propres pour RATISS : (a) fond complet = fiabilité immédiate ;
  (b) entraîner au régime d'inférence visé = la voie « native », à protocoliser.
- **GPS sur amorces libres** : faible (36 % au mieux, F2 rejeté) ; familles servies en
  oracle dans G1. Corpus d'amorces d'entraînement = chantier ouvert.
- **Mémoire du moteur** : V1 tient ~669 Mo au repos mono-famille (cible historique
  <300 Mo non atteinte ; quantification et élite réduite non testées).

## 🧭 Décisions qui appartiennent au chef demain
1. Le régime de fond de RATISS au démarrage (complet fiable, ou régime entraîné).
2. Les **critères de naissance** de RATISS VEGA (tâches, seuils, génération) — à écrire
   AVANT tout entraînement, comme toujours dans ce labo ⚖️
3. Le nom et le périmètre exacts de la construction (le présent dossier n'en préjuge pas).

*Dossier de veille scellé — RATISS Labs, Yaoundé. Demain on construit ; ce soir, on
sait exactement ce qui tient, ce qui casse, et ce qui reste à décider. 😜🥳🇨🇲*
