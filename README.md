# 🧠⚡ RATISS-X — Le LLM Mixer (pensée en chemins)

**RATISS Labs** · Jonathan Evina · Yaoundé, Cameroun 🇨🇲 · MIT

> 🌫️➡️🛤️ *On ne transforme pas le brouillard noir d'un LLM.
> On apprend à lire ses chemins — puis on le fait penser en chemins.*

## 🌟 L'hypothèse
Un LLM dense calcule 100 % de ses paramètres pour chaque mot (~20 W pour un cerveau
humain contre des mégawatts de GPU). RATISS-X explore l'inverse : **sparsité 1–2 %,
routage topologique, activations ciblées** — un modèle qui emprunte des *chemins*
(« accords » d'activation) au lieu de tout allumer.

## 🐜 Le cobaye (règle du chef)
**SmolLM2-360M** (le « grand frère », ×2,7) est le cobaye de référence depuis le 2 nov. 2026 —
le 135M, premier cobaye historique, reste dans l'historique git et les rapports.
Ici : aucun poids de modèle n'est versionné, le repo reste léger.
Une fois tout validé naîtra **RATISS VEGA**, premier LLM augmenté du labo. 🌟

## 📦 Contenu
| Élément | Rôle |
|---|---|
| `PROTOCOLE_GPS.md` | ⚖️ Critères figés AVANT l'Étape 1 (le GPS) |
| `src/gps_routeur.py` | 🗺️ Le GPS : embedding → famille d'accord → masque $M_k$, sans transformeur |
| `src/demo_gps.py` | ▶️ Démo en direct sur questions jamais vues |
| `data/questionnaire.json` | 📝 78 stimuli (6 familles × 10 + 18 paraphrases) |
| `artifacts/gps_modele.npz` | ⚡ Le GPS entraîné (centroïdes + masques, quelques Ko) |
| `results/RESULTATS_GPS.json` | 📏 Tous les chiffres mesurés de l'Étape 1 |
| `docs/RAPPORT_ETAPE1_GPS.md` | 📜 Le rapport complet, en mots simples |
| `tests/` | ✅ 3 tests unitaires |

## ▶️ Rejouer
```bash
pip install numpy scikit-learn transformers safetensors huggingface-hub pytest
python3 src/gps_routeur.py     # ré-entraîne le GPS depuis le banc d'essai n°1 et mesure tout
python3 src/demo_gps.py        # route des questions jamais vues, en direct
python3 -m pytest tests/ -q    # 3 tests
```
*Le banc d'essai n°1 (la carte des chemins) vit dans `banc_essai_mixer/` du poste du chef ;
ses chemins bruts ne sont pas versionnés ici — seuls les masques dérivés le sont.*

## 📸 Résultats en images (acquis et validé uniquement)
![Chemins et GPS](docs/images/fig1_chemins_gps.png)
![Bistouri](docs/images/fig2_bistouri.png)
![Traducteur](docs/images/fig3_traducteur.png)
![Efficacité CPU](docs/images/fig4_efficacite_cpu.png)

📜 **Reproductibilité pas-à-pas :** `docs/JOURNAL_REPRODUCTIBILITE.md` — chaque étape, chaque commande, chaque graine.

## 🗺️ Feuille de route (Phase 2 du doc RATISS X)
- ✅ **Étape 1 — Le GPS** : prédire l'accord en ~0,06 ms (voir rapport)
- ⚠️ Étapes 2 & 3 sur 135M : réussites partielles — `docs/RAPPORT_ETAPES_2_3.md`
- 🐘 **Tous les tests refaits sur le grand frère 360M** : presque tout progresse — `docs/RAPPORT_GRAND_FRERE.md`
- 🔪 Découpage physique D1–D3 : tranche exacte, 11× plus rapide au banc, **apprentissage égal au dense** sur segment — `docs/RAPPORT_DECOUPAGE_D1.md` · `D2` · `D3`
- 🗣️ Génération libre en tranches seules : preuve de concept obtenue — `docs/RAPPORT_GENERATION_TRANCHE.md`
- 🪜 Extension au masque causal : 4 couches −62,4 % → 8 couches −65,5 % → **16 couches −70,2 %** (toujours ÷20 en mises à jour/pas) — `docs/RAPPORT_EXTENSION_E1.md` · `E2` · `E3`
- 🧹 Série mémoire M1–M3 : Adam à états BF16 validé, passeport mémoire transféré au futur moteur natif — `docs/RAPPORT_M1.md` · `M2` · `M3`, `docs/DECLARATION_PASSEPORT_MOTEUR_NATIF.md`

*Ici c'est la joie et l'enthousiasme 😜🥳 — et la méthode : on mesure, on nomme, on publie.*
