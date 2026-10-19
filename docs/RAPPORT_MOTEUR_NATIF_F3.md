# 🏗️ MICRO-RAPPORT F3 — MOTEUR NATIF v0 : IL TOURNE, IL NE PARLE PAS ENCORE
**Protocole :** `PROTOCOLE_DUEL_ET_ROUTAGE_ET_MOTEUR.md`. Moteur **sans PyTorch**
(NumPy + tokenizers) : poids gelés lus par `read()` dans le safetensors, MLP en
**tranche seule entraînée** (export E4, 576 tenseurs), génération gloutonne, 3 prompts.

## 📏 Critères figés, jugés
| Critère | Cible | Mesuré | Verdict |
|---|---|---|---|
| RSS pic du processus | < 300 Mo | **366 Mo** | ❌ (3 familles chargées en un processus) |
| Génération stable | pas de boucle/charabia | 1 boucle, 3 sorties dégradées | ❌ |
| Vitesse | publiée | 0,55 s/token | 📏 (à titre indicatif) |

Sorties brutes : « La capitale de la France est » → « Gibraltar twentieth reinst recess… » ;
« Le ciel est » → « replaced substituted substituted… » (boucle) ;
« Two plus two equals » → « 1 correct correct correct incorrect Wrong… ».

## ⚖️ Lecture honnête — deux échecs distincts
1. **La mécanique, elle, fonctionne** : un processus sans PyTorch charge le modèle,
   exécute 32 couches (attention GQA + RoPE convention HF ré-implémentées) et produit
   des tokens en 0,55 s/token. Le RSS de 366 Mo inclut les tranches FP32 de **trois**
   familles cumulées dans le même processus (~41 Mo chacune) : le critère des 300 Mo
   devra être re-mesuré **une famille par processus** avant verdict définitif.
2. **La tranche seule à 32 couches déraille** — contraste majeur avec D4 (2 couches,
   génération cohérente). Deux explications non départagées à ce stade : (a) la branche
   inactive, absente ici, porte une part essentielle du signal à grande profondeur
   (les tranches E4 ont été entraînées **avec** elle, pas pour s'en passer) ;
   (b) un écart numérique du moteur v0 reste possible tant qu'un test de contrôle
   (logits moteur contre logits du segment PyTorch, même prompt) n'a pas été passé.
   Le labo ne tranche pas sans ce contrôle : il est protocolisé comme prochain pas.

## 🔜 Suite propre pour le moteur
Test de contrôle des logits (une heure de banc, pas un moteur neuf) → si le moteur est
fidèle, la conclusion scientifique sera : **l'inférence « tranche seule » ne passe pas
l'échelle en profondeur sans la branche inactive** — et le moteur natif devra calculer
l'inactive en BF16 à la demande (toujours sans PyTorch), ce qui reste compatible avec
la cible mémoire puisque les poids se lisent par `read()` sans résidence permanente.

## 📦 Fichiers
`src/moteur_natif.py` · `src/exporter_tranches.py` · `results/RESULTATS_MOTEUR_NATIF.json` ·
ce rapport. Fichiers lourds (segment, export 128 Mo) évacués. Commit local, rien de poussé.

*Micro-rapport scellé — RATISS Labs, Yaoundé. Un moteur qui tourne mais ne parle pas encore,
c'est une étape — publiée comme telle. 😜🥳*
