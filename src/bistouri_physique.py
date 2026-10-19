#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔪 BISTOURI PHYSIQUE — Marche D1 : extraction physique de la tranche active.
Proposition validée par le chef : docs/PROPOSITION_DECOUPAGE_PHYSIQUE.md

Principe (et rien de plus à cette marche) :
  À partir des trois matrices d'un MLP (W_gate, W_up, W_down) et des indices k
  de l'accord M_k fourni par le GPS, EXTRAIRE PHYSIQUEMENT les sous-matrices
      W_gate[k, :]   W_up[k, :]   W_down[:, k]
  et calculer le forward uniquement dans ce couloir étroit.
Aucun entraînement ici. Aucun modèle complet chargé. CPU uniquement.
RATISS Labs — Yaoundé — novembre 2026.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def normaliser_indices(indices) -> torch.Tensor:
    """Canonise les indices d'un accord : entiers longs, uniques, triés.

    L'ordre des neurones intermédiaires ne change pas le résultat final :
    permuter h et les colonnes de W_down dans le même ordre laisse y invariant.
    On trie donc pour avoir une tranche reproductible, toujours identique.
    """
    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(list(indices), dtype=torch.long)
    indices = indices.to(dtype=torch.long).reshape(-1)
    return torch.unique(indices, sorted=True)


@dataclass
class TrancheMLP:
    """Les trois sous-matrices physiques d'un accord. C'est TOUT ce qui calcule."""
    W_gate: torch.Tensor   # (n_actifs, dim_entree)
    W_up: torch.Tensor     # (n_actifs, dim_entree)
    W_down: torch.Tensor   # (dim_entree, n_actifs)
    indices: torch.Tensor  # (n_actifs,) positions dans la matrice globale

    @property
    def n_actifs(self) -> int:
        return int(self.indices.numel())

    def nombre_parametres(self) -> int:
        return int(self.W_gate.numel() + self.W_up.numel() + self.W_down.numel())


def extraire_tranche(W_gate: torch.Tensor, W_up: torch.Tensor,
                     W_down: torch.Tensor, indices) -> TrancheMLP:
    """Découpe PHYSIQUE aux ciseaux (index_select = vraie copie mémoire). ✂️

    La tranche renvoyée est indépendante des matrices globales : la modifier
    ne modifie pas W. C'est ce qui permettra (march D2) de ne différencier
    et de ne stocker chez l'optimiseur que ces paramètres-là.
    """
    if W_gate.shape != W_up.shape:
        raise ValueError(f"W_gate {tuple(W_gate.shape)} != W_up {tuple(W_up.shape)}")
    if W_down.shape != (W_gate.shape[1], W_gate.shape[0]):
        raise ValueError(f"W_down {tuple(W_down.shape)} incompatible avec "
                         f"W_gate {tuple(W_gate.shape)}")
    idx = normaliser_indices(indices)
    if idx.numel() == 0:
        raise ValueError("Accord vide : aucun neurone actif.")
    if int(idx.min()) < 0 or int(idx.max()) >= W_gate.shape[0]:
        raise IndexError("Indice d'accord hors de la matrice intermédiaire.")
    return TrancheMLP(
        W_gate=torch.index_select(W_gate, 0, idx).contiguous(),
        W_up=torch.index_select(W_up, 0, idx).contiguous(),
        W_down=torch.index_select(W_down, 1, idx).contiguous(),
        indices=idx,
    )


def forward_tranche(x: torch.Tensor, tranche: TrancheMLP) -> torch.Tensor:
    """y = [ SiLU(x·W_gate_actifᵀ) ⊙ (x·W_up_actifᵀ) ] · W_down_actifᵀ

    x : (..., dim_entree) — fonctionne en 2D (tokens, dim) et 3D (batch, tokens, dim).
    Seuls les neurones de l'accord participent au calcul. 🌫️➡️🛤️
    """
    h = F.silu(F.linear(x, tranche.W_gate)) * F.linear(x, tranche.W_up)
    return F.linear(h, tranche.W_down)


def forward_dense(x: torch.Tensor, W_gate: torch.Tensor, W_up: torch.Tensor,
                  W_down: torch.Tensor) -> torch.Tensor:
    """Référence dense (le bloc MLP complet), pour les tests d'égalité D1."""
    h = F.silu(F.linear(x, W_gate)) * F.linear(x, W_up)
    return F.linear(h, W_down)


def reinjecter_tranche(W_gate: torch.Tensor, W_up: torch.Tensor,
                       W_down: torch.Tensor, tranche: TrancheMLP) -> None:
    """Remet la tranche (éventuellement mise à jour) dans les matrices globales,
    EN PLACE, uniquement aux indices de l'accord (index_copy_). Les autres
    coefficients sont strictement inchangés. Testée unitairement (D1)."""
    idx = tranche.indices
    with torch.no_grad():
        W_gate.index_copy_(0, idx, tranche.W_gate)
        W_up.index_copy_(0, idx, tranche.W_up)
        W_down.index_copy_(1, idx, tranche.W_down)
