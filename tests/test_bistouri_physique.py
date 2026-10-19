# 🔪 Tests D1 — extracteur de tranche (aucun entraînement, aucun modèle chargé)
# Critère figé : égalité numérique ≤ 1e-5 sur le « masque plein ».
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bistouri_physique import (extraire_tranche, forward_tranche, forward_dense,
                               reinjecter_tranche)

torch.manual_seed(20261103)
DIM, INTER, TOK = 16, 24, 7


def bloc(dim=DIM, inter=INTER):
    return (torch.randn(inter, dim), torch.randn(inter, dim), torch.randn(dim, inter))


def test_masque_plein_egalite_numerique():
    Wg, Wu, Wd = bloc()
    x = torch.randn(TOK, DIM)
    tranche = extraire_tranche(Wg, Wu, Wd, list(range(INTER)))
    ecart = (forward_tranche(x, tranche) - forward_dense(x, Wg, Wu, Wd)).abs().max().item()
    assert ecart <= 1e-5, f"écart masque plein = {ecart}"


def test_masque_plein_3d_et_indices_melanges():
    Wg, Wu, Wd = bloc()
    x = torch.randn(2, TOK, DIM)
    indices = torch.randperm(INTER)              # ordre mélangé : doit être invariant
    tranche = extraire_tranche(Wg, Wu, Wd, indices)
    assert torch.allclose(forward_tranche(x, tranche), forward_dense(x, Wg, Wu, Wd), atol=1e-5)


def test_tranche_egale_composante_active_du_dense():
    Wg, Wu, Wd = bloc()
    x = torch.randn(TOK, DIM)
    accord = [1, 4, 9, 15, 20]
    tranche = extraire_tranche(Wg, Wu, Wd, accord)
    h = torch.nn.functional.silu(x @ Wg.T) * (x @ Wu.T)
    masque = torch.zeros(INTER); masque[accord] = 1.0
    reference = (h * masque) @ Wd.T               # dense dont on a éteint les inactifs
    assert torch.allclose(forward_tranche(x, tranche), reference, atol=1e-5)


def test_formes_et_copie_physique_independante():
    Wg, Wu, Wd = bloc()
    tranche = extraire_tranche(Wg, Wu, Wd, [3, 1, 3, 7])   # doublon dédupliqué
    assert tranche.W_gate.shape == (3, DIM) and tranche.W_down.shape == (DIM, 3)
    assert tranche.indices.tolist() == [1, 3, 7]
    avant = Wg.clone(); tranche.W_gate.add_(100.0)
    assert torch.equal(Wg, avant), "la tranche doit être une copie, pas une vue"


def test_reinjection_ne_touche_que_l_accord():
    Wg, Wu, Wd = bloc()
    g0, u0, d0 = Wg.clone(), Wu.clone(), Wd.clone()
    tranche = extraire_tranche(Wg, Wu, Wd, [2, 5])
    tranche.W_gate.add_(1.0); tranche.W_up.add_(2.0); tranche.W_down.add_(3.0)
    reinjecter_tranche(Wg, Wu, Wd, tranche)
    assert torch.equal(Wg[[2, 5]], tranche.W_gate) and torch.equal(Wd[:, [2, 5]], tranche.W_down)
    reste = [i for i in range(INTER) if i not in (2, 5)]
    assert torch.equal(Wg[reste], g0[reste]) and torch.equal(Wu[reste], u0[reste])
    assert torch.equal(Wd[:, reste], d0[:, reste])


def test_gradients_uniquement_dans_la_tranche():
    Wg, Wu, Wd = bloc()
    for W in (Wg, Wu, Wd): W.requires_grad_(False)
    tranche = extraire_tranche(Wg, Wu, Wd, [0, 8, 16])
    for p in (tranche.W_gate, tranche.W_up, tranche.W_down): p.requires_grad_(True)
    perte = forward_tranche(torch.randn(TOK, DIM), tranche).square().mean()
    perte.backward()
    assert tranche.W_gate.grad is not None and tranche.W_gate.grad.shape == tranche.W_gate.shape
    assert Wg.grad is None and Wu.grad is None and Wd.grad is None


def test_erreurs_propres():
    Wg, Wu, Wd = bloc()
    for mauvais in ([], [INTER + 1], [-1]):
        try:
            extraire_tranche(Wg, Wu, Wd, mauvais); assert False, mauvais
        except (ValueError, IndexError):
            pass


def test_dimensions_du_cobaye_360m():
    Wg, Wu, Wd = bloc(dim=960, inter=2560)
    accord = sorted(torch.randperm(2560)[:113].tolist())
    tranche = extraire_tranche(Wg, Wu, Wd, accord)
    y = forward_tranche(torch.randn(4, 960), tranche)
    assert y.shape == (4, 960) and tranche.nombre_parametres() == 3 * 113 * 960
