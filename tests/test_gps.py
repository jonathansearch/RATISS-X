# Tests du GPS — python -m pytest tests/ -q
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from gps_routeur import jaccard, chemin_vers_masques, similarite_masque_chemin, GPS, FAMILLES

def test_jaccard():
    assert jaccard({1, 2}, {2, 3}) == 1 / 3
    assert jaccard(set(), set()) == 0.0

def test_masque_prototype():
    chemins = [[{1, 2, 3}], [{1, 2, 4}], [{1, 5}]]
    assert chemin_vers_masques(chemins, seuil=0.5)[0] == {1, 2}

def test_gps_predire_et_sauvegarde(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 576)); y = np.array([FAMILLES[i % 6] for i in range(60)])
    for i, f in enumerate(y): X[i] += (FAMILLES.index(f) + 1)
    masques = {f: [{1, 2}] * 30 for f in FAMILLES}
    g = GPS().entrainer(X, y, masques)
    fam, conf, scores = g.predire(X[0])
    assert fam in FAMILLES and len(scores) == 6
    p = tmp_path / "g.npz"; g.sauvegarder(p)
    assert GPS.charger(p).masques["CALCUL"][0] == {1, 2}
