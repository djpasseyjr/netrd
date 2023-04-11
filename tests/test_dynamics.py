"""
test_dynamics.py
----------------

Test dynamics algorithms.

"""

import networkx as nx
from netrd import dynamics
from netrd.dynamics import BaseDynamics, LotkaVolterra, SISModel
import numpy as np


def test_dynamics_valid_dimensions():
    """Dynamics models should return N x L arrays."""

    G = nx.barbell_graph(10, 5)
    N = G.number_of_nodes()

    for L in [25, 100]:
        for obj in dynamics.__dict__.values():
            if isinstance(obj, type) and BaseDynamics in obj.__bases__:
                TS = obj().simulate(G, L)
                assert TS.shape == (N, L), "f{label} has wrong dimensions"

    assert BaseDynamics().simulate(G, 25).shape == (N, 25)
    assert BaseDynamics().simulate(G, 100).shape == (N, 100)


def test_lotka_volterra():
    """Test Lotka Volterra simulation in the deterministic case."""
    g = nx.fast_gnp_random_graph(10, 0.001)
    lv_model = LotkaVolterra()
    assert lv_model.simulate(g, 100, stochastic=False).shape == (10, 100)


def test_SIS_small_initial_infected():
    """Test that the SIS model works when initial seed is small."""
    N = 10
    # All to all graph
    g = nx.fast_gnp_random_graph(N, 1.0)
    sis_model = SISModel()
    L = 20
    X = sis_model.simulate(
        g,
        L,
        num_seeds=1,  # Only one initially sick
        beta=1.0,  # All neighbors are guarenteed to get infected bc beta=1
        mu=1.0,  # All sick guarenteed to recover because mu=1
    )
    # We make sure that the sickness spreads
    assert np.sum(X) > 1
