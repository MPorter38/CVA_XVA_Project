# Test function in simulation.py

import pytest
from irb_rwa.simulation import simulate_loss
import numpy as np

# Setup a random number generator for reproducibility
@pytest.fixture
def rng():
    return np.random.default_rng(42)

# Test that simulated loss is between 0 and LGD for various PD and R values
@pytest.mark.parametrize("pd", [0.001, 0.02, 0.2, 0.9])
@pytest.mark.parametrize("r", [0.05, 0.12, 0.24])
def test_loss_is_between_zero_and_lgd(rng, pd, r):
    loss = simulate_loss(pd, 0.45, r, N=5000, generator=rng)
    assert 0 <= loss <= 0.45

# Test that simulated loss is the same for the same seed and parameters and different for different seeds
def test_same_seed_gives_same_result():
    a = simulate_loss(0.02, 0.45, 0.1, N=1000, generator=np.random.default_rng(1))
    b = simulate_loss(0.02, 0.45, 0.1, N=1000, generator=np.random.default_rng(1))
    assert a == b
    
def test_different_seeds_give_different_results():
    a = simulate_loss(0.02, 0.45, 0.1, N=1000, generator=np.random.default_rng(1))
    b = simulate_loss(0.02, 0.45, 0.1, N=1000, generator=np.random.default_rng(2))
    assert a != b
    
