# Test IRB RWA Calculator Class 

# Setup
from src.irb_rwa import IRBCalculator
import pytest
from pytest import approx
import numpy as np

BASE = dict(pd=0.02, lgd=0.45, ead=1e6, beel=0.0, M=2.5)

def call(calc, **overrides):
    args = {**BASE, **overrides}
    return calc.rwa(args["pd"], args["lgd"], args["ead"], args["beel"], args["M"])

@pytest.fixture
def calc():
    return IRBCalculator()

# Test that scalar inputs return a scalar output
def test_scalar_returns_number(calc):
    assert np.ndim(call(calc)) == 0

# Test defaulted RWA floored at zero (RWA cannot be negative)
def test_defaulted_rwa_floored_at_zero(calc):
    assert call(calc, pd=1.0, lgd=0.45, beel=0.60) == 0.0

# Test that RWA scales linearly with EAD and LGD
def test_rwa_scales_linearly_with_ead(calc):
    assert call(calc, ead=2e6) == approx(2 * call(calc, ead=1e6))

def test_rwa_scales_linearly_with_lgd(calc):
    assert call(calc, lgd=0.40) == approx(call(calc, lgd=0.20) * 2)   # performing only

# Test that PD floor is applied (PD cannot be below 5bps)
def test_pd_floor_applied(calc):
    assert call(calc, pd=1e-6) == approx(call(calc, pd=calc.PD_FLOOR))

# Test that RWA increases with LGD 
def test_rwa_increases_with_lgd(calc):
    lgds = np.linspace(0.001, 0.99, 50)
    assert np.all(np.diff(calc.rwa(0.02, lgds, 1e6, 0.0, 2.5)) > 0)

# Test that array and scalars can be inputted together and broadcasted correctly
def test_scalar_broadcasts_against_array(calc):
    assert calc.rwa(np.full(5, 0.02), 0.45, 1e6, 0.0, 2.5).shape == (5,)

def test_list_input_accepted(calc):
    assert calc.rwa([0.02, 0.05], 0.45, 1e6, 0.0, 2.5).shape == (2,)

# Tests for inputs for variables being out of bounds 
def test_negative_ead_rejected():
    with pytest.raises(ValueError):
        call(calc, ead=-1)

def test_negative_pd_rejected():
    with pytest.raises(ValueError):
        call(calc, pd=-0.1)

def test_negative_lgd_rejected():
    with pytest.raises(ValueError):
        call(calc, lgd=-0.1)

def test_negative_beel_rejected():
    with pytest.raises(ValueError):
        call(calc, beel=-0.1)

def test_negative_M_rejected():
    with pytest.raises(ValueError):
        call(calc, M=-1)      
        
def test_pd_over_one_rejected():
    with pytest.raises(ValueError):
        call(calc, pd=2)

def test_lgd_over_one_rejected():
    with pytest.raises(ValueError):
        call(calc, lgd=2)    
        
# Test that RWA is always positive 
@pytest.mark.parametrize("pd", [0.001, 0.01, 0.05, 0.1, 0.5, 0.99, 1])
def test_rwa_postive_with_pd_range(pd):
    assert call(calc,pd=pd) > 0
    
@pytest.mark.parametrize("lgd", [0.001, 0.01, 0.05, 0.1, 0.5, 0.99, 1]) 
def test_rwa_positive_with_lgd_range(lgd):
    assert call(calc, lgd=lgd) > 0 
    
@pytest.mark.parametrize("M", [0,1,2,2.5,3,4,5])
def test_rwa_postive_with_M_range(M):
    assert call(calc, M=M) > 0
   
# Test default calculation 
def test_default_rwa():
    assert calc(pd=1, lgd=0.1, ead=100, beel=0.05, M=1).rwa() == approx(12.5*0.05*100, rel=1e-6)
    
# Test non-defaulted calculation 
def test_non_def_rwa():
    assert calc(pd=0.1, lgd=0.1, ead=100, beel=0.1, M=1).rwa() == approx(39.055708, rel=1e-6)