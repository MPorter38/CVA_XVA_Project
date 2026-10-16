# Test IRB RWA Calculator Class 

# Setup
from IRB_RWA_Calculator_Class import Exposure, IRBCalculator
import pytest
from pytest import approx

@pytest.fixture 
def example_exposure():
    return IRBCalculator(Exposure(pd=0.1, lgd=0.1, ead=100, beel=0.1, M=1))

# Tests for inputs for Exposure being out of bounds 
def test_negative_ead_rejected():
    with pytest.raises(ValueError):
        Exposure(pd=0.1, lgd=0.1, ead=-1, beel=0.1, M=1)

def test_negative_pd_rejected():
    with pytest.raises(ValueError):
        Exposure(pd=-0.1,lgd=0.1,ead=1, beel=0.1, M=1)

def test_negative_lgd_rejected():
    with pytest.raises(ValueError):
        Exposure(pd=0.1, lgd=-0.1, ead=1, beel=0.1, M=1)

def test_negative_beel_rejected():
    with pytest.raises(ValueError):
        Exposure(pd=1, lgd=0.1, ead=1, beel=-0.1, M=1)

def test_negative_M_rejected():
    with pytest.raises(ValueError):
        Exposure(pd=0.1, lgd=0.1, ead=1, beel=0.1, M=-1)      
        
def test_pd_over_one_rejected():
    with pytest.raises(ValueError):
        Exposure(pd=2, lgd=0.1, ead=1, beel=0.1, M=1)

def test_lgd_over_one_rejected():
    with pytest.raises(ValueError):
        Exposure(pd=0.1, lgd=2, ead=1, beel=0.1, M=1)    
        
# Test that RWA is always positive 
@pytest.mark.parametrize("pd", [0.001, 0.01, 0.05, 0.1, 0.5, 0.99, 1])
def test_rwa_postive_with_pd_range(pd):
    assert IRBCalculator(Exposure(pd=pd, lgd=0.1, ead=100, beel=0.05, M=1)).rwa() > 0
    
@pytest.mark.parametrize("lgd", [0.001, 0.01, 0.05, 0.1, 0.5, 0.99, 1]) 
def test_rwa_positive_with_lgd_range(lgd):
    assert IRBCalculator(Exposure(pd=0.1, lgd=lgd, ead=100, beel=0.1, M=1)).rwa() > 0 
    
@pytest.mark.parametrize("M", [0,1,2,2.5,3,4,5])
def test_rwa_postive_with_M_range(M):
    assert IRBCalculator(Exposure(pd=0.1, lgd=0.1, ead=100, beel=0.1, M=M)).rwa() > 0

# Test that RWA is linear with EAD (RWA doubles when EAD doubles)
def test_rwa_linear_with_ead(example_exposure):
    rwa_for_ead_200 = IRBCalculator(Exposure(pd=0.1, lgd=0.1, ead=200, beel=0.1, M=1)).rwa()
    assert (2 * example_exposure.rwa()) == approx(rwa_for_ead_200, rel=1e-6)   
    
# Test RWA is linear with LGD (RWA doubles when LGD doubles)
def test_rwa_linear_with_lgd(example_exposure):
    rwa_for_lgd_20 = IRBCalculator(Exposure(pd=0.1, lgd=0.2, ead=100, beel=0.1, M=1)).rwa()
    assert (2 * example_exposure.rwa()) == approx(rwa_for_lgd_20, rel=1e-6)
    
# Test PD floor is enforced (PD doesn't change when PD goes below 5bps)
def test_pd_floor():
    rwa_for_pd_3bps = IRBCalculator(Exposure(pd=0.03, lgd=0.1, ead=100, beel=0.1, M=1)).rwa()
    rwa_for_pd_5bps = IRBCalculator(Exposure(pd=0.05, lgd=0.1, ead=100, beel=0.1, M=1)).rwa()
    assert rwa_for_pd_3bps == approx(rwa_for_pd_5bps, rel=1e-6)
    
# Test default calculation 
def test_default_rwa():
    assert IRBCalculator(Exposure(pd=1, lgd=0.1, ead=100, beel=0.05, M=1)).rwa() == approx(12.5*0.05*100, rel=1e-6)
    
def test_default_rwa_floored():
    assert IRBCalculator(Exposure(pd=1, lgd=0.05, ead=100, beel=0.1, M=1)).rwa() == approx(0, rel=1e-6)
    
# Test non-defaulted calculation 
def test_non_def_rwa(example_exposure):
    assert example_exposure.rwa() == approx(39.055708, rel=1e-6)