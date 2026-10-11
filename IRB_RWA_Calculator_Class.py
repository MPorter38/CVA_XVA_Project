# This code defines a data class for exposures 
# and a class for calculating the IRB RWA for (non-FSE) Corporates, based on the Basel framework.

from dataclasses import dataclass
import numpy as np
import scipy.stats as norm 

@dataclass
class Exposure:
    pd: float  # Probability of Default
    lgd: float  # Loss Given Default
    ead: float  # Exposure at Default
    beel: float  # Best Estimate of Expected Loss (for PD=1 case)
    M : float  # Maturity of the exposure in years
    
    def __post_init__(self):
        if not (0 <= self.pd <= 1):
            raise ValueError("PD must be between 0 and 1")
        if not (0 <= self.lgd <= 1):
            raise ValueError("LGD must be between 0 and 1")
        if self.ead < 0:
            raise ValueError("EAD must be non-negative")
        if self.pd == 1 and not (0 <= self.beel <= 1):
            raise ValueError("BEEL must be between 0 and 1 if PD is 1")
        if self.M < 0 or self.M > 5:
            raise ValueError("Maturity must be non-negative and less than or equal to 5 years")

class IRBCalculator:
    def __init__(self, exposure: Exposure):
        self.exposure = exposure
    
    def asset_correlation(self):
        # Calculate asset correlation based on PD for Corporates 
        if self.pd < 1:
            weight = (1 - np.exp(-50 * self.pd)) / (1 - np.exp(-50))
            return 0.12 * weight + 0.24 * (1 - weight)
    
    def maturity_adjustment(self):
        # Calculate maturity adjustment based on PD for Coporates 
        if self.pd < 1:
            b = (0.11852 - 0.05478 * np.log(self.pd)) ** 2
            return (1+(self.M-2.5)*b) / (1-1.5*b)
            
    def capital_requirement(self):
        # Calculate capital requirement using the Basel formula
        if self.pd < 1:
            R = self.asset_correlation()
            barrier = (1/np.sqrt(1-R)) * norm.ppf(self.pd) + np.sqrt(R/(1-R)) * norm.ppf(0.999)
            return self.lgd * (norm.cdf(barrier) - self.pd)
        
    def rwa(self):
        # Calculate Risk-Weighted Assets (RWA)
        if self.pd < 1:
            K = self.capital_requirement()
            M = self.maturity_adjustment()
            return 12.5 * K * self.ead * M
        if self.pd == 1:
            return max(0, 12.5 * (self.lgd - self.beel)) * self.ead