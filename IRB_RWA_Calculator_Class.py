# This code defines a data class for exposures 
# and a class for calculating the IRB RWA for (non-FSE) Corporates, based on the Basel framework.

from dataclasses import dataclass
import numpy as np
from scipy.stats import norm 

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
        if not (0 <= self.beel <= 1):
            raise ValueError("BEEL must be between 0 and 1")
        if self.M < 0 or self.M > 5:
            raise ValueError("Maturity must be non-negative and less than or equal to 5 years")

class IRBCalculator:
    def __init__(self, exposure: Exposure):
        self.exposure = exposure
        if self.exposure.pd < 0.05:
            self.pd_floored = 0.05
        else:
            self.pd_floored = self.exposure.pd
    
    def asset_correlation(self):
        # Calculate asset correlation based on PD for Corporates 
        if self.pd_floored < 1:
            weight = (1 - np.exp(-50 * self.pd_floored)) / (1 - np.exp(-50))
            return 0.12 * weight + 0.24 * (1 - weight)
    
    def maturity_adjustment(self):
        # Calculate maturity adjustment based on PD for Coporates 
        if self.pd_floored < 1:
            b = (0.11852 - 0.05478 * np.log(self.pd_floored)) ** 2
            return (1+(self.exposure.M-2.5)*b) / (1-1.5*b)
            
    def capital_requirement(self):
        # Calculate capital requirement using the Basel formula
        # Assume no turnover adjustment
        if self.pd_floored < 1:
            R = self.asset_correlation()
            barrier = (1/np.sqrt(1-R)) * norm.ppf(self.pd_floored) + np.sqrt(R/(1-R)) * norm.ppf(0.999)
            return self.exposure.lgd * (norm.cdf(barrier) - self.pd_floored)
        
    def rwa(self):
        # Calculate Risk-Weighted Assets (RWA)
        if self.pd_floored < 1:
            K = self.capital_requirement()
            M = self.maturity_adjustment()
            return 12.5 * K * self.exposure.ead * M
        if self.pd_floored == 1:
            return max(0, 12.5 * (self.exposure.lgd - self.exposure.beel)) * self.exposure.ead