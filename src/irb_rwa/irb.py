"""This code defines a class for calculating the IRB RWA for (non-FSE) Corporates, based on the Basel framework.
 inputs are taken as numpy arrays, and the output is a numpy array of RWA values. """
 
import numpy as np
from scipy.stats import norm 

class IRBCalculator:
    
    pd_floor = 0.05 
    
    @staticmethod
    def _validated(x,name):
        arr = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite")
        if not np.all((arr >= 0)):
            raise ValueError(f"{name} must be non-negative")
        return arr

    def _check_input(self,pd,lgd,ead,beel,M):        
        pd = self._validated(pd, "PD") #probability of default
        lgd = self._validated(lgd, "LGD") #loss given default
        ead = self._validated(ead, "EAD") #exposure at default
        beel = self._validated(beel, "BEEL") #best estimate of expected loss
        M = self._validated(M, "Maturity")
        if np.any(pd > 1):
            raise ValueError("PD must be less than or equal to 1")
        if np.any(lgd > 1):
            raise ValueError("LGD must be less than or equal to 1") 
        if np.any(beel > 1):
            raise ValueError("BEEL must be less than or equal to 1")        
        if np.any((M > 5) | (M < 1)):
            raise ValueError("Maturity must be between 1 and 5 years")
        pd, lgd, ead, beel, M =  np.broadcast_arrays(pd, lgd, ead, beel, M) #check if shapes are compatible 
        pd, lgd, ead, beel, M = (np.atleast_1d(a) for a in (pd, lgd, ead, beel, M))
        return np.maximum(pd,self.pd_floor), lgd, ead, beel, M  

    def asset_correlation(self,pd):
        # Calculate asset correlation based on PD for Corporates 
        weight = (1 - np.exp(-50 * pd)) / (1 - np.exp(-50))
        return 0.12 * weight + 0.24 * (1 - weight)
    
    def maturity_adjustment(self,pd,M):
        # Calculate maturity adjustment based on PD for Coporates 
        b = (0.11852 - 0.05478 * np.log(pd)) ** 2
        return (1+(M-2.5)*b) / (1-1.5*b)
            
    def capital_requirement(self,pd,lgd):
        # Calculate capital requirement using the Basel formula
        # Assume no turnover adjustment
        R = self.asset_correlation(pd)
        barrier = (1/np.sqrt(1-R)) * norm.ppf(pd) + np.sqrt(R/(1-R)) * norm.ppf(0.999)
        return lgd * (norm.cdf(barrier) - pd)
        
    def rwa(self, pd, lgd, ead, beel, M):
        # Calculate Risk-Weighted Assets (RWA)

        is_scalar = all(np.ndim(a) == 0 for a in [pd, lgd, ead, beel, M]) # check if all inputs are scalars
        pd, lgd, ead, beel, M = self._check_input(pd, lgd, ead, beel, M) # validate inputs
        pd, lgd, ead, beel, M = (np.atleast_1d(a) for a in [pd, lgd, ead, beel, M]) # convert inputs to 1D arrays if scalar
        
        # identify defaulted and performing exposures
        out = np.empty_like(pd)
        defaulted = pd >= 1
        performing = ~defaulted
        
        #RWA calculation for performing and defaulted exposures
        if performing.any():
            K = self.capital_requirement(pd[performing], lgd[performing])
            MA = self.maturity_adjustment(pd[performing], M[performing])
            out[performing] = 12.5 * K * ead[performing] * MA
        if defaulted.any():
            out[defaulted] = np.maximum(0, 12.5 * (lgd[defaulted] - beel[defaulted])) * ead[defaulted]
            
        return out[0] if is_scalar else out