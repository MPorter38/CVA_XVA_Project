# Simulation Functions 
from scipy.stats import norm
import numpy as np

# Simulate loss from Vasicek model and calculate capital requirement K 
def simulate_loss(PD, LGD, R, N=1000, generator=None):
    Z = generator.standard_normal(N)
    p = norm.cdf((norm.ppf(PD) - np.sqrt(R)*Z)/np.sqrt(1-R))
    loss = LGD * p
    var_999 = np.quantile(loss, 0.999)
    expected_loss = loss.mean()
    simulated_K = var_999 - expected_loss
    return simulated_K
