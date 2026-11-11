# Show by simulation that the closed form capital requirement 'K' is equal
# to a simulation of the single factor Vasicek model. 
# Simulated unexpected loss (99.9% loss quantile less EL) should converage to 
# K within a few MC standard errors 
# Error shrinks like 1/sqrt(N) as N increases

# Maths to use:
# Asset Value X = sqrt(R)*Z + sqrt(1-R)*epsilon with Z as systematic factor and epsilon as idiosyncratic factor
# Default occurs if X < N^-1(PD)
# The default probability is p(Z) = N((N^-1(PD) - sqrt(R)*Z)/sqrt(1-R))
# Portfolio loss rate is LGD x p(Z) 
# Analytical target is capital_requirement(pd,lgd) = LGD x (99.9% conditional PD - PD)

# Libraries
from matplotlib.pylab import seed
from irb_rwa.irb import IRBCalculator
import numpy as np
from matplotlib import pyplot as plt
from irb_rwa.simulation import simulate_loss

def run_many(pd, lgd, r, n, n_runs, seed_seq):
    """ n_runs independent estimates of K, each using N draws"""
    rngs = [np.random.default_rng(s) for s in seed_seq.spawn(n_runs)]
    return np.array([simulate_loss(pd, lgd, r, N=n, generator=g) for g in rngs])

def main():
    # Test Case
    pd = 0.02 
    lgd = 0.45
    calc = IRBCalculator()
    r = calc.asset_correlation(pd)

    # Compare one anlytical K to one simulated K
    #seed = 42
    #rng = np.random.default_rng(seed)
    #print("Analytical K: ", calc.capital_requirement(pd, lgd))
    #print("Simulated K: ", simulate_loss(pd, lgd, r, N=1000000, generator=rng))
    
    # Compare for 30 seeds, each using 100000 draws
    # est = run_many(pd, lgd, r, 100000, 30 , np.random.SeedSequence(master_seed))
    # print (est.mean(), est.std(ddof=1), calc.capital_requirement(pd, lgd))
    
    # Compare for different N, each using 30 seeds
    master_seed = 2026
    Ns = [10**3, 10**4, 10**5, 10**6]
    n_runs = 30
    n_seeds = np.random.SeedSequence(master_seed).spawn(len(Ns))

    k_true = float(calc.capital_requirement(pd, lgd))
    results = {n: run_many(pd,lgd, r, n, n_runs, ss) for n, ss in zip(Ns, n_seeds)}

    print(f"Analytical K: {k_true:.6f}")
    print(f"{'N':>10} {'mean K':>10}{'std':>10}{'bias':>10}{'bias/se_mean':>13}")
    for n, est in results.items():
        mean, sd = est.mean(), est.std(ddof=1)
        se_mean = sd / np.sqrt(n_runs)
        print(f"{n:>10}{mean:>10.6f}{sd:>10.6f}{mean-k_true:>10.6f}{(mean-k_true)/se_mean:>13.2f}")
        
    # Plot distribution of estimates at the largest N
    n_big = Ns[-1]
    fig, ax = plt.subplots()
    ax.hist(results[n_big], bins=12, alpha=0.7)
    ax.axvline(k_true, color='red', label="Analytical K")
    ax.set_xlabel("Simulated K"); ax.set_ylabel("Runs")
    ax.set_title(f"Simulated K over {n_runs} runs, N={n_big:,}")
    ax.legend()
    fig.savefig("results/k_distribution.png", dpi=150, bbox_inches='tight')
    
    # Plot error vs N, log-log
    sds = np.array([results[n].std(ddof=1) for n in Ns])
    slope = np.polyfit(np.log(Ns), np.log(sds), 1)[0]
    fig, ax = plt.subplots()
    ax.loglog(Ns, sds, marker="o", label=f"Observed (slope {slope:.2f})")
    ax.loglog(Ns, sds[0] * np.sqrt(Ns[0] / np.array(Ns)), "--", label="1/sqrt(N) reference")
    ax.set_xlabel("N"); ax.set_ylabel("Std of K estimate")
    ax.legend()
    fig.savefig("results/convergence.png", dpi=150, bbox_inches='tight')
    
if __name__ == "__main__":
    main()