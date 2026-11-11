# CVA/XVA Capstone

IRB RWA calculator for non-FSE corporates, with a Monte Carlo check of the
Basel capital formula against the single-factor Vasicek model. CVA and exposure
modules to follow.

Uses synthetic data only.

## Install
    python -m pip install -e ".[dev]"

## Usage
    from irb_rwa.irb import IRBCalculator
    calc = IRBCalculator()
    calc.rwa(pd=0.02, lgd=0.45, ead=1e6, beel=0.0, M=2.5)

## Run the tests
    python -m pytest

## Monte Carlo check
    python scripts/irb_mc_check.py
    Results (master seed 2026, 30 runs per N):
    Analytical K: 0.076617
         N     mean K       std      bias bias/se_mean
      1000  0.071870  0.010479 -0.004747        -2.48
     10000  0.075950  0.004373 -0.000666        -0.83
    100000  0.076612  0.001638 -0.000005        -0.02
   1000000  0.076548  0.000600 -0.000069        -0.63

## Project layout
cva_xva_project/

├── pyproject.toml

├── requirements.txt

├── README.md

├── .gitignore

├── pytest.ini

├── src/

│   └── irb_rwa/

│       ├── __init__.py

│       ├── irb.py              # IRBCalculator

│       └── simulation.py       # simulate_loss

├── scripts/

│   └── irb_mc_check.py             # convergence study and plots

└── tests/

    ├── test_irb.py
    
    └── test_simulation.py


## Status and limitations
Infinitely granular portfolio; no CVA yet.
