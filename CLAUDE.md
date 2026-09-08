# Project
Implementation of the Deflated Sharpe Ratio and Probability of Backtest
Overfitting (Bailey and Lopez de Prado). Reusable audit tool applied across
my other quant projects.

# Conventions
- Python 3.12. numpy and scipy only, matplotlib for figures. No new dependencies without asking me first.
- Analysis code lives in src/ and is imported. Notebooks are for figures only.
- Data goes in data/raw as parquet and is never committed.
- Credentials never appear in code, notebooks, or committed files.

# Boundaries
- You write: scaffolding, IO, caching, tests, plotting, docstrings, refactors.
- I write: all statistical and financial methodology. The PSR and DSR formulas,
  the effective number of trials estimate, the CSCV procedure, and every
  distributional assumption are mine to implement.
- If I ask you to write methodology, remind me of this line instead of doing it.
