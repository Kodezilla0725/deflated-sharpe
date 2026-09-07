"""One entry point: a trial matrix in, both verdicts out.

The T x M matrix of trial returns is the smallest input that determines the
answer, and every other module wants it. `trials.average_correlation` needs the
columns to measure redundancy, `dsr` needs their dispersion, `pbo.cscv` needs
the whole matrix uncollapsed. Nothing here accepts a precomputed trial count,
because letting a caller supply their own would let them supply the raw M, and
suppressing exactly that is the point of the exercise.

The two verdicts attack the same problem from opposite sides. DSR is
parametric: how high must a Sharpe ratio be before the size of the search stops
explaining it. PBO is non-parametric and rank-based: does the configuration
that wins in sample keep winning out of sample. Neither subsumes the other, and
a strategy that clears one but fails the other is worth a second look.
"""

import numpy as np

from .dsr import TrialSet, deflated_sharpe_ratio, expected_max_sharpe
from .pbo import DEFAULT_N_SPLITS, cscv
from .psr import SharpeStats
from .trials import average_correlation, effective_num_trials

from dataclasses import dataclass


@dataclass(frozen=True)
class AuditResult:
    """Both verdicts, and every input that produced them.

    The inputs travel with the answer deliberately. Reporting a DSR without the
    trial count and the dispersion behind it reproduces the omission the DSR
    paper exists to attack, and the effective trial count is itself noisy - at
    M/T = 33 its spread runs a couple of hundred trials wide.
    """

    n_trials_raw: int
    avg_correlation: float
    n_effective_trials: float
    trial_sr_mean: float
    trial_sr_var: float
    best_trial: int
    stats: SharpeStats
    trials: TrialSet
    threshold_sr: float
    dsr: float
    cscv_result: object

    @property
    def pbo(self):
        return self.cscv_result.pbo

    @property
    def prob_loss(self):
        return self.cscv_result.prob_loss

    @property
    def periods_per_year(self):
        return self.stats.periods_per_year

    def summary(self):
        ppy = self.periods_per_year
        return "\n".join(
            [
                f"trials run                 {self.n_trials_raw}",
                f"average correlation        {self.avg_correlation:.3f}",
                f"effective independent      {self.n_effective_trials:.1f}",
                f"selected trial             #{self.best_trial}",
                f"  Sharpe (annualized)      {self.stats.annualized_sr:.2f}",
                f"  skew / kurtosis          {self.stats.skew:.2f} / "
                f"{self.stats.kurtosis:.2f}",
                f"  observations             {self.stats.n_obs:.0f}",
                f"hurdle SR0 (annualized)    "
                f"{self.threshold_sr * np.sqrt(ppy):.2f}",
                f"Deflated Sharpe Ratio      {self.dsr:.4f}",
                f"Probability of Overfitting {self.pbo:.4f}",
                f"OOS probability of loss    {self.prob_loss:.4f}",
            ]
        )


def audit(
    trial_returns,
    periods_per_year,
    n_splits=DEFAULT_N_SPLITS,
    selected=None,
    run_cscv=True,
):
    """Run both tests on one T x M matrix of trial returns.

    Parameters
    ----------
    trial_returns:
        T x M array. Rows are synchronous observations, columns are the trials.
        Every column the search actually produced, including the losers -
        hiding trials biases both statistics toward saying nothing is wrong.
    periods_per_year:
        Declared once here and propagated to `SharpeStats` and `TrialSet`,
        which cross-check that they agree.
    selected:
        Column index of the strategy under test. Defaults to the in-sample
        best, which is the case the papers are about.
    run_cscv:
        CSCV is by far the slowest step - about 5 seconds at S=16 against
        milliseconds for everything else. Set False for the DSR half alone.

    The threshold is computed with a trial mean of ZERO, not the observed mean
    of the sweep. DSR tests H0: SR = 0, so the hurdle is the expected maximum
    under that null. Feeding in the observed mean would ask a different and
    much harsher question - whether the winner beats the expected best of a
    strategy class already assumed to have skill. The observed mean is reported
    as `trial_sr_mean` if you want it.
    """
    X = np.asarray(trial_returns, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D T x M array, got shape {X.shape}")
    n_obs, n_trials_raw = X.shape
    if n_trials_raw < 2:
        raise ValueError("need at least 2 trials; a single backtest cannot be deflated")

    sd = X.std(axis=0)
    dead = sd <= 1e-12 * np.abs(X).max(axis=0)
    if np.any(dead):
        raise ValueError(f"trials {list(np.flatnonzero(dead))} have zero variance")
    trial_sr = X.mean(axis=0) / sd  # per observation, one per trial

    rho = average_correlation(X)
    n_eff = effective_num_trials(n_trials_raw, rho)

    best = int(np.argmax(trial_sr)) if selected is None else int(selected)
    if not 0 <= best < n_trials_raw:
        raise ValueError(f"selected={selected} is outside 0..{n_trials_raw - 1}")

    stats = SharpeStats.from_returns(X[:, best], periods_per_year)
    trials = TrialSet(n_eff, float(trial_sr.var()), periods_per_year, mean_sr=0.0)

    return AuditResult(
        n_trials_raw=n_trials_raw,
        avg_correlation=float(rho),
        n_effective_trials=float(n_eff),
        trial_sr_mean=float(trial_sr.mean()),
        trial_sr_var=float(trial_sr.var()),
        best_trial=best,
        stats=stats,
        trials=trials,
        threshold_sr=expected_max_sharpe(n_eff, trials.var_sr, trials.mean_sr),
        dsr=deflated_sharpe_ratio(stats, trials),
        cscv_result=(
            cscv(X, n_splits=n_splits, periods_per_year=periods_per_year)
            if run_cscv
            else None
        ),
    )