"""Expected maximum Sharpe ratio under multiple testing, and the DSR.

Bailey & Lopez de Prado (2014), Eq. (1) p. 7, derived in Appendix 1 p. 12;
Eq. (2) p. 8.

The threshold is a property of the search, not of the winner: it depends only
on the number of independent trials and the dispersion of their Sharpe ratios.
Nothing about the selected track record enters it.
"""

import math
import warnings
from dataclasses import dataclass

from scipy.stats import norm

from .psr import probabilistic_sharpe_ratio

EULER_MASCHERONI = 0.5772156649015329

# Below this, Appendix 2 (pp. 18-19) shows the approximation drifting: it
# overestimates the maximum by up to ~0.05 for a variance-1 process, falling to
# ~0.006 by 1000 trials. The proof assumes N >> 1.
SMALL_N_TRIALS = 50

# Eq. (1) returns a NEGATIVE correction below N = 1.2836, which would put the
# expected maximum below the mean. Impossible, so it is floored.
_GUMBEL_FLOOR_N = 1.2836


@dataclass(frozen=True)
class TrialSet:
    """The search that produced a strategy: how many independent trials, and
    how widely their Sharpe ratios were spread.

    Exists for the same reason `SharpeStats` does. `var_sr` used to arrive as a
    bare float guarded by a plausibility threshold, and that could not work: the
    error being detected is a factor of `periods_per_year`, so its size depends
    on the frequency, while any absolute threshold does not. At 250 periods a
    year a mistaken annualized 0.5 implies an annualized dispersion of 11.2; at
    12 periods it implies 2.4, indistinguishable from a merely wide sweep. No
    constant separates those. So units are declared at construction instead,
    and `deflated_sharpe_ratio` checks that the two declarations agree.

    Attributes
    ----------
    n_trials:
        Number of INDEPENDENT trials, from `trials.effective_num_trials`. May
        be fractional; it is an interpolation, not a count. Passing a raw
        backtest count overstates the threshold and rejects sound strategies.
    var_sr, mean_sr:
        Cross-sectional variance and mean of the trial Sharpe ratios, PER
        OBSERVATION. Under the null `mean_sr` is zero.
    """

    n_trials: float
    var_sr: float
    periods_per_year: float
    mean_sr: float = 0.0

    def __post_init__(self):
        if self.n_trials < 1:
            raise ValueError(f"n_trials must be at least 1, got {self.n_trials}")
        if self.var_sr < 0:
            raise ValueError(f"var_sr must be non-negative, got {self.var_sr}")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")

    @classmethod
    def from_annualized(
        cls, n_trials, var_sr_annual, periods_per_year, mean_sr_annual=0.0
    ):
        """Variance scales with periods_per_year; the mean with its square root."""
        return cls(
            n_trials,
            var_sr_annual / periods_per_year,
            periods_per_year,
            mean_sr_annual / math.sqrt(periods_per_year),
        )

    @classmethod
    def from_trial_sharpes(cls, sharpes, periods_per_year, n_trials):
        """Dispersion from the per-observation Sharpe ratios of every trial run.

        `sharpes` is all M of them; `n_trials` is the effective count. The two
        are different numbers and both are needed: dispersion comes from the
        whole sweep, the threshold from its independent content.
        """
        import numpy as np

        s = np.asarray(sharpes, dtype=float).ravel()
        if s.size < 2:
            raise ValueError("need at least 2 trial Sharpe ratios")
        return cls(n_trials, float(s.var()), periods_per_year, float(s.mean()))

    @property
    def annualized_sr_std(self):
        return math.sqrt(self.var_sr * self.periods_per_year)


def expected_max_sharpe(n_trials, var_trial_sr, mean_trial_sr=0.0):
    """Eq. (1): the Sharpe ratio you expect from the best of `n_trials`.

    Under the null (`mean_trial_sr=0`) this is the hurdle a selected strategy
    must clear to be evidence of anything, since the maximum of a set of
    estimates exceeds zero even when every true Sharpe ratio is zero.

    Deliberately unit-agnostic: a location-scale formula returns whatever units
    it is fed. `TrialSet` and `deflated_sharpe_ratio` pin units down; this is
    the bare arithmetic, and keeping it bare is what lets it be checked against
    the paper's own numbers directly.

    `n_trials` may be fractional, since `effective_num_trials` interpolates.

    Below `SMALL_N_TRIALS` this warns rather than refuses. The approximation
    errs by overestimating, which raises the hurdle and lowers DSR, so the
    failure is conservative - and effective counts under 50 are common once a
    correlated sweep is collapsed, so refusing would reject the ordinary case
    to guard against a mild error in the safe direction.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")
    if var_trial_sr < 0:
        raise ValueError(f"var_trial_sr must be non-negative, got {var_trial_sr}")

    if n_trials < SMALL_N_TRIALS and n_trials != 1:
        warnings.warn(
            f"n_trials={n_trials} is below {SMALL_N_TRIALS}, where Eq. (1) is "
            f"least accurate (Appendix 2, pp. 18-19). It overestimates, so the "
            f"threshold is conservative.",
            stacklevel=2,
        )

    if n_trials < _GUMBEL_FLOOR_N:
        # Exact at N=1, where the expected maximum of one draw is its mean.
        # Between 1 and 1.2836 the approximation is negative, so it is floored
        # at the mean rather than allowed to return an impossible value.
        return mean_trial_sr

    gumbel = (1 - EULER_MASCHERONI) * norm.ppf(1 - 1 / n_trials) + (
        EULER_MASCHERONI * norm.ppf(1 - 1 / (n_trials * math.e))
    )
    return mean_trial_sr + math.sqrt(var_trial_sr) * gumbel


def deflated_sharpe_ratio(stats, trials):
    """Eq. (2): PSR with the multiplicity-adjusted threshold substituted.

    Both arguments declare their own frequency, and this checks the two agree.
    That check is exact, which is the point: it replaces a magnitude heuristic
    that could not detect a scale error whose size varies with the frequency.
    """
    if not math.isclose(stats.periods_per_year, trials.periods_per_year):
        raise ValueError(
            f"frequency mismatch: stats is {stats.periods_per_year} periods per "
            f"year, trials is {trials.periods_per_year}. They must describe the "
            f"same clock."
        )
    threshold = expected_max_sharpe(trials.n_trials, trials.var_sr, trials.mean_sr)
    return probabilistic_sharpe_ratio(stats, benchmark_sr=threshold)