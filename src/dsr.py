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

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from .psr import probabilistic_sharpe_ratio

EULER_MASCHERONI = 0.5772156649015329

# Below this, Appendix 2 (pp. 18-19) shows the approximation drifting, and the
# proof assumes N >> 1. Measured against the exact order statistic, Eq. (1)
# overestimates by +0.036 at N=10, +0.023 at N=100 and +0.014 at N=1000.
SMALL_N_TRIALS = 50

# Below this, the exact order statistic is integrated instead of applying
# Eq. (1). Not the crossover itself: the sign change is measured at N = 2.7752,
# and by N = 1.2836 Eq. (1) has gone negative outright, which would put the
# expected maximum below the mean. Rounded up to a whole number because the
# integral is exact on both sides, so this constant only decides where to stop
# using the cheaper approximation - a margin above the measured crossover costs
# one quadrature call and nothing else.
EXACT_BRANCH_BELOW = 3.0


def _exact_expected_max_z(n_trials):
    """E[max] of n_trials standard Normals, by direct integration.

    The density of the maximum is n*phi(z)*Phi(z)^(n-1), which integrates to 1
    for any real n > 0, so this is the continuous extension of the order
    statistic and is exact at integers. Checked against the closed forms
    1/sqrt(pi) at n=2 and 3/(2*sqrt(pi)) at n=3, to 1e-9. Not against Monte
    Carlo: at 300k replications the sampling error is around 0.0016, the same
    size as the quantity being verified.

    There is no ground truth at fractional n - you cannot take the maximum of
    1.1 draws - and this is not the only defensible interpolation. Linear
    interpolation between the integer values runs lower (0.057 against 0.085 at
    n=1.1). Both are far above the zero that Eq. (1) implies there.
    """
    if n_trials == 1:
        return 0.0
    integrand = lambda z: z * n_trials * norm.pdf(z) * norm.cdf(z) ** (n_trials - 1)
    return quad(integrand, -12, 12, limit=200)[0]


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

    Two regimes, because Eq. (1) is not uniformly conservative:

    Above the measured crossover at N = 2.7752, Eq. (1) overestimates - by
    +0.036 at N=10, +0.023 at N=100 - which raises the hurdle and lowers DSR.
    That direction is safe, so Eq. (1) is used there, and the paper's published
    figures reproduce exactly. Between `EXACT_BRANCH_BELOW` and
    `SMALL_N_TRIALS` this warns, since the error is largest in that band, but
    it warns rather than refusing: effective counts under 50 are common once a
    correlated sweep is collapsed, and the error runs in the safe direction.

    Below the crossover the sign flips. At N=2 Eq. (1) UNDERSTATES by 0.044,
    and below N=1.2836 it returns a negative correction, implying an expected
    maximum below the mean. Understating the threshold raises DSR and makes the
    tool permissive in exactly the regime it exists to police, so the exact
    order statistic is integrated instead. This region is reachable, not
    theoretical: M=100 at rho=0.98 gives N = 2.98.

    The branch actually switches at `EXACT_BRANCH_BELOW` = 3.0 rather than at
    2.7752, so the narrow band between them uses the exact form where Eq. (1)
    would have been safe too. The two disagree by 0.0065 there, an order of
    magnitude below Eq. (1)'s own error at any N, so the seam is left
    unsmoothed.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")
    if var_trial_sr < 0:
        raise ValueError(f"var_trial_sr must be non-negative, got {var_trial_sr}")

    # Only warn on the branch the warning is about. Below the crossover the
    # exact integral is used, so nothing is approximated and nothing
    # overestimates - the old message told the caller the opposite of what
    # happened, in the very case cited as motivation (M=100 at rho=0.98 gives
    # N=2.98).
    if EXACT_BRANCH_BELOW <= n_trials < SMALL_N_TRIALS:
        warnings.warn(
            f"n_trials={n_trials} is below {SMALL_N_TRIALS}, where Eq. (1) is "
            f"least accurate (Appendix 2, pp. 18-19). It overestimates, so the "
            f"threshold is conservative.",
            stacklevel=2,
        )

    if n_trials < EXACT_BRANCH_BELOW:
        z = _exact_expected_max_z(n_trials)
    else:
        z = (1 - EULER_MASCHERONI) * norm.ppf(1 - 1 / n_trials) + (
            EULER_MASCHERONI * norm.ppf(1 - 1 / (n_trials * math.e))
        )
    return mean_trial_sr + math.sqrt(var_trial_sr) * z


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