"""Probabilistic Sharpe Ratio and Minimum Track Record Length.

Bailey & Lopez de Prado (2012a), used as the second half of the Deflated
Sharpe Ratio in Bailey & Lopez de Prado (2014), Eq. (2), p. 8.

Units: everything is PER OBSERVATION. Build stats through `from_returns` or
`from_annualized` so the frequency is declared at the boundary. An annualised
Sharpe paired with T in days is silently wrong and no downstream check can see
it, which is why the plain constructor guards against it.

Kurtosis is non-excess (Normal = 3.0), matching the (gamma_4 - 1)/4 term.
scipy defaults to Fisher, so `from_returns` passes fisher=False.

All four moments are the plain population estimators (ddof=0, bias=True). One
convention, chosen because gamma_3 and gamma_4 in Eq. (2) are population
moments; the only sample correction in the statistic is the sqrt(T - 1), and
putting a second one in sigma would be inconsistent with the derivation. At
any T where the difference is visible the statistic is not usable anyway.

That choice is coupled to the Pearson guard below and is not free. Biased
sample moments satisfy kurtosis >= skew^2 + 1 identically, so `from_returns`
can never trip the guard; the equality is exact for n=2 samples, which is why
the guard carries a 1e-9 tolerance rather than none. Unbiased moments do NOT
satisfy it: over 20,000 small fat-tailed samples the slack reached -4.0, so
mixing the conventions would make the guard reject real data.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import norm
from scipy.stats import skew as _skew


@dataclass(frozen=True)
class SharpeStats:
    """A track record reduced to what PSR needs. `sr` is per observation."""

    sr: float
    n_obs: float
    periods_per_year: float
    skew: float = 0.0
    kurtosis: float = 3.0

    def __post_init__(self):
        if not self.n_obs > 1:
            raise ValueError(f"n_obs must exceed 1, got {self.n_obs}")
        # Pearson's inequality, the general form. `kurtosis < 1` is only this
        # at zero skew, and lets (skew=-3, kurtosis=2) through to a PSR of 1.0.
        if self.kurtosis < self.skew**2 + 1 - 1e-9:
            raise ValueError(
                f"kurtosis {self.kurtosis} is below skew^2 + 1 = "
                f"{self.skew**2 + 1}; no distribution has these moments. "
                f"Kurtosis here is non-excess, so a Normal is 3.0 - add 3 if "
                f"you passed an excess kurtosis."
            )
        # A per-period SR above 1.0 is 15.8 annualized at daily frequency.
        # Skipped at ppy=1, where 2.5 is legitimate.
        if self.periods_per_year > 1 and abs(self.sr) > 1:
            raise ValueError(
                f"sr={self.sr} looks annualized: at {self.periods_per_year} "
                f"periods/year that is {self.annualized_sr:.1f} annualized. "
                f"Use SharpeStats.from_annualized(...)."
            )

    @classmethod
    def from_annualized(
        cls, sr_annual, periods_per_year, n_obs, skew=0.0, kurtosis=3.0
    ):
        sr = sr_annual / math.sqrt(periods_per_year)
        return cls(sr, n_obs, periods_per_year, skew, kurtosis)

    @classmethod
    def from_returns(cls, returns, periods_per_year):
        """Moments from a return series, assumed already in excess of cash."""
        r = np.asarray(returns, dtype=float).ravel()
        if r.size < 2 or not np.all(np.isfinite(r)):
            raise ValueError("need at least 2 finite observations")
        sd = r.std()
        # Relative, not `sd <= 0`: a constant series leaves a ~1e-19 residue
        # rather than an exact zero, which sails through an absolute check and
        # gives a Sharpe ratio of order 1e15.
        if sd <= 1e-12 * np.abs(r).max():
            raise ValueError("returns have zero variance; Sharpe ratio undefined")
        return cls(
            r.mean() / sd,
            r.size,
            periods_per_year,
            float(_skew(r, bias=True)),
            float(_kurtosis(r, fisher=False, bias=True)),
        )

    @property
    def annualized_sr(self):
        return self.sr * math.sqrt(self.periods_per_year)


def _se_factor(sr, skew, kurtosis):
    """sqrt(1 - g3*SR + (g4-1)/4 * SR^2), the Eq. (2) denominator.

    Collapses to sqrt(1 + SR^2/2) under Normality, which is Lo (2002).
    Negative skew makes -skew*sr positive for a profitable strategy, inflating
    the standard error and shrinking PSR. That direction is the point.
    """
    radicand = 1 - skew * sr + (kurtosis - 1) / 4 * sr**2
    # As a quadratic in sr this is minimised at sr* = 2*g3/(g4 - 1), where it
    # equals (g4 - 1 - g3^2)/(g4 - 1) - the Pearson slack. So once the
    # constructor enforces Pearson it can never go strictly negative. It can
    # still hit exactly zero, for moments sitting ON the boundary evaluated at
    # sr*: the paper's own g3=-3, g4=10 does this at sr=-2/3. That is a real
    # division by zero, not an impossible-moments case, hence the guard stays.
    if radicand <= 0:
        raise ValueError(
            f"standard error vanishes: skew={skew}, kurtosis={kurtosis} sit on "
            f"the Pearson boundary and sr={sr} is exactly 2*skew/(kurtosis-1), "
            f"where the non-Normality correction is degenerate"
        )
    return math.sqrt(radicand)


def probabilistic_sharpe_ratio(stats, benchmark_sr=0.0):
    """P(true SR > benchmark_sr). `benchmark_sr` is PER OBSERVATION.

    Zero asks whether the strategy beats nothing at all. Pass
    `dsr.expected_max_sharpe` instead and this returns the Deflated Sharpe
    Ratio; that substitution is the whole difference between the two.
    """
    z = (stats.sr - benchmark_sr) * math.sqrt(stats.n_obs - 1)
    return float(norm.cdf(z / _se_factor(stats.sr, stats.skew, stats.kurtosis)))


def min_track_record_length(stats, benchmark_sr=0.0, confidence=0.95):
    """Observations needed before PSR reaches `confidence`. Eq. (2) solved for T.

    Takes a `SharpeStats` rather than loose floats so units are established in
    exactly one place. On loose floats this function had no frequency guard at
    all: `min_track_record_length(2.5)` returned 2.79, quietly implying three
    days of data from an annualised Sharpe.

    `stats.n_obs` is ignored - this returns the T you would need, not the T you
    have. Returns a float; ceil it for whole periods. Undefined unless
    `stats.sr` exceeds `benchmark_sr`, since more data never rescues a strategy
    that does not clear the threshold.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    if stats.sr <= benchmark_sr:
        raise ValueError(
            f"sr ({stats.sr}) must exceed benchmark_sr ({benchmark_sr})"
        )
    edge = (stats.sr - benchmark_sr) / _se_factor(
        stats.sr, stats.skew, stats.kurtosis
    )
    return 1 + (norm.ppf(confidence) / edge) ** 2