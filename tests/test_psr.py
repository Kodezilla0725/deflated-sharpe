"""Tests for src.psr. Lo collapse first: it forces the signature to exist."""

import math

import dataclasses

import numpy as np
import pytest
from scipy.stats import norm

from src.psr import SharpeStats, min_track_record_length, probabilistic_sharpe_ratio

# Bailey & Lopez de Prado (2014) numerical example, pp. 9-10. SR0 values come
# from Eq. (1) with V[{SR_n}] = 1/2 annualised, hardcoded so this file has no
# dependency on dsr.py. test_dsr.py asserts dsr.expected_max_sharpe reproduces
# them, which is what keeps them from silently drifting.
PAPER = dict(sr_annual=2.5, periods_per_year=250, n_obs=1250, skew=-3.0, kurtosis=10.0)
SR0_N100, SR0_N46, SR0_N88_NORMAL = 0.11317200, 0.10036237, 0.11114576


def stats(sr=0.1, n_obs=1250, ppy=250, skew=0.0, kurtosis=3.0):
    return SharpeStats(sr, n_obs, ppy, skew, kurtosis)


# --- Lo collapse ----------------------------------------------------------- #


@pytest.mark.parametrize("sr", [0.02, 0.1, 0.3])
@pytest.mark.parametrize("n_obs", [30, 1250])
def test_lo_collapse(sr, n_obs):
    """gamma3=0, gamma4=3 must reproduce the sqrt(1 + SR^2/2) denominator."""
    expected = norm.cdf(sr * math.sqrt(n_obs - 1) / math.sqrt(1 + sr**2 / 2))
    assert probabilistic_sharpe_ratio(stats(sr, n_obs)) == pytest.approx(
        expected, rel=1e-12
    )


def test_lo_collapse_with_benchmark():
    expected = norm.cdf(0.1 * math.sqrt(1249) / math.sqrt(1 + 0.15**2 / 2))
    got = probabilistic_sharpe_ratio(stats(sr=0.15), benchmark_sr=0.05)
    assert got == pytest.approx(expected, rel=1e-12)


# --- the paper's numbers --------------------------------------------------- #


def test_paper_example():
    s = SharpeStats.from_annualized(**PAPER)
    assert probabilistic_sharpe_ratio(s, SR0_N100) == pytest.approx(0.90, abs=5e-4)
    assert probabilistic_sharpe_ratio(s, SR0_N46) == pytest.approx(0.9505, abs=5e-4)
    # Same returns, zero benchmark: the naive question looks unimpeachable.
    assert probabilistic_sharpe_ratio(s) > 0.999


def test_paper_example_under_normality():
    """Normal returns buy tolerance for 88 trials instead of 46."""
    s = SharpeStats.from_annualized(2.5, 250, 1250)
    assert probabilistic_sharpe_ratio(s, SR0_N88_NORMAL) == pytest.approx(
        0.9505, abs=5e-4
    )


# --- direction of the corrections ------------------------------------------ #


def test_moment_corrections_move_the_right_way():
    base = probabilistic_sharpe_ratio(stats())
    assert probabilistic_sharpe_ratio(stats(skew=-1.2)) < base
    assert probabilistic_sharpe_ratio(stats(skew=1.2)) > base
    assert probabilistic_sharpe_ratio(stats(kurtosis=9.0)) < base
    assert probabilistic_sharpe_ratio(stats(n_obs=2500)) > base


# --- min track record length ----------------------------------------------- #


@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
@pytest.mark.parametrize("skew,kurt", [(0.0, 3.0), (-1.2, 9.0)])
def test_min_trl_roundtrips_through_psr(confidence, skew, kurt):
    """Same formula solved for T, so it must invert exactly."""
    s = stats(sr=0.1, skew=skew, kurtosis=kurt)
    t = min_track_record_length(s, confidence=confidence)
    at_t = dataclasses.replace(s, n_obs=t)
    assert probabilistic_sharpe_ratio(at_t) == pytest.approx(confidence, rel=1e-9)


def test_min_trl_grows_with_negative_skew():
    assert min_track_record_length(stats(skew=-1.2)) > min_track_record_length(stats())


def test_min_trl_rejects_sr_below_benchmark():
    with pytest.raises(ValueError, match="benchmark"):
        min_track_record_length(stats(sr=0.05), benchmark_sr=0.05)


def test_min_trl_cannot_be_reached_with_an_annualised_sharpe():
    """Loose floats let min_track_record_length(2.5) return 2.79 observations."""
    with pytest.raises(ValueError, match="annualized"):
        min_track_record_length(SharpeStats(2.5, 1250, 250))


# --- frequency and moment conventions -------------------------------------- #


def test_annualised_sr_passed_as_per_period_is_rejected():
    """The bug the SharpeStats guard exists to prevent."""
    with pytest.raises(ValueError, match="annualized"):
        SharpeStats(2.5, 1250, 250)


def test_guard_skipped_for_annual_data():
    """At ppy=1 a per-period SR of 2.5 is legitimate, so do not cry wolf."""
    assert SharpeStats(2.5, 30, 1).annualized_sr == pytest.approx(2.5)


def test_from_annualized_stores_per_period():
    s = SharpeStats.from_annualized(**PAPER)
    assert s.sr == pytest.approx(2.5 / math.sqrt(250))
    assert s.annualized_sr == pytest.approx(2.5)


def test_from_returns_kurtosis_is_non_excess_and_sharpe_is_right():
    """scipy defaults to Fisher. The paper is Pearson: Normal is 3, not 0."""
    r = np.random.default_rng(0).normal(0.0005, 0.01, 200_000)
    s = SharpeStats.from_returns(r, 250)
    assert s.kurtosis == pytest.approx(3.0, abs=0.1)
    assert s.skew == pytest.approx(0.0, abs=0.05)
    assert s.sr == pytest.approx(0.05, rel=0.05)


def test_excess_kurtosis_passed_by_mistake_is_rejected():
    with pytest.raises(ValueError, match="excess"):
        stats(kurtosis=0.0)


def test_pearson_inequality_general_form():
    """`kurtosis < 1` is Pearson at zero skew only, and let this through."""
    with pytest.raises(ValueError, match="skew"):
        stats(sr=0.5, skew=-3.0, kurtosis=2.0)


def test_moments_on_the_pearson_boundary_still_build():
    """The paper's own skew=-3, kurtosis=10 sits exactly on skew^2 + 1."""
    assert stats(skew=-3.0, kurtosis=10.0).kurtosis == 10.0
    assert stats(skew=-1.2, kurtosis=2.44).kurtosis == 2.44


def test_moments_strictly_inside_the_pearson_boundary_build():
    assert stats(skew=-1.2, kurtosis=9.0).kurtosis == 9.0
    assert stats(skew=0.5, kurtosis=4.0).kurtosis == 4.0


def test_real_data_never_trips_the_pearson_guard():
    """Biased moments satisfy the inequality identically, which is why the
    ddof=0 / bias=True convention and this guard cannot be chosen separately."""
    # ppy=1 to isolate the Pearson guard: at 3 observations the sample Sharpe
    # is legitimately huge and would trip the frequency guard first.
    rng = np.random.default_rng(7)
    for _ in range(300):
        r = rng.standard_t(2.5, rng.integers(3, 40))
        SharpeStats.from_returns(r, 1)


def test_unbiased_moments_would_break_the_pearson_guard():
    """The other half of the coupling, which is why the convention is not free.

    Biased moments satisfy kurtosis >= skew^2 + 1 identically. Unbiased ones do
    not: on small fat-tailed samples the slack goes several units negative, so
    mixing conventions would make the guard reject ordinary data.
    """
    from scipy.stats import kurtosis, skew

    rng = np.random.default_rng(11)
    worst_biased = worst_unbiased = np.inf
    for _ in range(5000):
        r = rng.standard_t(2.5, rng.integers(4, 12))
        for bias, current in ((True, "b"), (False, "u")):
            g3 = skew(r, bias=bias)
            g4 = kurtosis(r, fisher=False, bias=bias)
            slack = g4 - g3**2 - 1
            if current == "b":
                worst_biased = min(worst_biased, slack)
            else:
                worst_unbiased = min(worst_unbiased, slack)
    assert worst_biased > -1e-9, "biased moments satisfy Pearson identically"
    assert worst_unbiased < -1.0, "unbiased moments violate it substantially"


def test_degenerate_standard_error_is_caught():
    """Boundary moments at sr = 2*skew/(kurtosis-1) zero the radicand."""
    with pytest.raises(ValueError, match="vanishes"):
        probabilistic_sharpe_ratio(stats(sr=-2 / 3, skew=-3.0, kurtosis=10.0))


def test_from_returns_rejects_zero_variance():
    with pytest.raises(ValueError, match="variance"):
        SharpeStats.from_returns(np.full(100, 0.001), 250)


def test_rejects_degenerate_sample_length():
    with pytest.raises(ValueError, match="n_obs"):
        stats(n_obs=1)