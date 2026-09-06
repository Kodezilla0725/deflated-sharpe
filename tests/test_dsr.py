"""Tests for src.dsr."""

import math
import warnings

import pytest

from src.dsr import (
    EULER_MASCHERONI,
    TrialSet,
    deflated_sharpe_ratio,
    expected_max_sharpe,
)
from src.psr import SharpeStats, probabilistic_sharpe_ratio

# Bailey & Lopez de Prado (2014), numerical example pp. 9-10.
PAPER = dict(sr_annual=2.5, periods_per_year=250, n_obs=1250, skew=-3.0, kurtosis=10.0)
PPY = 250
VAR_TRIAL_SR_ANNUAL = 0.5
VAR_TRIAL_SR = VAR_TRIAL_SR_ANNUAL / PPY  # per observation, as the module wants


def paper_stats(**kw):
    return SharpeStats.from_annualized(**{**PAPER, **kw})


def trial_set(n_trials, var=VAR_TRIAL_SR, ppy=PPY, mean_sr=0.0):
    return TrialSet(n_trials, var, ppy, mean_sr)


def quiet_emax(*a, **kw):
    """expected_max_sharpe with the small-N warning suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return expected_max_sharpe(*a, **kw)


def quiet_dsr(stats, n_trials, var=VAR_TRIAL_SR):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return deflated_sharpe_ratio(stats, trial_set(n_trials, var))


# --- cross-check: retires the hardcoded SR0 constants in test_psr.py ------- #


@pytest.mark.parametrize(
    "n_trials,expected_sr0",
    [(100, 0.11317200), (46, 0.10036237), (88, 0.11114576)],
)
def test_reproduces_the_thresholds_hardcoded_in_test_psr(n_trials, expected_sr0):
    assert quiet_emax(n_trials, VAR_TRIAL_SR) == pytest.approx(
        expected_sr0, abs=1e-8
    )


def test_paper_example_end_to_end():
    """The full pipeline, from trial count to the published DSR (p. 10)."""
    s = paper_stats()
    assert deflated_sharpe_ratio(s, trial_set(100)) == pytest.approx(0.90, abs=5e-4)
    assert quiet_dsr(s, 46) == pytest.approx(0.9505, abs=5e-4)


def test_normal_returns_buy_tolerance_for_more_trials():
    """Non-Normality alone roughly halves the tolerable search: 88 vs 46."""
    normal = SharpeStats.from_annualized(2.5, PPY, 1250)
    assert deflated_sharpe_ratio(normal, trial_set(88)) == pytest.approx(
        0.9505, abs=5e-4
    )
    assert quiet_dsr(paper_stats(), 46) == pytest.approx(0.9505, abs=5e-4)


# --- Eq. (1) structure ----------------------------------------------------- #


def test_gumbel_multiplier_matches_hand_calculation():
    """Strip the scaling: the bracket alone at N=100."""
    assert quiet_emax(100, 1.0) == pytest.approx(2.53060289, abs=1e-8)


def test_threshold_scales_with_sigma_not_variance():
    """Quadrupling the variance must double the hurdle, per Eq. (1)."""
    assert quiet_emax(100, 4.0) == pytest.approx(2 * quiet_emax(100, 1.0))


def test_threshold_grows_slowly_in_n_trials():
    """Tenfold more search raises the hurdle under 30%: the sqrt(2 ln N) rate."""
    ratio = quiet_emax(1000, 1.0) / quiet_emax(100, 1.0)
    assert 1.0 < ratio < 1.3


def test_threshold_is_monotone_in_n_trials():
    vals = [quiet_emax(n, 1.0) for n in (2, 10, 50, 100, 1000, 10_000)]
    assert vals == sorted(vals)


def test_mean_trial_sr_shifts_the_threshold():
    assert quiet_emax(100, 1.0, mean_trial_sr=0.5) == pytest.approx(
        quiet_emax(100, 1.0) + 0.5
    )


def test_zero_dispersion_gives_no_hurdle():
    """Identical trials carry no selection bias, however many you ran."""
    assert quiet_emax(1000, 0.0) == 0.0


def test_euler_mascheroni_constant():
    assert EULER_MASCHERONI == pytest.approx(0.5772156649, abs=1e-10)


# --- N = 1: DSR must collapse to PSR --------------------------------------- #


def test_single_trial_threshold_is_the_mean():
    """Expected max of one draw is its mean. Also dodges Z^-1(0) = -inf."""
    assert expected_max_sharpe(1, 0.002) == 0.0
    assert expected_max_sharpe(1, 0.002, mean_trial_sr=0.3) == 0.3


def test_dsr_collapses_to_psr_at_one_trial():
    s = paper_stats()
    assert deflated_sharpe_ratio(s, trial_set(1)) == probabilistic_sharpe_ratio(s)


# --- small N: warn, do not refuse ------------------------------------------ #


def test_warns_below_fifty_trials_but_still_returns():
    with pytest.warns(UserWarning, match="below 50"):
        got = expected_max_sharpe(20, 1.0)
    assert got > 0


def test_no_warning_at_or_above_fifty():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        expected_max_sharpe(50, 1.0)


def test_no_warning_at_one_trial():
    """N=1 is exact, not an approximation, so it must not warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        expected_max_sharpe(1, 1.0)


# --- units and validation -------------------------------------------------- #


def test_from_annualized_converts_variance_by_ppy():
    t = TrialSet.from_annualized(100, VAR_TRIAL_SR_ANNUAL, PPY)
    assert t.var_sr == pytest.approx(VAR_TRIAL_SR)
    assert t.annualized_sr_std == pytest.approx(math.sqrt(VAR_TRIAL_SR_ANNUAL))


def test_frequency_mismatch_is_rejected():
    """Exact check, replacing a magnitude heuristic that could not scale."""
    monthly_trials = TrialSet.from_annualized(100, VAR_TRIAL_SR_ANNUAL, 12)
    with pytest.raises(ValueError, match="frequency mismatch"):
        deflated_sharpe_ratio(paper_stats(), monthly_trials)


def test_monthly_annualised_variance_no_longer_slips_through():
    """The old absolute guard fired only at daily frequency, and barely.

    A monthly track record with an annualized variance passed by mistake
    returned DSR = 0.0 with no error - a silent, decisive-looking rejection.
    """
    monthly = SharpeStats.from_annualized(2.5, 12, 60, skew=-3.0, kurtosis=10.0)
    right = deflated_sharpe_ratio(monthly, TrialSet.from_annualized(100, 0.5, 12))
    wrong = deflated_sharpe_ratio(monthly, TrialSet(100, 0.5 / 12, 12))
    assert right == pytest.approx(wrong)
    assert 0 < right < 1


def test_correct_per_observation_var_passes():
    assert 0 < deflated_sharpe_ratio(paper_stats(), trial_set(100)) < 1


def test_accepts_fractional_trial_counts():
    """effective_num_trials interpolates, so N is not an integer."""
    assert quiet_emax(80.9, 1.0) == pytest.approx(quiet_emax(81, 1.0), abs=0.01)


def test_floors_the_negative_correction_near_one_trial():
    """Eq. (1) goes negative below N=1.2836; the max cannot be below the mean."""
    assert quiet_emax(1.1, 1.0, mean_trial_sr=0.3) == 0.3
    assert quiet_emax(1.5, 1.0) > 0


def test_rejects_non_positive_trial_counts():
    for bad in (0, -1, 0.5):
        with pytest.raises(ValueError, match="n_trials"):
            expected_max_sharpe(bad, 1.0)


def test_trialset_rejects_bad_inputs():
    with pytest.raises(ValueError, match="n_trials"):
        TrialSet(0, 0.002, 250)
    with pytest.raises(ValueError, match="var_sr"):
        TrialSet(100, -1.0, 250)


def test_rejects_negative_variance():
    with pytest.raises(ValueError, match="var_trial_sr"):
        expected_max_sharpe(100, -1.0)


# --- the point of the paper ------------------------------------------------ #


def test_more_search_lowers_confidence_in_the_same_track_record():
    s = paper_stats()
    dsrs = [quiet_dsr(s, n) for n in (1, 10, 46, 100, 1000)]
    assert dsrs == sorted(dsrs, reverse=True)
    # N=1 is the naive PSR; 1000 trials takes the same returns to ~0.64.
    assert dsrs[0] > 0.999 and dsrs[-1] < 0.7


def test_wider_sweep_lowers_confidence_at_fixed_trial_count():
    s = paper_stats()
    assert deflated_sharpe_ratio(s, trial_set(100, 4 * VAR_TRIAL_SR)) < (
        deflated_sharpe_ratio(s, trial_set(100))
    )