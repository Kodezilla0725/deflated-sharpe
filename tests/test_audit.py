"""Tests for src.audit."""

import warnings

import numpy as np
import pytest

from src.audit import audit
from src.dsr import TrialSet, deflated_sharpe_ratio, expected_max_sharpe
from src.trials import average_correlation, effective_num_trials

PPY = 250


def redundant_noise(n_obs=480, n_trials=60, rho=0.8, seed=0, edge=0.0, on=None):
    """Correlated trials with no edge, or one column given a real one."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n_obs, 1))
    e = rng.standard_normal((n_obs, n_trials))
    X = (np.sqrt(rho) * f + np.sqrt(1 - rho) * e) * 0.01
    if edge:
        X[:, on] += edge
    return X


def quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


# --- the two verdicts agree ------------------------------------------------ #


def test_both_are_calibrated_under_the_null():
    """With no real edge, DSR is a p-value and should be roughly uniform, and
    the IS winner's OOS rank should be roughly uniform too. Both means land on
    0.5.

    Averaged over seeds because a single draw says almost nothing: across 8
    seeds DSR ranges from 0.01 to 0.89 and PBO from 0.37 to 0.81. Reading
    either as a verdict on one dataset is the mistake this test exists to
    document.
    """
    verdicts = [
        quiet(audit, redundant_noise(seed=s), PPY, n_splits=10) for s in range(8)
    ]
    assert np.mean([v.dsr for v in verdicts]) == pytest.approx(0.5, abs=0.2)
    assert np.mean([v.pbo for v in verdicts]) == pytest.approx(0.5, abs=0.2)
    assert np.ptp([v.dsr for v in verdicts]) > 0.3, "one draw is not a verdict"


def test_both_accept_a_genuine_edge():
    r = quiet(audit, redundant_noise(edge=0.004, on=7), PPY, n_splits=10)
    assert r.dsr > 0.95
    assert r.pbo < 0.05
    assert r.best_trial == 7


def test_the_verdicts_move_together_across_edge_sizes():
    dsrs, pbos = [], []
    for edge in (0.0, 0.002, 0.004):
        r = quiet(audit, redundant_noise(edge=edge, on=7), PPY, n_splits=10)
        dsrs.append(r.dsr)
        pbos.append(r.pbo)
    assert dsrs == sorted(dsrs), "DSR rises with a real edge"
    assert pbos == sorted(pbos, reverse=True), "PBO falls with a real edge"


# --- each reachable independently ------------------------------------------ #


def test_dsr_half_runs_without_cscv():
    r = quiet(audit, redundant_noise(), PPY, run_cscv=False)
    assert 0 < r.dsr < 1
    assert r.cscv_result is None
    with pytest.raises(AttributeError):
        _ = r.pbo


def test_cscv_half_is_reachable_on_its_own():
    """S=16 here specifically, to pin the combination count end to end."""
    r = quiet(audit, redundant_noise(n_obs=480, n_trials=30), PPY, n_splits=16)
    assert 0 <= r.pbo <= 1
    assert 0 <= r.prob_loss <= 1
    assert r.cscv_result.n_combinations == 12870


# --- wiring ---------------------------------------------------------------- #


def test_matches_the_modules_called_by_hand():
    """audit is composition, not reimplementation."""
    X = redundant_noise()
    r = quiet(audit, X, PPY, run_cscv=False)

    rho = average_correlation(X)
    n = effective_num_trials(X.shape[1], rho)
    srs = X.mean(0) / X.std(0)
    trials = TrialSet(n, float(srs.var()), PPY, mean_sr=0.0)
    by_hand_threshold = quiet(expected_max_sharpe, n, trials.var_sr, 0.0)

    assert r.avg_correlation == pytest.approx(rho)
    assert r.n_effective_trials == pytest.approx(n)
    assert r.best_trial == int(np.argmax(srs))
    assert r.threshold_sr == pytest.approx(by_hand_threshold)
    assert r.dsr == pytest.approx(quiet(deflated_sharpe_ratio, r.stats, trials))


def test_threshold_uses_a_null_trial_mean_not_the_observed_one():
    """DSR tests H0: SR = 0, so the hurdle is the expected max under that null.

    The observed mean is still reported, since the class-relative question is a
    reasonable one to ask separately.
    """
    r = quiet(audit, redundant_noise(edge=0.004, on=7), PPY, run_cscv=False)
    assert r.trials.mean_sr == 0.0
    assert r.trial_sr_mean != 0.0
    assert r.threshold_sr == pytest.approx(
        quiet(expected_max_sharpe, r.n_effective_trials, r.trials.var_sr, 0.0)
    )


def test_frequency_propagates_to_both_objects():
    r = quiet(audit, redundant_noise(), 12, run_cscv=False)
    assert r.stats.periods_per_year == 12
    assert r.trials.periods_per_year == 12
    assert r.periods_per_year == 12


def test_redundancy_collapses_the_trial_count():
    r = quiet(audit, redundant_noise(n_trials=200, rho=0.8), PPY, run_cscv=False)
    assert r.n_trials_raw == 200
    assert 20 < r.n_effective_trials < 70


def test_selected_overrides_the_in_sample_best():
    X = redundant_noise(edge=0.004, on=7)
    r = quiet(audit, X, PPY, selected=3, run_cscv=False)
    assert r.best_trial == 3
    assert r.dsr < quiet(audit, X, PPY, run_cscv=False).dsr


def test_summary_reports_the_inputs_alongside_the_verdicts():
    """Reporting DSR without N and the dispersion repeats the paper's own
    complaint about backtests published without their trial counts."""
    text = quiet(audit, redundant_noise(n_trials=30), PPY, n_splits=8).summary()
    for line in ("trials run", "average correlation", "effective independent"):
        assert line in text
    assert "Deflated Sharpe Ratio" in text and "Probability of Overfitting" in text


# --- validation ------------------------------------------------------------ #


def test_rejects_a_single_trial():
    with pytest.raises(ValueError, match="at least 2 trials"):
        audit(np.random.default_rng(0).standard_normal((100, 1)), PPY)


def test_rejects_wrong_shape():
    with pytest.raises(ValueError, match="2-D"):
        audit(np.zeros(100), PPY)


def test_rejects_a_constant_trial():
    X = redundant_noise(n_trials=20)
    X[:, 4] = 0.001
    with pytest.raises(ValueError, match="zero variance"):
        audit(X, PPY, run_cscv=False)


def test_rejects_out_of_range_selection():
    with pytest.raises(ValueError, match="outside"):
        audit(redundant_noise(n_trials=20), PPY, selected=99, run_cscv=False)