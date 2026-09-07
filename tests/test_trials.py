"""Tests for src.trials."""

import warnings

import numpy as np
import pytest

from src.trials import average_correlation, effective_num_trials


def equicorrelated(n_obs, n_cols, rho, seed=0):
    """T x M returns with a known constant pairwise correlation."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n_obs, 1))
    e = rng.standard_normal((n_obs, n_cols))
    return np.sqrt(rho) * f + np.sqrt(1 - rho) * e


def quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


# --- Eq. (9) endpoints ----------------------------------------------------- #


def test_perfect_correlation_collapses_to_one_trial():
    assert effective_num_trials(400, 1.0) == 1.0


def test_zero_correlation_leaves_m_untouched():
    assert effective_num_trials(400, 0.0) == 400.0


def test_interpolates_linearly_between_the_limits():
    assert effective_num_trials(400, 0.8) == pytest.approx(0.8 + 0.2 * 400)


def test_exhibit_4_spot_check():
    """M=20,000 at rho=0.75 lands in the 0-5000 band of Exhibit 4, p. 20."""
    assert 4000 < effective_num_trials(20_000, 0.75) < 5100


def test_returns_a_float_without_rounding():
    n = effective_num_trials(400, 0.8)
    assert isinstance(n, float) and n != int(n)


def test_clamped_to_the_number_of_trials_actually_run():
    """A slightly negative sample rho must not imply more trials than exist."""
    assert effective_num_trials(400, -0.002) == 400.0


def test_rejects_impossible_inputs():
    with pytest.raises(ValueError, match="avg_corr"):
        effective_num_trials(400, 1.5)
    with pytest.raises(ValueError, match="n_trials_raw"):
        effective_num_trials(0, 0.5)


# --- Eq. (8) --------------------------------------------------------------- #


def test_matches_numpy_corrcoef():
    """The O(TM) form must equal the explicit off-diagonal mean exactly."""
    X = equicorrelated(500, 40, 0.6)
    C = np.corrcoef(X, rowvar=False)
    m = C.shape[0]
    expected = (C.sum() - m) / (m * (m - 1))
    assert average_correlation(X) == pytest.approx(expected, rel=1e-12)


def test_recovers_a_known_correlation():
    assert average_correlation(equicorrelated(2000, 50, 0.8)) == pytest.approx(
        0.8, abs=0.02
    )


def test_independent_trials_show_near_zero_correlation():
    rng = np.random.default_rng(3)
    assert average_correlation(rng.standard_normal((2000, 50))) == pytest.approx(
        0.0, abs=0.02
    )


def test_scale_guard_is_per_column():
    """Same bug as pbo.sharpe_columns had: a global max rejects real columns."""
    rng = np.random.default_rng(0)
    X = np.column_stack(
        [rng.standard_normal(300) * 1e6, rng.standard_normal(300) * 1e-9]
    )
    assert abs(average_correlation(X)) < 0.2


def test_rejects_a_constant_trial():
    X = equicorrelated(200, 5, 0.5)
    X[:, 2] = 0.01
    with pytest.raises(ValueError, match="zero variance"):
        average_correlation(X)


def test_rejects_degenerate_shapes():
    with pytest.raises(ValueError, match="2 trials"):
        average_correlation(np.zeros((100, 1)))
    with pytest.raises(ValueError, match="2-D"):
        average_correlation(np.zeros(100))


# --- M > T: the evidence for warning rather than refusing ------------------ #


def test_stays_unbiased_when_m_far_exceeds_t():
    """M/T = 33, rank-deficient by a factor of 33, and rho still lands.

    This is why `average_correlation` warns instead of refusing. The paper's
    ill-conditioning concern (p. 15) bites methods that invert or decompose C;
    an equal-weighted average does neither.
    """
    got = quiet(average_correlation, equicorrelated(60, 2000, 0.8))
    assert got == pytest.approx(0.8, abs=0.05)


def test_warns_when_m_exceeds_t():
    with pytest.warns(UserWarning, match="exceeds"):
        average_correlation(equicorrelated(50, 200, 0.5))


def test_no_warning_when_t_exceeds_m():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        average_correlation(equicorrelated(500, 50, 0.5))


def test_short_samples_widen_the_estimate():
    """Bias holds, dispersion does not. That is what the warning names."""
    spread = {}
    for n_obs in (1250, 60):
        vals = [
            quiet(average_correlation, equicorrelated(n_obs, 200, 0.8, seed=s))
            for s in range(12)
        ]
        spread[n_obs] = np.std(vals)
    assert spread[60] > 3 * spread[1250]


# --- the pipeline ---------------------------------------------------------- #


def test_end_to_end_400_trials_at_80_percent_correlation():
    """400 backtests, rho 0.8, collapse to roughly 81 independent trials."""
    X = equicorrelated(1250, 400, 0.8)
    n = effective_num_trials(400, average_correlation(X))
    assert n == pytest.approx(81, abs=3)