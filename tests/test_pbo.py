"""Tests for src.pbo."""

import warnings
from collections import Counter
from math import comb

import numpy as np
import pytest

from src.pbo import (
    CSCVResult,
    cscv,
    probability_of_backtest_overfitting,
    sharpe_columns,
)


def noise(n_obs=480, n_trials=50, seed=0, scale=0.01):
    """Informationless trials: no column has any edge over any other."""
    return np.random.default_rng(seed).standard_normal((n_obs, n_trials)) * scale


def with_one_skilled_column(n_obs=480, n_trials=50, seed=0, edge=0.004):
    X = noise(n_obs, n_trials, seed)
    X[:, 7] += edge
    return X


def quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


# --- combination count ----------------------------------------------------- #


@pytest.mark.parametrize("n_splits", [4, 6, 8, 12])
def test_number_of_combinations_is_the_binomial_coefficient(n_splits):
    r = cscv(noise(n_obs=n_splits * 40), n_splits=n_splits)
    assert r.n_combinations == comb(n_splits, n_splits // 2)


def test_s16_gives_12870_not_the_printed_12780():
    """The paper prints 12,780 twice (pp. 11, 22). C(16,8) is 12,870.

    Eq. (2.3)'s product form gives 12,870, and the paper's S=24 figure of
    2,704,156 is exactly C(24,12), so the 16-case is a transposed digit.
    """
    assert comb(16, 8) == 12870
    assert comb(24, 12) == 2_704_156


def test_figure_1_case():
    """S=4 gives the six combinations laid out in Figure 1, p. 13."""
    r = cscv(noise(n_obs=400, n_trials=20), n_splits=4)
    assert r.n_combinations == 6


def test_train_and_test_sets_are_complementary():
    """Symmetry: every training combination is reused as a testing set.

    That is what makes the split sets equal in size, so a performance decline
    can only come from overfitting rather than from unequal sample sizes.
    """
    from itertools import combinations

    s = 8
    combos = set(combinations(range(s), s // 2))
    for c in combos:
        assert tuple(sorted(set(range(s)) - set(c))) in combos


# --- the two baselines ----------------------------------------------------- #


def pbo_over_seeds(builder, n_seeds=12, n_splits=12, **kw):
    return np.array(
        [cscv(builder(seed=s, **kw), n_splits=n_splits).pbo for s in range(n_seeds)]
    )


def test_informationless_trials_give_pbo_near_one_half():
    """No column has an edge, so the IS winner's OOS rank is uniform.

    Averaged over seeds, because phi on any single dataset is far noisier than
    the paper implies - see test_phi_is_noisy_across_datasets below. The
    paper's own random-walk example lands at 0.55 (Figure 8, p. 28).
    """
    assert pbo_over_seeds(noise, n_trials=50).mean() == pytest.approx(0.5, abs=0.08)


def test_phi_is_noisy_across_datasets():
    """p. 22 puts the standard error of phi under 0.0045 at S=16.

    That comes from sqrt(p(1-p)/n) with n the logit count, which assumes the
    logits are independent. They are combinations of the same S blocks, so they
    are not. The measured spread is roughly thirty times the claimed one.
    """
    assert pbo_over_seeds(noise, n_trials=50).std() > 0.05


def test_more_combinations_barely_reduce_the_noise():
    """The tell for dependence: 14x the logits, essentially the same spread.

    Independent draws would shrink the standard error by sqrt(14) = 3.7.
    """
    coarse = pbo_over_seeds(noise, n_splits=8, n_trials=50).std()
    fine = pbo_over_seeds(noise, n_splits=12, n_trials=50).std()
    assert fine > coarse / 2


def test_genuine_skill_drives_pbo_to_zero():
    """One column with a real edge keeps winning OOS, so the logits go positive."""
    r = cscv(with_one_skilled_column(), n_splits=12)
    assert r.pbo < 0.05
    assert r.logits.mean() > 1.0
    assert (r.best_trial == 7).mean() > 0.9


def test_pbo_separates_the_two_cases_sharply():
    overfit = pbo_over_seeds(noise, n_trials=50).mean()
    skilled = pbo_over_seeds(with_one_skilled_column).mean()
    assert overfit - skilled > 0.3


# --- the logit ------------------------------------------------------------- #


def test_logits_are_always_finite():
    """Dividing the rank by N+1 rather than N is what guarantees this.

    Ranks run 1..N, so omega stays inside [1/(N+1), N/(N+1)] and neither
    argument of the logarithm can reach zero.
    """
    r = cscv(noise(n_trials=20), n_splits=10)
    assert np.all(np.isfinite(r.logits))
    bound = np.log(20)
    assert np.all(np.abs(r.logits) <= bound + 1e-9)


def test_informationless_logits_are_logistic_not_normal():
    """p. 22 says the logits "approximate the standard Normal" here.

    They approximate the standard LOGISTIC, sd pi/sqrt(3) = 1.814, not 1.
    Measured across seeds: 1.583 +- 0.248. Averaged, because seed 0 alone gives
    2.030 - above the logistic, which would reverse the reading that discrete
    ranks truncate the tails.
    """
    sds = np.array(
        [cscv(noise(seed=s, n_trials=50), n_splits=12).logits.std() for s in range(8)]
    )
    assert sds.mean() > 1.3, "far wider than a standard Normal"
    assert sds.mean() < 1.814, "below the logistic; discrete ranks truncate"
    assert sds.std() > 0.1, "one draw is not enough to quote"


def test_definition_2_2_threshold_is_off_by_one_rank():
    """Def 2.2 (p. 10) says rbar < N/2; the logit says rbar < (N+1)/2.

    Under the null the OOS rank of the IS winner is uniform on 1..N, so the
    answer must be 0.5. Only the logit form gives it. The two agree at odd N.
    """
    expected = {
        10: (0.5000, 0.4000),
        50: (0.5000, 0.4800),
        100: (0.5000, 0.4900),
        11: (0.4545, 0.4545),
        101: (0.4950, 0.4950),
    }
    for n_trials, (via_logit, via_def_2_2) in expected.items():
        rank = np.arange(1, n_trials + 1)
        assert (rank / (n_trials + 1) < 0.5).mean() == pytest.approx(
            via_logit, abs=5e-5
        )
        assert (rank < n_trials / 2).mean() == pytest.approx(via_def_2_2, abs=5e-5)

    # The logit form hits 0.5 exactly only at even N. At odd N one rank lands
    # on the median, so omega = 1/2, the logit is zero, and strict inequality
    # excludes it, leaving (N-1)/2N.
    for n_trials in (11, 51, 101, 999):
        rank = np.arange(1, n_trials + 1)
        assert (rank / (n_trials + 1) < 0.5).mean() == pytest.approx(
            (n_trials - 1) / (2 * n_trials)
        )


def test_scale_guard_is_per_column():
    """A global scale rejects a real column whenever another is far larger."""
    rng = np.random.default_rng(0)
    X = np.column_stack(
        [rng.standard_normal(200) * 1e6, rng.standard_normal(200) * 1e-9]
    )
    assert np.all(np.isfinite(sharpe_columns(X)))
    assert np.all(np.isfinite(quiet(cscv, X, n_splits=8).logits))
    X[:, 1] = 3.0
    with pytest.raises(ValueError, match="zero variance"):
        sharpe_columns(X)


def test_pbo_is_the_share_of_negative_logits():
    r = cscv(noise(n_trials=30), n_splits=8)
    assert r.pbo == pytest.approx((r.logits < 0).mean())


# --- invariances ----------------------------------------------------------- #


def test_deterministic():
    """p. 21: running CSCV twice on the same inputs gives identical results."""
    X = noise()
    a, b = cscv(X, n_splits=10), cscv(X, n_splits=10)
    assert np.array_equal(a.logits, b.logits)


def test_pbo_is_invariant_to_positive_rescaling():
    """Rank-based, so annualizing cannot move it."""
    X = noise()
    assert cscv(X, n_splits=10).pbo == cscv(X * 7.5, n_splits=10).pbo


def test_annualization_scales_reported_performance_only():
    X = noise()
    plain = cscv(X, n_splits=10)
    ann = cscv(X, n_splits=10, periods_per_year=250)
    assert ann.pbo == plain.pbo
    assert ann.perf_oos == pytest.approx(plain.perf_oos * np.sqrt(250))


def test_custom_metric_is_honoured():
    """The paper's central claim is that the procedure is metric-free."""
    mean_only = cscv(noise(), n_splits=8, metric=lambda b: b.mean(axis=0))
    assert 0.0 <= mean_only.pbo <= 1.0
    assert mean_only.n_combinations == comb(8, 4)


# --- the other three statistics -------------------------------------------- #


def test_prob_loss_is_high_for_informationless_trials():
    vals = [
        cscv(noise(seed=s, n_trials=50), n_splits=12).prob_loss for s in range(12)
    ]
    assert np.mean(vals) > 0.35


def test_prob_loss_is_low_when_there_is_a_real_edge():
    assert cscv(with_one_skilled_column(), n_splits=12).prob_loss < 0.2


def test_performance_degradation_slope_is_negative_when_overfit():
    """Section 3.2: better IS performance buys worse OOS performance.

    The paper attributes this to memory effects in financial series, but part
    of it is mechanical: IS and OOS are complements of one fixed sample, so a
    combination that puts the lucky rows in the training half necessarily
    leaves them out of the testing half. It shows up here in pure i.i.d. noise,
    which has no memory at all - negative in about 90% of draws, so this
    averages rather than trusting one.
    """
    slopes = [
        cscv(noise(seed=s, n_trials=50), n_splits=12).performance_degradation[0]
        for s in range(12)
    ]
    assert np.mean(slopes) < 0


def test_degradation_is_mechanical_conditional_on_the_winner():
    """The whole misattribution argument, in one test.

    Section 3.2 reads the negative IS/OOS slope as a memory effect in financial
    series. Conditional on WHICH trial wins, the relationship is near
    deterministic on i.i.d. noise, which has no memory at all: the two halves
    are equal-sized complements of one fixed sample, so a column's two means
    must sum to twice its full-sample mean - exactly, to floating point. The
    Sharpe pair departs from a slope of -1 only through the differing standard
    deviations of the halves.

    What survives pooling across winners with different full-sample means is
    far weaker, and its sign is not even stable across configurations. That gap
    is the finding, and the coloured bands in figures/is_vs_oos_scatter.png are
    the same fact drawn.
    """
    r = cscv(noise(n_trials=50), n_splits=10)
    counts = Counter(r.best_trial.tolist())
    top, wins = counts.most_common(1)[0]
    m = r.best_trial == top

    within = np.corrcoef(r.perf_is[m], r.perf_oos[m])[0, 1]
    pooled = np.corrcoef(r.perf_is, r.perf_oos)[0, 1]

    assert wins > 20, "need enough splits sharing a winner to correlate"
    assert within < -0.9, f"within-winner should be near-deterministic, got {within}"
    assert pooled > -0.3, f"pooled should be far weaker, got {pooled}"
    assert abs(within) > 3 * abs(pooled)


def test_the_complement_identity_that_drives_it():
    """A column's two half-means sum to twice its full-sample mean, exactly."""
    X = noise(n_obs=480, n_trials=50)
    full = X.mean(axis=0)
    halves = X[:240].mean(axis=0) + X[240:].mean(axis=0)
    assert np.abs(halves - 2 * full).max() < 1e-15


def test_a_few_columns_win_most_of_the_splits():
    """Why the scatter bands: the winner is not redrawn each combination."""
    r = cscv(noise(n_trials=50), n_splits=10)
    counts = Counter(r.best_trial.tolist())
    share = sum(n for _, n in counts.most_common(6)) / r.n_combinations
    assert share > 0.5
    assert len(counts) < 50


def test_top_level_helper_matches_the_result_object():
    X = noise()
    assert probability_of_backtest_overfitting(X, n_splits=8) == cscv(
        X, n_splits=8
    ).pbo


# --- validation ------------------------------------------------------------ #


def test_rejects_odd_number_of_splits():
    with pytest.raises(ValueError, match="even"):
        cscv(noise(), n_splits=7)


def test_rejects_splits_that_do_not_divide_the_sample():
    """The paper's own example trips this: T=1000 at S=16 is 62.5 rows a block.

    Unequal blocks would make IS and OOS different sizes for different
    combinations, which destroys the symmetry the method is built on.
    """
    with pytest.raises(ValueError, match="remainder of 8"):
        cscv(noise(n_obs=1000), n_splits=16)


def test_rejects_degenerate_shapes():
    with pytest.raises(ValueError, match="2 trials"):
        cscv(noise(n_trials=1), n_splits=4)
    with pytest.raises(ValueError, match="2-D"):
        cscv(np.zeros(100), n_splits=4)


def test_rejects_non_finite_input():
    X = noise()
    X[3, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        cscv(X, n_splits=8)


def test_rejects_a_constant_trial():
    X = noise()
    X[:, 4] = 0.001
    with pytest.raises(ValueError, match="zero variance"):
        cscv(X, n_splits=8)


def test_warns_when_too_few_trials_for_granularity():
    """p. 22: N >> 10 is needed before phi < 1/10 is even representable."""
    with pytest.warns(UserWarning, match="granularity|coarse"):
        cscv(noise(n_trials=5), n_splits=8)


def test_no_warning_at_a_reasonable_trial_count():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cscv(noise(n_trials=50), n_splits=8)


# --- the metric helper ----------------------------------------------------- #


def test_sharpe_columns_matches_a_hand_calculation():
    X = np.array([[0.01, 0.02], [0.03, -0.01], [-0.02, 0.04], [0.04, 0.01]])
    expected = X.mean(axis=0) / X.std(axis=0)
    assert sharpe_columns(X) == pytest.approx(expected)