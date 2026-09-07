"""Collapsing M backtests into the number of independent trials.

Bailey & Lopez de Prado (2014), Appendix 3, pp. 14-15, Eqs. (7)-(9).

Grid-search backtests are massively redundant: a 5-day and a 6-day holding
period on the same signal are nearly the same strategy. The maximum over 10,000
overlapping trials behaves like the maximum over a few hundred distinct ones,
so Eq. (1) needs the independent count. Note the direction of the error, which
is usually guessed backwards: passing raw M OVERSTATES the threshold and
rejects strategies that were fine.
"""

import warnings

import numpy as np


def average_correlation(trial_returns):
    """Equal-weighted mean of the off-diagonal correlations, Eq. (8).

    Only the unit-vector case. Eq. (7) defines rho generally, as the constant
    off-diagonal value leaving a quadratic form a'Ca unchanged, but there is no
    principled `a` here: the trials are unweighted candidates, which is exactly
    the condition under which Eq. (7) reduces to Eq. (8). A weight vector would
    matter if trials were weighted by capital or conviction, which is a
    different question from how much of the search was redundant. Adding the
    parameter would be configuration for a value that never varies.

    Computed as ||Z1||^2 / T rather than by forming the M x M matrix: the sum
    of all entries of C is what Eq. (8) needs, and it can be had in O(TM). At
    M=20,000 the explicit matrix is 3.2 GB.

    Parameters
    ----------
    trial_returns:
        T x M array. Rows are observations, columns are trials.

    Warns when M > T. The paper calls the matrix ill-conditioned there and
    warns that rho itself may be overfit (p. 15). That concern is real, but it
    is aimed at methods that invert or decompose C; an equal-weighted average
    does neither, and each pairwise correlation still rests on T observations
    however many columns there are.

    Measured over 30 seeds at M/T = 33 (T=60, M=2000, true rho 0.8), rho comes
    back at 0.8002 +- 0.0312: the BIAS is negligible. What degrades is the
    DISPERSION, and `effective_num_trials` multiplies it by M. At T=50, M=1650
    the same 30 seeds put Nhat anywhere from 218 to 457 against a true 331 -
    a standard deviation of 57 and an observed spread of 239. Compare T=1250,
    M=400: rho = 0.7986 +- 0.0060, Nhat 76 to 87.

    So this warns rather than refuses, and the warning names the dispersion
    rather than the bias, because the bias is not the problem.
    """
    X = np.asarray(trial_returns, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D T x M array, got shape {X.shape}")
    n_obs, n_cols = X.shape
    if n_cols < 2:
        raise ValueError("need at least 2 trials to have a correlation")
    if n_obs < 3:
        raise ValueError("need at least 3 observations")

    sd = X.std(axis=0)
    # Per column. Against a global maximum, a genuinely varying trial is
    # rejected whenever another trial is far larger in scale.
    dead = sd <= 1e-12 * np.abs(X).max(axis=0)
    if np.any(dead):
        raise ValueError(
            f"trials {list(np.flatnonzero(dead))} have zero variance; their "
            f"correlations are undefined"
        )
    if n_cols > n_obs:
        warnings.warn(
            f"M={n_cols} exceeds T={n_obs}; the average correlation stays "
            f"essentially unbiased but its dispersion grows sharply, and "
            f"effective_num_trials multiplies that error by M. At M/T=33 the "
            f"resulting Nhat spread was 218-457 against a true 331.",
            stacklevel=2,
        )

    Z = (X - X.mean(axis=0)) / sd
    row_sums = Z.sum(axis=1)
    total = float(row_sums @ row_sums) / n_obs  # sum of every entry of C
    return (total - n_cols) / (n_cols * (n_cols - 1))


def effective_num_trials(n_trials_raw, avg_corr):
    """Eq. (9): Nhat = rho + (1 - rho) * M.

    A linear interpolation between the two limits. Perfectly correlated trials
    collapse to one; uncorrelated trials stay at M.

    Returns a FLOAT, and deliberately does not round. Nhat is an interpolation
    between two regimes, not a count of anything, and Eq. (1) is continuous in
    N, so rounding would discard information to buy nothing. `TrialSet` and
    `expected_max_sharpe` both accept fractional N.

    Clamped to [1, M]. A slightly negative sample rho, which is admissible down
    to -1/(M-1), would otherwise imply more independent trials than trials run.
    """
    if n_trials_raw < 1:
        raise ValueError(f"n_trials_raw must be at least 1, got {n_trials_raw}")
    if not -1 <= avg_corr <= 1:
        raise ValueError(f"avg_corr must lie in [-1, 1], got {avg_corr}")
    return min(max(avg_corr + (1 - avg_corr) * n_trials_raw, 1.0), float(n_trials_raw))