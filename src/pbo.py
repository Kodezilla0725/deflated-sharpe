"""Probability of Backtest Overfitting via combinatorially symmetric CV.

Bailey, Borwein, Lopez de Prado & Zhu (2015), "The Probability of Backtest
Overfitting", SSRN 2326253. Algorithm 2.3, pp. 11-12; PBO as
phi = integral of f(lambda) from -inf to 0, Section 3.1, p. 13.

Attacks the same selection problem as `dsr.py` from the opposite side. DSR is
parametric: it asks how high a Sharpe ratio has to be before it stops being
explicable by the size of the search. CSCV is non-parametric and rank-based: it
asks whether the configuration that wins in-sample keeps winning out-of-sample,
across every symmetric split of the data. Neither subsumes the other, and a
strategy that clears one and fails the other is worth a second look.

Three departures from the paper as printed, each verified rather than assumed:

1. It states that S=16 yields 12,780 combinations (pp. 11, 22). C(16,8) is
   12,870. The paper's own Eq. (2.3) product form gives 12,870, and its S=24
   figure of 2,704,156 is exactly C(24,12), so this is a transposed digit.
2. Step (c) calls J "the testing set" after step (a) defined it as the training
   set. Step (d) settles it: (c) is in-sample, (d) is out-of-sample.
3. Definition 2.2 and Section 3.1 say the optimal IS strategy underperforms the
   MEDIAN out of sample; the Conclusions say the mean. Only the median has a
   rank-based estimator, which is what phi computes, so median it is.
4. p. 22 says the logits approximate the standard NORMAL when the backtest is
   informationless. The logit of a uniform is standard LOGISTIC, whose standard
   deviation is pi/sqrt(3) = 1.81. Only matters if you compare an observed
   spread against a Normal baseline, which would look far too wide.

And one claim that does not survive measurement. p. 22 puts the standard error
of phi at under 0.0045 for S=16, from sigma = sqrt(p(1-p)/n) with n the number
of logits. That formula assumes independent draws. The logits come from
overlapping combinations of the same S blocks, so they are heavily dependent
and the effective sample size is governed by S, not by C(S, S/2). Measured over
25 informationless datasets at T=960, N=100, S=16: phi has mean 0.468 - correct,
0.5 is the right answer - but a standard deviation of 0.139, some thirty times
the claimed figure, with individual draws ranging from 0.24 to 0.81. Dropping to
S=12 and 924 combinations, a fourteenth as many, gives 0.144. Raising the
combination count buys almost nothing, which is the signature of the
dependence. Treat phi as a coarse reading, not a precise one, and do not read
0.05 as a sharp cut-off on a single dataset.
"""

import math
import warnings
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import rankdata

DEFAULT_N_SPLITS = 16

# p. 22: N must be large enough that the relative rank has granularity. "If the
# investor is sensitive to values of phi < 1/10, it is clear that the range of
# values that the logits can adopt must be greater than 10, and so N >> 10."
MIN_USEFUL_TRIALS = 10


def sharpe_columns(block):
    """Per-observation Sharpe ratio of every column. The default metric.

    ddof=0 to match `psr.SharpeStats.from_returns`, though CSCV only uses the
    ranks, which any positive-scaling choice leaves alone.
    """
    sd = block.std(axis=0)
    if np.any(sd <= 1e-12 * np.abs(block).max()):
        raise ValueError(
            "a trial has zero variance within a split; its Sharpe ratio is "
            "undefined. Drop constant columns before running CSCV."
        )
    return block.mean(axis=0) / sd


@dataclass(frozen=True)
class CSCVResult:
    """One logit per combination, plus the IS/OOS pairs behind them."""

    logits: np.ndarray
    perf_is: np.ndarray
    perf_oos: np.ndarray
    best_trial: np.ndarray
    n_trials: int
    n_splits: int

    @property
    def pbo(self):
        """phi: the share of combinations whose IS winner lands below the OOS median.

        Strict inequality. lambda = 0 means the IS winner landed exactly on the
        OOS median, reachable only when N is odd, and the integral's endpoint
        carries no mass.
        """
        return float((self.logits < 0).mean())

    @property
    def prob_loss(self):
        """Prob[R_n* < 0]: how often the selected strategy loses money OOS.

        Section 3.2, p. 14: this can be high even when phi is near zero, in
        which case OOS performance is poor for reasons other than overfitting.
        """
        return float((self.perf_oos < 0).mean())

    @property
    def performance_degradation(self):
        """(slope, intercept) of OOS performance regressed on IS performance.

        Section 3.2: the slope is negative in most practical cases. The paper's
        reading is that an overfit backtest does not merely fail to help, it
        actively hurts - fit tightly enough to past noise, a model is rendered
        unfit for future signal.
        """
        slope, intercept = np.polyfit(self.perf_is, self.perf_oos, 1)
        return float(slope), float(intercept)

    @property
    def n_combinations(self):
        return len(self.logits)


def cscv(
    trial_returns,
    n_splits=DEFAULT_N_SPLITS,
    metric=sharpe_columns,
    periods_per_year=None,
):
    """Algorithm 2.3: every symmetric train/test split, one logit each.

    Parameters
    ----------
    trial_returns:
        T x N array. Rows are synchronous observations, columns are the trials.
        The same matrix `trials.average_correlation` takes.
    n_splits:
        S, the number of row blocks. Must be even and divide T. The paper
        recommends 16 (p. 22): enough combinations that the standard error of
        the estimated proportion stays under 0.0045, while blocks stay long
        enough to preserve serial structure - at four years of daily data,
        S=16 is quarterly.
    metric:
        Callable mapping a (rows x N) block to N performance values. Defaults
        to the Sharpe ratio. Kept configurable because the paper's central
        claim is that the procedure is metric-free (Section 2.2), and PBO on
        the Sortino ratio or PSR is a real use.
    periods_per_year:
        Annualizes the REPORTED performances only. PBO is rank-based, so any
        positive rescaling leaves it untouched; this exists so the degradation
        regression comes out in the units the paper's figures use.

    Rows keep their original order inside both sets. Irrelevant for the Sharpe
    ratio, but step (b) notes it matters for path-dependent metrics such as
    return-over-maximum-drawdown, and a custom `metric` may be one.
    """
    X = np.asarray(trial_returns, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D T x N array, got shape {X.shape}")
    n_obs, n_trials = X.shape
    if n_splits % 2:
        raise ValueError(f"n_splits must be even, got {n_splits}")
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")
    if n_obs % n_splits:
        raise ValueError(
            f"n_splits must divide T exactly: T={n_obs} leaves a remainder of "
            f"{n_obs % n_splits} at S={n_splits}. Trim {n_obs % n_splits} rows "
            f"or pick a divisor of {n_obs}."
        )
    if n_trials < 2:
        raise ValueError("need at least 2 trials to rank anything")
    if not np.all(np.isfinite(X)):
        raise ValueError("trial_returns contains non-finite values")
    if n_trials < MIN_USEFUL_TRIALS:
        warnings.warn(
            f"N={n_trials} trials gives the relative rank only {n_trials} "
            f"possible values, so f(lambda) is coarse and phi carries "
            f"discretization error (p. 22).",
            stacklevel=2,
        )

    scale = 1.0 if periods_per_year is None else math.sqrt(periods_per_year)
    blocks = np.arange(n_obs).reshape(n_splits, n_obs // n_splits)
    half = n_splits // 2

    logits, perf_is, perf_oos, best = [], [], [], []
    for combo in combinations(range(n_splits), half):
        mask = np.zeros(n_splits, dtype=bool)
        mask[list(combo)] = True
        # combinations() is ascending and blocks are contiguous, so ravelling
        # preserves the original row order in both halves.
        rows_is = blocks[mask].ravel()
        rows_oos = blocks[~mask].ravel()

        r_is = metric(X[rows_is])
        r_oos = metric(X[rows_oos])

        n_star = int(np.argmax(r_is))
        # Dividing by N+1, not N, is what keeps the relative rank inside (0,1)
        # and the logit finite: ranks run 1..N, so omega lands in
        # [1/(N+1), N/(N+1)] and neither log argument can reach zero.
        omega = rankdata(r_oos)[n_star] / (n_trials + 1)

        logits.append(math.log(omega / (1 - omega)))
        perf_is.append(r_is[n_star] * scale)
        perf_oos.append(r_oos[n_star] * scale)
        best.append(n_star)

    return CSCVResult(
        logits=np.array(logits),
        perf_is=np.array(perf_is),
        perf_oos=np.array(perf_oos),
        best_trial=np.array(best),
        n_trials=n_trials,
        n_splits=n_splits,
    )


def probability_of_backtest_overfitting(
    trial_returns, n_splits=DEFAULT_N_SPLITS, metric=sharpe_columns
):
    """phi, the probability of backtest overfitting.

    Section 3.1, p. 13, suggests rejecting models with PBO above 0.05, by
    analogy with a conventional significance level.

    Note the paper's own warning (Section 5.2, p. 25): PBO must never become
    the objective a search optimizes. Using it to select a strategy is a misuse
    that reintroduces the overfitting it measures.
    """
    return cscv(trial_returns, n_splits=n_splits, metric=metric).pbo
