![tests](https://github.com/Kodezilla0725/deflated-sharpe/actions/workflows/test.yml/badge.svg)

# Deflated Sharpe Ratio and Probability of Backtest Overfitting

Implementations of two tests for whether a backtest survived a search or merely
won one.

- **DSR** — Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*, SSRN
  [2460551](https://ssrn.com/abstract=2460551). Parametric: how high must a
  Sharpe ratio be before the size of the search stops explaining it.
- **PBO** — Bailey, Borwein, López de Prado & Zhu (2015), *The Probability of
  Backtest Overfitting*, SSRN
  [2326253](https://ssrn.com/abstract=2326253). Non-parametric and rank-based:
  does the configuration that wins in sample keep winning out of sample.

Both are reproduced against the papers' own published figures, to four decimals
where the papers print four. Along the way, five errata and one claim that does
not survive measurement.

---

## What the PBO paper gets wrong

Everything in this section is in the **PBO paper**. Five errata — Definition
2.2, the combination count, step (c), median vs mean, and logistic vs Normal —
plus one claim that does not survive measurement, and one finding attributed to
the wrong cause.

**I found no errors in the DSR paper.** Its published figures reproduce to four
decimals, and the section after this one documents a PDF extraction problem
rather than anything its authors got wrong.

### The CSCV standard error is understated by a factor of about thirty

PBO paper, p. 22, puts the standard error of φ under **0.0045** at S=16, from
σ = √(p(1−p)/n) with n the number of logits. That formula assumes independent
draws. The logits come from overlapping combinations of the same S row blocks,
so they are heavily dependent, and the effective sample size is governed by S,
not by C(S, S/2).

Measured over 25 informationless datasets at T=960, N=100, S=16:

| | φ |
|---|---|
| mean | 0.468 — correct, 0.5 is the right answer under the null |
| standard deviation | **0.139** |
| range across datasets | 0.24 to 0.81 |
| paper's claimed σ | 0.0045 |

![phi across 25 informationless datasets](figures/phi_dispersion.png)

The red strip is the paper's claimed standard error. The dots are the measured
values. Dropping to S=12 and 924 combinations — a fourteenth as many — gives
0.144.
Independent draws would have shrunk the spread by √14 ≈ 3.7×. Raising the
combination count buys essentially nothing, which is the signature of the
dependence.

The practical consequence: p. 13 suggests rejecting models with PBO above 0.05,
which reads as a sharp cut-off. On a single dataset it is not one.

### Definition 2.2 contradicts the paper's own logit

PBO paper, Definition 2.2 (p. 10) writes PBO as `Prob[r̄ₙ < N/2]`. The rank is
uniform on 1..N under the null, so that expression gives:

| N | via the logit | via Definition 2.2 |
|---|---|---|
| 10 | 0.500 | 0.400 |
| 50 | 0.500 | 0.480 |
| 100 | 0.500 | 0.490 |
| 101 | 0.495 | 0.495 |

Never 0.5, which is the correct answer when the in-sample winner carries no
information. φ as the paper actually *computes* it — the mass of the logit
distribution below zero — tests `r̄/(N+1) < ½`, equivalently `r̄ < (N+1)/2`, and
gives 0.5 at every N. The two differ by exactly one rank at even N and agree at
odd N. This implementation follows the logit.

### C(16,8) is 12,870, not the printed 12,780

PBO paper, pp. 11 and 22. The paper's own Eq. (2.3) product form gives 12,870,
and its S=24 figure of 2,704,156 is exactly C(24,12) — so the 16-case is a
transposed digit rather than a different definition.

### Two smaller ones

- **Algorithm 2.3 step (c)** calls J "the testing set" after step (a) defined it
  as the training set. Step (d) resolves it: (c) is in-sample.
- **Median vs mean.** Definition 2.2 and §3.1 say the optimal IS strategy
  underperforms the *median* OOS; the Conclusions say the *mean*. Only the
  median has a rank-based estimator, which is what φ computes.
### Logistic, not Normal

p. 22 says informationless logits approximate the standard Normal. The logit of
a uniform is standard *logistic*, σ = π/√3 = 1.814. Measured over 8 seeds:
1.583 ± 0.248. Below the logistic — discrete ranks truncate both tails — but
far above the Normal's 1.0.

![logit distribution with both reference curves](figures/logit_histogram.png)

This is the paper's Figure 8, with both candidate reference curves drawn. The
shaded mass below zero is φ.

### And one finding the PBO paper attributes to the wrong cause

§3.2 explains the negative in-sample/out-of-sample slope through *memory
effects* in financial series. Part of it is mechanical: IS and OOS are
complements of one fixed sample, so a combination that puts the lucky rows in
the training half necessarily leaves them out of the testing half. It appears
here in pure i.i.d. noise, which has no memory at all — negative in about 90%
of draws.

![in-sample against out-of-sample Sharpe ratios](figures/is_vs_oos_scatter.png)

The paper's Figure 7, on data containing no signal, coloured by which trial won
in sample. The diagonal bands are not an artifact — each one is a single
column, and six of them account for most of the 12,870 splits.

That colouring sharpens the claim considerably. **Conditional on the winner,
the relationship is not partly mechanical, it is almost entirely mechanical:
r = −0.994 within the top band against −0.151 pooled.** The reason is exact —
the two halves are equal-sized complements of one fixed sample, so a column's
two means must sum to twice its full-sample mean, to floating point. The Sharpe
pair departs from a slope of −1 only because the two halves have different
standard deviations. What survives pooling across winners with different
full-sample means is the much weaker −0.151, whose sign is not even stable
across configurations.

Pinned by `test_degradation_is_mechanical_conditional_on_the_winner`.

---

## Reading the DSR paper — a PDF problem, not a paper error

The PDF text layer destroys the Greek letters and every equation. Working from
the prose, an early draft of these notes recorded the trial variance in the
worked example as 1/100; at that value every case returns DSR = 1.0000.
Rendering p. 9 as an image settles it — the analyst's disclosure reads
`N = 100, V[{SR̂ₙ}] = 1/2, T = 1250, γ̂₃ = −3, γ̂₄ = 10` — and p. 10 writes the
threshold as `√(1/(2·250))(·)`, confirming the ½ is annualised and divided by
250 to reach per-observation units.

Rasterize before reconstructing. `pdftoppm -jpeg -r 200 -f 9 -l 10 paper.pdf p`.

Reproduced exactly:

| | paper (p. 10) | this code |
|---|---|---|
| SR₀ at N=100 | 0.1132 | 0.1132 |
| DSR at N=100 | 0.9004 | 0.9004 |
| DSR at N=46 | 0.9505 | 0.9505 |
| DSR at N=88, Normal returns | 0.9505 | 0.9505 |

---

## Install and run

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # 140 passed, ~36s
```

Run from the project root — imports are `from src.psr import ...`, and
`python -m pytest` puts the root on `sys.path` where plain `pytest` may not.
`src/__init__.py` is empty on purpose; deleting it breaks the imports.

Most of the runtime is `test_pbo.py`, which runs CSCV repeatedly across seeds
because φ on a single dataset has a standard deviation around 0.14. For a fast
edit loop: `python -m pytest tests/ -q --ignore=tests/test_pbo.py` gives 97
tests in about 4 seconds.

## Use

```python
from src.audit import audit

result = audit(trial_returns, periods_per_year=250)   # T x M, all trials
print(result.summary())
```

```
trials run                 400
average correlation        0.791
effective independent      84.4
selected trial             #85
  Sharpe (annualized)      0.31
  skew / kurtosis          0.08 / 3.20
  observations             960
hurdle SR0 (annualized)    0.59
Deflated Sharpe Ratio      0.2904
Probability of Overfitting 0.3852
OOS probability of loss    0.7455
```

That is 400 correlated backtests on pure noise. Both tests agree there is
nothing there, by different routes.

The inputs are reported alongside the verdicts deliberately. A DSR quoted
without the trial count and the dispersion behind it reproduces the omission
the DSR paper exists to attack.

### Modules

| | |
|---|---|
| `src/psr.py` | Probabilistic Sharpe Ratio, Minimum Track Record Length |
| `src/dsr.py` | Expected maximum Sharpe under the null, Deflated Sharpe Ratio |
| `src/trials.py` | Average correlation, effective number of independent trials |
| `src/pbo.py` | CSCV, Probability of Backtest Overfitting |
| `src/audit.py` | Composition: trial matrix in, both verdicts out |
| `scripts/make_figures.py` | Regenerates the three figures above |

---

## Design decisions

**Units are unconstructible when wrong, not merely asserted against.** An
earlier version guarded the trial dispersion with a plausibility threshold. That
cannot work: the error being detected is a factor of `periods_per_year`, so its
magnitude depends on the frequency while any absolute threshold does not. At 250
periods a year a mistaken annualised 0.5 implies an annualised dispersion of
11.2; at 12 periods it implies 2.4, indistinguishable from a merely wide sweep.
`SharpeStats` and `TrialSet` each declare their own frequency and
`deflated_sharpe_ratio` checks the two agree — an exact comparison with no
margin to erode.

**Kurtosis is non-excess** (Normal = 3.0), matching the (γ₄−1)/4 term. scipy
defaults to Fisher and would silently understate the penalty.

**Moments are the biased estimators, and that is coupled to the Pearson
guard.** Biased sample moments satisfy γ₄ ≥ γ₃² + 1 identically, so
`from_returns` can never trip it; the equality is exact for n=2 samples, which
is why the guard carries a 1e-9 tolerance. Unbiased moments do *not* satisfy it
— over 5,000 small fat-tailed samples the slack reaches −3.99 against a biased
minimum of +5.5e−4, so mixing the conventions would make the guard reject
ordinary data. Both figures are pinned by
`test_unbiased_moments_would_break_the_pearson_guard`.

**Eq. (1) is not uniformly conservative, so it is not used everywhere.** Above
the measured crossover at N = 2.7752 it overestimates the expected maximum —
+0.036 at N=10, +0.023 at N=100 — which raises the hurdle and is the safe
direction. Below it the sign flips: at N=2 it understates by 0.044, and below
N=1.2836 it returns a negative correction, implying an expected maximum below
the mean. Understating the threshold *raises* DSR and makes the tool permissive
in exactly the regime it exists to police, so the exact order statistic
`∫ z·N·φ(z)·Φ(z)^(N−1) dz` is integrated instead. Reachable, not theoretical:
M=100 at ρ̂=0.98 gives N = 2.98.

The branch switches at `EXACT_BRANCH_BELOW = 3.0`, not at 2.7752. The integral
is exact on both sides, so that constant only decides where to stop using the
cheaper approximation, and a margin above the measured crossover costs one
quadrature call. The small-N warning fires only on the Eq. (1) branch — below
it there is no approximation to warn about.

**M > T warns rather than refusing, and the warning names the right risk.** The
DSR paper (p. 15) calls the correlation matrix ill-conditioned there. That
concern is aimed at methods that invert or decompose it; an equal-weighted
average does neither, and each pairwise correlation still rests on T
observations however many columns exist. Measured over 30 seeds at M/T = 33:
ρ̂ = 0.8002 ± 0.0312, essentially unbiased. What degrades is the *dispersion*,
and `effective_num_trials` multiplies it by M — at M=1650 that puts N̂ anywhere
from 218 to 457 against a true 331.

**The effective trial count is a float.** N̂ is an interpolation between two
regimes, not a count, and Eq. (1) is continuous in N.

**Average correlation is computed as ‖Z1‖²/T**, never by forming the matrix.
The sum of all entries is what Eq. (8) needs and it is available in O(TM); at
M=20,000 the explicit matrix is 3.2 GB.

**CSCV requires S to divide T exactly.** The PBO paper's own example does not
satisfy this — T=1000 daily at S=16 is 62.5 rows per block. Unequal blocks
would make IS and OOS different sizes across combinations, destroying the
symmetry the method is named for.

**ω̄ divides the rank by N+1, not N.** Ranks run 1..N, so ω̄ stays inside
[1/(N+1), N/(N+1)] and the logit cannot reach ±∞.

**The DSR hurdle uses a trial mean of zero,** not the observed mean of the
sweep. DSR tests H₀: SR = 0. Feeding in the observed mean asks a different and
much harsher question — whether the winner beats the expected best of a strategy
class already assumed to have skill. The observed mean is reported separately.

---

## Limits

Both statistics are noisy on a single dataset. Under the null, DSR is a p-value
and should be roughly uniform: across 8 informationless datasets it ranges from
0.01 to 0.89 with a mean of 0.485. PBO ranges from 0.37 to 0.81. The means are
correct and the individual readings are not verdicts. Every statistical test
here averages across seeds for that reason.

The PBO paper's own warning (§5.2, p. 25) applies with full force: PBO must
never become the objective a search optimises. Using it to select a strategy is
a misuse that reintroduces the overfitting it measures.

Every figure above is generated by `scripts/make_figures.py` from fixed seeds
(0–24 at T=960, N=100, S=16). The values it computed are committed to
`figures/phi_dispersion.json`, and `tests/test_figures.py` checks them against
the numbers quoted here, so the images and the prose cannot drift apart.

The figures and the quoted measurements use that headline configuration; the
test suite measures the same effect at T=480, N=50, S=12 over 12 seeds instead,
because 25 datasets at S=16 take 165 seconds against a 36-second suite. The
claim is identical, only the cost is not.

Stochastic dominance (§3.3, the fourth statistic) is not implemented.
`CSCVResult.perf_oos_all` stores the full out-of-sample surface so it remains
reachable without recomputation.
