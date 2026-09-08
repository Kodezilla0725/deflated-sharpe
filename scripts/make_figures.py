"""Write the three README figures. Run once; commit the output.

    python scripts/make_figures.py

Computes nothing itself - every number comes from `src`. Seeds and shapes are
fixed and stated in `figures/phi_dispersion.json`, which
`tests/test_figures.py` checks against the numbers the README quotes, so the
committed images cannot drift away from the prose.

Figure 1 uses the configuration the README headline claim is stated at
(T=960, N=100, S=16, seeds 0-24). The test suite measures the same effect at
T=480, N=50, S=12 over 12 seeds instead, because 25 datasets at S=16 take 165
seconds against a 36-second suite. The claim is identical; only the cost is not.
"""

import json
import pathlib
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import logistic, norm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.pbo import cscv  # noqa: E402

FIGURES = pathlib.Path(__file__).resolve().parents[1] / "figures"

N_OBS, N_TRIALS, N_SPLITS, N_SEEDS = 960, 100, 16, 25
# Annualizes the reported Sharpe ratios only. phi and the logits are
# rank-based and invariant to it, so figures 1 and 2 are unaffected;
# figure 3 needs it or its axis labels would be false.
PPY = 250
PAPER_CLAIMED_SE = 0.0045  # PBO paper, p. 22


def informationless(seed):
    """Trials with no edge: whatever wins in sample won it by luck."""
    return np.random.default_rng(seed).standard_normal((N_OBS, N_TRIALS)) * 0.01


def figure_1_phi_dispersion(phis):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.axhspan(
        0.5 - PAPER_CLAIMED_SE,
        0.5 + PAPER_CLAIMED_SE,
        color="tab:red",
        alpha=0.5,
        zorder=2,
        label=f"paper's claimed $\\pm${PAPER_CLAIMED_SE} (p. 22)",
    )
    ax.axhline(0.5, color="0.4", lw=1, ls="--", zorder=1, label="0.5, the true value")
    ax.scatter(
        np.arange(len(phis)),
        phis,
        s=55,
        color="tab:blue",
        edgecolor="white",
        zorder=3,
        label="measured $\\phi$, one per dataset",
    )

    ax.errorbar(
        len(phis) + 1.5,
        phis.mean(),
        yerr=phis.std(),
        fmt="o",
        color="black",
        capsize=5,
        zorder=4,
        label=f"mean {phis.mean():.3f} $\\pm$ {phis.std():.3f}",
    )

    ax.set_xlim(-1.5, len(phis) + 3)
    ax.set_ylim(0, 1)
    # The summary marker sits off the end of the seed axis, so give it its own
    # tick rather than letting x=26.5 read as a 26th dataset.
    ax.set_xticks(list(range(0, len(phis), 5)) + [len(phis) + 1.5])
    ax.set_xticklabels([str(t) for t in range(0, len(phis), 5)] + ["all"])
    ax.set_xlabel("informationless dataset (seed)")
    ax.set_ylabel("$\\phi$  (probability of backtest overfitting)")
    ax.set_title(
        "$\\phi$ across 25 datasets that contain no signal at all\n"
        f"T={N_OBS}, N={N_TRIALS}, S={N_SPLITS}, seeds 0-{N_SEEDS - 1}",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)

    fig.text(
        0.5,
        -0.08,
        f"The mean is right: 0.5 is the correct answer when nothing is there. "
        f"The spread is {phis.std() / PAPER_CLAIMED_SE:.0f} times the paper's "
        f"claimed standard error, because the\nlogits come from overlapping "
        f"combinations of the same S blocks and are not independent draws. "
        f"Individual datasets run {phis.min():.2f} to {phis.max():.2f}.",
        ha="center",
        fontsize=9,
        color="0.3",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "phi_dispersion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_2_logit_histogram(result):
    fig, ax = plt.subplots(figsize=(9, 5))
    logits = result.logits

    ax.hist(
        logits,
        bins=45,
        density=True,
        color="tab:green",
        alpha=0.75,
        edgecolor="white",
        label="observed logits",
    )
    lo, hi = logits.min(), logits.max()
    ax.axvspan(lo - 0.5, 0, color="tab:red", alpha=0.10, zorder=0)
    ax.axvline(0, color="black", lw=1.5, zorder=5)

    grid = np.linspace(lo - 0.5, hi + 0.5, 400)
    ax.plot(grid, logistic.pdf(grid), "r-", lw=2, label="standard logistic (what it is)")
    ax.plot(
        grid, norm.pdf(grid), "b--", lw=2, label="standard Normal (what p. 22 says)"
    )

    ax.annotate(
        f"$\\phi$ = {result.pbo:.3f}\nmass below zero",
        xy=(-4.0, 0.155),
        fontsize=11,
        color="tab:red",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="tab:red", alpha=0.9),
    )
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_xlabel("logit  $\\lambda_c = \\ln\\,[\\bar\\omega_c / (1 - \\bar\\omega_c)]$")
    ax.set_ylabel("density")
    ax.set_title(
        "Distribution of rank logits, informationless case\n"
        f"T={N_OBS}, N={N_TRIALS}, S={N_SPLITS}, seed 0, "
        f"{result.n_combinations:,} combinations",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.text(
        0.5,
        -0.08,
        "The paper's Figure 8. Shaded mass below zero is $\\phi$: the share of "
        "splits where the in-sample winner landed below the\nout-of-sample "
        "median. Both reference curves are drawn because p. 22 names the "
        "Normal, but the logit of a uniform is logistic.",
        ha="center",
        fontsize=9,
        color="0.3",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "logit_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_3_is_vs_oos(result):
    """Coloured by winning column, because the diagonal bands ARE the columns.

    Each band is one trial that keeps winning in sample. Within a band the
    relationship is near-deterministic: the in-sample and out-of-sample halves
    are equal-sized complements of one fixed sample, so the two column means
    must sum to twice the full-sample mean - exactly, to 2e-18. The Sharpe pair
    departs from a slope of -1 only because the two halves have different
    standard deviations.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    slope, intercept = result.performance_degradation
    x, y = result.perf_is, result.perf_oos
    n_combos = len(x)

    counts = Counter(result.best_trial.tolist())
    top = [col for col, _ in counts.most_common(6)]
    palette = plt.get_cmap("tab10")

    rest = ~np.isin(result.best_trial, top)
    ax.scatter(
        x[rest], y[rest], s=6, color="0.75", alpha=0.5, edgecolor="none",
        label=f"{len(counts) - len(top)} other winners",
    )
    for i, col in enumerate(top):
        m = result.best_trial == col
        r_within = np.corrcoef(x[m], y[m])[0, 1]
        ax.scatter(
            x[m], y[m], s=7, color=palette(i), alpha=0.65, edgecolor="none",
            label=f"trial {col}: {m.sum() / n_combos:.0%} of splits, "
                  f"r = {r_within:.3f}",
        )

    grid = np.linspace(x.min(), x.max(), 100)
    ax.plot(
        grid,
        intercept + slope * grid,
        "k-",
        lw=2.5,
        label=(
            f"pooled: {intercept:.2f} {'-' if slope < 0 else '+'} "
            f"{abs(slope):.2f} $\\times$ [SR IS]"
        ),
    )
    ax.axhline(0, color="0.5", lw=1, ls="--")

    pooled_r = np.corrcoef(x, y)[0, 1]
    best_col = counts.most_common(1)[0][0]
    m = result.best_trial == best_col
    ax.text(
        0.03,
        0.05,
        f"within winner (trial {best_col}):  r = "
        f"{np.corrcoef(x[m], y[m])[0, 1]:.3f}\n"
        f"pooled across winners:      r = {pooled_r:.3f}\n"
        f"Prob[SR OOS < 0] = {result.prob_loss:.2f}",
        transform=ax.transAxes,
        fontsize=9.5,
        family="monospace",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.5"),
    )
    ax.set_xlabel("SR in sample (annualized)")
    ax.set_ylabel("SR out of sample (annualized)")
    ax.set_title(
        "Out-of-sample performance degradation\n"
        f"T={N_OBS}, N={N_TRIALS}, S={N_SPLITS}, seed 0, "
        f"one point per combination",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax.grid(alpha=0.3)

    fig.text(
        0.5,
        -0.17,
        "The paper's Figure 7, on data containing no signal. The diagonal bands "
        "are not an artifact: each one is a single trial that keeps\nwinning in "
        "sample, and six columns account for most of the 12,870 splits. The "
        "paper reads the negative slope as a memory\neffect in financial series. "
        "This data is i.i.d. noise with no memory, and CONDITIONAL ON THE WINNER "
        "the relationship is\nalmost entirely mechanical - the two halves are "
        "equal-sized complements of one fixed sample, so their means must sum to "
        "twice\nthe full-sample mean. What survives pooling across winners with "
        "different full-sample means is the much weaker -0.15.",
        ha="center",
        fontsize=8.5,
        color="0.3",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "is_vs_oos_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)

    print(f"running CSCV on {N_SEEDS} datasets (S={N_SPLITS}, this takes a while)")
    results = []
    for seed in range(N_SEEDS):
        results.append(
            cscv(informationless(seed), n_splits=N_SPLITS, periods_per_year=PPY)
        )
        print(f"  seed {seed:>2}  phi = {results[-1].pbo:.4f}")

    phis = np.array([r.pbo for r in results])
    figure_1_phi_dispersion(phis)
    figure_2_logit_histogram(results[0])
    figure_3_is_vs_oos(results[0])

    slope, intercept = results[0].performance_degradation
    r0 = results[0]
    top_col = Counter(r0.best_trial.tolist()).most_common(1)[0][0]
    top_mask = r0.best_trial == top_col
    (FIGURES / "phi_dispersion.json").write_text(
        json.dumps(
            {
                "n_obs": N_OBS,
                "n_trials": N_TRIALS,
                "n_splits": N_SPLITS,
                "seeds": list(range(N_SEEDS)),
                "paper_claimed_se": PAPER_CLAIMED_SE,
                "phi": [round(float(p), 6) for p in phis],
                "phi_mean": round(float(phis.mean()), 6),
                "phi_std": round(float(phis.std()), 6),
                "phi_min": round(float(phis.min()), 6),
                "phi_max": round(float(phis.max()), 6),
                "seed0_logit_std": round(float(results[0].logits.std()), 6),
                "seed0_prob_loss": round(float(results[0].prob_loss), 6),
                "seed0_degradation_slope": round(slope, 6),
                "seed0_top_winner": int(top_col),
                "seed0_top_winner_share": round(float(top_mask.mean()), 6),
                "seed0_within_winner_corr": round(
                    float(np.corrcoef(r0.perf_is[top_mask], r0.perf_oos[top_mask])[0, 1]), 6
                ),
                "seed0_pooled_corr": round(
                    float(np.corrcoef(r0.perf_is, r0.perf_oos)[0, 1]), 6
                ),
            },
            indent=2,
        )
        + "\n"
    )

    print(
        f"\nphi: mean {phis.mean():.3f}, sd {phis.std():.3f}, "
        f"range {phis.min():.2f}-{phis.max():.2f}"
    )
    print(f"that is {phis.std() / PAPER_CLAIMED_SE:.0f}x the claimed {PAPER_CLAIMED_SE}")
    print(f"wrote 3 figures and phi_dispersion.json to {FIGURES}")


if __name__ == "__main__":
    main()