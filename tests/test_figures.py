"""The committed figure data must agree with the numbers the README quotes.

`scripts/make_figures.py` runs 25 datasets at S=16, which takes 165 seconds
against a 36-second suite - too slow to regenerate here. So it writes
what it computed to `figures/phi_dispersion.json` and this checks the committed
values instead. Regenerating the figures without updating the README, or the
reverse, fails.
"""

import json
import pathlib

import pytest

FIGURES = pathlib.Path(__file__).resolve().parents[1] / "figures"

pytestmark = pytest.mark.skipif(
    not (FIGURES / "phi_dispersion.json").exists(),
    reason="figures not generated; run scripts/make_figures.py",
)


@pytest.fixture(scope="module")
def data():
    return json.loads((FIGURES / "phi_dispersion.json").read_text())


def test_all_three_figures_are_committed():
    for name in ("phi_dispersion", "logit_histogram", "is_vs_oos_scatter"):
        png = FIGURES / f"{name}.png"
        assert png.exists(), f"{name}.png missing"
        assert png.stat().st_size > 10_000, f"{name}.png looks truncated"


def test_configuration_is_the_one_the_readme_states(data):
    assert (data["n_obs"], data["n_trials"], data["n_splits"]) == (960, 100, 16)
    assert data["seeds"] == list(range(25))
    assert data["paper_claimed_se"] == 0.0045


def test_phi_summary_matches_the_readme(data):
    """README: mean 0.468, standard deviation 0.139, range 0.24 to 0.81."""
    assert data["phi_mean"] == pytest.approx(0.468, abs=5e-4)
    assert data["phi_std"] == pytest.approx(0.139, abs=5e-4)
    assert data["phi_min"] == pytest.approx(0.24, abs=5e-3)
    assert data["phi_max"] == pytest.approx(0.81, abs=5e-3)


def test_the_refutation_holds_at_the_quoted_magnitude(data):
    """README says roughly thirty times the claimed standard error."""
    ratio = data["phi_std"] / data["paper_claimed_se"]
    assert round(ratio) == 31


def test_mean_is_correct_even_though_single_draws_are_not(data):
    """0.5 is the right answer under the null; the spread is the finding."""
    assert data["phi_mean"] == pytest.approx(0.5, abs=0.05)
    assert data["phi_max"] - data["phi_min"] > 0.5


def test_seed0_logit_spread_is_logistic_not_normal(data):
    """Figure 2 draws both curves; the observed spread must sit near neither 1.0
    (Normal) nor above the logistic's 1.814."""
    assert 1.3 < data["seed0_logit_std"] < 1.814


def test_seed0_degradation_slope_is_negative(data):
    """Figure 3's regression line, on data with no memory effects at all."""
    assert data["seed0_degradation_slope"] < 0


def test_figure_3_bands_are_winning_columns(data):
    """The coloured bands: one trial wins 27% of splits, and within it the
    IS/OOS relationship is near deterministic. Pooling across winners leaves
    something far weaker. That gap is what the figure exists to show."""
    assert data["seed0_top_winner_share"] == pytest.approx(0.27, abs=0.01)
    assert data["seed0_within_winner_corr"] == pytest.approx(-0.994, abs=5e-3)
    assert data["seed0_pooled_corr"] == pytest.approx(-0.151, abs=5e-3)
    assert abs(data["seed0_within_winner_corr"]) > 6 * abs(data["seed0_pooled_corr"])