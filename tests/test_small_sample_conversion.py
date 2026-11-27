"""Fixed-horizon conversion analysis at small or sparse counts (issue #38)."""

import numpy as np
import pytest
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import confint_proportions_2indep

from ablift.engine import analyze, parse_payload


def fixed(c: tuple[int, int], *treatments: tuple[int, int], alpha: float = 0.05):
    variants = [{"name": "control", "is_control": True, "visitors": c[1], "conversions": c[0]}]
    variants += [
        {"name": f"t{i}", "visitors": n, "conversions": x}
        for i, (x, n) in enumerate(treatments, start=1)
    ]
    return analyze(
        parse_payload(
            {
                "method": "fixed",
                "primary_metric": "conversion_rate",
                "alpha": alpha,
                "variants": variants,
            }
        )
    )


def newcombe(c: tuple[int, int], t: tuple[int, int], alpha: float) -> tuple[float, float]:
    lo, hi = confint_proportions_2indep(
        t[0], t[1], c[0], c[1], method="newcombe", compare="diff", alpha=alpha
    )
    return float(lo), float(hi)


@pytest.mark.parametrize(
    ("c", "t"),
    [
        ((0, 10), (0, 10)),
        ((0, 20), (0, 20)),
        ((0, 50), (4, 50)),
        ((10, 10), (9, 10)),
        ((400, 10_000), (460, 10_000)),
    ],
)
def test_ci_matches_newcombe_reference(c, t):
    comp = fixed(c, t).comparisons[0]
    lo, hi = newcombe(c, t, alpha=0.05)
    assert comp.ci_low == pytest.approx(lo, abs=1e-9)
    assert comp.ci_high == pytest.approx(hi, abs=1e-9)


def test_zero_vs_zero_interval_is_not_degenerate():
    comp = fixed((0, 10), (0, 10)).comparisons[0]
    assert comp.ci_low < -0.2 and comp.ci_high > 0.2


def test_ci_uses_bonferroni_alpha_across_treatments():
    res = fixed((0, 50), (4, 50), (2, 50))
    lo, hi = newcombe((0, 50), (4, 50), alpha=0.025)
    assert res.comparisons[0].ci_low == pytest.approx(lo, abs=1e-9)
    assert res.comparisons[0].ci_high == pytest.approx(hi, abs=1e-9)


def test_sparse_cells_use_fisher_exact_p_value():
    from ablift.reporting import build_markdown_report

    res = fixed((0, 50), (4, 50))
    comp = res.comparisons[0]
    expected = fisher_exact([[4, 46], [0, 50]], alternative="two-sided").pvalue
    assert comp.raw_p_value == pytest.approx(expected, rel=1e-9)
    # The normal approximation gives p ~= 0.04 here; the exact test does not reject.
    assert comp.significant is False
    assert comp.test_statistic is None
    report = build_markdown_report(res)
    assert "Fisher's exact" in report
    assert "z-statistic" not in report


def test_sparse_cells_raise_risk_flag():
    res = fixed((0, 50), (4, 50))
    assert res.recommendation is not None
    assert "sparse_conversions" in res.recommendation.risk_flags


def test_large_counts_keep_z_test_and_no_flag():
    res = fixed((400, 10_000), (460, 10_000))
    comp = res.comparisons[0]
    assert comp.test_statistic is not None
    assert res.recommendation is not None
    assert "sparse_conversions" not in res.recommendation.risk_flags


def test_coverage_near_nominal_for_rare_events():
    rng = np.random.default_rng(38)
    n, p, sims = 50, 0.02, 2000
    covered = 0
    for _ in range(sims):
        xc, xt = rng.binomial(n, p, size=2)
        comp = fixed((int(xc), n), (int(xt), n)).comparisons[0]
        covered += comp.ci_low <= 0.0 <= comp.ci_high
    # Wald undercovers badly here (zero-width intervals at 0/50 vs 0/50).
    assert covered / sims >= 0.94
