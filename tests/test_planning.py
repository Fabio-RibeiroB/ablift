import unittest

import numpy as np

from ablift.engine import msprt_statistic, two_proportion_test
from ablift.planning import (
    bayesian_duration_conversion,
    fixed_horizon_duration_conversion,
    sequential_duration_conversion,
)


class PlanningTests(unittest.TestCase):
    def test_fixed_horizon_matches_textbook_value(self):
        # 4% -> 4.2%, alpha=0.05 two-sided, power=0.8: ~154k per arm.
        plan = fixed_horizon_duration_conversion(0.04, 0.05, 50000)
        self.assertTrue(150_000 < plan.n_per_variant < 158_000, plan.n_per_variant)
        self.assertAlmostEqual(plan.estimated_days, plan.n_per_variant / 25000)

    def test_bonferroni_increases_sample_size(self):
        two = fixed_horizon_duration_conversion(0.04, 0.05, 50000, n_variants=2)
        three = fixed_horizon_duration_conversion(0.04, 0.05, 50000, n_variants=3)
        self.assertGreater(three.n_per_variant, two.n_per_variant)
        self.assertAlmostEqual(three.diagnostics["alpha_per_comparison"], 0.025)

    def test_sequential_plan_achieves_target_power(self):
        baseline, mde, power = 0.05, 0.2, 0.8
        plan = sequential_duration_conversion(baseline, mde, 10000, power=power)
        n = plan.n_per_variant
        tau = plan.assumptions["sequential_tau"]
        rng = np.random.default_rng(1)
        sims = 2000
        c = rng.binomial(n, baseline, size=sims)
        t = rng.binomial(n, baseline * (1 + mde), size=sims)
        hits = 0
        for ci, ti in zip(c, t, strict=True):
            _, _, se = two_proportion_test(int(ci), n, int(ti), n)
            hits += msprt_statistic(ti / n - ci / n, se, tau) >= 20
        self.assertAlmostEqual(hits / sims, power, delta=0.04)
        self.assertGreater(n, plan.diagnostics["fixed_horizon_n_per_variant"])

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            sequential_duration_conversion(1.5, 0.05, 1000)
        with self.assertRaises(ValueError):
            fixed_horizon_duration_conversion(0.04, 0.0, 1000)
        with self.assertRaises(ValueError):
            fixed_horizon_duration_conversion(0.04, 0.05, 1000, n_variants=1)

    def test_bayesian_duration_returns_selection_or_none(self):
        plan = bayesian_duration_conversion(
            baseline_rate=0.04,
            relative_mde=0.08,
            daily_total_traffic=60000,
            n_variants=2,
            max_days=30,
            sims=120,
            posterior_draws=800,
        )
        self.assertIn("assurance_at_selected_day", plan.diagnostics)


if __name__ == "__main__":
    unittest.main()
