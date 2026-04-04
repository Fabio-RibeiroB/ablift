import unittest

from ablift.planning import bayesian_duration_conversion, frequentist_duration_conversion
from ablift.text_parser import parse_duration_prompt


class PlanningTests(unittest.TestCase):
    def test_frequentist_duration_returns_days(self):
        plan = frequentist_duration_conversion(
            baseline_rate=0.04,
            relative_mde=0.05,
            daily_total_traffic=50000,
            n_variants=2,
            max_looks=10,
        )
        self.assertIsNotNone(plan.estimated_days)
        self.assertGreater(plan.estimated_days, 0)
        self.assertGreater(plan.n_per_variant, 0)

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

    def test_parse_duration_prompt(self):
        parsed = parse_duration_prompt(
            "Traffic: 50000 visitors/day\nBaseline: 4%\nMDE: 5%\nPower: 0.8\n"
        )
        self.assertEqual(parsed["daily_total_traffic"], 50000)
        self.assertAlmostEqual(parsed["baseline_rate"], 0.04)
        self.assertAlmostEqual(parsed["relative_mde"], 0.05)


if __name__ == "__main__":
    unittest.main()
