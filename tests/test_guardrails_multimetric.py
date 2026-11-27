import json
import math
import unittest

from click.testing import CliRunner

from ablift.cli import cli
from ablift.engine import analyze, parse_payload
from ablift.reporting import build_markdown_report


def bayes_only(method, **settings):
    """Posterior-sampling settings are only accepted (and only used) for bayesian runs."""
    return settings if method == "bayesian" else {}


def _abn_primary_variants():
    # Both v1 (+15% rel) and v2 (+25% rel) clearly beat control (4.0%).
    return [
        {"name": "control", "visitors": 50000, "conversions": 2000, "is_control": True},
        {"name": "v1", "visitors": 50000, "conversions": 2300, "is_control": False},
        {"name": "v2", "visitors": 50000, "conversions": 2500, "is_control": False},
    ]


def _abn_bounce_guardrail_v2_regressed(max_relative_change: float = 0.03):
    # control bounce = 30.0%, v1 = 30.1% (within noise), v2 = 34.0% (+13.3% relative regression)
    return {
        "name": "bounce_rate",
        "metric": "conversion_rate",
        "direction": "decrease",
        "max_relative_change": max_relative_change,
        "variants": [
            {"name": "control", "conversions": 15000},
            {"name": "v1", "conversions": 15050},
            {"name": "v2", "conversions": 17000},
        ],
    }


def _policy():
    return {
        "enabled": True,
        "bayes_prob_beats_control": 0.95,
        "max_expected_loss": 0.005,
    }


class StatisticalGuardrailTests(unittest.TestCase):
    def test_conversion_rate_guardrail_all_methods_pass_and_fail(self):
        for method in ("bayesian", "sequential", "fixed"):
            payload = {
                "experiment_name": f"gr_{method}",
                "method": method,
                **bayes_only(method, samples=4000, random_seed=11),
                "variants": _abn_primary_variants(),
                "guardrails": [_abn_bounce_guardrail_v2_regressed()],
                "decision_policy": _policy(),
            }
            result = analyze(parse_payload(payload))
            self.assertEqual(len(result.guardrails), 2, method)
            by_var = {g.variant: g for g in result.guardrails}
            self.assertEqual(set(by_var), {"v1", "v2"}, method)

            self.assertEqual(by_var["v1"].mode, "statistical")
            self.assertTrue(by_var["v1"].passed, method)
            self.assertIsNotNone(by_var["v1"].comparison)

            self.assertEqual(by_var["v2"].mode, "statistical")
            self.assertFalse(by_var["v2"].passed, method)
            self.assertIsNotNone(by_var["v2"].comparison)
            self.assertFalse(result.guardrails_passed, method)

            # v2 has higher lift, but failed bounce_rate guardrail -> ship v1!
            self.assertIsNotNone(result.recommendation, method)
            assert result.recommendation is not None
            self.assertEqual(result.recommendation.action, "ship_v1", method)
            self.assertIn("other_variant_guardrail_failure", result.recommendation.risk_flags)
            self.assertNotIn("guardrail_failure", result.recommendation.risk_flags)
            self.assertIn("v2", result.recommendation.rationale)

    def test_arpu_guardrail_direction_increase_all_methods(self):
        # Guardrail: retention/revenue per user should not drop by > 3%.
        # v1 holds steady at 10.0; v2 drops to 8.5 (-15%).
        arpu_guardrail = {
            "name": "session_revenue",
            "metric": "arpu",
            "direction": "increase",
            "max_relative_change": 0.03,
            "variants": [
                {
                    "name": "control",
                    "visitors": 20000,
                    "revenue_sum": 200000.0,
                    "revenue_sum_squares": 2500000.0,
                },
                {
                    "name": "v1",
                    "visitors": 20000,
                    "revenue_sum": 200500.0,
                    "revenue_sum_squares": 2510000.0,
                },
                {
                    "name": "v2",
                    "visitors": 20000,
                    "revenue_sum": 170000.0,
                    "revenue_sum_squares": 1950000.0,
                },
            ],
        }
        for method in ("bayesian", "sequential", "fixed"):
            payload = {
                "experiment_name": f"arpu_gr_{method}",
                "method": method,
                **bayes_only(method, samples=4000, random_seed=7),
                "variants": _abn_primary_variants(),
                "guardrails": [arpu_guardrail],
                "decision_policy": _policy(),
            }
            result = analyze(parse_payload(payload))
            by_var = {g.variant: g for g in result.guardrails}
            self.assertTrue(by_var["v1"].passed, method)
            self.assertFalse(by_var["v2"].passed, method)
            assert result.recommendation is not None
            self.assertEqual(result.recommendation.action, "ship_v1", method)

    def test_zero_tolerance_statistical_guardrail(self):
        # max_relative_change == 0.0 tests statistically significant regression in the bad direction.
        for method in ("bayesian", "sequential", "fixed"):
            payload = {
                "method": method,
                **bayes_only(method, samples=4000, random_seed=5),
                "variants": _abn_primary_variants(),
                "guardrails": [_abn_bounce_guardrail_v2_regressed(max_relative_change=0.0)],
                "decision_policy": _policy(),
            }
            result = analyze(parse_payload(payload))
            by_var = {g.variant: g for g in result.guardrails}
            self.assertTrue(by_var["v1"].passed, method)
            self.assertFalse(by_var["v2"].passed, method)

    def test_noisy_small_sample_guardrail_does_not_false_alarm(self):
        # Point estimate is +10% worse (11/100 vs 10/100), exceeding 2% threshold,
        # yet statistically indistinguishable from noise at n=100 per arm.
        for method in ("bayesian", "sequential", "fixed"):
            payload = {
                "method": method,
                **bayes_only(method, samples=4000, random_seed=7),
                "variants": [
                    {"name": "control", "visitors": 50000, "conversions": 2000, "is_control": True},
                    {"name": "v1", "visitors": 50000, "conversions": 2300, "is_control": False},
                ],
                "guardrails": [
                    {
                        "name": "error_rate",
                        "metric": "conversion_rate",
                        "direction": "decrease",
                        "max_relative_change": 0.02,
                        "variants": [
                            {"name": "control", "visitors": 100, "conversions": 10},
                            {"name": "v1", "visitors": 100, "conversions": 11},
                        ],
                    }
                ],
                "decision_policy": _policy(),
            }
            result = analyze(parse_payload(payload))
            self.assertTrue(result.guardrails[0].passed, method)
            self.assertTrue(result.guardrails_passed, method)
            assert result.recommendation is not None
            self.assertEqual(result.recommendation.action, "ship_v1", method)

    def test_all_treatments_failing_per_variant_guardrail_blocks_ship(self):
        both_fail = {
            "name": "bounce_rate",
            "metric": "conversion_rate",
            "direction": "decrease",
            "max_relative_change": 0.03,
            "variants": [
                {"name": "control", "conversions": 15000},
                {"name": "v1", "conversions": 17000},
                {"name": "v2", "conversions": 17500},
            ],
        }
        for method in ("bayesian", "sequential", "fixed"):
            payload = {
                "method": method,
                **bayes_only(method, samples=4000, random_seed=7),
                "variants": _abn_primary_variants(),
                "guardrails": [both_fail],
                "decision_policy": _policy(),
            }
            result = analyze(parse_payload(payload))
            self.assertFalse(result.guardrails_passed, method)
            assert result.recommendation is not None
            self.assertEqual(result.recommendation.action, "do_not_ship", method)
            self.assertIn("guardrail_failure", result.recommendation.risk_flags)
            self.assertIn("v1", result.recommendation.rationale)
            self.assertIn("v2", result.recommendation.rationale)

    def test_remaining_eligible_treatment_is_loser_after_guardrail_exclusion(self):
        # v1 is a significant loser on primary, v2 is a winner on primary but fails guardrail.
        variants = [
            {"name": "control", "visitors": 50000, "conversions": 2500, "is_control": True},
            {"name": "v1", "visitors": 50000, "conversions": 2000, "is_control": False},
            {"name": "v2", "visitors": 50000, "conversions": 3000, "is_control": False},
        ]
        expected_action = {
            "bayesian": "do_not_ship",
            "fixed": "do_not_ship",
            "sequential": "stop_and_rollback",
        }
        for method, action in expected_action.items():
            payload = {
                "method": method,
                **bayes_only(method, samples=4000, random_seed=7),
                "variants": variants,
                "guardrails": [_abn_bounce_guardrail_v2_regressed()],
                "decision_policy": _policy(),
            }
            result = analyze(parse_payload(payload))
            assert result.recommendation is not None
            self.assertEqual(result.recommendation.action, action, method)
            self.assertIn("guardrail_failure", result.recommendation.risk_flags)

    def test_remaining_eligible_treatment_inconclusive_after_guardrail_exclusion(self):
        # v1 is flat/inconclusive on primary, v2 wins on primary but fails guardrail.
        variants = [
            {"name": "control", "visitors": 50000, "conversions": 2000, "is_control": True},
            {"name": "v1", "visitors": 50000, "conversions": 2010, "is_control": False},
            {"name": "v2", "visitors": 50000, "conversions": 2500, "is_control": False},
        ]
        expected_action = {
            "bayesian": "continue_collecting_data",
            "sequential": "continue_collecting_data",
            "fixed": "do_not_ship",
        }
        for method, action in expected_action.items():
            payload = {
                "method": method,
                **bayes_only(method, samples=4000, random_seed=7),
                "variants": variants,
                "guardrails": [_abn_bounce_guardrail_v2_regressed()],
                "decision_policy": _policy(),
            }
            result = analyze(parse_payload(payload))
            assert result.recommendation is not None
            self.assertEqual(result.recommendation.action, action, method)
            self.assertIn("some_variants_guardrail_failure", result.recommendation.risk_flags)

    def test_legacy_threshold_guardrail_remains_backwards_compatible(self):
        payload = {
            "method": "sequential",
            "variants": _abn_primary_variants(),
            "guardrails": [
                {
                    "name": "p95_latency_ms",
                    "control": 400.0,
                    "treatment": 450.0,
                    "direction": "decrease",
                    "max_relative_change": 0.05,
                }
            ],
            "decision_policy": {"enabled": True},
        }
        result = analyze(parse_payload(payload))
        self.assertEqual(len(result.guardrails), 1)
        g = result.guardrails[0]
        self.assertIsNone(g.variant)
        self.assertEqual(g.mode, "threshold")
        self.assertIsNone(g.comparison)
        self.assertFalse(g.passed)
        assert result.recommendation is not None
        self.assertEqual(result.recommendation.action, "do_not_ship")
        self.assertEqual(result.recommendation.risk_flags, ["guardrail_failure"])


class SecondaryMetricsTests(unittest.TestCase):
    def test_secondary_metrics_conversion_and_arpu_all_methods(self):
        secondary = [
            {
                "name": "add_to_cart",
                "metric": "conversion_rate",
                "variants": [
                    {"name": "control", "conversions": 10000},
                    {"name": "v1", "conversions": 11000},
                    {"name": "v2", "conversions": 9000},
                ],
            },
            {
                "name": "revenue_per_visitor",
                "metric": "arpu",
                "variants": [
                    {
                        "name": "control",
                        "revenue_sum": 250000.0,
                        "revenue_sum_squares": 4000000.0,
                    },
                    {
                        "name": "v1",
                        "revenue_sum": 275000.0,
                        "revenue_sum_squares": 4400000.0,
                    },
                    {
                        "name": "v2",
                        "revenue_sum": 180000.0,
                        "revenue_sum_squares": 3000000.0,
                    },
                ],
            },
        ]
        for method in ("bayesian", "sequential", "fixed"):
            payload = {
                "method": method,
                **bayes_only(method, samples=4000, random_seed=7),
                "variants": _abn_primary_variants(),
                "secondary_metrics": secondary,
                "decision_policy": _policy(),
            }
            result = analyze(parse_payload(payload))
            self.assertEqual(len(result.secondary_metrics), 2, method)
            atc, rev = result.secondary_metrics
            self.assertEqual(atc.name, "add_to_cart")
            self.assertEqual(atc.metric, "conversion_rate")
            self.assertEqual([c.treatment for c in atc.comparisons], ["v1", "v2"])

            self.assertEqual(rev.name, "revenue_per_visitor")
            self.assertEqual(rev.metric, "arpu")
            self.assertEqual([c.treatment for c in rev.comparisons], ["v1", "v2"])

            if method == "bayesian":
                self.assertIsNotNone(atc.bayesian_arms)
                self.assertIsNotNone(rev.bayesian_arms)
                assert atc.bayesian_arms is not None
                self.assertEqual(len(atc.bayesian_arms), 3)
            else:
                self.assertIsNone(atc.bayesian_arms)
                self.assertIsNone(rev.bayesian_arms)

            # Secondary metric drop on v2 does NOT block shipping v2 (primary winner).
            assert result.recommendation is not None
            self.assertEqual(result.recommendation.action, "ship_v2", method)


class GuardrailAndSecondaryValidationTests(unittest.TestCase):
    def _base(self):
        return {
            "method": "fixed",
            "variants": [
                {"name": "control", "visitors": 1000, "conversions": 100, "is_control": True},
                {"name": "v1", "visitors": 1000, "conversions": 120, "is_control": False},
            ],
        }

    def test_mixing_variants_and_scalar_control_rejected(self):
        payload = self._base()
        payload["guardrails"] = [
            {
                "name": "bounce",
                "metric": "conversion_rate",
                "control": 0.3,
                "variants": [
                    {"name": "control", "conversions": 300},
                    {"name": "v1", "conversions": 310},
                ],
            }
        ]
        with self.assertRaisesRegex(ValueError, "cannot specify both"):
            analyze(parse_payload(payload))

    def test_statistical_guardrail_requires_metric(self):
        payload = self._base()
        payload["guardrails"] = [
            {
                "name": "bounce",
                "variants": [
                    {"name": "control", "conversions": 300},
                    {"name": "v1", "conversions": 310},
                ],
            }
        ]
        with self.assertRaisesRegex(ValueError, "requires 'metric'"):
            analyze(parse_payload(payload))

    def test_threshold_guardrail_rejects_metric_without_variants(self):
        payload = self._base()
        payload["guardrails"] = [
            {
                "name": "bounce",
                "metric": "conversion_rate",
                "control": 0.3,
                "treatment": 0.31,
            }
        ]
        with self.assertRaisesRegex(ValueError, "requires 'variants'"):
            analyze(parse_payload(payload))

    def test_missing_and_unknown_variant_names_rejected(self):
        payload = self._base()
        payload["guardrails"] = [
            {
                "name": "bounce",
                "metric": "conversion_rate",
                "variants": [{"name": "control", "conversions": 300}],
            }
        ]
        with self.assertRaisesRegex(ValueError, "missing variant"):
            analyze(parse_payload(payload))

        payload["guardrails"] = [
            {
                "name": "bounce",
                "metric": "conversion_rate",
                "variants": [
                    {"name": "control", "conversions": 300},
                    {"name": "v1", "conversions": 310},
                    {"name": "ghost", "conversions": 300},
                ],
            }
        ]
        with self.assertRaisesRegex(ValueError, "unknown variant"):
            analyze(parse_payload(payload))

    def test_contradicting_is_control_rejected(self):
        payload = self._base()
        payload["secondary_metrics"] = [
            {
                "name": "clicks",
                "metric": "conversion_rate",
                "variants": [
                    {"name": "control", "conversions": 200, "is_control": False},
                    {"name": "v1", "conversions": 210},
                ],
            }
        ]
        with self.assertRaisesRegex(ValueError, "contradicts primary variant"):
            parse_payload(payload)

    def test_non_finite_and_negative_max_relative_change_rejected(self):
        for bad in (-0.01, math.nan, math.inf, True):
            payload = self._base()
            payload["guardrails"] = [
                {
                    "name": "latency",
                    "control": 100.0,
                    "treatment": 105.0,
                    "max_relative_change": bad,
                }
            ]
            with self.assertRaisesRegex(ValueError, "max_relative_change"):
                analyze(parse_payload(payload))

    def test_unknown_secondary_metric_key_rejected(self):
        payload = self._base()
        payload["secondary_metrics"] = [
            {
                "name": "clicks",
                "metric": "conversion_rate",
                "direction": "increase",
                "variants": [
                    {"name": "control", "conversions": 200},
                    {"name": "v1", "conversions": 210},
                ],
            }
        ]
        with self.assertRaisesRegex(ValueError, "unknown field"):
            parse_payload(payload)

    def test_non_list_guardrails_or_secondary_metrics_rejected(self):
        payload = self._base()
        payload["guardrails"] = "invalid"
        with self.assertRaisesRegex(ValueError, "'guardrails' must be a list"):
            parse_payload(payload)

        payload = self._base()
        payload["secondary_metrics"] = {"name": "clicks"}
        with self.assertRaisesRegex(ValueError, "'secondary_metrics' must be a list"):
            parse_payload(payload)


class GuardrailAndSecondaryReportingAndCliTests(unittest.TestCase):
    def test_markdown_report_renders_statistical_guardrails_and_secondary_metrics(self):
        payload = {
            "experiment_name": "multi_metric_demo",
            "method": "bayesian",
            "samples": 2000,
            "random_seed": 7,
            "variants": _abn_primary_variants(),
            "guardrails": [
                _abn_bounce_guardrail_v2_regressed(),
                {
                    "name": "p95_latency_ms",
                    "control": 400.0,
                    "treatment": 410.0,
                    "direction": "decrease",
                    "max_relative_change": 0.05,
                },
            ],
            "secondary_metrics": [
                {
                    "name": "add_to_cart",
                    "metric": "conversion_rate",
                    "variants": [
                        {"name": "control", "conversions": 10000},
                        {"name": "v1", "conversions": 10500},
                        {"name": "v2", "conversions": 10800},
                    ],
                }
            ],
            "decision_policy": _policy(),
        }
        report = build_markdown_report(analyze(parse_payload(payload)))
        self.assertIn("- `bounce_rate` (`v1`): pass=`True`", report)
        self.assertIn("- `bounce_rate` (`v2`): pass=`False`", report)
        self.assertIn("  - mode: `statistical`", report)
        self.assertIn("  - interval (absolute lift):", report)
        self.assertIn("- `p95_latency_ms`: pass=`True`", report)
        self.assertIn("## Secondary Metrics", report)
        self.assertIn("### add_to_cart (`conversion_rate`)", report)

    def test_example_input_cli_round_trip_includes_statistical_guardrail_and_secondary(self):
        runner = CliRunner()
        ex_res = runner.invoke(cli, ["example-input"])
        self.assertEqual(ex_res.exit_code, 0, ex_res.output)
        example = json.loads(ex_res.output)
        self.assertEqual(len(example["guardrails"]), 2)
        self.assertIn("variants", example["guardrails"][0])
        self.assertEqual(len(example["secondary_metrics"]), 1)

        an_res = runner.invoke(cli, ["analyze", "--input", "-"], input=ex_res.output)
        self.assertEqual(an_res.exit_code, 0, an_res.output)
        data = json.loads(an_res.output)
        self.assertTrue(data["guardrails_passed"])
        self.assertEqual(data["guardrails"][0]["mode"], "statistical")
        self.assertEqual(data["guardrails"][0]["variant"], "treatment_a")
        self.assertIsNotNone(data["guardrails"][0]["comparison"])
        self.assertEqual(data["guardrails"][1]["mode"], "threshold")
        self.assertIsNone(data["guardrails"][1]["variant"])
        self.assertEqual(len(data["secondary_metrics"]), 1)
        self.assertEqual(data["secondary_metrics"][0]["name"], "revenue_per_visitor")


if __name__ == "__main__":
    unittest.main()
