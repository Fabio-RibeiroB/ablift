import math
import unittest

import numpy as np
from scipy.stats import chi2, chisquare
from statsmodels.stats.proportion import proportions_ztest

from ablift.engine import (
    always_valid_ci_margin,
    analyze,
    evaluate_srm,
    mean_and_var_from_aggregates,
    msprt_statistic,
    parse_payload,
    two_proportion_test,
)
from ablift.models import VariantInput


class EngineTests(unittest.TestCase):
    def test_bayesian_recommends_ship_for_clear_winner(self):
        payload = {
            "experiment_name": "exp",
            "method": "bayesian",
            "variants": [
                {"name": "control", "visitors": 10000, "conversions": 500, "is_control": True},
                {"name": "treatment", "visitors": 10000, "conversions": 650, "is_control": False},
            ],
            "decision_thresholds": {
                "bayes_prob_beats_control": 0.9,
                "max_expected_loss": 0.01,
            },
            "samples": 30000,
            "random_seed": 11,
        }

        result = analyze(parse_payload(payload))
        self.assertTrue(result.recommendation.action.startswith("ship_"))
        self.assertGreaterEqual(result.recommendation.decision_confidence, 0.9)

    def test_guardrail_failure_blocks_ship(self):
        payload = {
            "experiment_name": "exp",
            "method": "sequential",
            "alpha": 0.05,
            "variants": [
                {"name": "control", "visitors": 50000, "conversions": 3000, "is_control": True},
                {"name": "treatment", "visitors": 50000, "conversions": 3500, "is_control": False},
            ],
            "guardrails": [
                {
                    "name": "p95_latency_ms",
                    "control": 200,
                    "treatment": 240,
                    "direction": "decrease",
                    "max_relative_change": 0.1,
                }
            ],
        }

        result = analyze(parse_payload(payload))
        self.assertEqual(result.recommendation.action, "do_not_ship")
        self.assertIn("guardrail_failure", result.recommendation.risk_flags)

    def test_srm_failure_blocks_decision(self):
        payload = {
            "experiment_name": "exp",
            "method": "bayesian",
            "variants": [
                {"name": "control", "visitors": 9900, "conversions": 500, "is_control": True},
                {"name": "treatment", "visitors": 100, "conversions": 20, "is_control": False},
            ],
            "samples": 5000,
            "random_seed": 1,
        }

        result = analyze(parse_payload(payload))
        self.assertFalse(result.srm.passed)
        self.assertEqual(result.recommendation.action, "investigate_data_quality")

    def test_bayesian_without_decision_policy_returns_no_recommendation(self):
        payload = {
            "experiment_name": "exp",
            "method": "bayesian",
            "variants": [
                {"name": "control", "visitors": 10000, "conversions": 500, "is_control": True},
                {"name": "treatment", "visitors": 10000, "conversions": 520, "is_control": False},
            ],
        }

        result = analyze(parse_payload(payload))
        self.assertIsNone(result.recommendation)
        self.assertEqual(result.analysis_settings.samples, 50000)
        self.assertEqual(result.analysis_settings.priors["family"], "beta_binomial")

    def test_bayesian_arpu_probability_to_win(self):
        payload = {
            "experiment_name": "arpu_exp",
            "method": "bayesian",
            "primary_metric": "arpu",
            "variants": [
                {
                    "name": "control",
                    "visitors": 5000,
                    "conversions": 300,
                    "revenue_sum": 10000.0,
                    "revenue_sum_squares": 60000.0,
                    "is_control": True,
                },
                {
                    "name": "treatment",
                    "visitors": 5000,
                    "conversions": 320,
                    "revenue_sum": 11500.0,
                    "revenue_sum_squares": 77000.0,
                    "is_control": False,
                },
            ],
            "samples": 10000,
            "random_seed": 7,
            "decision_thresholds": {
                "bayes_prob_beats_control": 0.8,
                "max_expected_loss": 0.2,
            },
        }
        result = analyze(parse_payload(payload))
        comp = result.comparisons[0]
        self.assertEqual(comp.metric, "arpu")
        self.assertIsNotNone(comp.probability_beats_control)
        self.assertGreater(comp.probability_beats_control, 0.5)

    def test_multi_variant_selects_best_treatment(self):
        payload = {
            "experiment_name": "multi_variant",
            "method": "bayesian",
            "primary_metric": "conversion_rate",
            "variants": [
                {"name": "control", "visitors": 20000, "conversions": 900, "is_control": True},
                {"name": "treatment_a", "visitors": 20000, "conversions": 940, "is_control": False},
                {
                    "name": "treatment_b",
                    "visitors": 20000,
                    "conversions": 1010,
                    "is_control": False,
                },
            ],
            "samples": 15000,
            "random_seed": 9,
            "decision_thresholds": {
                "bayes_prob_beats_control": 0.9,
                "max_expected_loss": 0.01,
            },
        }
        result = analyze(parse_payload(payload))
        self.assertEqual(len(result.comparisons), 2)
        self.assertEqual(result.recommendation.action, "ship_treatment_b")

    def test_srm_matches_scipy_chi_square_tail(self):
        variants = [
            VariantInput(name="control", visitors=5200, conversions=260, is_control=True),
            VariantInput(name="treatment", visitors=4800, conversions=250, is_control=False),
        ]

        result = evaluate_srm(variants)

        total = 5200 + 4800
        expected = total / 2
        chi2_stat = ((5200 - expected) ** 2) / expected + ((4800 - expected) ** 2) / expected
        expected_p_value = chi2.sf(chi2_stat, df=1)

        self.assertAlmostEqual(result.p_value, expected_p_value, places=4)
        self.assertEqual(result.passed, expected_p_value >= 0.001)
        self.assertEqual(result.expected_proportions, [0.5, 0.5])

    def test_srm_unequal_splits_match_scipy_chisquare(self):
        cases = [
            # 2-arm 90/10 matching
            ([("control", 9012), ("v1", 988)], [0.9, 0.1], True),
            # 2-arm 90/10 skewed (passes weights in percentage scale)
            ([("control", 8500), ("v1", 1500)], [90, 10], False),
            # 3-arm 50/25/25 matching
            ([("control", 5020), ("v1", 2495), ("v2", 2485)], [0.5, 0.25, 0.25], True),
            # 3-arm 50/25/25 skewed
            ([("control", 5000), ("v1", 2800), ("v2", 2200)], [50, 25, 25], False),
            # 4-arm 40/30/20/10 matching
            (
                [("control", 4010), ("v1", 2985), ("v2", 2015), ("v3", 990)],
                [40, 30, 20, 10],
                True,
            ),
            # 4-arm 40/30/20/10 skewed
            (
                [("control", 4200), ("v1", 3000), ("v2", 2000), ("v3", 800)],
                [0.4, 0.3, 0.2, 0.1],
                False,
            ),
        ]
        for arm_counts, weights, should_pass in cases:
            with self.subTest(arm_counts=arm_counts, weights=weights):
                variants = [
                    VariantInput(name=name, visitors=n, conversions=0, is_control=(i == 0))
                    for i, (name, n) in enumerate(arm_counts)
                ]
                result = evaluate_srm(variants, expected_split=weights)
                total = sum(n for _, n in arm_counts)
                weight_sum = sum(weights)
                proportions = [w / weight_sum for w in weights]
                expected_counts = [total * p for p in proportions]
                scipy_res = chisquare([n for _, n in arm_counts], f_exp=expected_counts)

                self.assertAlmostEqual(result.p_value, float(scipy_res.pvalue), places=10)
                self.assertEqual(result.passed, should_pass)
                for got_p, exp_p in zip(result.expected_proportions, proportions, strict=True):
                    self.assertAlmostEqual(got_p, exp_p, places=12)
                for got_e, exp_e in zip(result.expected, expected_counts, strict=True):
                    self.assertAlmostEqual(got_e, exp_e, places=9)

    def test_two_proportion_test_matches_statsmodels(self):
        x1, n1 = 500, 10_000
        x2, n2 = 650, 10_000

        p_value, z_value, unpooled_se = two_proportion_test(x1, n1, x2, n2)
        expected_z, expected_p_value = proportions_ztest([x2, x1], [n2, n1])

        p1 = x1 / n1
        p2 = x2 / n2
        expected_unpooled_se = math.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))

        self.assertAlmostEqual(z_value, expected_z, places=10)
        self.assertAlmostEqual(p_value, expected_p_value, places=10)
        self.assertAlmostEqual(unpooled_se, expected_unpooled_se, places=10)

    def test_sequential_conversion_reaches_significance_for_clear_winner(self):
        payload = {
            "experiment_name": "exp",
            "method": "sequential",
            "alpha": 0.05,
            "variants": [
                {"name": "control", "visitors": 50000, "conversions": 2000, "is_control": True},
                {"name": "treatment", "visitors": 50000, "conversions": 2400, "is_control": False},
            ],
        }
        result = analyze(parse_payload(payload))
        comp = result.comparisons[0]
        self.assertIsNotNone(comp.e_value)
        self.assertIsNotNone(comp.e_threshold)
        self.assertAlmostEqual(comp.e_threshold, 20.0)  # 1/0.05
        self.assertTrue(comp.significant)
        self.assertIsNotNone(comp.ci_low)
        self.assertIsNotNone(comp.ci_high)
        self.assertGreater(comp.ci_low, 0)  # always-valid CI excludes zero for clear winner

    def test_msprt_statistic_increases_with_effect_size(self):
        # Larger effect → higher e-value
        small = msprt_statistic(delta_hat=0.001, se=0.01, tau=0.005)
        large = msprt_statistic(delta_hat=0.05, se=0.01, tau=0.005)
        self.assertGreater(large, small)

    def test_always_valid_ci_wider_than_standard(self):
        # Always-valid CI should be wider than a standard 95% CI at the same alpha
        from statistics import NormalDist

        se = 0.01
        tau = 0.005
        alpha = 0.05
        standard_margin = NormalDist().inv_cdf(1 - alpha / 2) * se
        av_margin = always_valid_ci_margin(se, tau, alpha)
        self.assertGreater(av_margin, standard_margin)

    def test_mean_and_var_from_aggregates_matches_hand_worked_example(self):
        mean, variance = mean_and_var_from_aggregates(
            n=4,
            value_sum=14.0,
            value_sum_squares=54.0,
        )

        self.assertAlmostEqual(mean, 3.5)
        self.assertAlmostEqual(variance, 5 / 3)


class SequentialGuaranteeTests(unittest.TestCase):
    def _payload(self, variants, **extra):
        return {"experiment_name": "exp", "method": "sequential", "variants": variants, **extra}

    def test_bonferroni_threshold_scales_with_treatments(self):
        variants = [
            {"name": "control", "visitors": 10000, "conversions": 400, "is_control": True},
            {"name": "v1", "visitors": 10000, "conversions": 410},
            {"name": "v2", "visitors": 10000, "conversions": 420},
        ]
        result = analyze(parse_payload(self._payload(variants)))
        for comp in result.comparisons:
            self.assertAlmostEqual(comp.e_threshold, 40.0)  # 1 / (0.05 / 2)
        mc = result.analysis_settings.multiple_comparisons
        self.assertEqual(mc["correction"], "bonferroni")
        self.assertAlmostEqual(mc["alpha_per_comparison"], 0.025)

    def test_p_value_is_inverse_e_value_and_matches_ci(self):
        for treat_conv in (400, 430, 470, 520):
            variants = [
                {"name": "control", "visitors": 20000, "conversions": 400, "is_control": True},
                {"name": "v1", "visitors": 20000, "conversions": treat_conv},
            ]
            comp = analyze(parse_payload(self._payload(variants))).comparisons[0]
            self.assertAlmostEqual(comp.p_value, min(1.0, 1.0 / comp.e_value))
            # Always-valid CI excludes zero exactly when the mSPRT rejects.
            excludes_zero = comp.ci_low > 0 or comp.ci_high < 0
            self.assertEqual(excludes_zero, comp.significant, treat_conv)

    def test_type_one_error_controlled_under_continuous_monitoring(self):
        rng = np.random.default_rng(0)
        alpha, p, sims, looks, batch = 0.05, 0.05, 2000, 50, 500
        rejected = 0
        for _ in range(sims):
            c = np.cumsum(rng.binomial(batch, p, size=looks))
            t = np.cumsum(rng.binomial(batch, p, size=looks))
            for k in range(looks):
                n = batch * (k + 1)
                _, _, se = two_proportion_test(int(c[k]), n, int(t[k]), n)
                if msprt_statistic(t[k] / n - c[k] / n, se, 0.1 * p) >= 1 / alpha:
                    rejected += 1
                    break
        # Peeking 50 times with a naive z-test gives roughly 25-30% false positives.
        self.assertLess(rejected / sims, alpha + 0.01)

    def test_rejects_invalid_tau(self):
        variants = [
            {"name": "control", "visitors": 1000, "conversions": 40, "is_control": True},
            {"name": "v1", "visitors": 1000, "conversions": 45},
        ]
        with self.assertRaisesRegex(ValueError, "sequential_tau"):
            analyze(parse_payload(self._payload(variants, sequential_tau=0)))


class BayesianOutputTests(unittest.TestCase):
    def test_interval_is_absolute_with_separate_relative_interval(self):
        payload = {
            "experiment_name": "exp",
            "method": "bayesian",
            "variants": [
                {"name": "control", "visitors": 10000, "conversions": 500, "is_control": True},
                {"name": "v1", "visitors": 10000, "conversions": 600},
            ],
            "samples": 20000,
        }
        comp = analyze(parse_payload(payload)).comparisons[0]
        self.assertLess(comp.ci_low, comp.absolute_lift)
        self.assertGreater(comp.ci_high, comp.absolute_lift)
        self.assertLess(comp.relative_ci_low, comp.relative_lift)
        self.assertGreater(comp.relative_ci_high, comp.relative_lift)
        self.assertLess(comp.ci_high, 0.05)  # absolute scale, not relative (~20%)

    def test_missing_fields_raise_value_error(self):
        with self.assertRaisesRegex(ValueError, "variants"):
            parse_payload({"experiment_name": "x", "method": "bayesian"})
        with self.assertRaisesRegex(ValueError, "Invalid input"):
            parse_payload({"method": "bayesian", "variants": [{"name": "a", "bogus": 1}]})

    def test_two_variant_probability_best_matches_pairwise_for_cr_and_arpu(self):
        cr_payload = {
            "experiment_name": "two_arm_cr",
            "method": "bayesian",
            "primary_metric": "conversion_rate",
            "variants": [
                {"name": "v1", "visitors": 10000, "conversions": 560, "is_control": False},
                {"name": "control", "visitors": 10000, "conversions": 500, "is_control": True},
            ],
            "samples": 20000,
            "random_seed": 13,
        }
        arpu_payload = {
            "experiment_name": "two_arm_arpu",
            "method": "bayesian",
            "primary_metric": "arpu",
            "variants": [
                {
                    "name": "control",
                    "visitors": 5000,
                    "conversions": 300,
                    "revenue_sum": 10000.0,
                    "revenue_sum_squares": 60000.0,
                    "is_control": True,
                },
                {
                    "name": "v1",
                    "visitors": 5000,
                    "conversions": 320,
                    "revenue_sum": 11200.0,
                    "revenue_sum_squares": 73000.0,
                    "is_control": False,
                },
            ],
            "samples": 20000,
            "random_seed": 13,
        }

        for payload in (cr_payload, arpu_payload):
            with self.subTest(metric=payload["primary_metric"]):
                result = analyze(parse_payload(payload))
                comp = result.comparisons[0]
                self.assertIsNotNone(result.bayesian_arms)
                assert result.bayesian_arms is not None
                self.assertEqual(
                    [arm.variant for arm in result.bayesian_arms],
                    [v["name"] for v in payload["variants"]],
                )
                self.assertAlmostEqual(
                    sum(arm.probability_best for arm in result.bayesian_arms), 1.0, places=12
                )
                self.assertAlmostEqual(
                    comp.probability_best, comp.probability_beats_control, places=12
                )
                self.assertAlmostEqual(comp.multi_arm_expected_loss, comp.expected_loss, places=12)

    def test_multi_arm_joint_invariants_for_cr_and_arpu(self):
        cr_payload = {
            "experiment_name": "multi_cr",
            "method": "bayesian",
            "primary_metric": "conversion_rate",
            "variants": [
                {"name": "control", "visitors": 15000, "conversions": 750, "is_control": True},
                {"name": "v1", "visitors": 15000, "conversions": 810},
                {"name": "v2", "visitors": 15000, "conversions": 830},
            ],
            "samples": 25000,
            "random_seed": 21,
        }
        arpu_payload = {
            "experiment_name": "multi_arpu",
            "method": "bayesian",
            "primary_metric": "arpu",
            "variants": [
                {
                    "name": "control",
                    "visitors": 4000,
                    "conversions": 200,
                    "revenue_sum": 8000.0,
                    "revenue_sum_squares": 50000.0,
                    "is_control": True,
                },
                {
                    "name": "v1",
                    "visitors": 4000,
                    "conversions": 220,
                    "revenue_sum": 8800.0,
                    "revenue_sum_squares": 58000.0,
                },
                {
                    "name": "v2",
                    "visitors": 4000,
                    "conversions": 230,
                    "revenue_sum": 9100.0,
                    "revenue_sum_squares": 62000.0,
                },
            ],
            "samples": 25000,
            "random_seed": 21,
        }

        for payload in (cr_payload, arpu_payload):
            with self.subTest(metric=payload["primary_metric"]):
                result = analyze(parse_payload(payload))
                self.assertIsNotNone(result.bayesian_arms)
                assert result.bayesian_arms is not None
                self.assertEqual(len(result.bayesian_arms), 3)
                self.assertAlmostEqual(
                    sum(arm.probability_best for arm in result.bayesian_arms), 1.0, places=12
                )
                arm_map = {arm.variant: arm for arm in result.bayesian_arms}
                self.assertTrue(arm_map["control"].is_control)
                self.assertGreaterEqual(arm_map["control"].multi_arm_expected_loss, 0.0)
                for comp in result.comparisons:
                    assert comp.probability_best is not None
                    assert comp.probability_beats_control is not None
                    assert comp.multi_arm_expected_loss is not None
                    assert comp.expected_loss is not None
                    self.assertLessEqual(
                        comp.probability_best, comp.probability_beats_control + 1e-12
                    )
                    self.assertGreaterEqual(
                        comp.multi_arm_expected_loss + 1e-12, comp.expected_loss
                    )
                    self.assertAlmostEqual(
                        comp.probability_best, arm_map[comp.treatment].probability_best, places=12
                    )
                    self.assertAlmostEqual(
                        comp.multi_arm_expected_loss,
                        arm_map[comp.treatment].multi_arm_expected_loss,
                        places=12,
                    )

    def test_noisy_higher_lift_arm_does_not_mask_qualifying_winner(self):
        payload = {
            "experiment_name": "noisy_vs_clean",
            "method": "bayesian",
            "primary_metric": "conversion_rate",
            "expected_split": [10000, 10000, 20],
            "variants": [
                {"name": "control", "visitors": 10000, "conversions": 500, "is_control": True},
                {"name": "clean_winner", "visitors": 10000, "conversions": 650},
                {"name": "noisy_high", "visitors": 20, "conversions": 2},
            ],
            "decision_policy": {
                "enabled": True,
                "bayes_prob_beats_control": 0.95,
                "max_expected_loss": 0.001,
            },
            "samples": 25000,
            "random_seed": 7,
        }
        result = analyze(parse_payload(payload))
        self.assertTrue(result.srm.passed)
        by_name = {c.treatment: c for c in result.comparisons}
        # Confirm noisy_high really has a higher raw lift but fails Bayesian thresholds.
        self.assertGreater(
            by_name["noisy_high"].absolute_lift, by_name["clean_winner"].absolute_lift
        )
        self.assertLess(by_name["noisy_high"].probability_beats_control, 0.95)
        self.assertIsNotNone(result.recommendation)
        assert result.recommendation is not None
        self.assertEqual(result.recommendation.action, "ship_clean_winner")
        self.assertIn("P(best)=", result.recommendation.rationale)

    def test_multi_arm_partial_loser_continues_collecting_data(self):
        payload = {
            "experiment_name": "partial_loser",
            "method": "bayesian",
            "variants": [
                {"name": "control", "visitors": 10000, "conversions": 500, "is_control": True},
                {"name": "clear_loser", "visitors": 10000, "conversions": 300},
                {"name": "inconclusive", "visitors": 10000, "conversions": 505},
            ],
            "decision_policy": {
                "enabled": True,
                "bayes_prob_beats_control": 0.95,
                "max_expected_loss": 0.001,
            },
            "samples": 20000,
            "random_seed": 7,
        }
        result = analyze(parse_payload(payload))
        self.assertIsNotNone(result.recommendation)
        assert result.recommendation is not None
        self.assertEqual(result.recommendation.action, "continue_collecting_data")

    def test_multi_arm_all_losers_returns_do_not_ship(self):
        payload = {
            "experiment_name": "all_losers",
            "method": "bayesian",
            "variants": [
                {"name": "control", "visitors": 10000, "conversions": 600, "is_control": True},
                {"name": "loser_a", "visitors": 10000, "conversions": 400},
                {"name": "loser_b", "visitors": 10000, "conversions": 350},
            ],
            "decision_policy": {
                "enabled": True,
                "bayes_prob_beats_control": 0.95,
                "max_expected_loss": 0.001,
            },
            "samples": 20000,
            "random_seed": 7,
        }
        result = analyze(parse_payload(payload))
        self.assertIsNotNone(result.recommendation)
        assert result.recommendation is not None
        self.assertEqual(result.recommendation.action, "do_not_ship")
        self.assertIn(
            "Every treatment is unlikely to beat control", result.recommendation.rationale
        )

    def test_sequential_omits_bayesian_joint_fields(self):
        payload = {
            "experiment_name": "seq_check",
            "method": "sequential",
            "variants": [
                {"name": "control", "visitors": 10000, "conversions": 500, "is_control": True},
                {"name": "v1", "visitors": 10000, "conversions": 550},
            ],
        }
        result = analyze(parse_payload(payload))
        self.assertIsNone(result.bayesian_arms)
        self.assertIsNone(result.comparisons[0].probability_best)
        self.assertIsNone(result.comparisons[0].multi_arm_expected_loss)


if __name__ == "__main__":
    unittest.main()
