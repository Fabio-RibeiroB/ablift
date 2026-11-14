"""Edge-case and realistic-usage tests for ablift.engine and ablift.planning."""

import math
import unittest

import numpy as np

from ablift.engine import (
    always_valid_ci_margin,
    analyze,
    evaluate_guardrails,
    evaluate_srm,
    msprt_statistic,
    parse_payload,
    validate_input,
)
from ablift.models import GuardrailInput, VariantInput
from ablift.planning import (
    bayesian_duration_conversion,
    fixed_horizon_duration_conversion,
    sequential_duration_conversion,
)


def var(name, visitors, conversions=0, control=False, rev=None, sq=None):
    return {
        "name": name,
        "visitors": visitors,
        "conversions": conversions,
        "is_control": control,
        "revenue_sum": rev,
        "revenue_sum_squares": sq,
    }


def build(variants, **kw):
    payload = {"variants": variants, "samples": 3000}
    payload.update(kw)
    return parse_payload(payload)


def run(variants, **kw):
    return analyze(build(variants, **kw))


def two_arm(c_conv, t_conv, n=100_000, **kw):
    return [var("control", n, c_conv, True), var("v1", n, t_conv)]


def all_finite(testcase, comparison):
    for field in (
        "control_rate",
        "treatment_rate",
        "absolute_lift",
        "relative_lift",
        "p_value",
        "e_value",
        "probability_beats_control",
        "expected_loss",
        "ci_low",
        "ci_high",
        "relative_ci_low",
        "relative_ci_high",
    ):
        value = getattr(comparison, field)
        if value is not None:
            testcase.assertTrue(math.isfinite(value), f"{field}={value}")


class ValidateInputErrors(unittest.TestCase):
    def check(self, variants, message, **kw):
        with self.assertRaises(ValueError) as ctx:
            validate_input(build(variants, **kw))
        self.assertIn(message, str(ctx.exception))

    def test_single_variant(self):
        self.check([var("control", 100, 5, True)], "At least 2 variants")

    def test_no_variants(self):
        self.check([], "At least 2 variants")

    def test_no_control(self):
        self.check([var("a", 100, 5), var("b", 100, 6)], "Exactly one variant")

    def test_two_controls(self):
        self.check([var("a", 100, 5, True), var("b", 100, 6, True)], "Exactly one variant")

    def test_zero_visitors(self):
        self.check([var("control", 0, 0, True), var("v1", 100, 5)], "visitors must be positive")

    def test_negative_visitors(self):
        self.check(
            [var("control", 100, 5, True), var("v1", -5, 0)], "visitors must be positive for v1"
        )

    def test_negative_conversions(self):
        self.check(
            [var("control", 100, -1, True), var("v1", 100, 5)],
            "conversions must be non-negative for control",
        )

    def test_conversions_exceed_visitors(self):
        self.check(
            [var("control", 100, 5, True), var("v1", 100, 101)],
            "conversions cannot exceed visitors for v1",
        )

    def test_alpha_bounds(self):
        for alpha in (0, 1, -0.1, 1.5):
            with self.subTest(alpha=alpha):
                self.check(two_arm(5, 6, 100), "alpha must be in (0, 1)", alpha=alpha)

    def test_unknown_method(self):
        self.check(two_arm(5, 6, 100), "method must be", method="frequentist")

    def test_unknown_metric(self):
        self.check(two_arm(5, 6, 100), "primary_metric must be", primary_metric="aov")

    def test_bad_probability_threshold(self):
        for p in (0, 1, 1.2):
            with self.subTest(p=p):
                self.check(
                    two_arm(5, 6, 100),
                    "bayes_prob_beats_control must be in (0, 1)",
                    decision_policy={"bayes_prob_beats_control": p},
                )

    def test_negative_expected_loss_threshold(self):
        self.check(
            two_arm(5, 6, 100),
            "max_expected_loss must be non-negative",
            decision_thresholds={"max_expected_loss": -0.1},
        )

    def test_enabled_policy_without_thresholds(self):
        inp = build(two_arm(5, 6, 100))
        inp.decision_policy.enabled = True
        inp.decision_policy.thresholds = None
        with self.assertRaisesRegex(ValueError, "thresholds were not provided"):
            validate_input(inp)

    def test_arpu_missing_revenue(self):
        self.check(
            [var("control", 100, 5, True, rev=10.0, sq=20.0), var("v1", 100, 6)],
            "ARPU requires revenue_sum and revenue_sum_squares for v1",
            primary_metric="arpu",
        )

    def test_samples_non_positive(self):
        self.check(two_arm(5, 6, 100), "samples must be positive", samples=0)

    def test_tau_non_positive(self):
        self.check(
            two_arm(5, 6, 100),
            "sequential_tau must be positive",
            method="sequential",
            sequential_tau=0,
        )

    def test_disabled_policy_skips_threshold_validation(self):
        res = run(two_arm(5, 6, 100), decision_policy={"enabled": False})
        self.assertIsNone(res.recommendation)

    def test_parse_payload_errors(self):
        with self.assertRaisesRegex(ValueError, "missing required field"):
            parse_payload({})
        with self.assertRaisesRegex(ValueError, "JSON object"):
            parse_payload([])  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "Invalid input structure"):
            parse_payload({"variants": [{"name": "a", "visitors": 1, "bogus": 2}]})


class DegenerateData(unittest.TestCase):
    def test_zero_conversions_everywhere_bayesian(self):
        res = run(two_arm(0, 0))
        c = res.comparisons[0]
        all_finite(self, c)
        self.assertEqual(c.absolute_lift, 0.0)
        self.assertAlmostEqual(c.probability_beats_control, 0.5, delta=0.05)
        self.assertGreaterEqual(c.expected_loss, 0.0)

    def test_zero_conversions_everywhere_sequential(self):
        res = run(two_arm(0, 0), method="sequential")
        c = res.comparisons[0]
        all_finite(self, c)
        self.assertFalse(c.significant)
        self.assertEqual(res.recommendation.action, "continue_collecting_data")

    def test_hundred_percent_conversion_in_treatment(self):
        for method in ("bayesian", "sequential"):
            with self.subTest(method=method):
                res = run([var("control", 1000, 50, True), var("v1", 1000, 1000)], method=method)
                c = res.comparisons[0]
                all_finite(self, c)
                self.assertAlmostEqual(c.treatment_rate, 1.0)
                self.assertGreater(c.absolute_lift, 0.9)

    def test_hundred_percent_in_all_arms(self):
        for method in ("bayesian", "sequential"):
            with self.subTest(method=method):
                res = run([var("control", 500, 500, True), var("v1", 500, 500)], method=method)
                all_finite(self, res.comparisons[0])
                self.assertEqual(res.comparisons[0].absolute_lift, 0.0)

    def test_zero_control_conversions_positive_treatment(self):
        res = run([var("control", 1000, 0, True), var("v1", 1000, 10)], method="sequential")
        all_finite(self, res.comparisons[0])
        self.assertGreater(res.comparisons[0].absolute_lift, 0)

    def test_tiny_samples_are_finite(self):
        for n, c, t in ((1, 0, 1), (1, 1, 0), (2, 1, 1), (2, 0, 2)):
            for method in ("bayesian", "sequential"):
                with self.subTest(n=n, c=c, t=t, method=method):
                    res = run(two_arm(c, t, n), method=method)
                    all_finite(self, res.comparisons[0])

    def test_tiny_samples_bayesian_never_ships(self):
        for n, c, t in ((1, 0, 1), (2, 0, 2)):
            with self.subTest(n=n):
                res = run(two_arm(c, t, n), decision_policy={"enabled": True})
                self.assertEqual(res.recommendation.action, "continue_collecting_data")

    def test_tiny_samples_sequential_not_significant(self):
        # Regression (was a bug): engine.py two_proportion_test floors the unpooled variance at 1e-18, so
        # 0/1 vs 1/1 (or 0/2 vs 2/2) gives se=1e-9 and an astronomically large e-value:
        # the mSPRT "ships" a variant on one or two users.
        for n, c, t in ((1, 0, 1), (2, 0, 2)):
            res = run(two_arm(c, t, n), method="sequential")
            self.assertFalse(res.comparisons[0].significant, f"n={n}")
            self.assertNotEqual(res.recommendation.action, "ship_v1")

    def test_arpu_zero_variance_bayesian(self):
        # Every user spends exactly 10.
        n = 500
        res = run(
            [
                var("control", n, n, True, rev=10.0 * n, sq=100.0 * n),
                var("v1", n, n, rev=12.0 * n, sq=144.0 * n),
            ],
            primary_metric="arpu",
        )
        c = res.comparisons[0]
        all_finite(self, c)
        self.assertAlmostEqual(c.absolute_lift, 2.0)
        self.assertAlmostEqual(c.relative_lift, 0.2)
        self.assertGreater(c.probability_beats_control, 0.99)

    def test_arpu_zero_variance_sequential(self):
        n = 500
        res = run(
            [
                var("control", n, n, True, rev=10.0 * n, sq=100.0 * n),
                var("v1", n, n, rev=12.0 * n, sq=144.0 * n),
            ],
            primary_metric="arpu",
            method="sequential",
        )
        c = res.comparisons[0]
        all_finite(self, c)
        self.assertTrue(c.significant)
        self.assertEqual(res.recommendation.action, "ship_v1")

    def test_arpu_all_zero_revenue(self):
        for method in ("bayesian", "sequential"):
            with self.subTest(method=method):
                res = run(
                    [
                        var("control", 300, 0, True, rev=0.0, sq=0.0),
                        var("v1", 300, 0, rev=0.0, sq=0.0),
                    ],
                    primary_metric="arpu",
                    method=method,
                )
                c = res.comparisons[0]
                all_finite(self, c)
                self.assertEqual(c.absolute_lift, 0.0)

    def test_arpu_single_user_arms(self):
        res = run(
            [var("control", 1, 1, True, rev=5.0, sq=25.0), var("v1", 1, 1, rev=6.0, sq=36.0)],
            primary_metric="arpu",
            method="sequential",
        )
        all_finite(self, res.comparisons[0])

    def test_arpu_single_user_arms_not_significant(self):
        # Regression (was a bug): engine.py analyze_sequential_arpu: n=1 gives variance 0, se is floored at
        # 1e-9, so a single user per arm is declared significant.
        res = run(
            [var("control", 1, 1, True, rev=5.0, sq=25.0), var("v1", 1, 1, rev=6.0, sq=36.0)],
            primary_metric="arpu",
            method="sequential",
        )
        self.assertFalse(res.comparisons[0].significant)

    def test_arpu_float_rounding_negative_variance_is_clamped(self):
        # sum_squares slightly below sum^2/n due to rounding must not yield NaN.
        res = run(
            [
                var("control", 10, 10, True, rev=30.0, sq=89.9999999),
                var("v1", 10, 10, rev=31.0, sq=96.1),
            ],
            primary_metric="arpu",
            method="sequential",
        )
        all_finite(self, res.comparisons[0])


class SrmBehaviour(unittest.TestCase):
    def test_uneven_split_70_30_flagged(self):
        res = run([var("control", 70_000, 2800, True), var("v1", 30_000, 1200)])
        self.assertFalse(res.srm.passed)
        self.assertEqual(res.recommendation.action, "investigate_data_quality")
        self.assertIn("srm_detected", res.recommendation.risk_flags)

    def test_srm_assumes_equal_split_documented(self):
        # An intentional 90/10 rollout is indistinguishable from an SRM here: the check
        # always compares against an equal split.
        srm = evaluate_srm([VariantInput("a", 90_000), VariantInput("b", 10_000)])
        self.assertFalse(srm.passed)
        self.assertEqual(srm.expected, [50_000.0, 50_000.0])

    def test_mild_imbalance_passes(self):
        srm = evaluate_srm([VariantInput("a", 50_000), VariantInput("b", 50_300)])
        self.assertTrue(srm.passed)
        self.assertGreater(srm.p_value, 0.001)

    def test_perfect_split_p_value_near_one(self):
        srm = evaluate_srm([VariantInput("a", 1000), VariantInput("b", 1000)])
        self.assertTrue(srm.passed)
        self.assertGreater(srm.p_value, 0.9)

    def test_perfect_split_p_value_is_one(self):
        # Regression (was a bug): engine.py evaluate_srm: Wilson-Hilferty with chi2=0 is clamped to x=1e-12
        # and yields p~0.95 for a perfectly equal split (exact chi-square p-value is 1.0).
        srm = evaluate_srm([VariantInput("a", 1000), VariantInput("b", 1000)])
        self.assertGreater(srm.p_value, 0.99)

    def test_three_arm_srm(self):
        srm = evaluate_srm(
            [VariantInput("a", 10_000), VariantInput("b", 10_000), VariantInput("c", 7_000)]
        )
        self.assertFalse(srm.passed)
        self.assertEqual(len(srm.expected), 3)

    def test_srm_p_value_in_unit_interval(self):
        srm = evaluate_srm([VariantInput("a", 1), VariantInput("b", 10**9)])
        self.assertTrue(0.0 <= srm.p_value <= 1.0)
        self.assertFalse(srm.passed)


class SequentialRecommendations(unittest.TestCase):
    def arms(self, *convs, n=100_000):
        names = ["control"] + [f"v{i}" for i in range(1, len(convs))]
        return [var(nm, n, c, i == 0) for i, (nm, c) in enumerate(zip(names, convs, strict=True))]

    def test_ships_winner_when_other_arm_not_significant(self):
        res = run(self.arms(4000, 4500, 4010), method="sequential")
        sig = {c.treatment: c.significant for c in res.comparisons}
        self.assertEqual(sig, {"v1": True, "v2": False})
        self.assertEqual(res.recommendation.action, "ship_v1")
        self.assertEqual(res.recommendation.risk_flags, [])

    def test_ships_best_of_two_winners(self):
        res = run(self.arms(4000, 4400, 4600), method="sequential")
        self.assertEqual(res.recommendation.action, "ship_v2")

    def test_winner_with_loser_flags_other_variant_worse(self):
        res = run(self.arms(4000, 4500, 3500), method="sequential")
        self.assertEqual(res.recommendation.action, "ship_v1")
        self.assertIn("other_variant_significantly_worse", res.recommendation.risk_flags)

    def test_stop_and_rollback_when_all_arms_worse(self):
        res = run(self.arms(4000, 3500, 3400), method="sequential")
        self.assertTrue(all(c.significant for c in res.comparisons))
        self.assertEqual(res.recommendation.action, "stop_and_rollback")
        self.assertGreater(res.recommendation.decision_confidence, 0.99)

    def test_single_significant_loser_in_two_arm_test_rolls_back(self):
        res = run(self.arms(4000, 3500), method="sequential")
        self.assertEqual(res.recommendation.action, "stop_and_rollback")

    def test_some_worse_but_not_all_continues_with_flag(self):
        res = run(self.arms(4000, 3500, 4005), method="sequential")
        self.assertEqual(res.recommendation.action, "continue_collecting_data")
        self.assertIn("some_variants_significantly_worse", res.recommendation.risk_flags)

    def test_nothing_significant_continues(self):
        res = run(self.arms(4000, 4020, 3990), method="sequential")
        self.assertEqual(res.recommendation.action, "continue_collecting_data")
        self.assertEqual(res.recommendation.risk_flags, [])
        self.assertTrue(all(not c.significant for c in res.comparisons))

    def test_sequential_arpu_clear_winner(self):
        rng = np.random.default_rng(3)
        n = 5000
        ctrl = rng.gamma(2.0, 5.0, n)
        trt = rng.gamma(2.0, 6.0, n)
        res = run(
            [
                var(
                    "control",
                    n,
                    int((ctrl > 0).sum()),
                    True,
                    float(ctrl.sum()),
                    float((ctrl**2).sum()),
                ),
                var("v1", n, int((trt > 0).sum()), False, float(trt.sum()), float((trt**2).sum())),
            ],
            primary_metric="arpu",
            method="sequential",
        )
        self.assertTrue(res.comparisons[0].significant)
        self.assertGreater(res.comparisons[0].absolute_lift, 0)
        self.assertEqual(res.recommendation.action, "ship_v1")

    def test_sequential_arpu_identical_not_significant(self):
        res = run(
            [
                var("control", 1000, 1000, True, rev=10_000.0, sq=120_000.0),
                var("v1", 1000, 1000, rev=10_000.0, sq=120_000.0),
            ],
            primary_metric="arpu",
            method="sequential",
        )
        self.assertFalse(res.comparisons[0].significant)
        self.assertAlmostEqual(res.comparisons[0].p_value, 1.0, delta=0.2)

    def test_custom_tau_changes_evalue(self):
        small = run(self.arms(4000, 4100), method="sequential", sequential_tau=0.0001)
        big = run(self.arms(4000, 4100), method="sequential", sequential_tau=0.05)
        self.assertNotEqual(small.comparisons[0].e_value, big.comparisons[0].e_value)


class BayesianBehaviour(unittest.TestCase):
    policy = {"decision_policy": {"bayes_prob_beats_control": 0.95, "max_expected_loss": 0.001}}

    def test_do_not_ship_for_clear_loser(self):
        res = run(two_arm(4000, 3500), **self.policy)
        self.assertEqual(res.recommendation.action, "do_not_ship")
        self.assertLess(res.comparisons[0].probability_beats_control, 0.01)
        self.assertGreater(res.recommendation.decision_confidence, 0.99)

    def test_continue_for_inconclusive(self):
        res = run(two_arm(400, 410, 10_000), **self.policy)
        self.assertEqual(res.recommendation.action, "continue_collecting_data")
        self.assertTrue(0.3 < res.recommendation.decision_confidence < 0.9)

    def test_ship_requires_low_expected_loss(self):
        # High P(win) but a loose-enough loss cap is required: tighten it to zero.
        res = run(
            two_arm(4000, 4300),
            decision_policy={"bayes_prob_beats_control": 0.95, "max_expected_loss": 0.0},
        )
        self.assertEqual(res.recommendation.action, "continue_collecting_data")

    def test_same_seed_deterministic_different_seed_differs(self):
        a = run(two_arm(400, 430, 10_000), random_seed=11).comparisons[0]
        b = run(two_arm(400, 430, 10_000), random_seed=11).comparisons[0]
        c = run(two_arm(400, 430, 10_000), random_seed=12).comparisons[0]
        self.assertEqual(a, b)
        self.assertNotEqual(a.probability_beats_control, c.probability_beats_control)
        self.assertAlmostEqual(a.probability_beats_control, c.probability_beats_control, delta=0.05)

    def test_identical_arms_p_win_near_half(self):
        res = run(two_arm(2000, 2000), samples=5000)
        self.assertAlmostEqual(res.comparisons[0].probability_beats_control, 0.5, delta=0.05)
        self.assertGreater(res.comparisons[0].expected_loss, 0.0)

    def test_expected_loss_non_negative_and_complementary(self):
        res = run(
            [var("control", 5000, 250, True), var("v1", 5000, 300), var("v2", 5000, 200)],
            samples=5000,
        )
        by = {c.treatment: c for c in res.comparisons}
        for c in by.values():
            self.assertGreaterEqual(c.expected_loss, 0.0)
        self.assertLess(by["v1"].expected_loss, by["v2"].expected_loss)
        self.assertGreater(by["v1"].probability_beats_control, by["v2"].probability_beats_control)

    def test_credible_interval_contains_observed_lift(self):
        c = run(two_arm(400, 460, 10_000), samples=5000).comparisons[0]
        self.assertLess(c.ci_low, c.absolute_lift)
        self.assertGreater(c.ci_high, c.absolute_lift)
        self.assertLess(c.relative_ci_low, c.relative_ci_high)

    def test_arpu_posterior_mean_close_to_sample_mean(self):
        rng = np.random.default_rng(5)
        n = 4000
        x = rng.gamma(2.0, 4.0, n)
        y = x.copy()
        res = run(
            [
                var("control", n, n, True, float(x.sum()), float((x**2).sum())),
                var("v1", n, n, False, float(y.sum()) * 1.1, float((y**2).sum()) * 1.21),
            ],
            primary_metric="arpu",
            samples=5000,
        )
        c = res.comparisons[0]
        self.assertAlmostEqual(c.control_rate, x.mean())
        self.assertAlmostEqual(c.treatment_rate, 1.1 * x.mean())
        # Posterior mean of the difference ~ observed difference; CI must bracket it.
        self.assertLess(c.ci_low, c.absolute_lift)
        self.assertGreater(c.ci_high, c.absolute_lift)
        self.assertGreater(c.probability_beats_control, 0.95)

    def test_arpu_identical_arms_p_win_near_half(self):
        res = run(
            [
                var("control", 2000, 100, True, rev=10_000.0, sq=90_000.0),
                var("v1", 2000, 100, rev=10_000.0, sq=90_000.0),
            ],
            primary_metric="arpu",
            samples=5000,
        )
        self.assertAlmostEqual(res.comparisons[0].probability_beats_control, 0.5, delta=0.06)


class GuardrailTests(unittest.TestCase):
    def eval(self, **kw):
        inp = build(two_arm(5, 6, 100), guardrails=[dict(name="g", **kw)])
        return evaluate_guardrails(inp)[0]

    def test_decrease_direction_pass_and_fail(self):
        ok = self.eval(control=100, treatment=104, direction="decrease", max_relative_change=0.05)
        bad = self.eval(control=100, treatment=106, direction="decrease", max_relative_change=0.05)
        self.assertTrue(ok.passed)
        self.assertFalse(bad.passed)
        self.assertAlmostEqual(bad.relative_change, 0.06)
        self.assertIn("allowed increase", bad.reason)

    def test_increase_direction_pass_and_fail(self):
        ok = self.eval(control=100, treatment=96, direction="increase", max_relative_change=0.05)
        bad = self.eval(control=100, treatment=90, direction="increase", max_relative_change=0.05)
        self.assertTrue(ok.passed)
        self.assertFalse(bad.passed)
        self.assertIn("allowed decrease", bad.reason)

    def test_direction_is_normalised(self):
        g = self.eval(control=100, treatment=100, direction="  INCREASE ")
        self.assertEqual(g.direction, "increase")

    def test_invalid_direction(self):
        with self.assertRaisesRegex(ValueError, "direction must be 'increase' or 'decrease' for g"):
            self.eval(control=1, treatment=1, direction="sideways")

    def test_control_zero_uses_absolute_change(self):
        worse = self.eval(control=0, treatment=0.02, direction="decrease", max_relative_change=0.01)
        self.assertFalse(worse.passed)
        self.assertAlmostEqual(worse.relative_change, 0.02)
        fine = self.eval(control=0, treatment=0, direction="decrease")
        self.assertTrue(fine.passed)

    def test_negative_control_uses_abs(self):
        g = self.eval(control=-100, treatment=-90, direction="increase", max_relative_change=0.0)
        self.assertAlmostEqual(g.relative_change, 0.1)
        self.assertTrue(g.passed)

    def test_zero_tolerance_default(self):
        g = self.eval(control=100, treatment=100.01)
        self.assertFalse(g.passed)

    def test_guardrail_failure_overrides_winner(self):
        res = run(
            two_arm(4000, 4500),
            method="sequential",
            guardrails=[
                {"name": "latency", "control": 200, "treatment": 300, "max_relative_change": 0.1}
            ],
        )
        self.assertTrue(res.comparisons[0].significant)
        self.assertFalse(res.guardrails_passed)
        self.assertEqual(res.recommendation.action, "do_not_ship")
        self.assertIn("guardrail_failure", res.recommendation.risk_flags)
        self.assertEqual(res.recommendation.decision_confidence, 1.0)

    def test_guardrail_failure_applies_to_bayesian_even_without_policy(self):
        res = run(
            two_arm(4000, 4500),
            guardrails=[{"name": "g", "control": 1, "treatment": 2}],
        )
        self.assertEqual(res.recommendation.action, "do_not_ship")

    def test_srm_overrides_guardrail(self):
        res = run(
            [var("control", 70_000, 2800, True), var("v1", 30_000, 1500)],
            guardrails=[{"name": "g", "control": 1, "treatment": 5}],
        )
        self.assertEqual(res.recommendation.action, "investigate_data_quality")
        self.assertEqual(res.recommendation.risk_flags, ["srm_detected", "guardrail_failure"])

    def test_invalid_direction_surfaces_through_analyze(self):
        with self.assertRaises(ValueError):
            run(
                two_arm(5, 6, 100),
                guardrails=[{"name": "g", "control": 1, "treatment": 1, "direction": "up"}],
            )

    def test_guardrail_dataclass_defaults(self):
        g = GuardrailInput("x", 1.0, 1.0)
        self.assertEqual((g.direction, g.max_relative_change), ("decrease", 0.0))


class MsprtMath(unittest.TestCase):
    def test_non_positive_se_or_tau_returns_neutral(self):
        self.assertEqual(msprt_statistic(0.5, 0.0, 0.1), 1.0)
        self.assertEqual(msprt_statistic(0.5, -1.0, 0.1), 1.0)
        self.assertEqual(msprt_statistic(0.5, 0.1, 0.0), 1.0)
        self.assertEqual(msprt_statistic(0.5, 0.1, -0.1), 1.0)

    def test_margin_infinite_for_invalid_inputs(self):
        self.assertEqual(always_valid_ci_margin(0.0, 0.1, 0.05), float("inf"))
        self.assertEqual(always_valid_ci_margin(0.1, 0.0, 0.05), float("inf"))

    def test_zero_effect_evalue_below_one(self):
        e = msprt_statistic(0.0, 0.01, 0.01)
        self.assertLess(e, 1.0)
        self.assertGreater(e, 0.0)
        self.assertAlmostEqual(e, math.sqrt(0.5))  # sqrt(v/(v+t2)) with v == t2

    def test_symmetric_in_sign(self):
        self.assertAlmostEqual(
            msprt_statistic(0.02, 0.01, 0.02), msprt_statistic(-0.02, 0.01, 0.02)
        )

    def test_overflow_safe_for_huge_effects(self):
        e = msprt_statistic(1e6, 1e-3, 0.1)
        self.assertTrue(math.isfinite(e))
        self.assertGreater(e, 1e100)
        e2 = msprt_statistic(1e300, 1e-6, 1e-3)
        self.assertTrue(math.isfinite(e2))

    def test_huge_effect_end_to_end_is_finite(self):
        res = run(
            [var("control", 10**9, 1, True), var("v1", 10**9, 5 * 10**8)], method="sequential"
        )
        c = res.comparisons[0]
        all_finite(self, c)
        self.assertTrue(c.significant)
        self.assertEqual(c.p_value, 1.0 / c.e_value)

    def test_margin_shrinks_with_smaller_se(self):
        self.assertLess(
            always_valid_ci_margin(0.001, 0.01, 0.05), always_valid_ci_margin(0.01, 0.01, 0.05)
        )

    def test_margin_wider_for_smaller_alpha(self):
        self.assertGreater(
            always_valid_ci_margin(0.01, 0.01, 0.001), always_valid_ci_margin(0.01, 0.01, 0.05)
        )

    def test_ci_excludes_zero_iff_significant(self):
        for t_conv in (4000, 4050, 4150, 4300, 4600):
            with self.subTest(t=t_conv):
                c = run(two_arm(4000, t_conv), method="sequential").comparisons[0]
                excludes_zero = c.ci_low > 0 or c.ci_high < 0
                self.assertEqual(excludes_zero, c.significant)


class RealisticScenarios(unittest.TestCase):
    def test_three_arm_checkout_test_one_winner(self):
        n = 200_000
        # 4.0% baseline; variant A +5% relative (not clearly), variant B +10% relative.
        variants = [
            var("control", n, 8000, True),
            var("one_page", n, 8100),
            var("express_pay", n, 8800),
        ]
        bayes = run(
            variants,
            samples=4000,
            decision_policy={"bayes_prob_beats_control": 0.95, "max_expected_loss": 0.0005},
        )
        by = {c.treatment: c for c in bayes.comparisons}
        self.assertTrue(bayes.srm.passed)
        self.assertGreater(by["express_pay"].probability_beats_control, 0.999)
        self.assertLess(by["one_page"].probability_beats_control, 0.8)
        self.assertAlmostEqual(by["express_pay"].relative_lift, 0.10)
        self.assertEqual(bayes.recommendation.action, "ship_express_pay")

        seq = run(variants, method="sequential")
        sig = {c.treatment: c.significant for c in seq.comparisons}
        self.assertEqual(sig, {"one_page": False, "express_pay": True})
        self.assertEqual(seq.recommendation.action, "ship_express_pay")
        # Bonferroni with two treatments.
        self.assertAlmostEqual(seq.comparisons[0].e_threshold, 40.0)
        self.assertAlmostEqual(
            seq.analysis_settings.multiple_comparisons["alpha_per_comparison"], 0.025
        )

    def test_revenue_test_zero_inflated_lognormal(self):
        rng = np.random.default_rng(2024)
        n = 20_000

        def arm(rate, mu):
            buys = rng.random(n) < rate
            return np.where(buys, rng.lognormal(mu, 1.0, n), 0.0)

        ctrl = arm(0.05, 3.0)
        trt = arm(0.05, 3.15)  # about +16% order value
        tie = arm(0.05, 3.0)

        def v(name, x, is_ctrl=False):
            return var(name, n, int((x > 0).sum()), is_ctrl, float(x.sum()), float((x**2).sum()))

        variants = [v("control", ctrl, True), v("bigger_basket", trt), v("same", tie)]
        res = run(variants, primary_metric="arpu", samples=4000)
        by = {c.treatment: c for c in res.comparisons}
        all_finite(self, by["bigger_basket"])
        self.assertAlmostEqual(by["bigger_basket"].control_rate, ctrl.mean())
        self.assertGreater(by["bigger_basket"].absolute_lift, 0)
        self.assertGreater(
            by["bigger_basket"].probability_beats_control, by["same"].probability_beats_control
        )
        self.assertGreater(by["bigger_basket"].ci_high, by["bigger_basket"].ci_low)

        seq = run(variants, primary_metric="arpu", method="sequential")
        for c in seq.comparisons:
            all_finite(self, c)
        # Heavy-tailed revenue: the lift is real but noisy, so the test must not be
        # more confident about the null arm than the improved arm.
        sby = {c.treatment: c for c in seq.comparisons}
        self.assertGreaterEqual(sby["same"].p_value, sby["bigger_basket"].p_value)

    def test_dominated_variant_in_realistic_test(self):
        res = run(
            [var("control", 200_000, 8000, True), var("bad_ui", 200_000, 7200)],
            samples=3000,
            decision_policy={"enabled": True},
        )
        self.assertEqual(res.recommendation.action, "do_not_ship")
        self.assertAlmostEqual(res.comparisons[0].relative_lift, -0.10)

    def test_to_dict_is_json_safe(self):
        import json

        res = run(two_arm(4000, 4500), method="sequential")
        json.dumps(res.to_dict(), allow_nan=False)


class PlanningEdgeCases(unittest.TestCase):
    def test_tiny_mde_gives_huge_n(self):
        plan = fixed_horizon_duration_conversion(0.04, 0.001, 100_000)
        self.assertGreater(plan.n_per_variant, 10_000_000)
        self.assertGreater(plan.estimated_days, 100)

    def test_sequential_tiny_mde_either_huge_or_clean_error(self):
        try:
            plan = sequential_duration_conversion(0.04, 1e-7, 100_000)
        except ValueError as exc:
            self.assertIn("too small", str(exc))
        else:
            self.assertGreater(plan.n_per_variant, 10**9)

    def test_negative_relative_mde_is_symmetric_ish(self):
        up = fixed_horizon_duration_conversion(0.05, 0.10, 10_000)
        down = fixed_horizon_duration_conversion(0.05, -0.10, 10_000)
        self.assertEqual(down.diagnostics["p_treatment"], 0.05 * 0.9)
        self.assertGreater(down.diagnostics["absolute_delta"], 0)
        self.assertAlmostEqual(up.n_per_variant, down.n_per_variant, delta=0.2 * up.n_per_variant)
        seq = sequential_duration_conversion(0.05, -0.10, 10_000)
        self.assertGreater(seq.n_per_variant, 0)

    def test_extreme_negative_mde_clamped(self):
        plan = fixed_horizon_duration_conversion(0.05, -2.0, 10_000)
        self.assertGreater(plan.diagnostics["p_treatment"], 0.0)

    def test_n_variants_one_errors_everywhere(self):
        for fn in (fixed_horizon_duration_conversion, sequential_duration_conversion):
            with self.subTest(fn=fn.__name__):
                with self.assertRaisesRegex(ValueError, "n_variants must be at least 2"):
                    fn(0.04, 0.1, 1000, n_variants=1)
        with self.assertRaisesRegex(ValueError, "n_variants must be at least 2"):
            bayesian_duration_conversion(0.04, 0.1, 1000, n_variants=1)

    def test_alpha_and_power_validation(self):
        for fn in (fixed_horizon_duration_conversion, sequential_duration_conversion):
            for kw, msg in (
                ({"alpha": 0.0}, "alpha"),
                ({"alpha": 1.0}, "alpha"),
                ({"power": 0.0}, "power"),
                ({"power": 1.0}, "power"),
            ):
                with self.subTest(fn=fn.__name__, kw=kw):
                    with self.assertRaisesRegex(ValueError, msg):
                        fn(0.04, 0.1, 1000, **kw)

    def test_traffic_and_baseline_validation(self):
        with self.assertRaisesRegex(ValueError, "daily_total_traffic"):
            fixed_horizon_duration_conversion(0.04, 0.1, 0)
        with self.assertRaisesRegex(ValueError, "daily_total_traffic"):
            bayesian_duration_conversion(0.04, 0.1, -5)
        for b in (0.0, 1.0, -0.1):
            with self.subTest(baseline=b):
                with self.assertRaisesRegex(ValueError, "baseline_rate"):
                    fixed_horizon_duration_conversion(b, 0.1, 1000)

    def test_sequential_tau_override_changes_n(self):
        default = sequential_duration_conversion(0.04, 0.10, 20_000)
        tight = sequential_duration_conversion(0.04, 0.10, 20_000, sequential_tau=0.0040)
        wide = sequential_duration_conversion(0.04, 0.10, 20_000, sequential_tau=0.5)
        self.assertEqual(default.assumptions["sequential_tau"], 0.004)
        self.assertEqual(tight.assumptions["sequential_tau"], 0.004)
        self.assertEqual(tight.n_per_variant, default.n_per_variant)
        self.assertNotEqual(wide.n_per_variant, default.n_per_variant)
        self.assertGreater(wide.n_per_variant, default.n_per_variant)

    def test_sequential_negative_tau_rejected(self):
        with self.assertRaisesRegex(ValueError, "sequential_tau must be positive"):
            sequential_duration_conversion(0.04, 0.1, 1000, sequential_tau=-1)

    def test_sequential_needs_more_than_fixed(self):
        plan = sequential_duration_conversion(0.04, 0.10, 20_000)
        self.assertGreater(plan.diagnostics["sequential_inflation"], 1.0)
        self.assertEqual(plan.method, "sequential")

    def test_estimated_days_scale_with_traffic(self):
        a = fixed_horizon_duration_conversion(0.04, 0.1, 10_000)
        b = fixed_horizon_duration_conversion(0.04, 0.1, 20_000)
        self.assertEqual(a.n_per_variant, b.n_per_variant)
        self.assertAlmostEqual(a.estimated_days, 2 * b.estimated_days)

    def test_bayesian_max_days_too_small_returns_none(self):
        plan = bayesian_duration_conversion(
            0.04, 0.02, 2000, max_days=2, sims=40, posterior_draws=300
        )
        self.assertIsNone(plan.estimated_days)
        self.assertIsNone(plan.n_per_variant)
        self.assertIsNone(plan.diagnostics["assurance_at_selected_day"])

    def test_bayesian_easy_effect_found_early(self):
        plan = bayesian_duration_conversion(
            0.10, 0.5, 20_000, max_days=10, sims=40, posterior_draws=500, seed=1
        )
        self.assertIsNotNone(plan.estimated_days)
        self.assertLessEqual(plan.estimated_days, 10)
        self.assertGreaterEqual(plan.diagnostics["assurance_at_selected_day"], 0.8)

    def test_bayesian_zero_max_days(self):
        plan = bayesian_duration_conversion(0.04, 0.1, 1000, max_days=0, sims=5)
        self.assertIsNone(plan.estimated_days)

    def test_zero_mde_rejected(self):
        for fn in (fixed_horizon_duration_conversion, sequential_duration_conversion):
            with self.assertRaisesRegex(ValueError, "relative_mde must be non-zero"):
                fn(0.04, 0.0, 1000)


if __name__ == "__main__":
    unittest.main()
