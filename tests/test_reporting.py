import unittest

from ablift.engine import analyze, parse_payload
from ablift.reporting import _fmt_float, _fmt_pct, build_markdown_report


def _variants(*rows):
    out = []
    for i, (name, visitors, conversions) in enumerate(rows):
        out.append(
            {
                "name": name,
                "visitors": visitors,
                "conversions": conversions,
                "is_control": i == 0,
            }
        )
    return out


def _report(**overrides):
    payload = {
        "experiment_name": "report_exp",
        "method": "bayesian",
        "samples": 2000,
        "random_seed": 3,
        "variants": _variants(("control", 10000, 400), ("v1", 10000, 460)),
    }
    payload.update(overrides)
    return build_markdown_report(analyze(parse_payload(payload)))


def _arpu_variants():
    return [
        {
            "name": "control",
            "visitors": 1000,
            "conversions": 100,
            "revenue_sum": 5000.0,
            "revenue_sum_squares": 400000.0,
            "is_control": True,
        },
        {
            "name": "v1",
            "visitors": 1000,
            "conversions": 110,
            "revenue_sum": 5600.0,
            "revenue_sum_squares": 450000.0,
            "is_control": False,
        },
    ]


class FormatHelperTests(unittest.TestCase):
    def test_fmt_float_none_is_na(self):
        self.assertEqual(_fmt_float(None), "n/a")

    def test_fmt_float_digits(self):
        self.assertEqual(_fmt_float(0.123456789), "0.123457")
        self.assertEqual(_fmt_float(0.5, 2), "0.50")

    def test_fmt_pct(self):
        self.assertEqual(_fmt_pct(0.04), "4.000%")


class BayesianReportTests(unittest.TestCase):
    def setUp(self):
        self.report = _report()

    def test_title_and_sections(self):
        self.assertTrue(self.report.startswith("# A/B Decision Report: report_exp"))
        for header in ("## Summary", "## Data Quality", "## Variant Comparisons", "## Guardrails"):
            self.assertIn(header, self.report)

    def test_bayesian_summary_lines(self):
        self.assertIn("- Method: `bayesian`", self.report)
        self.assertIn("- Posterior samples: `2000`", self.report)
        self.assertIn("- Random seed: `3`", self.report)
        self.assertIn("- Priors:", self.report)
        self.assertIn("- Control: `control`", self.report)

    def test_comparison_lines(self):
        self.assertIn("- Treatment: `v1` vs Control: `control`", self.report)
        self.assertIn("95% credible interval (absolute lift): [", self.report)
        self.assertIn("95% credible interval (relative lift): [", self.report)
        self.assertIn("P(treatment > control):", self.report)
        self.assertIn("Expected loss:", self.report)
        self.assertIn("P(best overall):", self.report)
        self.assertIn("Multi-arm expected loss:", self.report)
        self.assertIn("## Joint Posterior Summary", self.report)
        self.assertIn("- `control` (control): P(best)=", self.report)
        self.assertIn("- `v1` (treatment): P(best)=", self.report)

    def test_no_sequential_lines(self):
        for text in ("e-value", "e-threshold", "Always-valid", "mSPRT", "Alpha (family-wise)"):
            self.assertNotIn(text, self.report)

    def test_conversion_formatting_is_percent(self):
        self.assertIn("- Control CR: 4.000%", self.report)
        self.assertIn("- Treatment CR: 4.600%", self.report)
        self.assertIn("- Absolute lift: 0.600%", self.report)
        self.assertIn("- Relative lift: 15.000%", self.report)

    def test_arpu_formatting_is_plain_float(self):
        report = _report(primary_metric="arpu", variants=_arpu_variants())
        self.assertIn("- Control ARPU: 5.000000", report)
        self.assertIn("- Treatment ARPU: 5.600000", report)
        self.assertIn("- Absolute lift: 0.600000", report)
        self.assertNotIn("Control CR", report)
        self.assertIn("`arpu`", report)


class SequentialReportTests(unittest.TestCase):
    def test_sequential_lines(self):
        report = _report(method="sequential", alpha=0.05)
        self.assertIn("- Method: `sequential`", report)
        self.assertIn("- Alpha (family-wise): `0.05`", report)
        self.assertIn("`bonferroni` across 1 comparison(s), alpha per comparison `0.05`", report)
        self.assertIn("e-value:", report)
        self.assertIn("e-threshold:", report)
        self.assertIn("Always-valid p-value:", report)
        self.assertIn("Always-valid confidence interval (absolute lift): [", report)
        self.assertIn("Significant now:", report)

    def test_sequential_has_no_bayesian_lines(self):
        report = _report(method="sequential")
        for text in (
            "Posterior samples",
            "Priors",
            "credible interval",
            "P(treatment",
            "Expected loss",
            "P(best overall)",
            "Multi-arm expected loss",
            "Joint Posterior Summary",
        ):
            self.assertNotIn(text, report)

    def test_tau_default_text_when_none(self):
        report = _report(method="sequential")
        self.assertIn("mSPRT tau: `default (10% of control baseline)`", report)

    def test_tau_value_when_set(self):
        report = _report(method="sequential", sequential_tau=0.02)
        self.assertIn("mSPRT tau: `0.02`", report)

    def test_bonferroni_with_multiple_variants(self):
        report = _report(
            method="sequential",
            alpha=0.06,
            variants=_variants(("control", 10000, 400), ("a", 10000, 420), ("b", 10000, 450)),
        )
        self.assertIn("across 2 comparison(s), alpha per comparison `0.03", report)

    def test_sequential_arpu(self):
        report = _report(method="sequential", primary_metric="arpu", variants=_arpu_variants())
        self.assertIn("Control ARPU: 5.000000", report)
        self.assertIn("Always-valid p-value:", report)


class RecommendationSectionTests(unittest.TestCase):
    def test_no_policy_not_evaluated(self):
        report = _report()
        self.assertIn("- Recommendation: `not_evaluated`", report)
        self.assertIn("No decision policy was provided", report)
        self.assertIn("Provide an explicit decision policy", report)
        self.assertIn("- Decision confidence: n/a", report)
        self.assertIn("- Risk flags: `none`", report)

    def test_with_policy(self):
        report = _report(decision_policy={"enabled": True, "bayes_prob_beats_control": 0.9})
        self.assertNotIn("not_evaluated", report)
        self.assertIn("- Recommendation: `", report)
        self.assertIn("- Rationale:", report)
        self.assertNotIn("No decision policy was provided", report)
        self.assertNotIn("- Decision confidence: n/a", report)

    def test_sequential_implicit_policy_has_recommendation(self):
        report = _report(method="sequential")
        self.assertNotIn("not_evaluated", report)
        self.assertIn("- Decision confidence: 0.", report)

    def test_risk_flags_listed(self):
        guardrails = [
            {
                "name": "bounce",
                "control": 0.3,
                "treatment": 0.6,
                "direction": "decrease",
                "max_relative_change": 0.05,
            }
        ]
        report = _report(method="sequential", guardrails=guardrails)
        self.assertIn("- Recommendation: `do_not_ship`", report)
        self.assertIn("- Risk flags: `guardrail_failure`", report)
        self.assertIn("- Decision confidence: 1.0000", report)

    def test_srm_flags_in_report(self):
        report = _report(
            method="sequential",
            variants=_variants(("control", 10000, 400), ("v1", 5000, 200)),
        )
        self.assertIn("- SRM passed: `False`", report)
        self.assertIn("srm_detected", report)
        self.assertIn("investigate_data_quality", report)


class GuardrailAndSrmSectionTests(unittest.TestCase):
    def test_no_guardrails(self):
        report = _report()
        self.assertIn("No guardrails provided.", report)
        self.assertIn("- Guardrails passed: `True`", report)

    def test_listed_guardrails_pass_and_fail(self):
        guardrails = [
            {
                "name": "latency",
                "control": 400.0,
                "treatment": 404.0,
                "direction": "decrease",
                "max_relative_change": 0.05,
            },
            {
                "name": "bounce",
                "control": 0.3,
                "treatment": 0.6,
                "direction": "decrease",
                "max_relative_change": 0.05,
            },
        ]
        report = _report(guardrails=guardrails)
        self.assertNotIn("No guardrails provided.", report)
        self.assertIn("- `latency`: pass=`True`", report)
        self.assertIn("- `bounce`: pass=`False`", report)
        self.assertIn("  - direction: `decrease`", report)
        self.assertIn("  - relative change: 100.000%", report)
        self.assertIn("  - allowed: 5.000%", report)
        self.assertIn("  - reason:", report)
        self.assertIn("- Guardrails passed: `False`", report)

    def test_srm_section_values(self):
        result = analyze(
            parse_payload(
                {
                    "method": "sequential",
                    "variants": _variants(("control", 10000, 400), ("v1", 10000, 460)),
                }
            )
        )
        report = build_markdown_report(result)
        self.assertIn("- SRM passed: `True`", report)
        self.assertIn(f"- SRM p-value: {result.srm.p_value:.6f}", report)
        self.assertIn("- SRM expected proportions: 50.000%, 50.000%", report)
        self.assertIn(f"- SRM reason: {result.srm.reason}", report)

    def test_srm_section_shows_configured_unequal_proportions(self):
        report = _report(
            method="sequential",
            expected_split=[0.9, 0.1],
            variants=_variants(("control", 90000, 3600), ("v1", 10000, 460)),
        )
        self.assertIn("- SRM passed: `True`", report)
        self.assertIn("- SRM expected proportions: 90.000%, 10.000%", report)

    def test_srm_perfectly_balanced_has_p_value_one(self):
        # Regression (was a bug): ablift/engine.py:evaluate_srm uses a Wilson-Hilferty approximation that
        # returns p ~= 0.95 (not 1.0) when chi2 == 0 (perfectly balanced traffic).
        result = analyze(
            parse_payload(
                {
                    "method": "sequential",
                    "variants": _variants(("control", 10000, 400), ("v1", 10000, 460)),
                }
            )
        )
        self.assertAlmostEqual(result.srm.p_value, 1.0, places=3)


class MultiVariantReportTests(unittest.TestCase):
    def test_one_block_per_treatment(self):
        for method in ("bayesian", "sequential"):
            report = _report(
                method=method,
                variants=_variants(
                    ("control", 10000, 400), ("a", 10000, 420), ("b", 10000, 450), ("c", 10000, 380)
                ),
            )
            self.assertEqual(report.count("- Treatment: `"), 3, method)
            for name in ("a", "b", "c"):
                self.assertIn(f"- Treatment: `{name}` vs Control: `control`", report)

    def test_report_is_newline_terminated_markdown(self):
        report = _report()
        self.assertTrue(report.endswith("\n"))


if __name__ == "__main__":
    unittest.main()
