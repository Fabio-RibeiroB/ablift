from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner
from test_schema import validate_json_schema

from ablift.cli import cli
from ablift.connectors import UsageOptionError, build_payload_from_rows
from ablift.engine import analyze, parse_payload
from ablift.reporting import build_markdown_report
from ablift.schemas import get_schema
from ablift.stats import compute_cuped_adjustment, validate_cuped_aggregates


def _arm_aggregates_arpu(
    name: str,
    is_control: bool,
    y: np.ndarray,
    x: np.ndarray | None = None,
) -> dict[str, Any]:
    obj: dict[str, Any] = {
        "name": name,
        "is_control": is_control,
        "visitors": int(y.size),
        "conversions": int(np.sum(y > 0.0)),
        "revenue_sum": float(np.sum(y)),
        "revenue_sum_squares": float(np.sum(y * y)),
    }
    if x is not None:
        obj["covariate_sum"] = float(np.sum(x))
        obj["covariate_sum_squares"] = float(np.sum(x * x))
        obj["outcome_covariate_sum"] = float(np.sum(y * x))
    return obj


def _arm_aggregates_conversion(
    name: str,
    is_control: bool,
    y: np.ndarray,
    x: np.ndarray | None = None,
) -> dict[str, Any]:
    obj: dict[str, Any] = {
        "name": name,
        "is_control": is_control,
        "visitors": int(y.size),
        "conversions": int(np.sum(y > 0.5)),
    }
    if x is not None:
        obj["covariate_sum"] = float(np.sum(x))
        obj["covariate_sum_squares"] = float(np.sum(x * x))
        obj["outcome_covariate_sum"] = float(np.sum(y * x))
    return obj


class TestCupedMathAndValidation:
    def test_pooled_within_arm_theta_matches_exact_formula_and_is_shift_invariant(self) -> None:
        rng = np.random.default_rng(101)
        n = 500
        x_c = rng.normal(10.0, 3.0, size=n)
        x_t = rng.normal(10.1, 3.0, size=n)
        y_c = 2.0 * x_c + rng.normal(0.0, 2.0, size=n)
        y_t = 2.0 * x_t + 1.5 + rng.normal(0.0, 2.0, size=n)

        arms_1, diag_1 = compute_cuped_adjustment(
            [
                (
                    "control",
                    True,
                    n,
                    float(np.sum(y_c)),
                    float(np.sum(y_c**2)),
                    float(np.sum(x_c)),
                    float(np.sum(x_c**2)),
                    float(np.sum(y_c * x_c)),
                ),
                (
                    "treatment",
                    False,
                    n,
                    float(np.sum(y_t)),
                    float(np.sum(y_t**2)),
                    float(np.sum(x_t)),
                    float(np.sum(x_t**2)),
                    float(np.sum(y_t * x_t)),
                ),
            ]
        )

        # Exact manual within-arm pooled theta
        s_xx = float(np.sum((x_c - x_c.mean()) ** 2) + np.sum((x_t - x_t.mean()) ** 2))
        s_xy = float(
            np.sum((x_c - x_c.mean()) * (y_c - y_c.mean()))
            + np.sum((x_t - x_t.mean()) * (y_t - y_t.mean()))
        )
        expected_theta = s_xy / s_xx
        assert math.isclose(diag_1["theta"], expected_theta, rel_tol=1e-12)
        assert diag_1["r_squared"] > 0.85
        assert math.isclose(
            diag_1["variance_reduction_pct"], diag_1["r_squared"] * 100.0, rel_tol=1e-9
        )

        # Adding a huge constant shift to treatment Y must NOT alter theta or adjusted variances
        y_t_shifted = y_t + 50.0
        arms_2, diag_2 = compute_cuped_adjustment(
            [
                (
                    "control",
                    True,
                    n,
                    float(np.sum(y_c)),
                    float(np.sum(y_c**2)),
                    float(np.sum(x_c)),
                    float(np.sum(x_c**2)),
                    float(np.sum(y_c * x_c)),
                ),
                (
                    "treatment",
                    False,
                    n,
                    float(np.sum(y_t_shifted)),
                    float(np.sum(y_t_shifted**2)),
                    float(np.sum(x_t)),
                    float(np.sum(x_t**2)),
                    float(np.sum(y_t_shifted * x_t)),
                ),
            ]
        )
        assert math.isclose(diag_2["theta"], diag_1["theta"], rel_tol=1e-10)
        assert math.isclose(arms_2["treatment"][0] - arms_1["treatment"][0], 50.0, abs_tol=1e-9)
        assert math.isclose(arms_2["treatment"][1], arms_1["treatment"][1], rel_tol=1e-9)

    def test_zero_variance_covariate_degrades_gracefully(self) -> None:
        n = 100
        x_c = np.full(n, 5.0)
        x_t = np.full(n, 5.0)
        rng = np.random.default_rng(7)
        y_c = rng.normal(10.0, 2.0, size=n)
        y_t = rng.normal(11.0, 2.0, size=n)

        arms, diag = compute_cuped_adjustment(
            [
                (
                    "control",
                    True,
                    n,
                    float(np.sum(y_c)),
                    float(np.sum(y_c**2)),
                    float(np.sum(x_c)),
                    float(np.sum(x_c**2)),
                    float(np.sum(y_c * x_c)),
                ),
                (
                    "treatment",
                    False,
                    n,
                    float(np.sum(y_t)),
                    float(np.sum(y_t**2)),
                    float(np.sum(x_t)),
                    float(np.sum(x_t**2)),
                    float(np.sum(y_t * x_t)),
                ),
            ]
        )
        assert diag["theta"] == 0.0
        assert diag["r_squared"] == 0.0
        assert diag["variance_reduction_pct"] == 0.0
        assert diag["covariate_imbalanced"] is False
        assert math.isclose(arms["control"][0], float(np.mean(y_c)), rel_tol=1e-12)
        assert math.isclose(arms["control"][1], float(np.var(y_c, ddof=1)), rel_tol=1e-12)

    def test_validate_cuped_aggregates_rejects_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="at least 2 visitors"):
            validate_cuped_aggregates(1, 1.0, 1.0, 1.0, 1.0, 1.0)

        with pytest.raises(ValueError, match="finite number"):
            validate_cuped_aggregates(10, 10.0, 20.0, float("nan"), 10.0, 10.0)

        with pytest.raises(ValueError, match="covariate_sum_squares"):
            validate_cuped_aggregates(10, 10.0, 20.0, 10.0, 5.0, 10.0)

        with pytest.raises(ValueError, match="outcome sum of squares"):
            validate_cuped_aggregates(10, 10.0, 5.0, 10.0, 20.0, 10.0)

        with pytest.raises(ValueError, match="Cauchy-Schwarz"):
            # s_yy = 10, s_xx = 10, so |s_xy| <= 10; xy_sum=100 gives s_xy = 90
            validate_cuped_aggregates(10, 10.0, 20.0, 10.0, 20.0, 100.0)


class TestCupedStatisticalProperties:
    def test_type_i_error_preserved_under_null_arpu_and_conversion(self) -> None:
        rng = np.random.default_rng(2025)
        n_trials = 300
        n_per_arm = 400
        false_positives_arpu = 0
        false_positives_cr = 0

        for _ in range(n_trials):
            x_c = rng.normal(50.0, 15.0, size=n_per_arm)
            x_t = rng.normal(50.0, 15.0, size=n_per_arm)
            y_c = 0.8 * x_c + rng.normal(0.0, 8.0, size=n_per_arm)
            y_t = 0.8 * x_t + rng.normal(0.0, 8.0, size=n_per_arm)
            res_arpu = analyze(
                parse_payload(
                    {
                        "experiment_name": "null_arpu",
                        "method": "fixed",
                        "primary_metric": "arpu",
                        "alpha": 0.05,
                        "variants": [
                            _arm_aggregates_arpu("control", True, y_c, x_c),
                            _arm_aggregates_arpu("treatment", False, y_t, x_t),
                        ],
                    }
                )
            )
            if res_arpu.comparisons[0].significant:
                false_positives_arpu += 1

            # Correlated binary outcome via latent threshold on X
            xb_c = rng.normal(0.0, 1.0, size=n_per_arm)
            xb_t = rng.normal(0.0, 1.0, size=n_per_arm)
            yb_c = (0.8 * xb_c + rng.normal(0.0, 0.6, size=n_per_arm) > 0.5).astype(float)
            yb_t = (0.8 * xb_t + rng.normal(0.0, 0.6, size=n_per_arm) > 0.5).astype(float)
            res_cr = analyze(
                parse_payload(
                    {
                        "experiment_name": "null_cr",
                        "method": "fixed",
                        "primary_metric": "conversion_rate",
                        "alpha": 0.05,
                        "variants": [
                            _arm_aggregates_conversion("control", True, yb_c, xb_c),
                            _arm_aggregates_conversion("treatment", False, yb_t, xb_t),
                        ],
                    }
                )
            )
            if res_cr.comparisons[0].significant:
                false_positives_cr += 1

        fpr_arpu = false_positives_arpu / n_trials
        fpr_cr = false_positives_cr / n_trials
        assert 0.01 <= fpr_arpu <= 0.09, f"Unexpected ARPU Type I error: {fpr_arpu}"
        assert 0.01 <= fpr_cr <= 0.09, f"Unexpected CR Type I error: {fpr_cr}"

    def test_cuped_narrows_ci_and_boosts_power_under_alternative(self) -> None:
        rng = np.random.default_rng(42)
        n_per_arm = 1000
        x_c = rng.normal(100.0, 30.0, size=n_per_arm)
        x_t = rng.normal(100.0, 30.0, size=n_per_arm)
        # r ~ 0.90 => variance reduction ~ 81%, CI width shrinks by ~55%
        y_c = 1.0 * x_c + rng.normal(0.0, 14.0, size=n_per_arm)
        y_t = 1.0 * x_t + 2.5 + rng.normal(0.0, 14.0, size=n_per_arm)

        raw = analyze(
            parse_payload(
                {
                    "experiment_name": "raw_vs_cuped",
                    "method": "fixed",
                    "primary_metric": "arpu",
                    "variants": [
                        _arm_aggregates_arpu("control", True, y_c),
                        _arm_aggregates_arpu("treatment", False, y_t),
                    ],
                }
            )
        )
        cuped = analyze(
            parse_payload(
                {
                    "experiment_name": "raw_vs_cuped",
                    "method": "fixed",
                    "primary_metric": "arpu",
                    "variants": [
                        _arm_aggregates_arpu("control", True, y_c, x_c),
                        _arm_aggregates_arpu("treatment", False, y_t, x_t),
                    ],
                }
            )
        )

        raw_c = raw.comparisons[0]
        cuped_c = cuped.comparisons[0]
        assert raw_c.ci_low is not None and raw_c.ci_high is not None
        assert cuped_c.ci_low is not None and cuped_c.ci_high is not None
        raw_width = raw_c.ci_high - raw_c.ci_low
        cuped_width = cuped_c.ci_high - cuped_c.ci_low
        assert cuped_width < 0.55 * raw_width
        assert cuped_c.significant is True
        assert cuped_c.p_value is not None and raw_c.p_value is not None
        assert cuped_c.p_value < raw_c.p_value
        cuped_meta = cuped.analysis_settings.input_interpretation["cuped"]
        assert cuped_meta["variance_reduction_pct"] > 70.0
        assert cuped_meta["covariate_imbalanced"] is False

    def test_uncorrelated_covariate_matches_unadjusted_closely(self) -> None:
        rng = np.random.default_rng(19)
        n = 5000
        x_c = rng.normal(0.0, 1.0, size=n)
        x_t = rng.normal(0.0, 1.0, size=n)
        y_c = rng.normal(20.0, 5.0, size=n)
        y_t = rng.normal(20.4, 5.0, size=n)

        raw = analyze(
            parse_payload(
                {
                    "experiment_name": "uncorr",
                    "method": "fixed",
                    "primary_metric": "arpu",
                    "variants": [
                        _arm_aggregates_arpu("control", True, y_c),
                        _arm_aggregates_arpu("treatment", False, y_t),
                    ],
                }
            )
        )
        cuped = analyze(
            parse_payload(
                {
                    "experiment_name": "uncorr",
                    "method": "fixed",
                    "primary_metric": "arpu",
                    "variants": [
                        _arm_aggregates_arpu("control", True, y_c, x_c),
                        _arm_aggregates_arpu("treatment", False, y_t, x_t),
                    ],
                }
            )
        )
        assert abs(cuped.comparisons[0].absolute_lift - raw.comparisons[0].absolute_lift) < 0.02
        assert cuped.analysis_settings.input_interpretation["cuped"]["variance_reduction_pct"] < 1.0


class TestCupedMethodsAndDiagnostics:
    @pytest.mark.parametrize("method", ["fixed", "sequential", "bayesian"])
    @pytest.mark.parametrize("metric", ["arpu", "conversion_rate"])
    def test_all_methods_and_metrics_with_multi_arm_and_schema(
        self, method: str, metric: str
    ) -> None:
        rng = np.random.default_rng(123)
        n = 1500
        x_c = rng.normal(0.0, 1.0, size=n)
        x_a = rng.normal(0.0, 1.0, size=n)
        x_b = rng.normal(0.0, 1.0, size=n)
        if metric == "arpu":
            y_c = 10.0 + 4.0 * x_c + rng.normal(0.0, 2.0, size=n)
            y_a = 10.0 + 4.0 * x_a + 0.5 + rng.normal(0.0, 2.0, size=n)
            y_b = 10.0 + 4.0 * x_b + 0.0 + rng.normal(0.0, 2.0, size=n)
            variants = [
                _arm_aggregates_arpu("control", True, y_c, x_c),
                _arm_aggregates_arpu("v1", False, y_a, x_a),
                _arm_aggregates_arpu("v2", False, y_b, x_b),
            ]
        else:
            y_c = (1.5 * x_c + rng.normal(0.0, 0.7, size=n) > 0.3).astype(float)
            y_a = (1.5 * x_a + 0.25 + rng.normal(0.0, 0.7, size=n) > 0.3).astype(float)
            y_b = (1.5 * x_b + rng.normal(0.0, 0.7, size=n) > 0.3).astype(float)
            variants = [
                _arm_aggregates_conversion("control", True, y_c, x_c),
                _arm_aggregates_conversion("v1", False, y_a, x_a),
                _arm_aggregates_conversion("v2", False, y_b, x_b),
            ]

        payload: dict[str, Any] = {
            "experiment_name": f"cuped_{method}_{metric}",
            "method": method,
            "primary_metric": metric,
            "variants": variants,
            "samples": 8000,
            "random_seed": 7,
        }
        if method == "bayesian":
            payload["decision_policy"] = {
                "enabled": True,
                "bayes_prob_beats_control": 0.95,
                "max_expected_loss": 0.1,
            }

        validate_json_schema(payload, get_schema("analyze-input"))
        res = analyze(parse_payload(payload))
        validate_json_schema(res.to_dict(), get_schema("analyze-output"))
        assert len(res.comparisons) == 2
        cuped_meta = res.analysis_settings.input_interpretation["cuped"]
        assert cuped_meta["enabled"] is True
        assert cuped_meta["variance_reduction_pct"] > 30.0
        assert res.recommendation is not None
        assert res.recommendation.action == "ship_v1"

        report = build_markdown_report(res)
        assert "CUPED variance reduction:" in report

        if method == "fixed":
            assert res.comparisons[0].degrees_of_freedom is not None
            assert "Welch t-statistic:" in report
        elif method == "sequential":
            assert res.comparisons[0].e_value is not None
            assert res.comparisons[0].e_value > res.comparisons[0].e_threshold
        else:
            assert res.analysis_settings.priors["family"] == "cuped_normal_inverse_gamma"
            assert res.bayesian_arms is not None
            assert math.isclose(
                sum(a.probability_best for a in res.bayesian_arms), 1.0, abs_tol=1e-6
            )

    def test_covariate_imbalance_risk_flag_and_report_warning(self) -> None:
        rng = np.random.default_rng(99)
        n = 1000
        # Severe pre-experiment imbalance between arms
        x_c = rng.normal(10.0, 2.0, size=n)
        x_t = rng.normal(11.5, 2.0, size=n)
        y_c = 1.5 * x_c + rng.normal(0.0, 1.0, size=n)
        y_t = 1.5 * x_t + rng.normal(0.0, 1.0, size=n)

        res = analyze(
            parse_payload(
                {
                    "experiment_name": "imbalanced_cuped",
                    "method": "fixed",
                    "primary_metric": "arpu",
                    "variants": [
                        _arm_aggregates_arpu("control", True, y_c, x_c),
                        _arm_aggregates_arpu("treatment", False, y_t, x_t),
                    ],
                }
            )
        )
        cuped_meta = res.analysis_settings.input_interpretation["cuped"]
        assert cuped_meta["covariate_imbalanced"] is True
        assert cuped_meta["covariate_balance_p_value"] < 0.001
        assert res.recommendation is not None
        assert "covariate_imbalance" in res.recommendation.risk_flags

        report = build_markdown_report(res)
        assert "CUPED covariate balance warning (`covariate_imbalance`)" in report


class TestCupedRowLevelAndCli:
    def test_row_level_csv_matches_aggregated_json_by_name_and_index(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(77)
        n = 300
        x_c = rng.normal(25.0, 8.0, size=n)
        x_t = rng.normal(25.0, 8.0, size=n)
        y_c = 0.7 * x_c + rng.normal(5.0, 3.0, size=n)
        y_t = 0.7 * x_t + 6.5 + rng.normal(5.0, 3.0, size=n)

        csv_path = tmp_path / "row_arpu_cuped.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["variant", "revenue", "pre_spend", "is_control"])
            for yv, xv in zip(y_c, x_c, strict=True):
                writer.writerow(["control", f"{yv:.10f}", f"{xv:.10f}", "true"])
            for yv, xv in zip(y_t, x_t, strict=True):
                writer.writerow(["treatment", f"{yv:.10f}", f"{xv:.10f}", "false"])

        runner = CliRunner()
        res_name = runner.invoke(
            cli,
            [
                "analyze",
                "--input",
                str(csv_path),
                "--method",
                "fixed",
                "--covariate-column",
                "pre_spend",
            ],
        )
        assert res_name.exit_code == 0, res_name.output
        out_name = json.loads(res_name.output)

        res_idx = runner.invoke(
            cli,
            [
                "analyze",
                "--input",
                str(csv_path),
                "--method",
                "fixed",
                "--covariate-column",
                "3",
            ],
        )
        assert res_idx.exit_code == 0, res_idx.output
        out_idx = json.loads(res_idx.output)

        # Direct JSON aggregates must match row-level CSV to floating-point precision
        json_res = analyze(
            parse_payload(
                {
                    "experiment_name": "row_arpu_cuped",
                    "method": "fixed",
                    "primary_metric": "arpu",
                    "variants": [
                        _arm_aggregates_arpu("control", True, y_c, x_c),
                        _arm_aggregates_arpu("treatment", False, y_t, x_t),
                    ],
                }
            )
        ).to_dict()

        assert (
            out_name["analysis_settings"]["input_interpretation"]["column_roles"]["covariate"]
            == "pre_spend"
        )
        assert math.isclose(
            out_name["comparisons"][0]["absolute_lift"],
            out_idx["comparisons"][0]["absolute_lift"],
            rel_tol=1e-12,
        )
        assert math.isclose(
            out_name["comparisons"][0]["absolute_lift"],
            json_res["comparisons"][0]["absolute_lift"],
            abs_tol=1e-8,
        )
        assert math.isclose(
            out_name["comparisons"][0]["p_value"],
            json_res["comparisons"][0]["p_value"],
            abs_tol=1e-8,
        )

    def test_row_level_conversion_with_binary_covariate_without_control_col(
        self, tmp_path: Path
    ) -> None:
        rng = np.random.default_rng(55)
        n = 400
        x_c = rng.binomial(1, 0.3, size=n).astype(float)
        x_t = rng.binomial(1, 0.3, size=n).astype(float)
        y_c = ((0.6 * x_c + rng.uniform(0, 1, size=n)) > 0.75).astype(int)
        y_t = ((0.6 * x_t + 0.1 + rng.uniform(0, 1, size=n)) > 0.75).astype(int)

        csv_path = tmp_path / "row_cr_cuped.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["arm", "converted", "pre_control_period_converted"])
            for yv, xv in zip(y_c, x_c, strict=True):
                writer.writerow(["control", int(yv), int(xv)])
            for yv, xv in zip(y_t, x_t, strict=True):
                writer.writerow(["treatment", int(yv), int(xv)])

        runner = CliRunner()
        res = runner.invoke(
            cli,
            [
                "analyze",
                "--input",
                str(csv_path),
                "--method",
                "fixed",
                "--covariate-column",
                "pre_control_period_converted",
            ],
        )
        assert res.exit_code == 0, res.output
        out = json.loads(res.output)
        assert out["control_variant"] == "control"
        assert out["analysis_settings"]["primary_metric"] == "conversion_rate"
        assert (
            out["analysis_settings"]["input_interpretation"]["cuped"]["variance_reduction_pct"]
            > 10.0
        )

    def test_winsorize_and_cuped_compose_on_row_level_arpu(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(88)
        n = 400
        x_c = rng.normal(20.0, 5.0, size=n)
        x_t = rng.normal(20.0, 5.0, size=n)
        y_c = np.maximum(0.0, 1.2 * x_c + rng.normal(0.0, 3.0, size=n))
        y_t = np.maximum(0.0, 1.2 * x_t + 1.5 + rng.normal(0.0, 3.0, size=n))
        # Inject whale outliers
        y_c[0] = 5000.0
        y_t[0] = 6000.0

        csv_path = tmp_path / "whale_cuped.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["variant", "revenue", "pre_revenue"])
            for yv, xv in zip(y_c, x_c, strict=True):
                writer.writerow(["control", f"{yv:.4f}", f"{xv:.4f}"])
            for yv, xv in zip(y_t, x_t, strict=True):
                writer.writerow(["treatment", f"{yv:.4f}", f"{xv:.4f}"])

        runner = CliRunner()
        res = runner.invoke(
            cli,
            [
                "analyze",
                "--input",
                str(csv_path),
                "--method",
                "fixed",
                "--winsorize",
                "0.99",
                "--covariate-column",
                "pre_revenue",
            ],
        )
        assert res.exit_code == 0, res.output
        out = json.loads(res.output)
        interp = out["analysis_settings"]["input_interpretation"]
        assert "winsorization" in interp
        assert interp["cuped"]["enabled"] is True
        assert interp["cuped"]["variance_reduction_pct"] > 50.0

    def test_usage_and_validation_rejections(self, tmp_path: Path) -> None:
        runner = CliRunner()

        # 1. --covariate-column on JSON input -> exit code 2
        json_path = tmp_path / "agg.json"
        json_path.write_text(
            json.dumps(
                {
                    "experiment_name": "j",
                    "method": "fixed",
                    "primary_metric": "conversion_rate",
                    "variants": [
                        {"name": "c", "visitors": 100, "conversions": 10, "is_control": True},
                        {"name": "t", "visitors": 100, "conversions": 15, "is_control": False},
                    ],
                }
            ),
            encoding="utf-8",
        )
        r_json = runner.invoke(
            cli, ["analyze", "--input", str(json_path), "--covariate-column", "pre_x"]
        )
        assert r_json.exit_code == 2

        # 2. --covariate-column on aggregated CSV -> exit code 2
        agg_csv = tmp_path / "agg.csv"
        agg_csv.write_text("variant,visitors,conversions\nc,100,10\nt,100,15\n", encoding="utf-8")
        r_agg = runner.invoke(
            cli, ["analyze", "--input", str(agg_csv), "--covariate-column", "pre_x"]
        )
        assert r_agg.exit_code == 2

        # 3. --covariate-column with --primary-metric ratio -> exit code 2
        ratio_csv = tmp_path / "ratio.csv"
        ratio_csv.write_text(
            "variant,num,den\nc,1,2\nc,2,3\nt,2,2\nt,3,3\n",
            encoding="utf-8",
        )
        r_ratio = runner.invoke(
            cli,
            [
                "analyze",
                "--input",
                str(ratio_csv),
                "--primary-metric",
                "ratio",
                "--covariate-column",
                "3",
            ],
        )
        assert r_ratio.exit_code == 2

        # 4. Bad covariate column spec (missing name, out of range index, colliding with col 1/2)
        row_csv = tmp_path / "row.csv"
        row_csv.write_text(
            "variant,outcome,pre_x\nc,1,0.5\nc,0,0.1\nt,1,0.6\nt,0,0.2\n",
            encoding="utf-8",
        )
        for bad_spec in ["missing_col", "9", "variant", "outcome", "   "]:
            r_bad = runner.invoke(
                cli, ["analyze", "--input", str(row_csv), "--covariate-column", bad_spec]
            )
            assert r_bad.exit_code == 2, f"Expected exit 2 for {bad_spec!r}, got {r_bad.exit_code}"

        # 5. Missing covariate value in row -> error
        missing_cov_csv = tmp_path / "missing_cov.csv"
        missing_cov_csv.write_text(
            "variant,outcome,pre_x\nc,1,0.5\nc,0,\nt,1,0.6\nt,0,0.2\n",
            encoding="utf-8",
        )
        r_miss = runner.invoke(
            cli, ["analyze", "--input", str(missing_cov_csv), "--covariate-column", "pre_x"]
        )
        assert r_miss.exit_code == 1

        # 6. Incomplete CUPED fields on one variant in JSON -> ValueError
        with pytest.raises(ValueError, match="incomplete for 't'"):
            analyze(
                parse_payload(
                    {
                        "experiment_name": "incomplete",
                        "method": "fixed",
                        "primary_metric": "conversion_rate",
                        "variants": [
                            {
                                "name": "c",
                                "visitors": 100,
                                "conversions": 10,
                                "is_control": True,
                                "covariate_sum": 10.0,
                                "covariate_sum_squares": 10.0,
                                "outcome_covariate_sum": 5.0,
                            },
                            {"name": "t", "visitors": 100, "conversions": 15, "is_control": False},
                        ],
                    }
                )
            )

        # 7. CUPED fields on ratio primary metric in JSON -> ValueError
        with pytest.raises(ValueError, match="not 'ratio'"):
            analyze(
                parse_payload(
                    {
                        "experiment_name": "ratio_cuped",
                        "method": "fixed",
                        "primary_metric": "ratio",
                        "variants": [
                            {
                                "name": "c",
                                "visitors": 10,
                                "is_control": True,
                                "numerator_sum": 10.0,
                                "numerator_sum_squares": 20.0,
                                "denominator_sum": 20.0,
                                "denominator_sum_squares": 50.0,
                                "numerator_denominator_sum": 25.0,
                                "covariate_sum": 5.0,
                                "covariate_sum_squares": 5.0,
                                "outcome_covariate_sum": 2.0,
                            },
                            {
                                "name": "t",
                                "visitors": 10,
                                "is_control": False,
                                "numerator_sum": 12.0,
                                "numerator_sum_squares": 25.0,
                                "denominator_sum": 20.0,
                                "denominator_sum_squares": 50.0,
                                "numerator_denominator_sum": 28.0,
                                "covariate_sum": 5.0,
                                "covariate_sum_squares": 5.0,
                                "outcome_covariate_sum": 2.0,
                            },
                        ],
                    }
                )
            )

        # 8. Direct connector call with 2-column row-level data and covariate_column -> UsageOptionError
        rows = [
            {"variant": "c", "outcome": "1"},
            {"variant": "c", "outcome": "0"},
            {"variant": "t", "outcome": "1"},
            {"variant": "t", "outcome": "0"},
        ]
        with pytest.raises(UsageOptionError, match="at least 3 columns"):
            build_payload_from_rows(rows, defaults={"covariate_column": "pre_x"})
