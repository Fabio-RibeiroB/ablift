"""Edge-case tests for CSV/XLSX parsing and payload building in ablift.connectors."""

import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from ablift.connectors import (
    build_duration_request_from_rows,
    build_payload_from_rows,
    read_table,
)
from ablift.engine import analyze, parse_payload


def agg(name, visitors, conversions, **extra):
    return {"name": name, "visitors": visitors, "conversions": conversions, **extra}


def by_name(payload):
    return {v["name"]: v for v in payload["variants"]}


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def write_csv(self, text, name="in.csv", encoding="utf-8"):
        path = self.dir / name
        path.write_text(text, encoding=encoding, newline="")
        return str(path)

    def write_xlsx(self, sheets, name="in.xlsx", active=None):
        wb = Workbook()
        wb.remove(wb.active)
        for title, rows in sheets.items():
            ws = wb.create_sheet(title)
            for row in rows:
                ws.append(row)
        if active is not None:
            wb.active = wb.sheetnames.index(active)
        path = self.dir / name
        wb.save(path)
        return str(path)


class ReadTableCsv(TempDirCase):
    def test_header_only_csv_is_empty_and_rejected_downstream(self):
        rows = read_table(self.write_csv("variant,visitors,conversions\n"))
        self.assertEqual(rows, [])
        with self.assertRaisesRegex(ValueError, "empty"):
            build_payload_from_rows(rows)

    def test_completely_empty_file(self):
        self.assertEqual(read_table(self.write_csv("")), [])

    def test_uppercase_extension_is_accepted(self):
        path = self.write_csv("a,b\n1,2\n", name="DATA.CSV")
        self.assertEqual(read_table(path), [{"a": "1", "b": "2"}])

    def test_unsupported_extension(self):
        path = self.write_csv("a,b\n", name="data.txt")
        with self.assertRaisesRegex(ValueError, "Unsupported file type"):
            read_table(path)
        with self.assertRaisesRegex(ValueError, "Unsupported file type"):
            read_table(str(self.dir / "noextension"))

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            read_table(str(self.dir / "nope.csv"))

    def test_quoted_fields_and_embedded_commas(self):
        path = self.write_csv('name,visitors,conversions\n"Control, v0",100,5\n"v1",100,6\n')
        rows = read_table(path)
        self.assertEqual(rows[0]["name"], "Control, v0")

    def test_crlf_line_endings(self):
        path = self.write_csv("name,visitors,conversions\r\nc,100,5\r\nv,100,7\r\n")
        payload = build_payload_from_rows(read_table(path))
        self.assertEqual(by_name(payload)["v"]["conversions"], 7)

    def test_bom_header_still_builds_correct_variants(self):
        path = self.write_csv(
            "variant,visitors,conversions\ncontrol,1000,40\nv1,1000,50\n", encoding="utf-8-sig"
        )
        payload = build_payload_from_rows(read_table(path))
        self.assertEqual([v["name"] for v in payload["variants"]], ["control", "v1"])
        self.assertTrue(payload["variants"][0]["is_control"])
        self.assertEqual(payload["variants"][1]["conversions"], 50)

    def test_bom_is_stripped_from_first_header(self):
        # Regression (was a bug): connectors.py read_table opens CSVs with encoding="utf-8", so a UTF-8 BOM
        # (written by Excel "CSV UTF-8") stays glued to the first column name
        # ('﻿variant'); it leaks into input_interpretation.column_roles.
        path = self.write_csv(
            "variant,visitors,conversions\ncontrol,1000,40\nv1,1000,50\n", encoding="utf-8-sig"
        )
        rows = read_table(path)
        self.assertEqual(list(rows[0])[0], "variant")

    def test_bom_duration_csv_first_column(self):
        # Regression (was a bug): same BOM problem; a duration CSV whose first column is 'method' is rejected
        # with "Unknown duration column(s): ﻿method".
        path = self.write_csv("method,baseline_rate\nbayesian,0.04\n", encoding="utf-8-sig")
        self.assertEqual(
            build_duration_request_from_rows(read_table(path)),
            {"method": "bayesian", "baseline_rate": 0.04},
        )


class ReadTableXlsx(TempDirCase):
    def test_active_sheet_is_default(self):
        path = self.write_xlsx(
            {"first": [["a", "b"], [1, 2]], "second": [["x", "y"], [3, 4]]}, active="second"
        )
        self.assertEqual(read_table(path), [{"x": 3, "y": 4}])

    def test_named_sheet(self):
        path = self.write_xlsx({"first": [["a", "b"], [1, 2]], "second": [["x", "y"], [3, 4]]})
        self.assertEqual(read_table(path, sheet="second"), [{"x": 3, "y": 4}])
        self.assertEqual(read_table(path, sheet="first"), [{"a": 1, "b": 2}])

    def test_unknown_sheet_raises_clean_value_error(self):
        # Regression (was a bug): connectors.py read_table does wb[sheet] and leaks a raw KeyError
        # ("Worksheet nope does not exist.") instead of a ValueError listing the sheets.
        path = self.write_xlsx({"first": [["a"], [1]]})
        with self.assertRaises(ValueError):
            read_table(path, sheet="nope")

    def test_empty_workbook(self):
        path = self.write_xlsx({"Sheet": []})
        self.assertEqual(read_table(path), [])

    def test_header_only(self):
        path = self.write_xlsx({"Sheet": [["variant", "visitors", "conversions"]]})
        self.assertEqual(read_table(path), [])

    def test_blank_and_none_header_cells_are_dropped(self):
        path = self.write_xlsx(
            {"Sheet": [["variant", None, "visitors", "  conversions  "], ["c", "junk", 10, 1]]}
        )
        rows = read_table(path)
        self.assertEqual(rows, [{"variant": "c", "visitors": 10, "conversions": 1}])

    def test_short_rows_padded_with_none(self):
        path = self.write_xlsx({"Sheet": [["a", "b", "c"], [1]]})
        self.assertEqual(read_table(path), [{"a": 1, "b": None, "c": None}])

    def test_numeric_header_is_stringified(self):
        path = self.write_xlsx({"Sheet": [[2024, "x"], [1, 2]]})
        self.assertEqual(read_table(path), [{"2024": 1, "x": 2}])

    def test_xlsx_numbers_keep_native_types(self):
        path = self.write_xlsx(
            {"Sheet": [["name", "visitors", "conversions"], ["c", 1000, 40], ["v", 1000, 55]]}
        )
        payload = build_payload_from_rows(read_table(path))
        self.assertEqual(by_name(payload)["v"]["visitors"], 1000)
        res = analyze(parse_payload({**payload, "samples": 2000}))
        self.assertGreater(res.comparisons[0].absolute_lift, 0)

    def test_trailing_none_rows_are_ignored_downstream(self):
        # Regression (was a bug): Excel sheets often carry trailing formatted-but-empty rows. read_table keeps
        # them as all-None dicts and build_payload_from_rows then fails with
        # "Missing required integer field 'visitors'" instead of skipping them.
        path = self.write_xlsx(
            {
                "Sheet": [
                    ["variant", "visitors", "conversions"],
                    ["c", 10, 1],
                    ["v", 10, 2],
                    [None, None, None],
                    [None, None, None],
                ]
            }
        )
        payload = build_payload_from_rows(read_table(path))
        self.assertEqual([v["name"] for v in payload["variants"]], ["c", "v"])

    def test_xlsm_extension_supported_via_xlsx_content(self):
        path = self.write_xlsx({"Sheet": [["a"], [1]]}, name="macro.xlsx")
        renamed = self.dir / "macro.xlsm"
        Path(path).rename(renamed)
        # openpyxl refuses macro-less files with a .xlsm suffix only when keep_vba is set;
        # reading values must work.
        self.assertEqual(read_table(str(renamed)), [{"a": 1}])


class AggregatedInput(unittest.TestCase):
    def test_whitespace_in_names_and_numbers(self):
        rows = [agg("  control ", " 1000 ", " 40 "), agg(" v1", "1000", "45 ")]
        payload = build_payload_from_rows(rows)
        self.assertEqual([v["name"] for v in payload["variants"]], ["control", "v1"])
        self.assertEqual(payload["variants"][0]["visitors"], 1000)
        self.assertEqual(payload["variants"][0]["conversions"], 40)

    def test_blank_rows_skipped_and_first_real_row_is_control(self):
        rows = [
            agg("", "", ""),
            agg("  ", "", ""),
            agg("c", "10", "1"),
            agg("", "", ""),
            agg("v", "10", "2"),
        ]
        payload = build_payload_from_rows(rows)
        self.assertEqual([v["name"] for v in payload["variants"]], ["c", "v"])
        self.assertEqual([v["is_control"] for v in payload["variants"]], [True, False])
        self.assertEqual(payload["input_interpretation"]["variant_count"], 2)

    def test_control_column_truthy_spellings(self):
        for yes, no in (("yes", "no"), ("TRUE", "FALSE"), ("1", "0"), ("Y", "N"), (True, False)):
            with self.subTest(yes=yes):
                rows = [
                    agg("a", "100", "5", is_control=no),
                    agg("b", "100", "6", is_control=yes),
                ]
                payload = build_payload_from_rows(rows)
                self.assertEqual(
                    [v["is_control"] for v in payload["variants"]], [False, True], (yes, no)
                )
                self.assertEqual(
                    payload["input_interpretation"]["column_roles"]["is_control"], "is_control"
                )

    def test_control_column_in_last_position_with_five_columns(self):
        rows = [
            {
                "name": "a",
                "visitors": "100",
                "conversions": "10",
                "rev": "100",
                "sq": "2000",
                "is_control": "false",
            },
            {
                "name": "b",
                "visitors": "100",
                "conversions": "12",
                "rev": "120",
                "sq": "2500",
                "is_control": "true",
            },
        ]
        payload = build_payload_from_rows(rows)
        self.assertEqual(payload["primary_metric"], "arpu")
        v = by_name(payload)
        self.assertTrue(v["b"]["is_control"])
        self.assertFalse(v["a"]["is_control"])
        self.assertEqual(v["b"]["conversions"], 12)
        self.assertEqual(v["b"]["revenue_sum"], 120.0)
        self.assertEqual(v["b"]["revenue_sum_squares"], 2500.0)
        roles = payload["input_interpretation"]["column_roles"]
        self.assertEqual(roles["revenue_sum"], "rev")
        self.assertEqual(roles["is_control"], "is_control")

    def test_four_columns_revenue_only_has_no_conversions(self):
        rows = [
            {"name": "a", "visitors": "100", "rev": "100", "sq": "2000"},
            {"name": "b", "visitors": "100", "rev": "120", "sq": "2500"},
        ]
        payload = build_payload_from_rows(rows)
        self.assertEqual(payload["primary_metric"], "arpu")
        self.assertNotIn("conversions", payload["variants"][0])
        res = analyze(parse_payload({**payload, "samples": 2000}))
        self.assertEqual(res.comparisons[0].metric, "arpu")

    def test_no_control_column_first_row_is_control(self):
        payload = build_payload_from_rows(
            [agg("x", "10", "1"), agg("y", "10", "2"), agg("z", "10", "3")]
        )
        self.assertEqual([v["is_control"] for v in payload["variants"]], [True, False, False])
        self.assertNotIn("is_control", payload["input_interpretation"]["column_roles"])

    def test_two_controls_rejected_at_analysis_time(self):
        rows = [
            agg("a", "100", "1", c="yes"),
            agg("b", "100", "1", c="yes"),
            agg("c", "100", "1", c="no"),
        ]
        payload = build_payload_from_rows(rows)
        self.assertEqual(sum(v["is_control"] for v in payload["variants"]), 2)
        with self.assertRaisesRegex(ValueError, "Exactly one variant"):
            analyze(parse_payload(payload))

    def test_non_numeric_visitors_is_value_error(self):
        with self.assertRaises(ValueError):
            build_payload_from_rows([agg("a", "lots", "1"), agg("b", "10", "1")])

    def test_non_numeric_conversions_is_value_error(self):
        with self.assertRaises(ValueError):
            build_payload_from_rows([agg("a", "10", "n/a"), agg("b", "10", "1")])

    def test_non_numeric_visitors_message_names_the_column(self):
        # Regression (was a bug): connectors.py _to_int: int(float("lots")) leaks Python's
        # "could not convert string to float: 'lots'" with no column or row context
        # (the missing-value branch does name the column).
        with self.assertRaisesRegex(ValueError, "visitors"):
            build_payload_from_rows([agg("a", "lots", "1"), agg("b", "10", "1")])

    def test_thousands_separator_is_rejected_not_misparsed(self):
        with self.assertRaises(ValueError):
            build_payload_from_rows([agg("a", "1,000", "1"), agg("b", "1000", "1")])

    def test_float_formatted_ints(self):
        payload = build_payload_from_rows([agg("a", "1000.0", "40.0"), agg("b", 1000.0, 41.0)])
        for v in payload["variants"]:
            self.assertIsInstance(v["visitors"], int)
            self.assertIsInstance(v["conversions"], int)
        self.assertEqual(payload["variants"][0]["visitors"], 1000)
        self.assertEqual(payload["variants"][1]["conversions"], 41)

    def test_missing_visitors_or_conversions(self):
        with self.assertRaisesRegex(ValueError, "Missing required integer field 'conversions'"):
            build_payload_from_rows([agg("a", "10", ""), agg("b", "10", "1")])
        with self.assertRaisesRegex(ValueError, "Missing required integer field 'visitors'"):
            build_payload_from_rows([agg("a", None, "1"), agg("b", "10", "1")])

    def test_fractional_conversions_looks_like_revenue(self):
        with self.assertRaisesRegex(ValueError, "looks like revenue"):
            build_payload_from_rows([agg("a", "100", "12.5"), agg("b", "100", "13.5")])

    def test_conversions_above_visitors_flagged_at_build_time(self):
        with self.assertRaisesRegex(ValueError, "looks like revenue"):
            build_payload_from_rows([agg("a", "100", "500"), agg("b", "100", "5")])

    def test_two_columns_rejected_with_layout_help(self):
        with self.assertRaisesRegex(ValueError, "at least 3 columns"):
            build_payload_from_rows(
                [{"name": "a", "visitors": "1"}, {"name": "b", "visitors": "2"}]
            )

    def test_primary_metric_override_mismatches(self):
        rows = [agg("a", "10", "1"), agg("b", "10", "2")]
        with self.assertRaisesRegex(ValueError, "needs revenue_sum"):
            build_payload_from_rows(rows, {"primary_metric": "arpu"})
        rev = [
            {"name": "a", "visitors": "10", "r": "5", "s": "30"},
            {"name": "b", "visitors": "10", "r": "6", "s": "40"},
        ]
        with self.assertRaisesRegex(ValueError, "needs a conversions column"):
            build_payload_from_rows(rev, {"primary_metric": "conversion_rate"})

    def test_defaults_propagate_to_payload(self):
        payload = build_payload_from_rows(
            [agg("a", "10", "1"), agg("b", "10", "2")],
            {"method": "sequential", "alpha": 0.01, "random_seed": 3, "samples": 777},
        )
        self.assertEqual(payload["method"], "sequential")
        self.assertEqual(payload["alpha"], 0.01)
        self.assertEqual(payload["random_seed"], 3)
        self.assertEqual(payload["samples"], 777)
        self.assertEqual(payload["input_interpretation"]["source_type"], "aggregated")

    def test_all_false_control_column_is_not_misdetected_silently(self):
        # A control column with only "no" values is not recognised (it needs at least one
        # true and one false), so it is mistaken for a metric column and the user gets an
        # error instead of a silent mis-analysis.
        rows = [agg("a", "10", "1", c="no"), agg("b", "10", "1", c="no")]
        with self.assertRaises(ValueError):
            build_payload_from_rows(rows)

    def test_single_row_builds_but_fails_validation(self):
        payload = build_payload_from_rows([agg("a", "10", "1")])
        with self.assertRaisesRegex(ValueError, "At least 2 variants"):
            analyze(parse_payload(payload))

    def test_end_to_end_realistic_aggregated_table(self):
        rows = [
            agg("control", "200000", "8000"),
            agg("one_page", "200000", "8100"),
            agg("express_pay", "200000", "8800"),
        ]
        payload = build_payload_from_rows(rows, {"method": "sequential"})
        res = analyze(parse_payload(payload))
        self.assertEqual(res.recommendation.action, "ship_express_pay")


class RowLevelInput(unittest.TestCase):
    @staticmethod
    def rows(*pairs):
        return [{"variant": a, "outcome": b} for a, b in pairs]

    def test_blank_outcome_cells_count_as_visitors_without_conversion(self):
        payload = build_payload_from_rows(
            self.rows(("c", "1"), ("c", ""), ("c", "0"), ("v", "1"), ("v", None), ("v", "1"))
        )
        v = by_name(payload)
        self.assertEqual((v["c"]["visitors"], v["c"]["conversions"]), (3, 1))
        self.assertEqual((v["v"]["visitors"], v["v"]["conversions"]), (3, 2))

    def test_blank_revenue_cells_are_dropped_from_visitors(self):
        # Documented asymmetry: for revenue outcomes a blank cell removes the row from the
        # denominator, whereas for binary outcomes it stays as a non-converter.
        payload = build_payload_from_rows(
            self.rows(("c", "0"), ("c", "12.5"), ("c", "0"), ("v", "0"), ("v", "3"), ("v", ""))
        )
        v = by_name(payload)
        self.assertEqual(payload["primary_metric"], "arpu")
        self.assertEqual(v["c"]["visitors"], 3)
        self.assertEqual(v["v"]["visitors"], 2)
        self.assertEqual(v["c"]["conversions"], 1)
        self.assertAlmostEqual(v["c"]["revenue_sum"], 12.5)
        self.assertAlmostEqual(v["c"]["revenue_sum_squares"], 156.25)

    def test_boolean_string_outcomes(self):
        payload = build_payload_from_rows(
            self.rows(
                ("c", "true"),
                ("c", "FALSE"),
                ("c", "False"),
                ("v", "True"),
                ("v", "yes"),
                ("v", "no"),
            )  # fmt: skip
        )
        v = by_name(payload)
        self.assertEqual(payload["primary_metric"], "conversion_rate")
        self.assertEqual(v["c"]["conversions"], 1)
        self.assertEqual(v["v"]["conversions"], 2)

    def test_python_bool_outcomes(self):
        payload = build_payload_from_rows(
            self.rows(("c", True), ("c", False), ("v", True), ("v", True))
        )
        self.assertEqual(by_name(payload)["v"]["conversions"], 2)

    def test_override_arpu_on_binary_outcomes_errors(self):
        with self.assertRaisesRegex(ValueError, "binary; use primary_metric conversion_rate"):
            build_payload_from_rows(
                self.rows(("c", "1"), ("c", "0"), ("v", "1")), {"primary_metric": "arpu"}
            )

    def test_override_conversion_on_continuous_outcomes_errors(self):
        with self.assertRaisesRegex(ValueError, "continuous; conversion_rate needs 0/1"):
            build_payload_from_rows(
                self.rows(("c", "1.5"), ("c", "0"), ("v", "7")),
                {"primary_metric": "conversion_rate"},
            )

    def test_variant_order_is_first_appearance(self):
        payload = build_payload_from_rows(
            self.rows(("z", "1"), ("a", "0"), ("z", "0"), ("m", "1"), ("a", "1"), ("m", "0"))
        )
        self.assertEqual([v["name"] for v in payload["variants"]], ["z", "a", "m"])
        self.assertEqual([v["is_control"] for v in payload["variants"]], [True, False, False])

    def test_many_variants(self):
        rows = [{"variant": f"v{i}", "outcome": str(j % 2)} for i in range(12) for j in range(4)]
        payload = build_payload_from_rows(rows)
        self.assertEqual(len(payload["variants"]), 12)
        self.assertEqual(sum(v["is_control"] for v in payload["variants"]), 1)
        self.assertTrue(
            all(v["visitors"] == 4 and v["conversions"] == 2 for v in payload["variants"])
        )
        self.assertEqual(payload["input_interpretation"]["variant_count"], 12)

    def test_interleaved_rows_group_correctly(self):
        rows = self.rows(*[(("a", "b")[i % 2], str(i % 3 == 0 and 1 or 0)) for i in range(10)])
        payload = build_payload_from_rows(rows)
        v = by_name(payload)
        self.assertEqual(v["a"]["visitors"] + v["b"]["visitors"], 10)

    def test_whitespace_in_variant_names_merges_groups(self):
        payload = build_payload_from_rows(
            self.rows((" c", "1"), ("c ", "0"), ("v", "1"), ("v", "0"))
        )
        self.assertEqual([v["name"] for v in payload["variants"]], ["c", "v"])
        self.assertEqual(by_name(payload)["c"]["visitors"], 2)

    def test_row_level_control_column(self):
        rows = [
            {"variant": "a", "outcome": "1", "ctl": "yes"},
            {"variant": "a", "outcome": "0", "ctl": "yes"},
            {"variant": "b", "outcome": "1", "ctl": "no"},
            {"variant": "b", "outcome": "0", "ctl": "no"},
        ]
        payload = build_payload_from_rows(rows)
        self.assertTrue(by_name(payload)["a"]["is_control"])
        self.assertFalse(by_name(payload)["b"]["is_control"])
        self.assertEqual(payload["input_interpretation"]["column_roles"]["is_control"], "ctl")

    def test_single_column_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 2 columns"):
            build_payload_from_rows([{"variant": "a"}, {"variant": "a"}])

    def test_all_blank_variant_names_rejected(self):
        with self.assertRaisesRegex(ValueError, "no variant data"):
            build_payload_from_rows(self.rows(("", "1"), ("", "0")))

    def test_zero_one_revenue_is_treated_as_conversion(self):
        payload = build_payload_from_rows(self.rows(("c", "0"), ("c", "1"), ("v", "1"), ("v", "1")))
        self.assertEqual(payload["primary_metric"], "conversion_rate")

    def test_non_numeric_revenue_is_value_error(self):
        with self.assertRaises(ValueError):
            build_payload_from_rows(self.rows(("c", "12.5"), ("c", "oops"), ("v", "3"), ("v", "4")))

    def test_row_level_revenue_end_to_end(self):
        rows = [{"variant": "c", "revenue": str(x)} for x in (0, 0, 10, 20, 0, 30)]
        rows += [{"variant": "v", "revenue": str(x)} for x in (0, 15, 25, 35, 0, 45)]
        payload = build_payload_from_rows(rows, {"method": "sequential"})
        v = by_name(payload)
        self.assertEqual(v["c"]["revenue_sum"], 60.0)
        self.assertEqual(v["v"]["revenue_sum_squares"], 15**2 + 25**2 + 35**2 + 45**2)
        res = analyze(parse_payload(payload))
        self.assertEqual(res.comparisons[0].control_rate, 10.0)


class DurationRequests(unittest.TestCase):
    def test_case_insensitive_headers_and_types(self):
        out = build_duration_request_from_rows(
            [
                {
                    " Method ": " Sequential ",
                    "BASELINE_RATE": "0.04",
                    "Relative_MDE": "0.1",
                    "Daily_Traffic": "20000.0",
                    "N_VARIANTS": "3",
                    "max_days": "30",
                }
            ]
        )
        self.assertEqual(
            out,
            {
                "method": "sequential",
                "baseline_rate": 0.04,
                "relative_mde": 0.1,
                "daily_traffic": 20000,
                "n_variants": 3,
                "max_days": 30,
            },
        )
        self.assertIsInstance(out["daily_traffic"], int)

    def test_empty_cells_skipped(self):
        out = build_duration_request_from_rows(
            [{"method": "", "baseline_rate": None, "alpha": "0.1", "power": ""}]
        )
        self.assertEqual(out, {"alpha": 0.1})

    def test_unknown_columns_error_lists_them_sorted(self):
        with self.assertRaises(ValueError) as ctx:
            build_duration_request_from_rows([{"zeta": 1, "alpha": "0.1", "beta": 2}])
        msg = str(ctx.exception)
        self.assertIn("beta, zeta", msg)
        self.assertIn("Expected any of", msg)

    def test_empty_table(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            build_duration_request_from_rows([])

    def test_only_first_row_is_used(self):
        out = build_duration_request_from_rows([{"alpha": "0.1"}, {"alpha": "0.9"}])
        self.assertEqual(out, {"alpha": 0.1})

    def test_none_header_from_extra_cells_is_ignored(self):
        out = build_duration_request_from_rows([{None: ["stray"], "alpha": "0.1"}])
        self.assertEqual(out, {"alpha": 0.1})

    def test_non_numeric_value_is_value_error(self):
        with self.assertRaises(ValueError):
            build_duration_request_from_rows([{"baseline_rate": "four percent"}])

    def test_all_known_fields_round_trip(self):
        row = {
            "method": "bayesian",
            "baseline_rate": "0.05",
            "relative_mde": "-0.1",
            "daily_traffic": "1000",
            "n_variants": "2",
            "alpha": "0.05",
            "power": "0.8",
            "sequential_tau": "0.01",
            "prob_threshold": "0.95",
            "max_expected_loss": "0.001",
            "assurance_target": "0.8",
            "max_days": "45",
        }
        out = build_duration_request_from_rows([row])
        self.assertEqual(set(out), set(row))
        self.assertEqual(out["relative_mde"], -0.1)

    def test_xlsx_duration_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            wb = Workbook()
            ws = wb.active
            ws.append(["method", "baseline_rate", "daily_traffic", "max_days", "alpha"])
            ws.append(["sequential", 0.04, 20000, 30, None])
            path = Path(d) / "dur.xlsx"
            wb.save(path)
            out = build_duration_request_from_rows(read_table(str(path)))
        self.assertEqual(
            out,
            {"method": "sequential", "baseline_rate": 0.04, "daily_traffic": 20000, "max_days": 30},
        )


if __name__ == "__main__":
    unittest.main()
