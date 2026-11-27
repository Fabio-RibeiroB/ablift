import unittest
from unittest.mock import patch

from ablift.cli import _version_meets_minimum, run_doctor


class DoctorTests(unittest.TestCase):
    def test_doctor_has_expected_checks(self):
        result = run_doctor()
        names = {item["name"] for item in result["checks"]}
        self.assertIn("python_version", names)
        self.assertIn("dependency_click", names)
        self.assertIn("dependency_numpy", names)
        self.assertIn("dependency_openpyxl", names)
        self.assertTrue(result["ok"])

    def test_version_meets_minimum(self):
        self.assertTrue(_version_meets_minimum("1.22", ">=1.22"))
        self.assertTrue(_version_meets_minimum("1.22.0", ">=1.22"))
        self.assertTrue(_version_meets_minimum("2.4.3", ">=1.22"))
        self.assertTrue(_version_meets_minimum("3.1.0.post1", ">=3.1.0"))
        self.assertFalse(_version_meets_minimum("1.21.9", ">=1.22"))
        self.assertFalse(_version_meets_minimum("3.0.10", ">=3.1.0"))
        self.assertFalse(_version_meets_minimum("1.22.0a1", ">=1.22"))
        self.assertFalse(_version_meets_minimum("3.1.0rc1", ">=3.1.0"))
        self.assertFalse(_version_meets_minimum("not-a-version", ">=1.0"))

    def test_doctor_fails_when_dependency_is_below_minimum(self):
        fake_versions = {"click": "8.1.7", "numpy": "1.21.5", "openpyxl": "3.1.5"}
        with patch("ablift.cli.importlib.metadata.version", side_effect=fake_versions.__getitem__):
            result = run_doctor()
        self.assertFalse(result["ok"])
        by_name = {c["name"]: c for c in result["checks"]}
        self.assertFalse(by_name["dependency_numpy"]["passed"])
        self.assertTrue(by_name["dependency_click"]["passed"])
        self.assertTrue(by_name["dependency_openpyxl"]["passed"])


if __name__ == "__main__":
    unittest.main()
