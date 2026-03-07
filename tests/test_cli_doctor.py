import unittest

from bayestest.cli import run_doctor


class DoctorTests(unittest.TestCase):
    def test_doctor_has_expected_checks(self):
        result = run_doctor()
        names = {item["name"] for item in result["checks"]}
        self.assertIn("python_version", names)
        self.assertIn("dependency_numpy", names)
        self.assertIn("dependency_openpyxl", names)


if __name__ == "__main__":
    unittest.main()
