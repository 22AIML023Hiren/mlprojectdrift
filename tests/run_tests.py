"""
Standard test runner using Python built-in unittest framework.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import generate_synthetic_benchmark, get_chronological_splits
from src.data.validator import validate_dataset_schema
from src.data.shifts import ShiftGenerator


class TestPhase1DataPipeline(unittest.TestCase):

    def setUp(self):
        self.sample_dataset = generate_synthetic_benchmark(n_samples=5000, fraud_rate=0.01, random_state=42)

    def test_schema_validation(self):
        report = validate_dataset_schema(self.sample_dataset)
        self.assertTrue(report["is_valid"], f"Schema validation failed: {report['issues']}")
        self.assertEqual(report["n_samples"], 5000)
        self.assertEqual(report["n_features"], 30)
        self.assertEqual(report["n_positives"], 50)
        self.assertTrue(report["is_monotonic_time"])
        self.assertEqual(len(report["issues"]), 0)

    def test_chronological_ordering_and_leakage(self):
        splits_unscaled = get_chronological_splits(self.sample_dataset, scale_features=False)
        time_train = splits_unscaled["X_train"]["Time"]
        time_calib = splits_unscaled["X_calib"]["Time"]
        time_test = splits_unscaled["X_test"]["Time"]

        self.assertLessEqual(time_train.max(), time_calib.min(), "Leakage: Train time overlaps with Calib time")
        self.assertLessEqual(time_calib.max(), time_test.min(), "Leakage: Calib time overlaps with Test time")

    def test_split_proportions(self):
        splits = get_chronological_splits(self.sample_dataset, train_ratio=0.70, calib_ratio=0.15, test_ratio=0.15)
        n_train = len(splits["X_train"])
        n_calib = len(splits["X_calib"])
        n_test = len(splits["X_test"])

        self.assertEqual(n_train, 3500)
        self.assertEqual(n_calib, 750)
        self.assertEqual(n_test, 750)
        self.assertEqual(n_train + n_calib + n_test, 5000)

    def test_scaler_isolation(self):
        splits = get_chronological_splits(self.sample_dataset, scale_features=True)
        scaler = splits["scaler"]
        self.assertIsNotNone(scaler)

        raw_time_train = self.sample_dataset.iloc[:3500]["Time"]
        expected_time_median = np.median(raw_time_train)
        self.assertAlmostEqual(scaler.center_[0], expected_time_median, places=3)

    def test_shift_generator_integrity(self):
        splits = get_chronological_splits(self.sample_dataset, scale_features=True)
        X_test, y_test = splits["X_test"], splits["y_test"]

        sg = ShiftGenerator(random_state=42)
        testbeds = sg.generate_all_shift_testbeds(X_test, y_test)

        expected_testbeds = [
            "Clean_ID", "Covariate_Mild", "Covariate_Severe",
            "Subpopulation_HighAmount", "Prevalence_Shift", "Concept_Drift"
        ]
        for name in expected_testbeds:
            self.assertIn(name, testbeds)
            X_s, y_s = testbeds[name]
            self.assertEqual(X_s.shape[1], X_test.shape[1], f"Testbed {name} altered feature dimension!")
            self.assertEqual(len(X_s), len(y_s), f"Testbed {name} has mismatched lengths!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
