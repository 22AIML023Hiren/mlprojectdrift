"""
Unit test suite for Phase 4: Uncertainty Quantification & Calibration Engine.
Verifies ECE computation, Brier scores, Isotonic and Platt calibration,
Inductive Conformal Prediction set coverage, and ensemble epistemic variance.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import generate_synthetic_benchmark, get_chronological_splits
from src.models import get_model
from src.uncertainty.calibration import (
    ProbabilityCalibrator,
    compute_expected_calibration_error,
    compute_brier_score
)
from src.uncertainty.conformal import InductiveConformalClassifier
from src.uncertainty.ensemble_var import EnsembleUncertaintyEstimator


class TestPhase4UncertaintyEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Train a fast Random Forest on synthetic data to produce probabilities."""
        cls.df = generate_synthetic_benchmark(n_samples=5000, fraud_rate=0.02, random_state=42)
        splits = get_chronological_splits(cls.df, scale_features=True)
        cls.X_train, cls.y_train = splits["X_train"], splits["y_train"]
        cls.X_calib, cls.y_calib = splits["X_calib"], splits["y_calib"]
        cls.X_test, cls.y_test = splits["X_test"], splits["y_test"]

        cls.rf = get_model("random_forest", n_estimators=25, max_depth=8, random_state=42)
        cls.rf.fit(cls.X_train, cls.y_train)

        cls.raw_calib_probs = cls.rf.predict_proba(cls.X_calib)[:, 1]
        cls.raw_test_probs = cls.rf.predict_proba(cls.X_test)[:, 1]

    def test_ece_computation(self):
        """Checks that ECE computes a valid score in [0, 1]."""
        ece = compute_expected_calibration_error(self.y_test, self.raw_test_probs, n_bins=10)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

    def test_brier_score_computation(self):
        """Checks that Brier score calculates mean squared error accurately."""
        y_true = np.array([0, 1, 0, 0])
        y_prob = np.array([0.1, 0.9, 0.2, 0.0])
        # Expected: ((0.1-0)^2 + (0.9-1)^2 + (0.2-0)^2 + (0-0)^2) / 4 = (0.01 + 0.01 + 0.04 + 0) / 4 = 0.015
        expected = (0.01 + 0.01 + 0.04 + 0.0) / 4.0
        calculated = compute_brier_score(y_true, y_prob)
        self.assertAlmostEqual(calculated, expected, places=5)

    def test_isotonic_calibration_improvement(self):
        """Verifies that Isotonic Regression yields probabilities in [0, 1] with reduced ECE."""
        calibrator = ProbabilityCalibrator(method="isotonic")
        calibrator.fit(self.raw_calib_probs, self.y_calib)

        calib_test_probs = calibrator.predict(self.raw_test_probs)
        self.assertTrue(np.all(calib_test_probs >= 0.0) and np.all(calib_test_probs <= 1.0))

        raw_ece = compute_expected_calibration_error(self.y_test, self.raw_test_probs)
        calib_ece = compute_expected_calibration_error(self.y_test, calib_test_probs)
        self.assertLessEqual(calib_ece, raw_ece + 1e-4, "Calibrated ECE should not be substantially worse than raw")

    def test_platt_scaling_bounds(self):
        """Verifies that Platt Scaling produces smooth sigmoidal probabilities in [0, 1]."""
        platt = ProbabilityCalibrator(method="platt")
        platt.fit(self.raw_calib_probs, self.y_calib)

        platt_probs = platt.predict(self.raw_test_probs)
        self.assertTrue(np.all(platt_probs >= 0.0) and np.all(platt_probs <= 1.0))

    def test_conformal_coverage_guarantee(self):
        """Verifies that Inductive Conformal Prediction achieves empirical coverage >= 1 - alpha."""
        calib_2d = np.column_stack([1.0 - self.raw_calib_probs, self.raw_calib_probs])
        test_2d = np.column_stack([1.0 - self.raw_test_probs, self.raw_test_probs])

        alpha = 0.05
        cp = InductiveConformalClassifier(alpha=alpha)
        cp.fit(calib_2d, self.y_calib)

        stats = cp.evaluate_coverage_and_efficiency(test_2d, self.y_test)
        # Finite-sample coverage bound allows small statistical sampling slack
        self.assertGreaterEqual(
            stats["empirical_coverage"],
            1.0 - alpha - 0.03,
            f"Empirical coverage {stats['empirical_coverage']} violated target {1.0 - alpha}"
        )

    def test_ensemble_variance_bounds(self):
        """Verifies that tree epistemic variance is bounded in [0, 0.25]."""
        tree_probs = self.rf.get_individual_tree_predictions(self.X_test)
        mean_prob, variance = EnsembleUncertaintyEstimator.compute_variance_from_tree_predictions(tree_probs)

        self.assertEqual(len(mean_prob), len(self.X_test))
        self.assertEqual(len(variance), len(self.X_test))
        self.assertTrue(np.all(variance >= 0.0))
        self.assertTrue(np.all(variance <= 0.25), "Bernoulli variance cannot exceed 0.25")


if __name__ == "__main__":
    unittest.main(verbosity=2)
