"""
Unit test suite for Phase 5: Risk-Aware Selective Classification.
Verifies coverage monotonicity, selective error mitigation, economic cost math,
AURC calculation, and edge-case robustness.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.decision.risk_scorer import RiskScorer
from src.decision.selector import SelectiveClassifier


class TestSelectiveClassification(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n_samples = 500

        # Synthetic ground truth: 5% fraud rate
        self.y_true = np.zeros(self.n_samples, dtype=int)
        self.y_true[:25] = 1
        np.random.shuffle(self.y_true)

        # Synthetic probabilities with some error on ambiguous points
        self.probs = np.random.uniform(0.01, 0.99, size=self.n_samples)
        # Ensure higher probabilities on actual fraud with noise
        self.probs[self.y_true == 1] = np.random.beta(5, 2, size=25)
        self.probs[self.y_true == 0] = np.random.beta(1, 10, size=self.n_samples - 25)

        # Risk scorer & synthetic variance
        self.scorer = RiskScorer(fit_ood_detector=False)
        self.variance = np.random.uniform(0.0, 0.05, size=self.n_samples)

        self.composite_risk = self.scorer.compute_composite_risk(
            calibrated_probs=self.probs,
            ensemble_variance=self.variance,
            w_conf=0.6,
            w_var=0.4
        )

        self.selector = SelectiveClassifier(cost_fn=100.0, cost_fp=5.0, cost_defer=2.0)

    def test_coverage_monotonicity(self):
        """Increasing threshold tau must produce monotonically non-decreasing coverage."""
        thresholds = np.linspace(0.05, 0.95, 20)
        coverages = []
        for tau in thresholds:
            decision = self.selector.predict_selective(self.probs, self.composite_risk, threshold=tau)
            coverages.append(decision["coverage"])

        for i in range(len(coverages) - 1):
            self.assertLessEqual(
                coverages[i], coverages[i + 1] + 1e-9,
                f"Monotonicity violated: Coverage({thresholds[i]:.2f})={coverages[i]} > Coverage({thresholds[i+1]:.2f})={coverages[i+1]}"
            )

    def test_selective_error_reduction(self):
        """Selective classification at c=80% must achieve lower or equal error than c=100%."""
        tau_100 = 1.01
        tau_80 = self.selector.find_threshold_for_target_coverage(self.composite_risk, 0.80)

        dec_100 = self.selector.predict_selective(self.probs, self.composite_risk, threshold=tau_100)
        dec_80 = self.selector.predict_selective(self.probs, self.composite_risk, threshold=tau_80)

        y_pred_100 = dec_100["predictions"]
        err_100 = (y_pred_100 != self.y_true).sum() / len(self.y_true)

        acc_80 = dec_80["accepted_mask"]
        y_pred_80 = dec_80["predictions"][acc_80]
        y_true_80 = self.y_true[acc_80]
        err_80 = (y_pred_80 != y_true_80).sum() / len(y_true_80)

        self.assertLessEqual(
            err_80, err_100,
            f"Selective error ({err_80:.4f}) exceeded unselective error ({err_100:.4f})!"
        )

    def test_cost_accounting_exactness(self):
        """Verifies mathematical exactness of total operational loss."""
        tau = 0.30
        decision = self.selector.predict_selective(self.probs, self.composite_risk, threshold=tau)
        acc = decision["accepted_mask"]
        deferred = decision["deferred_mask"]

        y_acc_pred = decision["predictions"][acc]
        y_acc_true = self.y_true[acc]

        fn = int(((y_acc_true == 1) & (y_acc_pred == 0)).sum())
        fp = int(((y_acc_true == 0) & (y_acc_pred == 1)).sum())
        n_def = int(deferred.sum())

        expected_loss = 100.0 * fn + 5.0 * fp + 2.0 * n_def
        actual_loss = self.selector.compute_economic_loss(
            self.y_true, self.probs, self.composite_risk, threshold=tau
        )

        self.assertAlmostEqual(actual_loss, expected_loss, places=4)

    def test_aurc_bounds_and_validity(self):
        """Verifies that AURC is bounded within [0, 1] and strictly positive."""
        rc_curve = self.selector.compute_risk_coverage_curve(
            self.y_true, self.probs, self.composite_risk, n_points=50
        )
        aurc = rc_curve["aurc"]
        self.assertGreater(aurc, 0.0, "AURC must be greater than 0")
        self.assertLess(aurc, 1.0, "AURC must be less than 1.0")
        self.assertEqual(len(rc_curve["coverages"]), 50)
        self.assertEqual(len(rc_curve["error_rates"]), 50)

    def test_extreme_boundary_thresholds(self):
        """Handles edge case thresholds (tau=0.0 and tau=1.0) cleanly."""
        # Minimum threshold (abstain on all)
        dec_zero = self.selector.predict_selective(self.probs, self.composite_risk, threshold=-0.1)
        self.assertEqual(dec_zero["coverage"], 0.0)
        self.assertEqual(dec_zero["n_accepted"], 0)
        self.assertEqual(dec_zero["n_deferred"], self.n_samples)
        self.assertTrue((dec_zero["predictions"] == -1).all())

        # Maximum threshold (accept all)
        dec_full = self.selector.predict_selective(self.probs, self.composite_risk, threshold=1.1)
        self.assertEqual(dec_full["coverage"], 1.0)
        self.assertEqual(dec_full["n_accepted"], self.n_samples)
        self.assertEqual(dec_full["n_deferred"], 0)
        self.assertFalse((dec_full["predictions"] == -1).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
