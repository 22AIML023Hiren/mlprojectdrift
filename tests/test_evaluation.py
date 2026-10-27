"""
Unit test suite for Phase 6: Comprehensive Evaluation & Analysis.
Verifies metric mathematical bounds, dataframe structures, and publication plotting routines.
"""

import sys
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_comprehensive_metrics
from src.evaluation.plots import (
    plot_risk_coverage_frontier,
    plot_calibration_diagram,
    plot_disentanglement_analysis,
    plot_multi_model_aurc_comparison
)


class TestEvaluationSuite(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.y_true = np.array([0] * 95 + [1] * 5)
        self.y_prob = np.random.uniform(0.0, 1.0, size=100)
        # Add stronger signal for positives
        self.y_prob[self.y_true == 1] = np.random.uniform(0.6, 0.95, size=5)
        self.y_prob[self.y_true == 0] = np.random.uniform(0.05, 0.4, size=95)

    def test_comprehensive_metrics_bounds(self):
        """Checks mathematical bounds: AUROC in [0, 1], AUPRC in [0, 1], ECE in [0, 1], Brier in [0, 1]."""
        metrics = compute_comprehensive_metrics(self.y_true, self.y_prob)

        self.assertGreaterEqual(metrics["auroc"], 0.0)
        self.assertLessEqual(metrics["auroc"], 1.0)

        self.assertGreaterEqual(metrics["auprc"], 0.0)
        self.assertLessEqual(metrics["auprc"], 1.0)

        self.assertGreaterEqual(metrics["ece"], 0.0)
        self.assertLessEqual(metrics["ece"], 1.0)

        self.assertGreaterEqual(metrics["brier_score"], 0.0)
        self.assertLessEqual(metrics["brier_score"], 1.0)

    def test_single_class_edge_case(self):
        """Handles subsets with zero fraud gracefully without throwing exceptions."""
        all_zeros = np.zeros(50, dtype=int)
        probs = np.random.uniform(0.0, 0.1, size=50)
        metrics = compute_comprehensive_metrics(all_zeros, probs)
        self.assertEqual(metrics["auroc"], 0.5)
        self.assertEqual(metrics["auprc"], 0.0)

    def test_disentanglement_plot_generation(self):
        """Verifies that plot_disentanglement_analysis generates a valid image file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_png = Path(tmp_dir) / "test_disentangle.png"
            df = pd.DataFrame({
                "testbed": ["Clean_ID", "Shift_A", "Shift_B"],
                "spearman_decision_risk_error": [0.45, 0.35, 0.60],
                "spearman_ood_distance_error": [0.12, 0.08, 0.15]
            })
            plot_disentanglement_analysis(df, save_path=out_png)
            self.assertTrue(out_png.exists())
            self.assertGreater(out_png.stat().st_size, 1000)

    def test_multi_model_aurc_plot_generation(self):
        """Verifies that plot_multi_model_aurc_comparison generates a valid image file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_png = Path(tmp_dir) / "test_aurc.png"
            df = pd.DataFrame([
                {"model": "Random_Forest", "testbed": "Clean_ID", "aurc": 0.0001},
                {"model": "Random_Forest", "testbed": "Shift_A", "aurc": 0.0005},
                {"model": "LightGBM", "testbed": "Clean_ID", "aurc": 0.0002},
                {"model": "LightGBM", "testbed": "Shift_A", "aurc": 0.0008},
            ])
            plot_multi_model_aurc_comparison(df, save_path=out_png)
            self.assertTrue(out_png.exists())
            self.assertGreater(out_png.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
