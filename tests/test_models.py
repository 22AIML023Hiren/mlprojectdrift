"""
Unit test suite for Phase 3: Baseline Classifiers Training.
Verifies model convergence, probability constraints, tree disagreement extraction,
Monte Carlo Dropout variance, and checkpoint serialization.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import generate_synthetic_benchmark, get_chronological_splits
from src.models import get_model, save_model, load_model
from src.evaluation.metrics import compute_comprehensive_metrics


class TestPhase3BaselineModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Train models once on a fast synthetic slice for unit testing."""
        cls.df = generate_synthetic_benchmark(n_samples=5000, fraud_rate=0.02, random_state=42)
        splits = get_chronological_splits(cls.df, scale_features=True)
        cls.X_train, cls.y_train = splits["X_train"], splits["y_train"]
        cls.X_test, cls.y_test = splits["X_test"], splits["y_test"]

        # Train models
        cls.rf = get_model("random_forest", n_estimators=30, max_depth=8, random_state=42)
        cls.rf.fit(cls.X_train, cls.y_train)

        cls.lr = get_model("logistic_regression", C=1.0, random_state=42)
        cls.lr.fit(cls.X_train, cls.y_train)

        cls.gbm = get_model("lightgbm", n_estimators=40, learning_rate=0.1, random_state=42)
        cls.gbm.fit(cls.X_train, cls.y_train)

        cls.mlp = get_model("tabular_mlp", hidden_dim=32, epochs=5, random_state=42)
        cls.mlp.fit(cls.X_train, cls.y_train)

    def test_probability_constraints(self):
        """Checks that predict_proba returns [N, 2] array summing to 1.0 with values in [0, 1]."""
        for name, model in [("RF", self.rf), ("LR", self.lr), ("GBM", self.gbm), ("MLP", self.mlp)]:
            probs = model.predict_proba(self.X_test)
            self.assertEqual(probs.shape, (len(self.X_test), 2), f"{name} output shape mismatch")
            self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0), f"{name} probability out of [0, 1]")
            self.assertTrue(np.allclose(probs.sum(axis=1), 1.0, atol=1e-5), f"{name} probabilities do not sum to 1")

    def test_random_forest_tree_variance(self):
        """Verifies tree-level prediction extraction and epistemic variance calculation."""
        tree_probs = self.rf.get_individual_tree_predictions(self.X_test)
        self.assertEqual(tree_probs.shape, (30, len(self.X_test)), "Tree predictions shape mismatch")

        # Tree variance should be non-negative and non-zero
        variance = np.var(tree_probs, axis=0)
        self.assertEqual(len(variance), len(self.X_test))
        self.assertTrue(np.all(variance >= 0.0))
        self.assertGreater(float(np.mean(variance)), 0.0, "Expected non-zero tree variance")

    def test_tabular_mlp_mc_dropout(self):
        """Verifies Monte Carlo Dropout uncertainty sampling on PyTorch Tabular MLP."""
        mean_probs, variance = self.mlp.mc_dropout_predict(self.X_test, n_samples=20)
        self.assertEqual(mean_probs.shape, (len(self.X_test), 2))
        self.assertEqual(len(variance), len(self.X_test))
        self.assertTrue(np.all(variance >= 0.0))
        self.assertGreater(float(np.mean(variance)), 0.0, "Expected non-zero MC-Dropout variance")

    def test_discriminative_thresholds(self):
        """Verifies all models achieve in-distribution AUROC >= 0.80 and AUPRC >= 0.40."""
        for name, model in [("RF", self.rf), ("GBM", self.gbm), ("LR", self.lr)]:
            probs = model.predict_proba(self.X_test)[:, 1]
            metrics = compute_comprehensive_metrics(self.y_test, probs)
            self.assertGreaterEqual(metrics["auroc"], 0.80, f"{name} AUROC below 0.80: {metrics['auroc']}")
            self.assertGreaterEqual(metrics["auprc"], 0.40, f"{name} AUPRC below 0.40: {metrics['auprc']}")

    def test_model_serialization(self):
        """Verifies that saving and reloading a model preserves identical predictions."""
        tmp_dir = PROJECT_ROOT / "models" / "tmp_test"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        rf_path = tmp_dir / "test_rf.joblib"

        save_model(self.rf, rf_path)
        reloaded_rf = load_model(rf_path)

        orig_probs = self.rf.predict_proba(self.X_test)
        reloaded_probs = reloaded_rf.predict_proba(self.X_test)
        self.assertTrue(np.allclose(orig_probs, reloaded_probs, atol=1e-5), "Reloaded model outputs differ!")

        # Clean up tmp file
        rf_path.unlink()
        tmp_dir.rmdir()


if __name__ == "__main__":
    unittest.main(verbosity=2)
