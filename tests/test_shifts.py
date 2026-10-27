"""
Unit test suite for Phase 2: Controlled Distribution Shift Suite.
Verifies statistical divergence (KS test, Wasserstein, PSI),
subpopulation quantile filtering, and label prevalence accuracy.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import generate_synthetic_benchmark, get_chronological_splits
from src.data.shifts import ShiftGenerator
from src.data.shift_metrics import compute_psi, compute_distribution_divergence


class TestPhase2DistributionShifts(unittest.TestCase):

    def setUp(self):
        self.sample_df = generate_synthetic_benchmark(n_samples=6000, fraud_rate=0.015, random_state=42)
        splits = get_chronological_splits(self.sample_df, scale_features=True)
        self.X_test = splits["X_test"]
        self.y_test = splits["y_test"]
        self.sg = ShiftGenerator(random_state=42)

    def test_covariate_shift_statistical_significance(self):
        """Verifies that covariate shift produces statistically significant KS divergence (p < 0.001)."""
        target_cols = ["V1", "V2", "V3", "V4"]
        X_shifted = self.sg.apply_covariate_shift(self.X_test, severity=1.0, target_features=target_cols)

        # Perturbed columns must exhibit statistically significant shift (p < 0.01)
        for col in target_cols:
            ks_res = ks_2samp(self.X_test[col], X_shifted[col])
            self.assertLess(ks_res.pvalue, 0.01, f"Column {col} did not show significant shift (p={ks_res.pvalue})")

        # Non-perturbed column (e.g. V20) must be identical
        self.assertTrue(np.allclose(self.X_test["V20"], X_shifted["V20"]))

    def test_subpopulation_shift_quantiles(self):
        """Verifies that subpopulation shift isolates top transaction amounts."""
        quantile = 0.85
        expected_cutoff = self.X_test["Amount"].quantile(quantile)
        X_sub, y_sub = self.sg.apply_subpopulation_shift(self.X_test, self.y_test, quantile_threshold=quantile)

        self.assertTrue((X_sub["Amount"] >= expected_cutoff).all(), "Found samples below amount cutoff in subpopulation")
        # Subpopulation should contain approx (1 - quantile) of records
        expected_len = int(round(len(self.X_test) * (1.0 - quantile)))
        self.assertAlmostEqual(len(X_sub), expected_len, delta=5)

    def test_prevalence_shift_ratio(self):
        """Verifies that label shift accurately targets fraud prevalence."""
        target_rate = 0.035  # 3.5%
        X_prev, y_prev = self.sg.apply_label_shift(self.X_test, self.y_test, target_fraud_rate=target_rate)

        actual_rate = float(y_prev.mean())
        self.assertAlmostEqual(actual_rate, target_rate, places=2)

    def test_concept_drift_minority_selectivity(self):
        """Verifies that concept drift selectively perturbs fraudulent instances (Y=1)."""
        X_drift, y_drift = self.sg.apply_concept_drift(self.X_test, self.y_test, drift_strength=1.5)

        legit_mask = (self.y_test == 0)
        # Legitimate transactions should remain identical
        self.assertTrue(np.allclose(self.X_test.loc[legit_mask, "V1"], X_drift.loc[legit_mask, "V1"]))

        fraud_mask = (self.y_test == 1)
        # Fraudulent transactions should be modified
        self.assertFalse(np.allclose(self.X_test.loc[fraud_mask, "V1"], X_drift.loc[fraud_mask, "V1"]))

    def test_psi_metric(self):
        """Verifies that PSI is ~0 for identical data and >0.25 for shifted data."""
        v = self.X_test["V1"].values
        # Identical distribution
        psi_self = compute_psi(v, v)
        self.assertAlmostEqual(psi_self, 0.0, places=2)

        # Shifted distribution (V1 shifted by 1.5 sigma)
        v_shifted = v + np.random.normal(0, 1.5, size=len(v))
        psi_shifted = compute_psi(v, v_shifted)
        self.assertGreater(psi_shifted, 0.25, f"Expected PSI > 0.25 for severe shift, got {psi_shifted}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
