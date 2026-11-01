"""
Focused tests for Bonus 3: Shift-Severity Sweep.
Verifies output files exist, CSV schema is correct, and correlation values are finite
and within the valid [-1, 1] range. Does NOT re-run the full experiment.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


class TestBonusSeveritySweep:
    """Validate output files produced by scripts/bonus_severity_sweep.py."""

    CSV_PATH = RESULTS_DIR / "bonus_severity_sweep.csv"
    FIG_PATH = FIGURES_DIR / "bonus_severity_sweep.png"
    MD_PATH  = RESULTS_DIR / "bonus_severity_sweep_analysis.md"

    REQUIRED_COLUMNS = [
        "Severity", "N_Samples", "N_Errors", "Fraud_Count",
        "Error_Rate_%", "Rho_Risk_vs_Error", "P_Risk",
        "Rho_OOD_vs_Error", "P_OOD", "Corr_Risk_Valid", "Corr_OOD_Valid",
    ]
    EXPECTED_SEVERITIES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

    def test_csv_exists(self):
        assert self.CSV_PATH.exists(), f"CSV output missing: {self.CSV_PATH}"

    def test_figure_exists(self):
        assert self.FIG_PATH.exists(), f"Figure missing: {self.FIG_PATH}"
        assert self.FIG_PATH.stat().st_size > 10_000, "Figure file suspiciously small"

    def test_analysis_markdown_exists(self):
        assert self.MD_PATH.exists(), f"Analysis markdown missing: {self.MD_PATH}"

    def test_csv_schema(self):
        df = pd.read_csv(self.CSV_PATH)
        for col in self.REQUIRED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_csv_severity_levels(self):
        df = pd.read_csv(self.CSV_PATH)
        assert list(df["Severity"].values) == self.EXPECTED_SEVERITIES, (
            f"Severity levels mismatch: {list(df['Severity'].values)}"
        )

    def test_csv_row_count(self):
        df = pd.read_csv(self.CSV_PATH)
        assert len(df) == len(self.EXPECTED_SEVERITIES), (
            f"Expected {len(self.EXPECTED_SEVERITIES)} rows, got {len(df)}"
        )

    def test_correlation_range(self):
        """All non-NaN correlations must lie in [-1, 1]."""
        df = pd.read_csv(self.CSV_PATH)
        for col in ["Rho_Risk_vs_Error", "Rho_OOD_vs_Error"]:
            valid = df[col].dropna()
            assert (valid >= -1.0).all() and (valid <= 1.0).all(), (
                f"Out-of-range correlation in {col}: {valid.values}"
            )

    def test_error_rate_increases_with_severity(self):
        """Error rate should be monotonically non-decreasing with severity."""
        df = pd.read_csv(self.CSV_PATH)
        rates = df["Error_Rate_%"].values
        for i in range(1, len(rates)):
            assert rates[i] >= rates[i - 1] - 1e-6, (
                f"Error rate decreased from severity {df['Severity'].iloc[i-1]} "
                f"to {df['Severity'].iloc[i]}: {rates[i-1]:.2f}% → {rates[i]:.2f}%"
            )

    def test_risk_correlation_increases_with_severity(self):
        """Risk correlation should generally increase with severity (not strictly enforced)."""
        df = pd.read_csv(self.CSV_PATH)
        valid = df[df["Corr_Risk_Valid"] == 1]
        if len(valid) >= 2:
            first = valid.iloc[0]["Rho_Risk_vs_Error"]
            last  = valid.iloc[-1]["Rho_Risk_vs_Error"]
            assert last >= first, (
                f"Risk correlation did not increase overall: {first:.4f} → {last:.4f}"
            )

    def test_spearman_helper_constant_input(self):
        """_spearman_safe should return NaN for constant score arrays."""
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from bonus_severity_sweep import _spearman_safe
        rho, pval = _spearman_safe(np.ones(100), np.random.randint(0, 2, 100))
        assert np.isnan(rho), "Expected NaN for constant scores"

    def test_spearman_helper_known_correlation(self):
        """_spearman_safe should return ρ ≈ 1 for perfectly correlated arrays."""
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from bonus_severity_sweep import _spearman_safe
        x = np.arange(50, dtype=float)
        rho, _ = _spearman_safe(x, x)
        assert abs(rho - 1.0) < 1e-6, f"Expected ρ≈1, got {rho}"
