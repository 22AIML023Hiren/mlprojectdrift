"""
Unit test suite for Phase 1: Data Pipeline & Temporal Partitioning.
Verifies schema validity, chronological non-leakage, split proportions,
and scaler isolation.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import generate_synthetic_benchmark, get_chronological_splits
from src.data.validator import validate_dataset_schema
from src.data.shifts import ShiftGenerator


@pytest.fixture
def sample_dataset():
    """Generates a reproducible 5,000-sample benchmark dataset."""
    return generate_synthetic_benchmark(n_samples=5000, fraud_rate=0.01, random_state=42)


def test_schema_validation(sample_dataset):
    """Checks that the generated dataset matches the 30-feature fraud benchmark schema."""
    report = validate_dataset_schema(sample_dataset)
    assert report["is_valid"] is True
    assert report["n_samples"] == 5000
    assert report["n_features"] == 30
    assert report["n_positives"] == 50
    assert report["is_monotonic_time"] is True
    assert len(report["issues"]) == 0


def test_chronological_ordering_and_leakage(sample_dataset):
    """
    Verifies strict chronological splitting without temporal overlap:
    max(Time_train) <= min(Time_calib) <= min(Time_test).
    """
    # Note: Before scaling, inspect raw time splits
    splits_unscaled = get_chronological_splits(sample_dataset, scale_features=False)
    
    time_train = splits_unscaled["X_train"]["Time"]
    time_calib = splits_unscaled["X_calib"]["Time"]
    time_test = splits_unscaled["X_test"]["Time"]

    assert time_train.max() <= time_calib.min(), "Leakage detected: Train time overlaps with Calib time"
    assert time_calib.max() <= time_test.min(), "Leakage detected: Calib time overlaps with Test time"


def test_split_proportions(sample_dataset):
    """Verifies 70% Train, 15% Calib, 15% Test partitioning math."""
    splits = get_chronological_splits(sample_dataset, train_ratio=0.70, calib_ratio=0.15, test_ratio=0.15)
    
    n_train = len(splits["X_train"])
    n_calib = len(splits["X_calib"])
    n_test = len(splits["X_test"])

    assert n_train == 3500  # 70% of 5000
    assert n_calib == 750   # 15% of 5000
    assert n_test == 750    # 15% of 5000
    assert n_train + n_calib + n_test == 5000


def test_scaler_isolation(sample_dataset):
    """
    Verifies RobustScaler is fitted strictly on the Training set.
    The scaler center_ and scale_ attributes must equal X_train statistics,
    NOT the combined or test statistics.
    """
    splits = get_chronological_splits(sample_dataset, scale_features=True)
    scaler = splits["scaler"]

    assert scaler is not None
    # Training Time median
    raw_time_train = sample_dataset.iloc[:3500]["Time"]
    expected_time_median = np.median(raw_time_train)

    # Scaler center_ for 'Time' (first scaled column)
    assert np.isclose(scaler.center_[0], expected_time_median, atol=1e-3)


def test_shift_generator_integrity(sample_dataset):
    """Verifies that shift testbeds maintain column count and dimensions."""
    splits = get_chronological_splits(sample_dataset, scale_features=True)
    X_test, y_test = splits["X_test"], splits["y_test"]

    sg = ShiftGenerator(random_state=42)
    testbeds = sg.generate_all_shift_testbeds(X_test, y_test)

    assert "Clean_ID" in testbeds
    assert "Covariate_Mild" in testbeds
    assert "Covariate_Severe" in testbeds
    assert "Subpopulation_HighAmount" in testbeds
    assert "Prevalence_Shift" in testbeds
    assert "Concept_Drift" in testbeds

    for name, (X_s, y_s) in testbeds.items():
        assert X_s.shape[1] == X_test.shape[1], f"Testbed {name} altered feature dimension!"
        assert len(X_s) == len(y_s), f"Testbed {name} has mismatched X and y lengths!"
