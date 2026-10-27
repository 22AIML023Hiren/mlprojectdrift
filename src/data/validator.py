"""
Data schema validation and integrity verification module.
Ensures zero nulls, correct column datatypes, monotonic temporal ordering,
and valid positive class distributions.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd


EXPECTED_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


def validate_dataset_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates that a DataFrame strictly adheres to the credit card fraud schema.
    
    Checks:
    1. All 30 input features + Class exist.
    2. Zero missing / NaN values.
    3. 'Class' is binary containing {0, 1}.
    4. 'Time' is monotonically non-decreasing.
    5. 'Amount' has non-negative values.
    """
    issues: List[str] = []

    # 1. Column presence check
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    # 2. Null values check
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0].to_dict()
    if cols_with_nulls:
        issues.append(f"Columns with null values detected: {cols_with_nulls}")

    # 3. Target class check
    if "Class" in df.columns:
        unique_classes = set(df["Class"].unique())
        if not unique_classes.issubset({0, 1}):
            issues.append(f"Target 'Class' contains non-binary values: {unique_classes}")
        n_pos = int((df["Class"] == 1).sum())
        n_neg = int((df["Class"] == 0).sum())
        fraud_ratio = float(n_pos / len(df))
    else:
        n_pos = 0
        n_neg = 0
        fraud_ratio = 0.0

    # 4. Temporal ordering check
    if "Time" in df.columns:
        is_monotonic = bool(df["Time"].is_monotonic_increasing)
        if not is_monotonic:
            issues.append("Feature 'Time' is not monotonically non-decreasing.")
    else:
        is_monotonic = False

    # 5. Amount check
    if "Amount" in df.columns:
        negative_amounts = int((df["Amount"] < 0).sum())
        if negative_amounts > 0:
            issues.append(f"Detected {negative_amounts} negative transaction amounts.")

    is_valid = len(issues) == 0

    return {
        "is_valid": is_valid,
        "n_samples": len(df),
        "n_features": len(df.columns) - 1 if "Class" in df.columns else len(df.columns),
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "fraud_rate": fraud_ratio,
        "is_monotonic_time": is_monotonic,
        "issues": issues
    }
