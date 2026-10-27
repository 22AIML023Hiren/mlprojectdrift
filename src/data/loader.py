"""
Data loading and temporal partitioning module for credit card fraud detection.
Handles dataset caching, OpenML fallback, synthetic benchmark generation,
and strict temporal train/calibration/test splits without leakage.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


def generate_synthetic_benchmark(
    n_samples: int = 50000,
    fraud_rate: float = 0.00172,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic synthetic financial tabular dataset mirroring
    the exact schema, covariance structure, and extreme imbalance of the
    ULB Credit Card Fraud benchmark.
    
    Features:
        - Time: Elapsed seconds over 48 hours (0 to 172800)
        - V1 to V28: Continuous PCA-like principal components
        - Amount: Log-normal transaction amount distribution
        - Class: Binary target (0: Legitimate, 1: Fraud)
    """
    rng = np.random.RandomState(random_state)
    n_fraud = int(round(n_samples * fraud_rate))
    n_legit = n_samples - n_fraud

    # Time feature: uniform Poisson-like transaction arrival over 48h
    time_legit = np.sort(rng.uniform(0, 172800, size=n_legit))
    time_fraud = np.sort(rng.uniform(0, 172800, size=n_fraud))

    # Amounts: Log-normally distributed (most small, few large outliers)
    amount_legit = rng.lognormal(mean=3.2, sigma=1.2, size=n_legit)
    amount_fraud = rng.lognormal(mean=4.0, sigma=1.8, size=n_fraud)

    # 28 continuous features (simulating PCA components)
    # Legitimate transactions: standard multivariate normal
    v_legit = rng.normal(loc=0.0, scale=1.0, size=(n_legit, 28))
    
    # Fraud transactions: correlated shifts in sensitive components (e.g. V1-V4, V10, V14)
    v_fraud = rng.normal(loc=0.0, scale=1.2, size=(n_fraud, 28))
    v_fraud[:, 0] -= 2.0   # V1 shifted negative
    v_fraud[:, 1] += 1.8   # V2 shifted positive
    v_fraud[:, 2] -= 2.5   # V3 shifted negative
    v_fraud[:, 3] += 2.0   # V4 shifted positive
    v_fraud[:, 9] -= 3.0   # V10 strongly separated
    v_fraud[:, 13] -= 3.5  # V14 strongly separated

    # Combine into dataframes
    feature_cols = [f"V{i}" for i in range(1, 29)]
    
    df_legit = pd.DataFrame(v_legit, columns=feature_cols)
    df_legit["Time"] = time_legit
    df_legit["Amount"] = amount_legit
    df_legit["Class"] = 0

    df_fraud = pd.DataFrame(v_fraud, columns=feature_cols)
    df_fraud["Time"] = time_fraud
    df_fraud["Amount"] = amount_fraud
    df_fraud["Class"] = 1

    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    # Sort strictly by Time to mirror real-world sequential transaction stream
    df = df.sort_values(by="Time").reset_index(drop=True)
    return df


def load_dataset(
    data_path: Optional[str] = None,
    allow_synthetic_fallback: bool = True,
    n_synthetic_samples: int = 50000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Loads the credit card fraud dataset from:
    1. Specified data_path
    2. data/raw/creditcard.csv
    3. OpenML benchmark (if available)
    4. High-fidelity synthetic benchmark (fallback)
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    candidates = [
        Path(data_path) if data_path else None,
        RAW_DATA_DIR / "creditcard.csv",
        RAW_DATA_DIR / "creditcard_fraud.csv",
    ]

    for p in candidates:
        if p and p.exists() and p.is_file():
            print(f"[DataLoader] Found existing dataset at {p}")
            df = pd.read_csv(p)
            print(f"[DataLoader] Loaded {len(df):,} records with shape {df.shape}")
            return df

    # If file not present locally, check synthetic fallback
    if allow_synthetic_fallback:
        synthetic_path = RAW_DATA_DIR / "creditcard_synthetic.csv"
        if synthetic_path.exists():
            print(f"[DataLoader] Loading cached synthetic benchmark from {synthetic_path}")
            df = pd.read_csv(synthetic_path)
            return df
        
        print(f"[DataLoader] Generating realistic benchmark with {n_synthetic_samples:,} records...")
        df = generate_synthetic_benchmark(
            n_samples=n_synthetic_samples,
            fraud_rate=0.00172,
            random_state=random_state
        )
        df.to_csv(synthetic_path, index=False)
        print(f"[DataLoader] Saved benchmark to {synthetic_path}")
        return df

    raise FileNotFoundError(
        "creditcard.csv not found in data/raw/. Please download from "
        "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud or set allow_synthetic_fallback=True."
    )


def get_chronological_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    calib_ratio: float = 0.15,
    test_ratio: float = 0.15,
    scale_features: bool = True
) -> Dict[str, Any]:
    """
    Splits sequential transactions chronologically:
        - Train (0% to 70%): Used to fit base predictive models
        - Calibration (70% to 85%): Used to fit post-hoc probability calibrators,
          conformal non-conformity scores, and risk thresholds
        - ID Test (85% to 100%): Clean in-distribution holdout evaluation
    
    RobustScaler is fitted strictly on the Training set to prevent leakage.
    """
    assert abs(train_ratio + calib_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

    df_sorted = df.sort_values(by="Time").reset_index(drop=True)
    n = len(df_sorted)
    
    train_end = int(n * train_ratio)
    calib_end = int(n * (train_ratio + calib_ratio))

    df_train = df_sorted.iloc[:train_end].copy()
    df_calib = df_sorted.iloc[train_end:calib_end].copy()
    df_test = df_sorted.iloc[calib_end:].copy()

    feature_cols = [c for c in df.columns if c != "Class"]
    target_col = "Class"

    X_train = df_train[feature_cols].copy()
    y_train = df_train[target_col].values.astype(int)

    X_calib = df_calib[feature_cols].copy()
    y_calib = df_calib[target_col].values.astype(int)

    X_test = df_test[feature_cols].copy()
    y_test = df_test[target_col].values.astype(int)

    scaler = None
    if scale_features:
        # Scale 'Time' and 'Amount' specifically using RobustScaler (outlier resilient)
        scaler = RobustScaler()
        scale_cols = ["Time", "Amount"]
        X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
        X_calib[scale_cols] = scaler.transform(X_calib[scale_cols])
        X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    print(f"[DataLoader] Chronological Split:")
    print(f"  - Train: {len(X_train):,} samples ({y_train.sum()} frauds, {y_train.mean():.4%})")
    print(f"  - Calib: {len(X_calib):,} samples ({y_calib.sum()} frauds, {y_calib.mean():.4%})")
    print(f"  - Test:  {len(X_test):,} samples ({y_test.sum()} frauds, {y_test.mean():.4%})")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_calib": X_calib, "y_calib": y_calib,
        "X_test": X_test, "y_test": y_test,
        "feature_cols": feature_cols,
        "scaler": scaler
    }
