"""
CLI script to fit, benchmark, and serialize post-hoc probability calibrators
and evaluate Inductive Conformal Prediction sets on clean and shifted testbeds.
"""

import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dataset, get_chronological_splits
from src.models import load_model
from src.uncertainty.calibration import (
    ProbabilityCalibrator,
    compute_expected_calibration_error,
    compute_brier_score
)
from src.uncertainty.conformal import InductiveConformalClassifier


def main():
    print("=" * 80)
    print("  UNCERTAINTY QUANTIFICATION & CALIBRATION ENGINE")
    print("=" * 80)

    calib_models_dir = PROJECT_ROOT / "models" / "calibrators"
    results_dir = PROJECT_ROOT / "results"
    calib_models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset & splits
    print("[1/4] Loading dataset and extracting calibration and test partitions...")
    df = load_dataset(n_synthetic_samples=40000, random_state=42)
    splits = get_chronological_splits(df, train_ratio=0.70, calib_ratio=0.15, test_ratio=0.15)

    X_calib, y_calib = splits["X_calib"], splits["y_calib"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    # 2. Load primary trained model (Random Forest)
    model_path = PROJECT_ROOT / "models" / "random_forest.joblib"
    if not model_path.exists():
        print(f"Error: {model_path} not found. Please run scripts/train_models.py first.")
        sys.exit(1)

    print(f"[2/4] Loading trained model from {model_path}...")
    model = load_model(model_path)

    raw_calib_probs = model.predict_proba(X_calib)[:, 1]
    raw_test_probs = model.predict_proba(X_test)[:, 1]

    # 3. Fit post-hoc calibrators on calibration split
    print("[3/4] Fitting Isotonic Regression and Platt Scaling calibrators...")
    calibrators = {
        "Isotonic": ProbabilityCalibrator(method="isotonic").fit(raw_calib_probs, y_calib),
        "Platt_Scaling": ProbabilityCalibrator(method="platt").fit(raw_calib_probs, y_calib)
    }

    # Save fitted calibrators
    for name, calib in calibrators.items():
        out_path = calib_models_dir / f"{name.lower()}_calibrator.joblib"
        joblib.dump(calib, out_path)
        print(f"  - Saved {name} calibrator to {out_path}")

    # Evaluate calibration quality on held-out test set
    calib_records = []
    # Baseline: Raw uncalibrated
    raw_ece = compute_expected_calibration_error(y_test, raw_test_probs)
    raw_brier = compute_brier_score(y_test, raw_test_probs)
    calib_records.append({
        "Method": "Raw_Uncalibrated",
        "Test_ECE": round(raw_ece, 5),
        "ECE_Reduction_%": 0.0,
        "Brier_Score": round(raw_brier, 6),
        "Mean_Confidence": round(float(np.mean(raw_test_probs)), 5),
        "Empirical_Fraud_Rate": round(float(np.mean(y_test)), 5)
    })

    for name, calib in calibrators.items():
        calib_probs = calib.predict(raw_test_probs)
        c_ece = compute_expected_calibration_error(y_test, calib_probs)
        c_brier = compute_brier_score(y_test, calib_probs)
        ece_reduction = (raw_ece - c_ece) / max(raw_ece, 1e-6) * 100.0

        calib_records.append({
            "Method": name,
            "Test_ECE": round(c_ece, 5),
            "ECE_Reduction_%": round(ece_reduction, 1),
            "Brier_Score": round(c_brier, 6),
            "Mean_Confidence": round(float(np.mean(calib_probs)), 5),
            "Empirical_Fraud_Rate": round(float(np.mean(y_test)), 5)
        })

    calib_df = pd.DataFrame(calib_records)
    calib_csv = results_dir / "calibration_comparison.csv"
    calib_df.to_csv(calib_csv, index=False)

    print("\nCalibration Benchmark on Clean In-Distribution Test Set:")
    print("=" * 85)
    print(calib_df.to_string(index=False))
    print("=" * 85)

    # 4. Evaluate Inductive Conformal Prediction across alpha levels
    print("\n[4/4] Evaluating Inductive Conformal Prediction Sets across alpha levels...")
    conformal_records = []
    alphas = [0.01, 0.05, 0.10, 0.15]

    calib_probs_2d = np.column_stack([1.0 - raw_calib_probs, raw_calib_probs])
    test_probs_2d = np.column_stack([1.0 - raw_test_probs, raw_test_probs])

    for alpha in alphas:
        cp = InductiveConformalClassifier(alpha=alpha)
        cp.fit(calib_probs_2d, y_calib)
        stats = cp.evaluate_coverage_and_efficiency(test_probs_2d, y_test)

        conformal_records.append({
            "Alpha": alpha,
            "Target_Coverage_%": f"{(1 - alpha)*100:.1f}%",
            "Empirical_Coverage_%": f"{stats['empirical_coverage']*100:.2f}%",
            "Coverage_Met": stats['empirical_coverage'] >= (1.0 - alpha - 0.01),
            "Single_Label_Ratio_%": f"{stats['single_label_ratio']*100:.2f}%",
            "Double_Label_Ambiguous_%": f"{stats['double_label_ratio']*100:.2f}%",
            "Empty_Set_Ratio_%": f"{stats['empty_set_ratio']*100:.2f}%",
            "Quantile_Threshold": round(cp.quantile_threshold, 5)
        })

    conformal_df = pd.DataFrame(conformal_records)
    conformal_csv = results_dir / "conformal_coverage_analysis.csv"
    conformal_df.to_csv(conformal_csv, index=False)

    print("\nInductive Conformal Prediction Coverage & Efficiency Analysis:")
    print("=" * 105)
    print(conformal_df.to_string(index=False))
    print("=" * 105)

    print(f"\n[Done] All uncertainty artifacts saved to '{results_dir}' and '{calib_models_dir}'")


if __name__ == "__main__":
    main()
