"""
Main Execution Script for Risk-Aware Machine Learning Under Distribution Shift.
Executes the full experimental pipeline:
1. Data loading and temporal split
2. Controlled distribution shift testbed generation
3. Training classifiers (Logistic Regression, Random Forest, LightGBM, Tabular MLP)
4. Uncertainty quantification (Isotonic calibration, Ensemble variance, Conformal sets)
5. Risk-gated selective classification
6. Comprehensive evaluation, shift resilience benchmarking, and figure generation.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data import load_dataset, get_chronological_splits, ShiftGenerator
from src.models import get_model
from src.uncertainty import (
    ProbabilityCalibrator,
    compute_expected_calibration_error,
    InductiveConformalClassifier
)
from src.decision import RiskScorer, SelectiveClassifier
from src.evaluation import (
    compute_comprehensive_metrics,
    evaluate_shift_resilience,
    plot_risk_coverage_frontier,
    plot_calibration_diagram,
    plot_shift_resilience_comparison
)


def run_experiment(
    n_samples: int = 40000,
    random_state: int = 42,
    target_coverage: float = 0.85
):
    print("=" * 80)
    print("  RISK-AWARE MACHINE LEARNING UNDER DISTRIBUTION SHIFT")
    print("  End-to-End Experimental Pipeline & Benchmark Suite")
    print("=" * 80)

    start_time = time.time()
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. LOAD DATA & TEMPORAL PARTITIONING
    # -------------------------------------------------------------
    print("\n[Step 1/6] Ingesting and Partitioning Data...")
    df = load_dataset(n_synthetic_samples=n_samples, random_state=random_state)
    splits = get_chronological_splits(df, train_ratio=0.70, calib_ratio=0.15, test_ratio=0.15)

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_calib, y_calib = splits["X_calib"], splits["y_calib"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    # -------------------------------------------------------------
    # 2. GENERATE CONTROLLED DISTRIBUTION SHIFTS
    # -------------------------------------------------------------
    print("\n[Step 2/6] Generating Controlled Distribution Shift Testbeds...")
    shift_gen = ShiftGenerator(random_state=random_state)
    testbeds = shift_gen.generate_all_shift_testbeds(X_test, y_test)
    for name, (X_s, y_s) in testbeds.items():
        print(f"  - Testbed '{name}': {len(X_s):,} samples | Fraud Rate: {y_s.mean():.3%}")

    # -------------------------------------------------------------
    # 3. TRAIN BASE CLASSIFIERS ON SOURCE DISTRIBUTION
    # -------------------------------------------------------------
    print("\n[Step 3/6] Training Base Classifiers on Source Distribution...")
    models = {
        "Random_Forest": get_model("random_forest", n_estimators=80, max_depth=10, random_state=random_state),
        "LightGBM": get_model("lightgbm", n_estimators=100, learning_rate=0.05, random_state=random_state),
        "Logistic_Regression": get_model("logistic_regression", C=1.0, random_state=random_state),
        "Tabular_MLP": get_model("tabular_mlp", hidden_dim=64, epochs=10, random_state=random_state)
    }

    trained_models = {}
    for name, model in models.items():
        t0 = time.time()
        print(f"  - Fitting {name}...", end="", flush=True)
        model.fit(X_train, y_train)
        print(f" done ({time.time() - t0:.2f}s)")
        trained_models[name] = model

    # Select primary benchmark model (Random Forest / LightGBM)
    primary_model = trained_models["Random_Forest"]

    # -------------------------------------------------------------
    # 4. FIT UNCERTAINTY & CALIBRATION ENGINES ON CALIBRATION SPLIT
    # -------------------------------------------------------------
    print("\n[Step 4/6] Calibrating Probabilities & Conformal Sets on Calibration Split...")
    calib_raw_probs = primary_model.predict_proba(X_calib)[:, 1]

    # Post-hoc Probability Calibrator (Isotonic Regression)
    calibrator = ProbabilityCalibrator(method="isotonic")
    calibrator.fit(calib_raw_probs, y_calib)

    ece_before = compute_expected_calibration_error(y_calib, calib_raw_probs)
    ece_after = compute_expected_calibration_error(y_calib, calibrator.predict(calib_raw_probs))
    print(f"  - Expected Calibration Error (ECE) on Calib Split: {ece_before:.4f} -> {ece_after:.4f} ({(1 - ece_after/max(ece_before, 1e-5))*100:.1f}% improvement)")

    # Inductive Conformal Prediction
    calib_probs_2d = np.column_stack([1.0 - calib_raw_probs, calib_raw_probs])
    conformal = InductiveConformalClassifier(alpha=0.05)
    conformal.fit(calib_probs_2d, y_calib)

    test_raw_probs = primary_model.predict_proba(X_test)[:, 1]
    test_probs_2d = np.column_stack([1.0 - test_raw_probs, test_raw_probs])
    conf_stats = conformal.evaluate_coverage_and_efficiency(test_probs_2d, y_test)
    print(f"  - Conformal Prediction Set Coverage on Test Set: {conf_stats['empirical_coverage']:.1%} (Target: 95.0%)")
    print(f"  - Ambiguous / Double-Label Set Ratio: {conf_stats['double_label_ratio']:.2%}")

    # Feature-space OOD detector (Isolation Forest) for comparison
    risk_scorer = RiskScorer(fit_ood_detector=True, random_state=random_state)
    risk_scorer.fit(X_train)

    # -------------------------------------------------------------
    # 5. RISK-AWARE SELECTIVE CLASSIFICATION BENCHMARK
    # -------------------------------------------------------------
    print(f"\n[Step 5/6] Benchmarking Risk-Aware Selective Classification across Shifts (Target Coverage: {target_coverage:.0%})...")
    benchmark_df = evaluate_shift_resilience(
        model=primary_model,
        calibrator=calibrator,
        risk_scorer=risk_scorer,
        testbeds=testbeds,
        target_coverage=target_coverage
    )

    # Save benchmark table
    csv_out = results_dir / "shift_benchmark_results.csv"
    benchmark_df.to_csv(csv_out, index=False)
    print(f"  - Saved benchmark results to {csv_out}")

    # Display results table
    print("\n" + "=" * 115)
    display_cols = [
        "Testbed", "Fraud_Rate", "Vanilla_AUPRC", "Calibrated_ECE",
        "Vanilla_Error", "Selective_Error_at_85Cov", "Vanilla_Missed_Fraud",
        "Selective_Missed_Fraud", "Loss_Reduction_%", "AURC",
        "Corr_Risk_to_Error", "Corr_OOD_to_Error"
    ]
    print(benchmark_df[display_cols].to_string(index=False))
    print("=" * 115)

    # -------------------------------------------------------------
    # 6. GENERATE VISUALIZATION FIGURES
    # -------------------------------------------------------------
    print("\n[Step 6/6] Generating Publication-Quality Figures...")
    
    # Severe shift testbed for risk-coverage curve
    X_cov, y_cov = testbeds["Covariate_Severe"]
    probs_cov = calibrator.predict(primary_model.predict_proba(X_cov)[:, 1])
    tree_probs_cov = primary_model.get_individual_tree_predictions(X_cov)
    var_cov = np.var(tree_probs_cov, axis=0)
    risk_cov = risk_scorer.compute_composite_risk(probs_cov, ensemble_variance=var_cov)

    selector = SelectiveClassifier()
    rc_risk_aware = selector.compute_risk_coverage_curve(y_cov, probs_cov, risk_cov)
    # Random / uncalibrated baseline curve using inverted raw distance
    rc_vanilla = selector.compute_risk_coverage_curve(y_cov, probs_cov, 1.0 - np.abs(probs_cov - 0.5))

    plot_risk_coverage_frontier(rc_vanilla, rc_risk_aware)
    plot_calibration_diagram(y_test, primary_model.predict_proba(X_test)[:, 1], calibrator.predict(primary_model.predict_proba(X_test)[:, 1]))
    plot_shift_resilience_comparison(benchmark_df)

    elapsed = time.time() - start_time
    print(f"\n[Completed] Full experimental pipeline executed successfully in {elapsed:.1f} seconds!")
    print(f"Results and plots are available in '{results_dir.resolve()}'")


if __name__ == "__main__":
    run_experiment()
