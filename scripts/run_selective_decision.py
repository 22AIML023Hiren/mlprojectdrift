"""
Phase 5 Execution Script: Risk-Aware Selective Classification & Cost Optimization.
Evaluates selective decision policies across coverage levels and shift testbeds.
Computes AURC, economic loss mitigation, and disentanglement correlations.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dataset, get_chronological_splits
from src.uncertainty.ensemble_var import EnsembleUncertaintyEstimator
from src.decision.risk_scorer import RiskScorer
from src.decision.selector import SelectiveClassifier


def run_selective_evaluation():
    print("=" * 80)
    print(" PHASE 5: RISK-AWARE SELECTIVE CLASSIFICATION BENCHMARK ")
    print("=" * 80)

    # 1. Load trained model & calibrator
    model_path = PROJECT_ROOT / "models" / "random_forest.joblib"
    calibrator_path = PROJECT_ROOT / "models" / "calibrators" / "isotonic_calibrator.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Train models first.")
    if not calibrator_path.exists():
        raise FileNotFoundError(f"Calibrator not found at {calibrator_path}. Calibrate models first.")

    model = joblib.load(model_path)
    calibrator = joblib.load(calibrator_path)
    print(f"[Model] Loaded Random Forest from {model_path.name}")
    print(f"[Calibrator] Loaded Isotonic Calibrator from {calibrator_path.name}")

    # 2. Load In-Distribution Training data to fit RiskScorer OOD detector
    splits_path = PROJECT_ROOT / "data" / "processed" / "splits.joblib"
    if splits_path.exists():
        splits = joblib.load(splits_path)
        X_train = splits["X_train"]
    else:
        df_raw = load_dataset(random_state=42)
        splits = get_chronological_splits(df_raw, scale_features=True)
        X_train = splits["X_train"]

    risk_scorer = RiskScorer(fit_ood_detector=True, random_state=42)
    risk_scorer.fit(X_train)
    print("[RiskScorer] Fitted Isolation Forest OOD detector on X_train (D=30)")

    # 3. Setup SelectiveClassifier with economic cost matrix
    # FN = $100 (missed fraud), FP = $5 (wrongful decline), Defer = $2 (analyst review)
    selector = SelectiveClassifier(cost_fn=100.0, cost_fp=5.0, cost_defer=2.0)

    # 4. Ingest Shift Testbeds
    shifted_dir = PROJECT_ROOT / "data" / "shifted"
    manifest_path = shifted_dir / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    testbeds = manifest["testbeds"]
    target_coverages = [0.70, 0.80, 0.85, 0.90, 0.95, 1.00]
    benchmark_records = []
    shift_comparison_records = []

    print(f"\nEvaluating across {len(testbeds)} shift testbeds...")

    for testbed_name, meta in testbeds.items():
        csv_file = shifted_dir / meta["file"]
        df_shift = pd.read_csv(csv_file)
        feature_cols = [c for c in df_shift.columns if c != "Class"]
        X_test = df_shift[feature_cols]
        y_test = df_shift["Class"].values

        # Predictions & Uncertainties
        raw_probs = model.predict_proba(X_test)[:, 1]
        calib_probs = calibrator.predict(raw_probs)
        tree_preds = model.get_individual_tree_predictions(X_test)
        _, tree_var = EnsembleUncertaintyEstimator.compute_variance_from_tree_predictions(tree_preds)
        ood_scores = risk_scorer.compute_feature_ood_score(X_test)

        # Risk Scores
        r_conf_raw = risk_scorer.compute_confidence_risk(raw_probs)
        r_conf_calib = risk_scorer.compute_confidence_risk(calib_probs)
        r_composite = risk_scorer.compute_composite_risk(
            calibrated_probs=calib_probs,
            ensemble_variance=tree_var,
            w_conf=0.60,
            w_var=0.40
        )

        risk_methods = {
            "Raw_Confidence": r_conf_raw,
            "Calibrated_Confidence": r_conf_calib,
            "Feature_OOD": ood_scores,
            "Composite_Risk": r_composite
        }

        # Baseline loss at full automation (coverage = 1.00)
        baseline_loss = selector.compute_economic_loss(
            y_true=y_test,
            probs=calib_probs,
            risk_scores=r_composite,
            threshold=1.01  # Accept all
        )

        # Disentanglement: Rank correlation with ground-truth error
        y_pred_unselective = (calib_probs >= 0.5).astype(int)
        errors = (y_pred_unselective != y_test).astype(int)

        corr_comp, _ = spearmanr(r_composite, errors)
        corr_ood, _ = spearmanr(ood_scores, errors)
        corr_conf, _ = spearmanr(r_conf_calib, errors)

        # Evaluate across coverage levels
        for method_name, r_scores in risk_methods.items():
            rc_curve = selector.compute_risk_coverage_curve(
                y_true=y_test,
                probs=calib_probs,
                risk_scores=r_scores,
                n_points=50
            )
            aurc = rc_curve["aurc"]

            for target_cov in target_coverages:
                if target_cov == 1.00:
                    tau = 1.01
                else:
                    tau = selector.find_threshold_for_target_coverage(r_scores, target_cov)

                decision = selector.predict_selective(
                    probs=calib_probs,
                    risk_scores=r_scores,
                    threshold=tau
                )

                acc = decision["accepted_mask"]
                n_acc = decision["n_accepted"]
                n_def = decision["n_deferred"]
                actual_cov = decision["coverage"]

                if n_acc > 0:
                    y_acc_pred = decision["predictions"][acc]
                    y_acc_true = y_test[acc]
                    fn = int(((y_acc_true == 1) & (y_acc_pred == 0)).sum())
                    fp = int(((y_acc_true == 0) & (y_acc_pred == 1)).sum())
                    err_rate = float((y_acc_pred != y_acc_true).sum() / n_acc)
                else:
                    fn, fp, err_rate = 0, 0, 0.0

                loss = selector.compute_economic_loss(
                    y_true=y_test,
                    probs=calib_probs,
                    risk_scores=r_scores,
                    threshold=tau
                )
                loss_reduc = max(0.0, (baseline_loss - loss) / max(baseline_loss, 1e-7) * 100.0)

                benchmark_records.append({
                    "testbed": testbed_name,
                    "risk_method": method_name,
                    "target_coverage": target_cov,
                    "actual_coverage": round(actual_cov, 4),
                    "threshold_tau": round(tau, 5),
                    "accepted_samples": n_acc,
                    "deferred_samples": n_def,
                    "missed_fraud_fn": fn,
                    "false_alarm_fp": fp,
                    "accepted_error_rate": round(err_rate, 5),
                    "economic_loss": round(loss, 2),
                    "loss_reduction_pct": round(loss_reduc, 2),
                    "aurc": round(aurc, 5)
                })

        # Summary for 85% operational coverage target (Composite Risk)
        tau_85 = selector.find_threshold_for_target_coverage(r_composite, 0.85)
        loss_85 = selector.compute_economic_loss(y_test, calib_probs, r_composite, tau_85)
        loss_red_85 = max(0.0, (baseline_loss - loss_85) / max(baseline_loss, 1e-7) * 100.0)

        # Baseline missed fraud at 100% coverage
        fn_base = int(((y_test == 1) & (y_pred_unselective == 0)).sum())
        # Missed fraud at 85% coverage
        dec_85 = selector.predict_selective(calib_probs, r_composite, tau_85)
        acc_85 = dec_85["accepted_mask"]
        fn_85 = int(((y_test[acc_85] == 1) & (dec_85["predictions"][acc_85] == 0)).sum())

        shift_comparison_records.append({
            "testbed": testbed_name,
            "baseline_loss": round(baseline_loss, 2),
            "selective_loss_85": round(loss_85, 2),
            "loss_reduction_85_pct": round(loss_red_85, 2),
            "baseline_missed_fraud": fn_base,
            "selective_missed_fraud": fn_85,
            "spearman_decision_risk_error": round(corr_comp, 4),
            "spearman_ood_distance_error": round(corr_ood, 4),
            "spearman_conf_risk_error": round(corr_conf, 4)
        })

    # 5. Export benchmark CSVs
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df_bench = pd.DataFrame(benchmark_records)
    df_shift_comp = pd.DataFrame(shift_comparison_records)

    bench_csv = results_dir / "selective_decision_benchmark.csv"
    shift_csv = results_dir / "selective_shift_comparison.csv"

    df_bench.to_csv(bench_csv, index=False)
    df_shift_comp.to_csv(shift_csv, index=False)

    print(f"\n[Export] Saved full benchmark table to {bench_csv}")
    print(f"[Export] Saved shift comparison table to {shift_csv}")

    # 6. Display Executive Summary Table
    print("\n" + "=" * 90)
    print(" SELECTIVE CLASSIFICATION UNDER DISTRIBUTION SHIFT (85% Target Operational Coverage)")
    print("=" * 90)
    print(f"{'Shift Testbed':<26} | {'Base Loss':>9} | {'85% Loss':>9} | {'Savings':>8} | {'Base FN':>7} | {'85% FN':>6} | {'Corr(R,Err)':>11} | {'Corr(OOD,Err)':>13}")
    print("-" * 90)
    for r in shift_comparison_records:
        print(f"{r['testbed']:<26} | ${r['baseline_loss']:>8.2f} | ${r['selective_loss_85']:>8.2f} | {r['loss_reduction_85_pct']:>7.1f}% | {r['baseline_missed_fraud']:>7} | {r['selective_missed_fraud']:>6} | {r['spearman_decision_risk_error']:>11.4f} | {r['spearman_ood_distance_error']:>13.4f}")
    print("=" * 90)

    # 7. Print AURC Comparison Across Methods on Severe Covariate Shift
    severe_records = df_bench[(df_bench["testbed"] == "Covariate_Severe") & (df_bench["target_coverage"] == 0.85)]
    print("\nAURC Comparison on 'Covariate_Severe' Testbed:")
    for _, row in severe_records.iterrows():
        print(f"  - {row['risk_method']:<22}: AURC = {row['aurc']:.5f} | Loss @ 85% = ${row['economic_loss']:.2f}")

    print("\n[OK] Phase 5 selective classification evaluation successfully completed.")


if __name__ == "__main__":
    run_selective_evaluation()
