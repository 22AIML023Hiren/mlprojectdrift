"""
Phase 6 Execution Script: Comprehensive Multi-Model Benchmarking & Disentanglement Analysis.
Evaluates all 4 model families (Logistic Regression, Random Forest, LightGBM, Tabular MLP)
across in-distribution and shifted testbeds. Generates comparative tables and publication figures.
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
from src.models import load_model, TabularMLP
from src.decision.risk_scorer import RiskScorer
from src.decision.selector import SelectiveClassifier
from src.evaluation.metrics import compute_comprehensive_metrics
from src.evaluation.plots import (
    plot_risk_coverage_frontier,
    plot_calibration_diagram,
    plot_shift_resilience_comparison,
    plot_disentanglement_analysis,
    plot_multi_model_aurc_comparison
)


def run_comprehensive_evaluation():
    print("=" * 95)
    print("  PHASE 6: COMPREHENSIVE MULTI-MODEL EVALUATION & SHIFT DISENTANGLEMENT SUITE  ")
    print("=" * 95)

    models_dir = PROJECT_ROOT / "models"
    results_dir = PROJECT_ROOT / "results"
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load All Trained Models
    model_paths = {
        "Random_Forest": models_dir / "random_forest.joblib",
        "LightGBM": models_dir / "lightgbm.joblib",
        "Logistic_Regression": models_dir / "logistic_regression.joblib",
        "Tabular_MLP": models_dir / "tabular_mlp.pt"
    }

    models = {}
    for name, path in model_paths.items():
        if path.exists():
            models[name] = load_model(str(path))
            print(f"[ModelRegistry] Loaded {name} from {path.name}")
        else:
            print(f"[ModelRegistry] Warning: {path.name} not found, skipping {name}")

    if not models:
        raise RuntimeError("No model checkpoints found under models/. Train models first.")

    # 2. Load Probability Calibrator
    calibrator_path = models_dir / "calibrators" / "isotonic_calibrator.joblib"
    if not calibrator_path.exists():
        raise FileNotFoundError(f"Calibrator not found at {calibrator_path}")
    calibrator = joblib.load(calibrator_path)
    print(f"[Calibrator] Loaded Isotonic Calibrator from {calibrator_path.name}")

    # 3. Load In-Distribution Data & Fit OOD Detector
    splits_path = PROJECT_ROOT / "data" / "processed" / "splits.joblib"
    if splits_path.exists():
        splits = joblib.load(splits_path)
        X_train = splits["X_train"]
        X_test_id = splits["X_test"]
        y_test_id = splits["y_test"]
    else:
        df_raw = load_dataset(random_state=42)
        splits = get_chronological_splits(df_raw, scale_features=True)
        X_train = splits["X_train"]
        X_test_id = splits["X_test"]
        y_test_id = splits["y_test"]

    risk_scorer = RiskScorer(fit_ood_detector=True, random_state=42)
    risk_scorer.fit(X_train)
    print("[RiskScorer] Fitted Isolation Forest feature-space OOD detector on X_train (D=30)")

    selector = SelectiveClassifier(cost_fn=100.0, cost_fp=5.0, cost_defer=2.0)

    # 4. Ingest All Shift Testbeds
    shifted_dir = PROJECT_ROOT / "data" / "shifted"
    manifest_path = shifted_dir / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    testbeds = manifest["testbeds"]
    all_testbeds = {}
    for tb_name, meta in testbeds.items():
        csv_p = shifted_dir / meta["file"]
        df_tb = pd.read_csv(csv_p)
        f_cols = [c for c in df_tb.columns if c != "Class"]
        all_testbeds[tb_name] = (df_tb[f_cols], df_tb["Class"].values)

    print(f"\nBenchmarking {len(models)} models across {len(all_testbeds)} shift testbeds...")

    # Data collection records
    model_comparison_records = []
    disentanglement_records = []
    shift_resilience_records = []

    for tb_name, (X_test, y_test) in all_testbeds.items():
        ood_scores = risk_scorer.compute_feature_ood_score(X_test)

        for model_name, model in models.items():
            # Model inference
            raw_probs = model.predict_proba(X_test)[:, 1]
            calib_probs = calibrator.predict(raw_probs)

            # Extract variance depending on model architecture
            if hasattr(model, "get_individual_tree_predictions"):
                tree_preds = model.get_individual_tree_predictions(X_test)
                var = np.var(tree_preds, axis=0)
            elif hasattr(model, "mc_dropout_predict"):
                _, var = model.mc_dropout_predict(X_test, n_samples=15)
            else:
                var = np.zeros_like(raw_probs)

            # Composite Risk
            r_comp = risk_scorer.compute_composite_risk(
                calibrated_probs=calib_probs,
                ensemble_variance=var,
                w_conf=0.60,
                w_var=0.40
            )

            # Discriminative & Calibration Metrics
            raw_metrics = compute_comprehensive_metrics(y_test, raw_probs)
            calib_metrics = compute_comprehensive_metrics(y_test, calib_probs)

            # Risk-Coverage Curve & AURC
            rc_curve = selector.compute_risk_coverage_curve(y_test, calib_probs, r_comp, n_points=50)
            aurc = rc_curve["aurc"]

            # Selective Decision at 85% Target Coverage
            tau_85 = selector.find_threshold_for_target_coverage(r_comp, 0.85)
            dec_85 = selector.predict_selective(calib_probs, r_comp, threshold=tau_85)

            acc_85 = dec_85["accepted_mask"]
            n_acc_85 = dec_85["n_accepted"]
            n_def_85 = dec_85["n_deferred"]
            actual_cov = dec_85["coverage"]

            if n_acc_85 > 0:
                y_acc_pred = dec_85["predictions"][acc_85]
                y_acc_true = y_test[acc_85]
                fn_85 = int(((y_acc_true == 1) & (y_acc_pred == 0)).sum())
                fp_85 = int(((y_acc_true == 0) & (y_acc_pred == 1)).sum())
                err_rate_85 = float((y_acc_pred != y_acc_true).sum() / n_acc_85)
            else:
                fn_85, fp_85, err_rate_85 = 0, 0, 0.0

            loss_85 = selector.compute_economic_loss(y_test, calib_probs, r_comp, threshold=tau_85)
            loss_full = selector.compute_economic_loss(y_test, calib_probs, r_comp, threshold=1.01)
            loss_reduction = max(0.0, (loss_full - loss_85) / max(loss_full, 1e-7) * 100.0)

            # Disentanglement: Rank correlation with ground-truth error
            y_pred_unselective = (calib_probs >= 0.5).astype(int)
            errors = (y_pred_unselective != y_test).astype(int)
            corr_comp, _ = spearmanr(r_comp, errors)
            corr_ood, _ = spearmanr(ood_scores, errors)

            record = {
                "model": model_name,
                "testbed": tb_name,
                "n_samples": len(y_test),
                "fraud_count": int((y_test == 1).sum()),
                "auroc": round(raw_metrics["auroc"], 4),
                "auprc": round(raw_metrics["auprc"], 4),
                "raw_ece": round(raw_metrics["ece"], 4),
                "calib_ece": round(calib_metrics["ece"], 4),
                "brier_score": round(calib_metrics["brier_score"], 6),
                "aurc": round(aurc, 5),
                "loss_full_auto": round(loss_full, 2),
                "loss_selective_85": round(loss_85, 2),
                "loss_reduction_pct": round(loss_reduction, 2),
                "missed_fraud_full": int(((y_test == 1) & (y_pred_unselective == 0)).sum()),
                "missed_fraud_85": fn_85,
                "selective_error_85": round(err_rate_85, 5),
                "spearman_decision_risk_error": round(corr_comp, 4) if not np.isnan(corr_comp) else 0.0,
                "spearman_ood_distance_error": round(corr_ood, 4) if not np.isnan(corr_ood) else 0.0
            }
            model_comparison_records.append(record)

            # Record Random Forest for shift resilience summary
            if model_name == "Random_Forest":
                shift_resilience_records.append({
                    "Testbed": tb_name,
                    "N_Samples": len(y_test),
                    "Fraud_Rate": f"{y_test.mean():.3%}",
                    "Vanilla_AUPRC": round(raw_metrics["auprc"], 4),
                    "Calibrated_ECE": round(calib_metrics["ece"], 4),
                    "Vanilla_Loss_$": round(loss_full, 1),
                    "Selective_Loss_$": round(loss_85, 1),
                    "Loss_Reduction_%": round(loss_reduction, 1),
                    "AURC": round(aurc, 4),
                    "Corr_Risk_to_Error": round(corr_comp, 3) if not np.isnan(corr_comp) else 0.0,
                    "Corr_OOD_to_Error": round(corr_ood, 3) if not np.isnan(corr_ood) else 0.0
                })
                disentanglement_records.append({
                    "testbed": tb_name,
                    "spearman_decision_risk_error": round(corr_comp, 4) if not np.isnan(corr_comp) else 0.0,
                    "spearman_ood_distance_error": round(corr_ood, 4) if not np.isnan(corr_ood) else 0.0
                })

    # 5. Export Benchmark CSVs
    df_model_comp = pd.DataFrame(model_comparison_records)
    df_shift_resil = pd.DataFrame(shift_resilience_records)
    df_disentangle = pd.DataFrame(disentanglement_records)

    comp_csv = results_dir / "comprehensive_model_comparison.csv"
    resil_csv = results_dir / "comprehensive_shift_resilience.csv"
    disentangle_csv = results_dir / "disentanglement_analysis.csv"

    df_model_comp.to_csv(comp_csv, index=False)
    df_shift_resil.to_csv(resil_csv, index=False)
    df_disentangle.to_csv(disentangle_csv, index=False)

    print(f"\n[Export] Saved comprehensive model comparison to {comp_csv}")
    print(f"[Export] Saved shift resilience summary to {resil_csv}")
    print(f"[Export] Saved disentanglement analysis to {disentangle_csv}")

    # 6. Render All 5 Publication Figures
    print("\nGenerating publication-grade figures...")

    # Figure 1: Multi-Model AURC Comparison
    plot_multi_model_aurc_comparison(df_model_comp, save_path=figures_dir / "multi_model_aurc_comparison.png")

    # Figure 2: Shift vs Risk Disentanglement Bar Chart
    plot_disentanglement_analysis(df_disentangle, save_path=figures_dir / "disentanglement_correlation_comparison.png")

    # Figure 3: Shift Resilience Loss Comparison
    plot_shift_resilience_comparison(df_shift_resil, save_path=figures_dir / "shift_resilience_loss_comparison.png")

    # Figure 4: Reliability Diagram (RF on In-Distribution Test)
    rf_model = models.get("Random_Forest")
    if rf_model is not None:
        raw_test_p = rf_model.predict_proba(X_test_id)[:, 1]
        calib_test_p = calibrator.predict(raw_test_p)
        plot_calibration_diagram(y_test_id, raw_test_p, calib_test_p, save_path=figures_dir / "calibration_reliability_diagram.png")

    # Figure 5: Risk-Coverage Frontier (Severe Covariate Shift)
    X_sev, y_sev = all_testbeds["Covariate_Severe"]
    rf_p_sev = calibrator.predict(rf_model.predict_proba(X_sev)[:, 1])
    rf_tree_preds = rf_model.get_individual_tree_predictions(X_sev)
    rf_var_sev = np.var(rf_tree_preds, axis=0)
    rf_risk_sev = risk_scorer.compute_composite_risk(rf_p_sev, ensemble_variance=rf_var_sev)

    rc_risk_aware = selector.compute_risk_coverage_curve(y_sev, rf_p_sev, rf_risk_sev)
    rc_vanilla = selector.compute_risk_coverage_curve(y_sev, rf_p_sev, 1.0 - np.abs(rf_p_sev - 0.5))
    plot_risk_coverage_frontier(rc_vanilla, rc_risk_aware, save_path=figures_dir / "risk_coverage_curve.png")

    # 7. Print Executive Summary
    print("\n" + "=" * 105)
    print(" COMPREHENSIVE CROSS-MODEL BENCHMARK (Clean In-Distribution vs. Severe Covariate Shift)")
    print("=" * 105)
    display_sub = df_model_comp[df_model_comp["testbed"].isin(["Clean_ID", "Covariate_Severe"])]
    cols = ["model", "testbed", "auroc", "auprc", "raw_ece", "calib_ece", "aurc", "loss_full_auto", "loss_selective_85", "loss_reduction_pct"]
    print(display_sub[cols].to_string(index=False))
    print("=" * 105)

    print("\n[OK] Phase 6 Comprehensive Evaluation & Analysis completed successfully.")


if __name__ == "__main__":
    run_comprehensive_evaluation()
