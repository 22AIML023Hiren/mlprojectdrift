"""
Evaluation metrics suite for risk-aware machine learning under distribution shift.
Computes discriminative metrics (AUPRC, AUROC), calibration metrics (ECE, Brier Score),
selective metrics (AURC, Effective Error under Coverage), and shift disentanglement correlation.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)
from src.uncertainty.calibration import compute_expected_calibration_error, compute_brier_score
from src.decision.selector import SelectiveClassifier


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    decision_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes discriminative and calibration metrics for imbalanced binary classification.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # Handle single-class edge cases in small subsets
    if len(np.unique(y_true)) < 2:
        return {
            "auroc": 0.5,
            "auprc": float(np.mean(y_true)),
            "ece": compute_expected_calibration_error(y_true, y_prob),
            "brier_score": compute_brier_score(y_true, y_prob),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }

    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=15)
    brier = compute_brier_score(y_true, y_prob)

    y_pred = (y_prob >= decision_threshold).astype(int)
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "auroc": auroc,
        "auprc": auprc,
        "ece": ece,
        "brier_score": brier,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


def evaluate_shift_resilience(
    model: Any,
    calibrator: Any,
    risk_scorer: Any,
    testbeds: Dict[str, Tuple[pd.DataFrame, np.ndarray]],
    target_coverage: float = 0.85
) -> pd.DataFrame:
    """
    Benchmarks model across all distribution shift testbeds comparing:
    1. Vanilla Non-Selective Classifier (100% automated, uncalibrated)
    2. Calibrated Classifier
    3. Risk-Aware Selective Classifier (Target 85% coverage, deferring high-risk cases)
    """
    selector = SelectiveClassifier()
    records = []

    for name, (X_test, y_test) in testbeds.items():
        # Raw model predictions
        raw_probs = model.predict_proba(X_test)[:, 1]
        raw_metrics = compute_comprehensive_metrics(y_test, raw_probs)

        # Calibrated predictions
        calib_probs = calibrator.predict(raw_probs)
        calib_metrics = compute_comprehensive_metrics(y_test, calib_probs)

        # Inter-tree / epistemic variance if supported
        variance = None
        if hasattr(model, "get_individual_tree_predictions"):
            tree_probs = model.get_individual_tree_predictions(X_test)
            variance = np.var(tree_probs, axis=0)

        # Compute composite risk score
        risk_scores = risk_scorer.compute_composite_risk(
            calibrated_probs=calib_probs,
            ensemble_variance=variance
        )

        # Find threshold to maintain target operational coverage (e.g. 85%)
        tau = selector.find_threshold_for_target_coverage(risk_scores, target_coverage=target_coverage)
        decision = selector.predict_selective(calib_probs, risk_scores, threshold=tau)

        # Errors on accepted subset
        acc_mask = decision["accepted_mask"]
        if acc_mask.sum() > 0:
            y_acc_true = y_test[acc_mask]
            y_acc_pred = decision["predictions"][acc_mask]
            selective_err = float((y_acc_true != y_acc_pred).mean())
            selective_fn = int(((y_acc_true == 1) & (y_acc_pred == 0)).sum())
        else:
            selective_err = 0.0
            selective_fn = 0

        # Vanilla baseline error (non-selective, 100% coverage)
        vanilla_pred = (raw_probs >= 0.5).astype(int)
        vanilla_err = float((y_test != vanilla_pred).mean())
        vanilla_fn = int(((y_test == 1) & (vanilla_pred == 0)).sum())

        # Economic loss comparison
        vanilla_loss = selector.compute_economic_loss(y_test, raw_probs, np.zeros_like(risk_scores), threshold=1.0)
        selective_loss = selector.compute_economic_loss(y_test, calib_probs, risk_scores, threshold=tau)
        loss_reduction = (vanilla_loss - selective_loss) / max(vanilla_loss, 1.0)

        # Risk-coverage curve & AURC
        rc_curve = selector.compute_risk_coverage_curve(y_test, calib_probs, risk_scores)

        # Disentanglement analysis: Correlation with actual misclassification
        misclassified = (vanilla_pred != y_test).astype(int)
        corr_risk_err, _ = spearmanr(risk_scores, misclassified)
        
        corr_ood_err = 0.0
        if risk_scorer.ood_detector is not None:
            ood_scores = risk_scorer.compute_feature_ood_score(X_test)
            corr_ood_err, _ = spearmanr(ood_scores, misclassified)

        records.append({
            "Testbed": name,
            "N_Samples": len(y_test),
            "Fraud_Rate": f"{y_test.mean():.3%}",
            "Vanilla_AUPRC": round(raw_metrics["auprc"], 4),
            "Calibrated_ECE": round(calib_metrics["ece"], 4),
            "Vanilla_ECE": round(raw_metrics["ece"], 4),
            "Vanilla_Error": round(vanilla_err, 4),
            "Selective_Error_at_85Cov": round(selective_err, 4),
            "Vanilla_Missed_Fraud": vanilla_fn,
            "Selective_Missed_Fraud": selective_fn,
            "Vanilla_Loss_$": round(vanilla_loss, 1),
            "Selective_Loss_$": round(selective_loss, 1),
            "Loss_Reduction_%": round(loss_reduction * 100, 1),
            "AURC": round(rc_curve["aurc"], 4),
            "Corr_Risk_to_Error": round(corr_risk_err, 3),
            "Corr_OOD_to_Error": round(corr_ood_err, 3)
        })

    return pd.DataFrame(records)
