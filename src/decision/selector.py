"""
Selective Classification and Risk-Gated Decision Engine.
Implements abstention policies, Risk-Coverage trade-off curve generation,
Area Under Risk-Coverage (AURC) computation, and economic loss evaluation.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np


class SelectiveClassifier:
    """
    Selective classifier with abstention mechanism:
    - If Risk(x) <= tau: Automate prediction y_hat (Accepted)
    - If Risk(x) >  tau: Defer decision to human investigator (Abstained)
    """
    def __init__(
        self,
        cost_fn: float = 100.0,
        cost_fp: float = 5.0,
        cost_defer: float = 2.0
    ):
        self.cost_fn = cost_fn        # Cost of missed fraud (catastrophic false negative)
        self.cost_fp = cost_fp        # Cost of wrongful card freeze (false positive)
        self.cost_defer = cost_defer  # Cost of manual human analyst triage

    def predict_selective(
        self,
        probs: np.ndarray,
        risk_scores: np.ndarray,
        threshold: float = 0.5,
        decision_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Executes risk-gated decision rule.
        
        Args:
            probs: P(Y=1) predicted probabilities [N]
            risk_scores: Calculated risk scores R(x) in [0, 1] [N]
            threshold: Abstention threshold tau
            decision_threshold: Classification threshold for accepted predictions
        """
        probs = np.asarray(probs)
        risk_scores = np.asarray(risk_scores)

        accepted_mask = (risk_scores <= threshold)
        deferred_mask = ~accepted_mask

        n_total = len(probs)
        n_accepted = int(accepted_mask.sum())
        n_deferred = int(deferred_mask.sum())
        coverage = float(n_accepted / n_total) if n_total > 0 else 0.0

        predictions = np.full(n_total, fill_value=-1, dtype=int)
        if n_accepted > 0:
            predictions[accepted_mask] = (probs[accepted_mask] >= decision_threshold).astype(int)

        return {
            "predictions": predictions,
            "accepted_mask": accepted_mask,
            "deferred_mask": deferred_mask,
            "coverage": coverage,
            "n_accepted": n_accepted,
            "n_deferred": n_deferred,
            "threshold": threshold
        }

    def compute_economic_loss(
        self,
        y_true: np.ndarray,
        probs: np.ndarray,
        risk_scores: np.ndarray,
        threshold: float = 0.5,
        decision_threshold: float = 0.5
    ) -> float:
        """
        Computes total operational cost:
        Loss = C_fn * FN_accepted + C_fp * FP_accepted + C_defer * N_deferred
        """
        decision = self.predict_selective(probs, risk_scores, threshold, decision_threshold)
        accepted = decision["accepted_mask"]
        deferred = decision["deferred_mask"]

        y_true = np.asarray(y_true)
        if accepted.sum() > 0:
            y_pred_acc = decision["predictions"][accepted]
            y_true_acc = y_true[accepted]

            fn = int(((y_true_acc == 1) & (y_pred_acc == 0)).sum())
            fp = int(((y_true_acc == 0) & (y_pred_acc == 1)).sum())
        else:
            fn = 0
            fp = 0

        n_deferred = int(deferred.sum())
        total_loss = self.cost_fn * fn + self.cost_fp * fp + self.cost_defer * n_deferred
        return float(total_loss)

    def compute_risk_coverage_curve(
        self,
        y_true: np.ndarray,
        probs: np.ndarray,
        risk_scores: np.ndarray,
        n_points: int = 50,
        decision_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Constructs the Risk-Coverage trade-off curve across coverage levels c in [0.1, 1.0].
        Calculates Area Under Risk-Coverage (AURC).
        """
        y_true = np.asarray(y_true)
        probs = np.asarray(probs)
        risk_scores = np.asarray(risk_scores)
        n = len(y_true)

        # Sort instances by risk ascending (most confident to most risky)
        sorted_indices = np.argsort(risk_scores)
        coverages = []
        error_rates = []
        economic_losses = []
        fn_rates = []

        coverage_levels = np.linspace(0.10, 1.0, n_points)

        for cov in coverage_levels:
            k = max(int(round(cov * n)), 1)
            acc_idx = sorted_indices[:k]

            y_pred_k = (probs[acc_idx] >= decision_threshold).astype(int)
            y_true_k = y_true[acc_idx]

            errors = (y_pred_k != y_true_k).sum()
            err_rate = float(errors / k)
            error_rates.append(err_rate)
            coverages.append(float(cov))

            # Missed fraud among accepted
            fraud_in_acc = (y_true_k == 1).sum()
            missed_fraud = ((y_true_k == 1) & (y_pred_k == 0)).sum()
            fn_rate = float(missed_fraud / fraud_in_acc) if fraud_in_acc > 0 else 0.0
            fn_rates.append(fn_rate)

            # Economic loss
            fp = int(((y_true_k == 0) & (y_pred_k == 1)).sum())
            n_def = n - k
            loss = self.cost_fn * missed_fraud + self.cost_fp * fp + self.cost_defer * n_def
            economic_losses.append(float(loss))

        coverages = np.array(coverages)
        error_rates = np.array(error_rates)

        # Area Under Risk-Coverage Curve (trapezoidal integration)
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        aurc = float(trapz_fn(error_rates, coverages))

        return {
            "coverages": coverages,
            "error_rates": error_rates,
            "economic_losses": np.array(economic_losses),
            "fn_rates": np.array(fn_rates),
            "aurc": aurc
        }

    def find_threshold_for_target_coverage(
        self,
        risk_scores: np.ndarray,
        target_coverage: float = 0.85
    ) -> float:
        """
        Finds the risk threshold tau that yields target automation coverage.
        """
        tau = float(np.quantile(risk_scores, target_coverage))
        return tau
