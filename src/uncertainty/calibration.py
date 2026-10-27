"""
Post-Hoc Probability Calibration module for aligning model confidence scores
with true empirical likelihoods under class imbalance and distribution shifts.
Supports: Isotonic Regression, Platt Scaling (Sigmoid), and Temperature Scaling.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Computes Expected Calibration Error (ECE):
    ECE = sum_k (|B_k| / N) * |acc(B_k) - conf(B_k)|
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_limits[i]
        bin_upper = bin_limits[i + 1]
        
        # Select samples within bin
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_size = in_bin.sum()
        if bin_size > 0:
            bin_acc = y_true[in_bin].mean()
            bin_conf = y_prob[in_bin].mean()
            ece += (bin_size / n_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes mean squared error between predicted probabilities and binary ground truth."""
    return float(np.mean((y_prob - y_true) ** 2))


class ProbabilityCalibrator:
    """
    Fits and applies probability calibration functions on held-out calibration data.
    """
    def __init__(self, method: str = "isotonic"):
        self.method = method.lower()
        self.calibrator = None

    def fit(self, val_probs: np.ndarray, y_val: np.ndarray) -> "ProbabilityCalibrator":
        """
        Fits calibration model mapping raw probabilities to empirical probabilities.
        
        Args:
            val_probs: Uncalibrated P(Y=1) predicted by model on calibration split [N]
            y_val: Ground truth binary labels [N]
        """
        val_probs = np.clip(np.asarray(val_probs), 1e-7, 1.0 - 1e-7)
        y_val = np.asarray(y_val)

        if self.method == "isotonic":
            # Isotonic regression: non-parametric piece-wise constant monotonic fit
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(val_probs, y_val)
        elif self.method in ["platt", "sigmoid"]:
            # Platt scaling: 1D logistic regression over logit
            logits = np.log(val_probs / (1.0 - val_probs)).reshape(-1, 1)
            self.calibrator = LogisticRegression(solver="lbfgs", max_iter=500)
            self.calibrator.fit(logits, y_val)
        else:
            raise ValueError(f"Unknown calibration method '{self.method}'. Use 'isotonic' or 'platt'.")

        return self

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        """
        Transforms raw probabilities into calibrated probabilities.
        """
        raw_probs = np.clip(np.asarray(raw_probs), 1e-7, 1.0 - 1e-7)
        if self.method == "isotonic":
            calibrated = self.calibrator.predict(raw_probs)
        elif self.method in ["platt", "sigmoid"]:
            logits = np.log(raw_probs / (1.0 - raw_probs)).reshape(-1, 1)
            calibrated = self.calibrator.predict_proba(logits)[:, 1]
        
        return np.clip(calibrated, 0.0, 1.0)
