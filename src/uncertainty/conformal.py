"""
Inductive Conformal Prediction for distribution-free uncertainty quantification.
Constructs valid prediction sets Gamma^alpha(X) with guaranteed coverage 1 - alpha.
"""

from typing import List, Tuple, Dict, Any
import numpy as np


class InductiveConformalClassifier:
    """
    Inductive Conformal Prediction for binary classification.
    Calculates non-conformity scores on calibration data to produce
    rigorous confidence sets with finite-sample coverage guarantees.
    """
    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Desired miscoverage rate (e.g. 0.05 implies 95% guaranteed coverage).
        """
        self.alpha = alpha
        self.quantile_threshold = None

    def fit(self, calib_probs: np.ndarray, y_calib: np.ndarray) -> "InductiveConformalClassifier":
        """
        Computes non-conformity scores on calibration set:
        s_i = 1 - P(Y = y_i | x_i)
        
        Args:
            calib_probs: Predicted probabilities [N, 2] on calibration set
            y_calib: True binary class labels [N]
        """
        calib_probs = np.asarray(calib_probs)
        y_calib = np.asarray(y_calib, dtype=int)
        n = len(y_calib)

        # Non-conformity score: 1 minus the probability assigned to the true label
        true_class_probs = calib_probs[np.arange(n), y_calib]
        scores = 1.0 - true_class_probs

        # Compute empirical (1 - alpha) quantile with finite-sample correction
        # q_val = np.ceil((n + 1) * (1 - alpha)) / n
        quantile_level = np.clip(np.ceil((n + 1) * (1.0 - self.alpha)) / n, 0.0, 1.0)
        self.quantile_threshold = float(np.quantile(scores, quantile_level, method="higher"))
        return self

    def predict_sets(self, test_probs: np.ndarray) -> List[List[int]]:
        """
        Constructs prediction set Gamma(x) for each test sample.
        Label y is included in Gamma(x) if: 1 - P(Y=y|x) <= quantile_threshold
        
        Returns:
            List of prediction sets, where each set is a subset of {0, 1}.
        """
        if self.quantile_threshold is None:
            raise RuntimeError("Conformal predictor must be fit on calibration data first.")

        test_probs = np.asarray(test_probs)
        prediction_sets = []

        for p in test_probs:
            s_0 = 1.0 - p[0]
            s_1 = 1.0 - p[1]
            c_set = []
            if s_0 <= self.quantile_threshold:
                c_set.append(0)
            if s_1 <= self.quantile_threshold:
                c_set.append(1)
            prediction_sets.append(c_set)

        return prediction_sets

    def evaluate_coverage_and_efficiency(
        self,
        test_probs: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluates empirical coverage (is true label in set?) and efficiency (set size).
        """
        pred_sets = self.predict_sets(test_probs)
        y_test = np.asarray(y_test, dtype=int)
        n = len(y_test)

        covered = sum(y_test[i] in pred_sets[i] for i in range(n))
        emp_coverage = covered / n

        set_sizes = [len(s) for s in pred_sets]
        mean_set_size = float(np.mean(set_sizes))
        single_label_ratio = sum(s == 1 for s in set_sizes) / n
        double_label_ratio = sum(s == 2 for s in set_sizes) / n
        empty_set_ratio = sum(s == 0 for s in set_sizes) / n

        return {
            "empirical_coverage": float(emp_coverage),
            "target_coverage": float(1.0 - self.alpha),
            "mean_set_size": mean_set_size,
            "single_label_ratio": float(single_label_ratio),
            "double_label_ratio": float(double_label_ratio),
            "empty_set_ratio": float(empty_set_ratio)
        }
