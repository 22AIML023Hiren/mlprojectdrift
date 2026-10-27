"""
Risk Scoring module for quantifying prediction unreliability.
Combines calibrated decision confidence, ensemble epistemic disagreement,
and feature-space out-of-distribution (OOD) distance into composite risk metrics.
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class RiskScorer:
    """
    Computes normalized risk scores R(x) in [0, 1] representing the likelihood
    that a model's prediction is unreliable or incorrect under distribution shift.
    """
    def __init__(self, fit_ood_detector: bool = True, random_state: int = 42):
        self.fit_ood_detector = fit_ood_detector
        self.random_state = random_state
        self.ood_detector = None
        self.ood_score_min = None
        self.ood_score_max = None

    def fit(self, X_train: pd.DataFrame) -> "RiskScorer":
        """
        Fits baseline feature-space OOD detector (Isolation Forest) on in-distribution training data.
        Allows empirical disentanglement between OOD novelty and predictive risk.
        """
        if self.fit_ood_detector:
            self.ood_detector = IsolationForest(
                n_estimators=100,
                contamination=0.01,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.ood_detector.fit(X_train)
            raw_scores = -self.ood_detector.score_samples(X_train)
            self.ood_score_min = float(np.min(raw_scores))
            self.ood_score_max = float(np.max(raw_scores))

        return self

    def compute_confidence_risk(self, probs: np.ndarray) -> np.ndarray:
        """
        Computes aleatoric boundary risk: R_conf(x) = 1 - max(p, 1 - p).
        High risk when p is near 0.5 (maximum ambiguity); low risk when confident (near 0 or 1).
        Normalized to [0, 1] by multiplying by 2.
        """
        p = np.clip(probs, 0.0, 1.0)
        max_p = np.maximum(p, 1.0 - p)
        # Margin from complete certainty (0.5 -> 1.0, 1.0 -> 0.0)
        risk = 2.0 * (1.0 - max_p)
        return np.clip(risk, 0.0, 1.0)

    def compute_epistemic_risk(self, variance: np.ndarray) -> np.ndarray:
        """
        Normalizes inter-tree / ensemble variance into [0, 1] risk score.
        Max theoretical variance for Bernoulli trial is 0.25 (at p=0.5).
        """
        # Scale variance (max 0.25) to [0, 1]
        norm_var = np.clip(variance / 0.25, 0.0, 1.0)
        return norm_var

    def compute_feature_ood_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Computes feature-space anomaly score using Isolation Forest.
        Higher score indicates out-of-distribution distance P(X) != P_train(X).
        """
        if self.ood_detector is None:
            raise RuntimeError("OOD detector not fitted. Call fit(X_train) first.")

        raw = -self.ood_detector.score_samples(X)
        denom = max(self.ood_score_max - self.ood_score_min, 1e-7)
        norm_ood = np.clip((raw - self.ood_score_min) / denom, 0.0, 1.0)
        return norm_ood

    def compute_composite_risk(
        self,
        calibrated_probs: np.ndarray,
        ensemble_variance: Optional[np.ndarray] = None,
        w_conf: float = 0.60,
        w_var: float = 0.40
    ) -> np.ndarray:
        """
        Synthesizes confidence risk and epistemic variance into composite decision risk score:
        R(x) = w_conf * R_conf(x) + w_var * R_var(x)
        """
        r_conf = self.compute_confidence_risk(calibrated_probs)

        if ensemble_variance is not None:
            r_var = self.compute_epistemic_risk(ensemble_variance)
            composite = w_conf * r_conf + w_var * r_var
        else:
            composite = r_conf

        return np.clip(composite, 0.0, 1.0)
