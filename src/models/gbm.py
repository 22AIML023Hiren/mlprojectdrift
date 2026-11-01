"""
LightGBM Gradient Boosted Decision Tree Classifier for tabular fraud detection.
Configured for high performance on imbalanced streams with leaf probability extraction.
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _lgb_available = True
except Exception as _lgb_import_error:
    lgb = None
    _lgb_available = False
    _lgb_import_error_msg = str(_lgb_import_error)


class LightGBMModel:
    """
    LightGBM model wrapper tailored for imbalanced tabular streams.
    """
    def __init__(
        self,
        n_estimators: int = 30,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.model = None

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        eval_set: Optional[list] = None
    ) -> "LightGBMModel":
        if not _lgb_available:
            raise RuntimeError(f"LightGBM import failed: {_lgb_import_error_msg}")

        # Convert y to a contiguous C-order float32 array.
        # LightGBM 4.7+ on Windows can raise an access violation when the
        # label array is not a proper contiguous C-array.
        y = np.ascontiguousarray(y, dtype=np.float32)
        # Ensure X is a plain numpy array to avoid pandas memory-view issues
        # with LightGBM 4.x on Windows.
        if hasattr(X, "values"):
            X = np.ascontiguousarray(X.values, dtype=np.float32)

        # Calculate dynamic positive weight if not specified
        if self.scale_pos_weight is None:
            n_neg = (y == 0).sum()
            n_pos = max((y == 1).sum(), 1)
            scale_pos = min(float(n_neg / n_pos), 50.0)  # Clip to avoid over-calibration distortion
        else:
            scale_pos = self.scale_pos_weight

        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            scale_pos_weight=scale_pos,
            random_state=self.random_state,
            n_jobs=1,          # n_jobs=-1 triggers an access violation in LightGBM 4.x on Windows
            verbose=-1
        )

        if eval_set:
            # Also sanitize eval labels and convert X arrays
            eval_set = [
                (
                    np.ascontiguousarray(xs.values if hasattr(xs, "values") else xs, dtype=np.float32),
                    np.ascontiguousarray(ys, dtype=np.float32)
                )
                for xs, ys in eval_set
            ]
            self.model.fit(
                X, y,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
            )
        else:
            self.model.fit(X, y)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities.
        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        return self.model.predict_proba(X)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def get_individual_tree_predictions(self, X: pd.DataFrame, n_checkpoints: int = 10) -> np.ndarray:
        """Return calibrated probability predictions at evenly-spaced boosting checkpoints.

        Background
        ----------
        LightGBM is a gradient-boosted model: each tree adds a raw-score increment
        (in log-odds space) rather than producing an independent probability estimate.
        Therefore there is no single "individual tree probability" analogous to a
        Random Forest tree vote.

        This method instead evaluates the *cumulative* ensemble at `n_checkpoints`
        evenly-spaced boosting iterations (e.g. iterations 10, 20, ..., n_trees for
        100 trees and 10 checkpoints).  Each checkpoint prediction is converted to a
        probability via the booster's built-in sigmoid transform (raw_score=False).
        Taking the variance of these checkpoint probabilities captures *training
        instability* — how much the ensemble's decision changes as more trees are
        added — which is a meaningful proxy for epistemic uncertainty under the
        boosting framework.

        Interpretation for downstream consumers
        ----------------------------------------
        • Values are in [0, 1] (sigmoid-transformed), so Var ∈ [0, 0.25]; the
          downstream `compute_epistemic_risk` normalisation divisor of 0.25 remains valid.
        • Variance is higher when early stopping would have been beneficial (the model
          continues to shift its predictions after reasonable convergence), and lower
          for well-converged, stable predictions.
        • Limitation: this is NOT equivalent to independent-tree disagreement (as in
          Random Forest). It measures boosting convergence stability, not true Bayesian
          epistemic uncertainty.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        n_checkpoints : int
            Number of evenly-spaced boosting iteration checkpoints to evaluate.
            Must be >= 2.  Defaults to 10.

        Returns
        -------
        np.ndarray of shape (n_checkpoints, n_samples)
            Predicted P(Y=1) at each checkpoint.  Variance across axis=0 gives
            the boosting-stability uncertainty score per sample.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted.")

        booster = self.model.booster_
        n_trees = booster.num_trees()

        # Ensure X is a plain numpy array for the booster's predict API
        if hasattr(X, "values"):
            X_arr = np.ascontiguousarray(X.values, dtype=np.float32)
        else:
            X_arr = np.ascontiguousarray(X, dtype=np.float32)

        n_checkpoints = max(2, min(n_checkpoints, n_trees))
        # Evenly-spaced iteration checkpoints from 1 to n_trees (inclusive)
        checkpoints = np.linspace(1, n_trees, n_checkpoints, dtype=int)
        # Remove duplicates that can appear when n_trees < n_checkpoints
        checkpoints = list(dict.fromkeys(checkpoints.tolist()))

        preds = []
        for iter_idx in checkpoints:
            # num_iteration=iter_idx: use first iter_idx trees (cumulative ensemble).
            # raw_score=False: LightGBM applies sigmoid internally -> output in [0, 1].
            pred_i = booster.predict(X_arr, num_iteration=iter_idx, raw_score=False)
            preds.append(pred_i)

        # Shape: (n_checkpoints, n_samples)
        return np.array(preds)
