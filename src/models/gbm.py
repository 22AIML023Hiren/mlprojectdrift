"""
LightGBM Gradient Boosted Decision Tree Classifier for tabular fraud detection.
Configured for high performance on imbalanced streams with leaf probability extraction.
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import lightgbm as lgb


class LightGBMModel:
    """
    LightGBM model wrapper tailored for imbalanced tabular streams.
    """
    def __init__(
        self,
        n_estimators: int = 150,
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
        # Calculate dynamic positive weight if not specified
        if self.scale_pos_weight is None:
            n_neg = (y == 0).sum()
            n_pos = max((y == 1).sum(), 1)
            scale_pos = min(float(n_neg / n_pos), 50.0) # Clip to avoid over-calibration distortion
        else:
            scale_pos = self.scale_pos_weight

        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            scale_pos_weight=scale_pos,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )

        if eval_set:
            self.model.fit(
                X, y,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
            )
        else:
            self.model.fit(X, y)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
