"""
Baseline predictive models for imbalanced tabular classification:
- Logistic Regression (L2 / ElasticNet regularized with class weight balancing)
- Random Forest (Bagged decision trees with tree prediction inspection)
"""

from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class LogisticRegressionBaseline:
    """
    Logistic Regression baseline with balanced class weights to address
    severe fraud skew (0.172%).
    """
    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state
        )
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LogisticRegressionBaseline":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


class RandomForestBaseline:
    """
    Random Forest ensemble of decorrelated decision trees.
    Provides methods to inspect individual tree predictions for inter-tree variance.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 12,
        min_samples_split: int = 5,
        random_state: int = 42
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state
        )
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RandomForestBaseline":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def get_individual_tree_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns predictions of each individual tree in the ensemble.
        Shape: (n_estimators, n_samples) containing predicted P(Y=1).
        """
        X_arr = np.asarray(X)
        tree_probs = []
        for estimator in self.model.estimators_:
            # Tree estimator predict_proba returns [n_samples, 2] if both classes seen, else [n_samples, 1]
            p = estimator.predict_proba(X_arr)
            if p.shape[1] == 2:
                tree_probs.append(p[:, 1])
            else:
                # Fallback if tree only has single class node
                single_class = estimator.classes_[0]
                p_fraud = np.full(len(X_arr), 1.0 if single_class == 1 else 0.0)
                tree_probs.append(p_fraud)

        return np.array(tree_probs)
