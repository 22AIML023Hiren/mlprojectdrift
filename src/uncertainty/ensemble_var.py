"""
Ensemble Disagreement and Epistemic Uncertainty Estimation.
Extracts prediction variance across tree estimators or multiple models
to isolate regions of the feature space where the model lacks knowledge.
"""

from typing import List, Union, Tuple
import numpy as np
import pandas as pd


class EnsembleUncertaintyEstimator:
    """
    Computes ensemble disagreement, variance, and entropy across individual
    estimators (e.g. Random Forest trees, Deep Ensembles, or multiple algorithms).
    """

    @staticmethod
    def compute_variance_from_tree_predictions(tree_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates ensemble mean and epistemic variance from individual tree predictions.
        
        Args:
            tree_probs: Array of shape [n_estimators, n_samples] with P(Y=1)
            
        Returns:
            mean_prob: Mean ensemble prediction [n_samples]
            variance: Inter-tree epistemic variance [n_samples]
        """
        mean_prob = np.mean(tree_probs, axis=0)
        variance = np.var(tree_probs, axis=0)
        return mean_prob, variance

    @staticmethod
    def compute_multi_model_disagreement(model_prob_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates disagreement across distinct model families (e.g. LogReg, RF, LightGBM).
        
        Args:
            model_prob_list: List of P(Y=1) predictions from each model [M arrays of shape (N,)]
            
        Returns:
            mean_prob: Mean ensemble prediction across models [N]
            disagreement: Pairwise standard deviation / variance [N]
        """
        stacked = np.array(model_prob_list)  # Shape: [M, N]
        mean_prob = np.mean(stacked, axis=0)
        disagreement = np.std(stacked, axis=0)
        return mean_prob, disagreement

    @staticmethod
    def compute_predictive_entropy(probs: np.ndarray) -> np.ndarray:
        """
        Computes Shannon entropy: H(p) = -p*log(p) - (1-p)*log(1-p).
        Maximum entropy at p=0.5 (uncertain decision boundary).
        """
        p = np.clip(probs, 1e-7, 1.0 - 1e-7)
        entropy = - (p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
        return entropy
