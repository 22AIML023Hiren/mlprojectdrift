"""
Controlled Distribution Shift Generation Suite.
Implements systematic synthetic and subpopulation shifts for tabular streams:
1. Covariate Shift (Gaussian perturbation on sensitive PCA components)
2. Subpopulation Shift (Transaction amount tail strata)
3. Prevalence / Label Shift (Simulating fraud waves with varying base rates)
4. Concept Drift (Adversarial feature rotation in fraud patterns)
"""

from typing import Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd


class ShiftGenerator:
    """
    Produces controlled, reproducible distribution shifts on tabular test sets.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def apply_covariate_shift(
        self,
        X: pd.DataFrame,
        severity: float = 1.0,
        target_features: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Applies Covariate Shift: P(X) shifts while P(Y|X) remains fixed.
        Injects calibrated Gaussian perturbation into selected sensitive features.
        
        Args:
            X: Input feature DataFrame
            severity: Standard deviation multiplier for injected noise (sigma)
            target_features: Subset of features to perturb (default: V1 to V4 and Amount)
        """
        rng = np.random.RandomState(self.random_state)
        X_shifted = X.copy()
        
        if target_features is None:
            # V1 to V4 are primary principal components with highest variance
            target_features = [col for col in ["V1", "V2", "V3", "V4", "Amount"] if col in X.columns]

        for col in target_features:
            col_std = np.std(X[col])
            noise = rng.normal(loc=0.0, scale=severity * col_std, size=len(X))
            X_shifted[col] += noise

        return X_shifted

    def apply_subpopulation_shift(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        quantile_threshold: float = 0.85
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Applies Subpopulation Shift: Isolates high-value tail transactions
        where fraud distribution and fraudster strategies deviate from normal transactions.
        
        Args:
            X: Input feature DataFrame
            y: Binary target array
            quantile_threshold: Quantile cutoff for transaction amount (e.g. top 15%)
        """
        if "Amount" not in X.columns:
            raise KeyError("Amount column required for subpopulation shift.")

        cutoff = X["Amount"].quantile(quantile_threshold)
        mask = (X["Amount"] >= cutoff).values

        X_shifted = X[mask].copy().reset_index(drop=True)
        y_shifted = y[mask].copy()

        return X_shifted, y_shifted

    def apply_label_shift(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        target_fraud_rate: float = 0.025
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Applies Label / Prevalence Shift: Alters the positive class proportion
        P(Y=1) to simulate sudden surges in organized fraud attack waves.
        
        Args:
            X: Input feature DataFrame
            y: Binary target array
            target_fraud_rate: Desired fraud rate (e.g. 2.5% vs standard 0.17%)
        """
        rng = np.random.RandomState(self.random_state)
        fraud_idx = np.where(y == 1)[0]
        legit_idx = np.where(y == 0)[0]

        n_fraud = len(fraud_idx)
        if n_fraud == 0:
            return X.copy(), y.copy()

        # Compute required number of legitimate samples to reach target fraud rate
        # target_fraud_rate = n_fraud / (n_fraud + n_legit_needed)
        n_legit_needed = int(round(n_fraud * (1.0 - target_fraud_rate) / target_fraud_rate))
        n_legit_needed = min(n_legit_needed, len(legit_idx))

        selected_legit = rng.choice(legit_idx, size=n_legit_needed, replace=False)
        combined_idx = np.concatenate([fraud_idx, selected_legit])
        rng.shuffle(combined_idx)

        X_shifted = X.iloc[combined_idx].copy().reset_index(drop=True)
        y_shifted = y[combined_idx].copy()

        return X_shifted, y_shifted

    def apply_concept_drift(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        drift_strength: float = 1.5
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Applies Concept Drift: P(Y|X) changes because fraudulent actors
        adopt new evasion tactics, altering feature correlation specifically
        in the fraudulent subpopulation.
        """
        rng = np.random.RandomState(self.random_state)
        X_shifted = X.copy()
        fraud_mask = (y == 1)

        # Perturb fraud instances to mimic evasion: invert sign and add shift
        evasion_cols = [c for c in ["V1", "V2", "V3", "V10", "V14"] if c in X.columns]
        for col in evasion_cols:
            noise = rng.normal(loc=drift_strength, scale=0.5, size=fraud_mask.sum())
            X_shifted.loc[fraud_mask, col] = -0.5 * X_shifted.loc[fraud_mask, col] + noise

        return X_shifted, y.copy()

    def generate_all_shift_testbeds(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray
    ) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
        """
        Constructs the complete evaluation suite containing clean In-Distribution (ID)
        and all 4 controlled Out-of-Distribution (OOD) shift testbeds.
        """
        testbeds = {}

        # 0. Clean In-Distribution Test Set
        testbeds["Clean_ID"] = (X_test.copy(), y_test.copy())

        # 1. Covariate Shift (Mild sigma=0.5 and Severe sigma=1.5)
        testbeds["Covariate_Mild"] = (
            self.apply_covariate_shift(X_test, severity=0.5),
            y_test.copy()
        )
        testbeds["Covariate_Severe"] = (
            self.apply_covariate_shift(X_test, severity=1.5),
            y_test.copy()
        )

        # 2. Subpopulation Shift (High Transaction Amount Strata)
        X_sub, y_sub = self.apply_subpopulation_shift(X_test, y_test, quantile_threshold=0.85)
        testbeds["Subpopulation_HighAmount"] = (X_sub, y_sub)

        # 3. Prevalence / Label Shift (Surge to 2.5% fraud rate)
        X_prev, y_prev = self.apply_label_shift(X_test, y_test, target_fraud_rate=0.025)
        testbeds["Prevalence_Shift"] = (X_prev, y_prev)

        # 4. Concept Drift (Adversarial pattern evasion)
        X_drift, y_drift = self.apply_concept_drift(X_test, y_test, drift_strength=1.5)
        testbeds["Concept_Drift"] = (X_drift, y_drift)

        return testbeds
