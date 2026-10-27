"""
Distribution shift quantification and statistical divergence metrics.
Implements:
1. Wasserstein Distance (Earth Mover's Distance)
2. Two-sample Kolmogorov-Smirnov (KS) Test with p-values
3. Population Stability Index (PSI)
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp


def compute_psi(ref: np.ndarray, shift: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Population Stability Index (PSI) between reference and shifted distributions:
    PSI = sum((P_b - Q_b) * ln(P_b / Q_b))
    
    Standard Interpretation:
    - PSI < 0.1: Negligible shift
    - 0.1 <= PSI < 0.25: Moderate shift
    - PSI >= 0.25: Significant shift
    """
    ref = np.asarray(ref)
    shift = np.asarray(shift)

    # Establish quantile bin edges from reference distribution
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(ref, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Calculate frequency proportions
    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    shift_counts, _ = np.histogram(shift, bins=bin_edges)

    # Add small epsilon to avoid log(0) or division by zero
    eps = 1e-4
    p = (ref_counts + eps) / (len(ref) + eps * n_bins)
    q = (shift_counts + eps) / (len(shift) + eps * n_bins)

    psi = np.sum((q - p) * np.log(q / p))
    return float(psi)


def compute_distribution_divergence(
    X_ref: pd.DataFrame,
    X_shift: pd.DataFrame,
    feature_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive statistical divergence profile between reference
    in-distribution data and shifted data across all continuous features.
    
    Returns:
        Per-feature metrics (Wasserstein distance, KS statistic, p-value, PSI)
        and summary aggregation statistics.
    """
    if feature_cols is None:
        feature_cols = [c for c in X_ref.columns if c in X_shift.columns and c != "Class"]

    feature_metrics = {}
    total_wasserstein = 0.0
    total_psi = 0.0
    significant_shifts_ks = 0

    for col in feature_cols:
        v_ref = X_ref[col].values
        v_shift = X_shift[col].values

        # 1. Wasserstein distance
        w_dist = float(wasserstein_distance(v_ref, v_shift))
        total_wasserstein += w_dist

        # 2. Kolmogorov-Smirnov 2-sample test
        ks_res = ks_2samp(v_ref, v_shift)
        ks_stat = float(ks_res.statistic)
        ks_pval = float(ks_res.pvalue)
        if ks_pval < 0.01:
            significant_shifts_ks += 1

        # 3. Population Stability Index
        psi_val = compute_psi(v_ref, v_shift)
        total_psi += psi_val

        feature_metrics[col] = {
            "wasserstein_distance": round(w_dist, 4),
            "ks_statistic": round(ks_stat, 4),
            "ks_pvalue": ks_pval,
            "is_significant_shift": ks_pval < 0.01,
            "psi": round(psi_val, 4)
        }

    n_feats = max(len(feature_cols), 1)
    mean_w1 = total_wasserstein / n_feats
    mean_psi = total_psi / n_feats

    return {
        "mean_wasserstein_distance": round(mean_w1, 4),
        "mean_psi": round(mean_psi, 4),
        "significant_ks_features_count": significant_shifts_ks,
        "total_features_evaluated": len(feature_cols),
        "feature_details": feature_metrics
    }
