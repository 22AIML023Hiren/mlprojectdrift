"""
Visualization and plotting module for generating publication-quality figures:
1. Risk-Coverage Pareto Curves
2. Reliability Diagrams (Calibration plots)
3. Shift Resilience Comparison Charts
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "figures"


def plot_risk_coverage_frontier(
    rc_curve_vanilla: Dict[str, Any],
    rc_curve_risk_aware: Dict[str, Any],
    save_path: Optional[str] = None
) -> None:
    """
    Plots the Risk-Coverage trade-off curve comparing Random/Vanilla abstention
    against the Risk-Aware Selective Classification policy.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = save_path or (RESULTS_DIR / "risk_coverage_curve.png")

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    # Risk-Aware curve
    ax.plot(
        rc_curve_risk_aware["coverages"] * 100,
        rc_curve_risk_aware["error_rates"] * 100,
        label=f"Risk-Aware Selection (AURC = {rc_curve_risk_aware['aurc']:.4f})",
        color="#2563eb",
        linewidth=2.2
    )

    # Vanilla / Random abstention curve
    ax.plot(
        rc_curve_vanilla["coverages"] * 100,
        rc_curve_vanilla["error_rates"] * 100,
        label=f"Vanilla Confidence (AURC = {rc_curve_vanilla['aurc']:.4f})",
        color="#dc2626",
        linestyle="--",
        linewidth=2.0
    )

    # Highlight 85% target coverage line
    ax.axvline(x=85, color="#10b981", linestyle=":", linewidth=1.5, label="Target 85% Throughput")

    ax.set_title("Selective Classification: Risk vs. Coverage Frontier Under Shift", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Coverage (% of Transactions Automated)", fontsize=10, fontweight="semibold")
    ax.set_ylabel("Classification Error on Accepted (%)", fontsize=10, fontweight="semibold")
    ax.set_xlim(20, 100)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Plotting] Saved Risk-Coverage plot to {out_path}")


def plot_calibration_diagram(
    y_true: np.ndarray,
    uncalib_prob: np.ndarray,
    calib_prob: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plots Reliability Diagram comparing uncalibrated vs. calibrated probability outputs.
    """
    from sklearn.calibration import calibration_curve

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = save_path or (RESULTS_DIR / "calibration_reliability_diagram.png")

    prob_true_uncal, prob_pred_uncal = calibration_curve(y_true, uncalib_prob, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, calib_prob, n_bins=10)

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", label="Perfect Calibration")

    ax.plot(
        prob_pred_uncal,
        prob_true_uncal,
        marker="s",
        color="#dc2626",
        linewidth=1.8,
        label="Raw Uncalibrated (GBDT)"
    )
    ax.plot(
        prob_pred_cal,
        prob_true_cal,
        marker="o",
        color="#2563eb",
        linewidth=2.0,
        label="Post-Hoc Calibrated (Isotonic)"
    )

    ax.set_title("Reliability Diagram: Calibration Under Imbalance", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Mean Predicted Confidence", fontsize=10, fontweight="semibold")
    ax.set_ylabel("Empirical Fraction of Positives (Fraud)", fontsize=10, fontweight="semibold")
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Plotting] Saved Reliability diagram to {out_path}")


def plot_shift_resilience_comparison(
    benchmark_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Plots economic loss comparison across shift testbeds between Vanilla and Selective policies.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = save_path or (RESULTS_DIR / "shift_resilience_loss_comparison.png")

    testbeds = benchmark_df["Testbed"].values
    vanilla_loss = benchmark_df["Vanilla_Loss_$"].values
    selective_loss = benchmark_df["Selective_Loss_$"].values

    x = np.arange(len(testbeds))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    rects1 = ax.bar(x - width/2, vanilla_loss, width, label="Vanilla Non-Selective", color="#f87171")
    rects2 = ax.bar(x + width/2, selective_loss, width, label="Risk-Aware Selective (85% Cov)", color="#60a5fa")

    ax.set_ylabel("Total Operational Loss ($)", fontsize=10, fontweight="semibold")
    ax.set_title("Economic Loss Mitigation Across Distribution Shift Testbeds", fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(testbeds, rotation=15, ha="right", fontsize=9)
    ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Plotting] Saved Shift Resilience Comparison plot to {out_path}")


def plot_disentanglement_analysis(
    disentangle_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Plots Spearman rank correlation comparing:
    1. Composite Decision Risk vs. Error
    2. Feature OOD Anomaly Distance vs. Error
    Proves that decision risk unreliability strongly outperforms pure feature shift detection.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = save_path or (RESULTS_DIR / "disentanglement_correlation_comparison.png")

    testbeds = disentangle_df["testbed"].values
    corr_risk = disentangle_df["spearman_decision_risk_error"].values
    corr_ood = disentangle_df["spearman_ood_distance_error"].values

    x = np.arange(len(testbeds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    rects1 = ax.bar(x - width/2, corr_risk, width, label="Decision Risk Correlation ρ(R, Error)", color="#2563eb")
    rects2 = ax.bar(x + width/2, corr_ood, width, label="Feature OOD Distance ρ(OOD, Error)", color="#f59e0b")

    ax.set_ylabel("Spearman Rank Correlation (ρ)", fontsize=10, fontweight="semibold")
    ax.set_title("Shift vs. Decision Risk Disentanglement: Predictive Unreliability vs. Feature Novelty", fontsize=11, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(testbeds, rotation=15, ha="right", fontsize=9)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(frameon=True, fontsize=9)

    # Add numeric labels on top of bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Plotting] Saved Disentanglement Comparison plot to {out_path}")


def plot_multi_model_aurc_comparison(
    multi_model_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Plots multi-model AURC comparison across distribution shifts.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = save_path or (RESULTS_DIR / "multi_model_aurc_comparison.png")

    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=300)

    models = multi_model_df["model"].unique()
    testbeds = multi_model_df["testbed"].unique()
    x = np.arange(len(testbeds))
    n_models = len(models)
    width = 0.8 / n_models

    colors = ["#3b82f6", "#10b981", "#f97316", "#8b5cf6"]

    for i, model_name in enumerate(models):
        sub_df = multi_model_df[multi_model_df["model"] == model_name]
        aurcs = [sub_df[sub_df["testbed"] == tb]["aurc"].values[0] if len(sub_df[sub_df["testbed"] == tb]) > 0 else 0.0 for tb in testbeds]
        ax.bar(x + (i - n_models / 2 + 0.5) * width, aurcs, width, label=model_name, color=colors[i % len(colors)])

    ax.set_ylabel("Area Under Risk-Coverage (AURC, Lower is Better)", fontsize=10, fontweight="semibold")
    ax.set_title("Multi-Model Selective Classification Efficiency (AURC) Under Shift", fontsize=11, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(testbeds, rotation=15, ha="right", fontsize=9)
    ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Plotting] Saved Multi-Model AURC comparison to {out_path}")
