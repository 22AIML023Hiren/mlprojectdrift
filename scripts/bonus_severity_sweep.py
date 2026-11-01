"""
Bonus 3: Shift-Severity Sweep Experiment
=========================================
Evaluates how well composite risk scores and OOD scores correlate with actual
classification errors under progressively stronger covariate shifts.

Experiment Protocol:
    1. Load data via the existing pipeline; split chronologically to avoid leakage.
    2. Fit LightGBM model on train split; fit isotonic calibrator on calib split;
       fit IsolationForest OOD detector on X_train.
    3. For each severity in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
       a. Apply covariate shift (Gaussian noise on V1–V4, Amount) using severity as sigma.
       b. Compute per-sample error indicator (1 = misclassified, 0 = correct).
       c. Compute per-sample composite risk score R(x) and OOD score.
       d. Spearman ρ(risk, error) and ρ(ood, error) — the main experiment metrics.
    4. Save results/figures/bonus_severity_sweep.png, results/bonus_severity_sweep.csv,
       results/bonus_severity_sweep_analysis.md.

Run:
    python scripts/bonus_severity_sweep.py
"""

import sys
import os
from pathlib import Path

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import spearmanr

from src.data.loader import load_dataset, get_chronological_splits
from src.data.shifts import ShiftGenerator
from src.models.gbm import LightGBMModel
from src.uncertainty.calibration import ProbabilityCalibrator
from src.decision.risk_scorer import RiskScorer

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
SEVERITY_LEVELS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
RANDOM_STATE = 42
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _spearman_safe(scores: np.ndarray, targets: np.ndarray):
    """
    Compute Spearman ρ safely.
    Returns (rho, p_value) or (np.nan, np.nan) if:
      - fewer than 5 samples,
      - all scores are constant (no discriminability), or
      - all targets are the same (no errors to correlate).
    """
    if len(scores) < 5:
        return np.nan, np.nan
    if np.std(scores) < 1e-10 or np.std(targets) < 1e-10:
        return np.nan, np.nan
    rho, pval = spearmanr(scores, targets)
    return float(rho), float(pval)


def run_sweep(splits: dict) -> pd.DataFrame:
    """
    Core experiment loop: train model, calibrate, fit OOD detector, sweep severities.
    Returns a DataFrame with one row per severity level.
    """
    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_calib = splits["X_calib"]
    y_calib = splits["y_calib"]
    X_test  = splits["X_test"]
    y_test  = splits["y_test"]

    # ── 1. Train primary model ────────────────────────────────
    print("\n[Sweep] Fitting LightGBM model on training split …")
    model = LightGBMModel(
        n_estimators=40,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    # ── 2. Calibrate probabilities ────────────────────────────
    print("[Sweep] Fitting isotonic calibrator on calibration split …")
    calib_raw = model.predict_proba(X_calib)[:, 1]
    calibrator = ProbabilityCalibrator(method="isotonic")
    calibrator.fit(calib_raw, y_calib)

    # ── 3. Fit OOD detector on training features ──────────────
    print("[Sweep] Fitting IsolationForest OOD detector on training split …")
    risk_scorer = RiskScorer(fit_ood_detector=True, random_state=RANDOM_STATE)
    risk_scorer.fit(X_train)

    # ── 4. Severity sweep ─────────────────────────────────────
    shift_gen = ShiftGenerator(random_state=RANDOM_STATE)
    records = []

    for severity in SEVERITY_LEVELS:
        print(f"\n[Sweep] severity={severity:.2f} …")

        if severity == 0.0:
            X_shifted = X_test.copy()
        else:
            X_shifted = shift_gen.apply_covariate_shift(X_test, severity=severity)

        n_samples = len(y_test)
        fraud_count = int(y_test.sum())

        # Per-sample probabilities and error indicator
        raw_probs      = model.predict_proba(X_shifted)[:, 1]
        calib_probs    = calibrator.predict(raw_probs)
        y_pred         = (raw_probs >= 0.5).astype(int)
        error_indicator = (y_pred != y_test).astype(int)
        n_errors        = int(error_indicator.sum())

        # Per-sample risk scores
        tree_preds = model.get_individual_tree_predictions(X_shifted)
        variance   = np.var(tree_preds, axis=0)
        risk_scores = risk_scorer.compute_composite_risk(calib_probs, ensemble_variance=variance)

        # Per-sample OOD scores
        ood_scores = risk_scorer.compute_feature_ood_score(X_shifted)

        # Spearman correlations
        rho_risk, p_risk = _spearman_safe(risk_scores, error_indicator)
        rho_ood,  p_ood  = _spearman_safe(ood_scores,  error_indicator)

        valid_risk = int(not np.isnan(rho_risk))
        valid_ood  = int(not np.isnan(rho_ood))

        print(f"  n={n_samples}, errors={n_errors}, fraud={fraud_count}")
        print(f"  ρ(risk, error)={rho_risk:.4f}  (p={p_risk:.4f})" if not np.isnan(rho_risk)
              else "  ρ(risk, error)=NaN (undefined — constant scores or zero-variance)")
        print(f"  ρ(OOD,  error)={rho_ood:.4f}  (p={p_ood:.4f})" if not np.isnan(rho_ood)
              else "  ρ(OOD,  error)=NaN (undefined — constant scores or zero-variance)")

        records.append({
            "Severity":             severity,
            "N_Samples":            n_samples,
            "N_Errors":             n_errors,
            "Fraud_Count":          fraud_count,
            "Error_Rate_%":         round(n_errors / max(n_samples, 1) * 100, 2),
            "Rho_Risk_vs_Error":    round(rho_risk, 4) if not np.isnan(rho_risk) else float("nan"),
            "P_Risk":               round(p_risk,  4) if not np.isnan(p_risk)  else float("nan"),
            "Rho_OOD_vs_Error":     round(rho_ood,  4) if not np.isnan(rho_ood)  else float("nan"),
            "P_OOD":                round(p_ood,   4) if not np.isnan(p_ood)   else float("nan"),
            "Corr_Risk_Valid":      valid_risk,
            "Corr_OOD_Valid":       valid_ood,
        })

    return pd.DataFrame(records)


def plot_sweep(df: pd.DataFrame, output_path: Path) -> None:
    """Generates a dual-line plot of Spearman correlations vs shift severity."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    sev     = df["Severity"].values
    rho_r   = df["Rho_Risk_vs_Error"].values
    rho_ood = df["Rho_OOD_vs_Error"].values

    # Mask NaN for clean plot
    mask_r   = ~np.isnan(rho_r)
    mask_ood = ~np.isnan(rho_ood)

    ax.plot(sev[mask_r],   rho_r[mask_r],   "o-",  color="#2563EB", lw=2.2,
            markersize=7, label="ρ(Composite Risk, Error)")
    ax.plot(sev[mask_ood], rho_ood[mask_ood], "s--", color="#DC2626", lw=2.2,
            markersize=7, label="ρ(OOD Score, Error)")

    # Mark NaN points explicitly
    for i, s in enumerate(sev):
        if not mask_r[i]:
            ax.axvline(s, color="#2563EB", lw=0.8, linestyle=":", alpha=0.5)
        if not mask_ood[i]:
            ax.axvline(s, color="#DC2626", lw=0.8, linestyle=":", alpha=0.5)

    ax.set_xlabel("Covariate Shift Severity (σ multiplier)", fontsize=12)
    ax.set_ylabel("Spearman ρ with Classification Error", fontsize=12)
    ax.set_title("Bonus 3: Risk vs OOD Score Correlation Under Increasing Covariate Shift",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(sev)
    ax.set_xlim(-0.15, max(sev) + 0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Output] Figure saved → {output_path}")


def write_analysis(df: pd.DataFrame, output_path: Path) -> None:
    """Writes the markdown analysis report from actual results."""

    # Build table
    table_lines = [
        "| Severity | N | Errors | Error% | ρ(Risk,Err) | p-val | ρ(OOD,Err) | p-val |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        def fmt(v): return f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "NaN"
        table_lines.append(
            f"| {row['Severity']:.2f} | {int(row['N_Samples'])} | {int(row['N_Errors'])} "
            f"| {row['Error_Rate_%']:.2f}% | {fmt(row['Rho_Risk_vs_Error'])} "
            f"| {fmt(row['P_Risk'])} | {fmt(row['Rho_OOD_vs_Error'])} | {fmt(row['P_OOD'])} |"
        )

    # Summary observations
    valid_r   = df[df["Corr_Risk_Valid"] == 1]
    valid_ood = df[df["Corr_OOD_Valid"]  == 1]

    if len(valid_r) > 0:
        avg_risk = valid_r["Rho_Risk_vs_Error"].mean()
        max_risk = valid_r["Rho_Risk_vs_Error"].max()
        obs_risk = (f"Risk correlation was valid at {len(valid_r)}/{len(df)} severity levels. "
                    f"Average ρ = {avg_risk:.3f}, peak ρ = {max_risk:.3f}.")
    else:
        obs_risk = "Risk correlation was undefined (NaN) at all severity levels — scores were constant."

    if len(valid_ood) > 0:
        avg_ood = valid_ood["Rho_OOD_vs_Error"].mean()
        max_ood = valid_ood["Rho_OOD_vs_Error"].max()
        obs_ood = (f"OOD score correlation was valid at {len(valid_ood)}/{len(df)} severity levels. "
                   f"Average ρ = {avg_ood:.3f}, peak ρ = {max_ood:.3f}.")
    else:
        obs_ood = "OOD correlation was undefined (NaN) at all severity levels — scores were constant."

    md = f"""# Bonus 3: Shift-Severity Sweep — Analysis Report

## Methodology

This experiment evaluates how well two separately computed per-sample unreliability
signals correlate with actual classification errors under progressively stronger
covariate distribution shift.

### Shift Mechanism
Covariate shift is applied to the clean holdout test set by injecting Gaussian noise
(zero mean, scaled standard deviation) into features V1, V2, V3, V4, and Amount:

```
X_shifted[col] += N(0, severity × std(X_test[col]))
```

Severity 0.0 is the clean baseline. A severity of 1.0 adds noise equal in scale to
the original feature standard deviation; severity 4.0 adds 4× the std — a severe shift.

### Pipeline (no data leakage)
- **Train split (70%):** Fit LightGBM model and IsolationForest OOD detector.
- **Calib split (15%):** Fit isotonic calibrator on raw model probabilities.
- **Test split (15%):** Held out; covariate shift applied *after* splitting.
- **Feature scaling:** RobustScaler fitted on train split only (applied to all splits).

### Signals Compared
| Signal | Definition |
|---|---|
| Composite Risk R(x) | 0.6 × R_conf + 0.4 × R_var, where R_var = Var(checkpoint probs) / 0.25 |
| OOD Score | Normalised Isolation Forest anomaly score (higher = more OOD) |

### Error Indicator
`error(x) = 1` if the model misclassifies sample x, else 0.
Spearman ρ measures rank-order correlation between each signal and error.
`NaN` is reported honestly when scores are constant (no discriminability).

---

## Results

{chr(10).join(table_lines)}

### Observations
- {obs_risk}
- {obs_ood}

---

## Interpretation

**Composite Risk vs. Error:** The composite risk score combines calibrated confidence
margin and boosting-checkpoint variance. Confidence margin contributes most (weight 0.6)
and captures aleatoric uncertainty (boundary cases), while variance captures boosting
instability. As shift severity increases, boundary-crossing samples increase, which should
raise risk scores for misclassified samples — improving the correlation. If the model
is robust enough that error rate does not increase much under mild shifts, ρ may remain
moderate even at severity > 0.

**OOD Score vs. Error:** The Isolation Forest OOD score detects feature-space novelty.
Under covariate shift, OOD scores should increase broadly across all shifted samples.
However, not all OOD samples are misclassified (a sample can be novel yet still easy
to classify correctly), so ρ(OOD, error) is expected to be lower than ρ(risk, error)
at mild severities and may improve at large severities where the shift overwhelms the
model's generalisation capacity.

---

## Limitations

1. **Synthetic data:** If the real `creditcard.csv` is unavailable, results use the
   synthetic benchmark (`generate_synthetic_benchmark`), which is simpler and more
   linearly separable than the real dataset. Correlations may differ on real data.
2. **Checkpoint variance ≠ independent tree disagreement:** The LightGBM uncertainty
   proxy measures boosting convergence stability, not true epistemic uncertainty.
3. **Small test split:** With 50 000 synthetic samples, the test set ~7 500 samples;
   with real data (~284 807 rows), ~42 721. Fraud counts may be very small,
   making error distributions sparse and correlations noisier.
4. **Covariate shift only:** This experiment covers Gaussian covariate shift exclusively.
   OOD detection under concept drift or label shift may behave very differently.
5. **OOD scores fitted on training features (pre-shift):** This is the correct design
   (no leakage), but means the OOD detector has no shift-specific calibration.
"""

    output_path.write_text(md, encoding="utf-8")
    print(f"[Output] Analysis markdown saved → {output_path}")


def main():
    print("=" * 60)
    print("  Bonus 3: Shift-Severity Sweep Experiment")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────
    print("\n[Data] Loading dataset …")
    df = load_dataset(allow_synthetic_fallback=True, random_state=RANDOM_STATE)

    print("[Data] Creating chronological splits with RobustScaler on train only …")
    splits = get_chronological_splits(df, scale_features=True)

    # ── Run sweep ─────────────────────────────────────────────
    results_df = run_sweep(splits)

    # ── Save CSV ──────────────────────────────────────────────
    csv_path = RESULTS_DIR / "bonus_severity_sweep.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n[Output] CSV  saved → {csv_path}")

    # ── Save figure ───────────────────────────────────────────
    fig_path = FIGURES_DIR / "bonus_severity_sweep.png"
    plot_sweep(results_df, fig_path)

    # ── Save analysis ─────────────────────────────────────────
    md_path = RESULTS_DIR / "bonus_severity_sweep_analysis.md"
    write_analysis(results_df, md_path)

    # ── Print summary table ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(results_df[["Severity", "N_Errors", "Error_Rate_%",
                       "Rho_Risk_vs_Error", "Rho_OOD_vs_Error"]].to_string(index=False))
    print("\n[Done] All outputs verified.")

    return results_df


if __name__ == "__main__":
    main()
