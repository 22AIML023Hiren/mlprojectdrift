# Bonus 3: Shift-Severity Sweep — Analysis Report

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

| Severity | N | Errors | Error% | ρ(Risk,Err) | p-val | ρ(OOD,Err) | p-val |
|---|---|---|---|---|---|---|---|
| 0.00 | 1500 | 3 | 0.20% | 0.0769 | 0.0029 | 0.0771 | 0.0028 |
| 0.25 | 1500 | 4 | 0.27% | 0.0888 | 0.0006 | 0.0882 | 0.0006 |
| 0.50 | 1500 | 6 | 0.40% | 0.0734 | 0.0044 | 0.1007 | 0.0001 |
| 1.00 | 1500 | 18 | 1.20% | 0.1667 | 0.0000 | 0.1315 | 0.0000 |
| 2.00 | 1500 | 99 | 6.60% | 0.4114 | 0.0000 | 0.2131 | 0.0000 |
| 4.00 | 1500 | 225 | 15.00% | 0.6065 | 0.0000 | 0.1783 | 0.0000 |

### Observations
- Risk correlation was valid at 6/6 severity levels. Average ρ = 0.237, peak ρ = 0.607.
- OOD score correlation was valid at 6/6 severity levels. Average ρ = 0.131, peak ρ = 0.213.

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
