# System Architecture

## Architecture Overview: Risk-Aware Selective Classification Pipeline (RASCP)

The proposed system addresses the vulnerability of standard classifiers under distribution shifts by wrapping predictive models inside a three-tiered risk-aware pipeline:

```
[Input Transaction x]
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Tier 1: Predictive Inference Engine                    │
│ - Logistic Regression (L1/L2 ElasticNet Baseline)      │
│ - Random Forest (Tree Ensemble)                        │
│ - LightGBM (Gradient Boosted Decision Trees)           │
│ - Tabular MLP (Deep Learning baseline with MC-Dropout) │
└──────────────────────────┬─────────────────────────────┘
                           │ Raw Probabilities / Logits
                           ▼
┌────────────────────────────────────────────────────────┐
│ Tier 2: Uncertainty & Risk Quantifier                  │
│ - Post-Hoc Probability Calibration:                   │
│     * Isotonic Regression                              │
│     * Platt Scaling / Temperature Scaling              │
│ - Ensemble Epistemic Disagreement:                    │
│     * Inter-tree variance across estimators            │
│     * Deep Ensemble predictive variance                │
│ - Inductive Conformal Prediction:                      │
│     * Non-conformity score calculation                 │
│     * Statistical coverage guarantee (1 - alpha)       │
└──────────────────────────┬─────────────────────────────┘
                           │ Calibrated Risk Metric R(x)
                           ▼
┌────────────────────────────────────────────────────────┐
│ Tier 3: Risk-Gated Decision Engine                     │
│ Evaluate: Risk(x) <= tau                               │
└────────────┬───────────────────────────────┬───────────┘
             │                               │
             ▼ (Low Risk)                    ▼ (High Risk)
  [Decision: ACCEPT]              [Decision: ABSTAIN]
  - Automated prediction ŷ        - Route to Human Fraud Analyst
  - High speed, high confidence   - Zero-tolerance safety net
```

## Architectural Components

1. **Data Pipeline & Shift Generation**:
   - `DataLoader`: Loads chronological tabular transactions (ULB credit card dataset).
   - `ShiftGenerator`: Produces controlled synthetic shifts:
     - Covariate shift: Gaussian perturbation on key PCA features ($V_1$–$V_4$).
     - Subpopulation shift: Stratified high-value transaction brackets.
     - Prevalence / Label shift: Artificially modified fraud incidence rate (0.17% to 2.5%).
     - Temporal drift: Chronological evaluation on hold-out time windows.

2. **Model Training & Storage**:
   - Modular training harness saving checkpoints/models under `models/`.
   - Unified interface exposing `fit(X, y)` and `predict_proba(X)`.

3. **Uncertainty & Calibration Modules**:
   - `ProbabilityCalibrator`: Fits calibration curves on held-out calibration split.
   - `EnsembleVarianceEstimator`: Extracts tree/model disagreement.
   - `ConformalPredictor`: Computes prediction sets $\Gamma^\alpha(x) \subseteq \{0, 1\}$.

4. **Selective Decision Engine**:
   - Computes composite risk metric $R(x)$.
   - Evaluates coverage-risk trade-offs and applies threshold $\tau$.
   - Routes instances to Automated Decision vs. Human Review Queue.
