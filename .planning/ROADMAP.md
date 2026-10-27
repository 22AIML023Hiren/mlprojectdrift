# Project Roadmap

## Phase Overview

| Phase | Title | Focus Area | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Data Pipeline & Temporal Partitioning | Dataset ingestion, preprocessing, and chronological train/calib/test split | **Complete** |
| **Phase 2** | Controlled Distribution Shift Suite | Covariate, subpopulation, label ratio, and temporal shift generators | **Complete** |
| **Phase 3** | Baseline Classifiers Training | Logistic Regression, Random Forest, LightGBM, and Tabular MLP | **Complete** |
| **Phase 4** | Uncertainty Quantification & Calibration | Platt/Isotonic calibration, ensemble variance, and conformal prediction | **Complete** |
| **Phase 5** | Risk-Aware Selective Classification | Risk-gated decision engine, abstention thresholds, and triage queue | **Complete** |
| **Phase 6** | Comprehensive Evaluation & Analysis | AURC curves, ECE calibration, cost analysis, and shift disentanglement | Ready |
| **Phase 7** | Final Project Packaging & Poster Delivery | LaTeX paper, presentation artifacts, and code release | Planned |

---

## Detailed Phase Breakdown

### Phase 1: Data Pipeline & Temporal Partitioning
- Set up directory structure (`data/raw/`, `data/processed/`, `src/`).
- Ingest and validate Kaggle / ULB Credit Card Fraud dataset (284,807 records, 30 features).
- Establish chronological split: 70% Train ($N \approx 199\text{k}$), 15% Calibration ($N \approx 42.7\text{k}$), 15% ID Test ($N \approx 42.7\text{k}$).
- *Exit Criteria*: Clean, reproducible data loader module passing verification checks.

### Phase 2: Controlled Distribution Shift Suite
- Implement `ShiftGenerator` supporting 4 shift paradigms:
  1. Covariate shift (feature perturbation on $V_1$–$V_4$).
  2. Subpopulation shift (transaction amount tail strata).
  3. Label ratio shift (fraud prevalence $0.17\% \to 2.5\%$).
  4. Temporal concept drift (chronological hold-out evaluation).
- Export parameterized test sets under `data/shifted/`.
- *Exit Criteria*: Verified synthetic shift test sets with automated summary statistics.

### Phase 3: Baseline Classifiers Training
- Train Logistic Regression (ElasticNet), Random Forest, LightGBM, and Tabular MLP on $\mathcal{D}_{\text{ID}}^{\text{train}}$.
- Optimize hyperparameters on calibration/validation split.
- Benchmark clean in-distribution performance (AUROC, AUPRC).
- *Exit Criteria*: Serialized model checkpoints and in-distribution performance table.

### Phase 4: Uncertainty Quantification & Calibration
- Implement post-hoc calibration (Isotonic Regression, Platt Scaling) on calibration split.
- Extract inter-tree ensemble variance for Random Forest & LightGBM.
- Implement Inductive Conformal Prediction for distribution-free prediction sets ($1 - \alpha$).
- Compute Expected Calibration Error (ECE) and generate reliability diagrams.
- *Exit Criteria*: Uncertainty module producing calibrated probabilities and confidence intervals.

### Phase 5: Risk-Aware Selective Classification
- Construct composite risk metric $R(x)$ from calibrated probabilities and epistemic variance.
- Build risk-gated decision engine: Accept if $R(x) \le \tau$, Abstain / Route if $R(x) > \tau$.
- Sweep $\tau$ to construct Risk-Coverage curves ($c \in [0.6, 1.0]$).
- Optimize threshold under operational constraint ($\ge 85\%$ automated clearance).
- *Exit Criteria*: Functional selective classifier with tunable abstention threshold.

### Phase 6: Comprehensive Evaluation & Analysis
- Benchmark all models and uncertainty methods across the 4 shifted test sets.
- Calculate AURC (Area Under Risk-Coverage curve), cost-weighted loss, and ECE degradation.
- Empirically test disentanglement: Correlate predictive risk with classification error vs. feature-space OOD distance.
- *Exit Criteria*: Publication-grade figures, risk-coverage trade-off plots, and ablation tables.

### Phase 7: Final Project Packaging & Poster Delivery
- Synchronize empirical findings into final project report (`project_proposal.tex` / `main.tex`).
- Update single-page presentation poster (`poster.html`, `poster.pdf`) with empirical results.
- Create automated test runner and README documentation.
- *Exit Criteria*: Complete, reproducible course submission package.
