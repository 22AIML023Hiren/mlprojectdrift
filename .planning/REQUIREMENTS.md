# Project Requirements

## Epic 1: Data Pipeline & Controlled Shift Benchmark
- **REQ-1.1 (Data Ingestion)**: Implement loader for the ULB Credit Card Fraud dataset with automated caching, checksum validation, and schema enforcement.
- **REQ-1.2 (Chronological Partitioning)**: Enforce strict temporal splits: 70% Train ($N \approx 199\text{k}$), 15% Calibration/Validation ($N \approx 42.7\text{k}$), 15% In-Distribution Test ($N \approx 42.7\text{k}$).
- **REQ-1.3 (Controlled Shift Suite)**: Implement reproducible generators for:
  - *Covariate Shift*: Gaussian noise injection on principal components $V_1$–$V_4$ ($X_{\text{shifted}} = X + \mathcal{N}(0, \sigma^2 \Sigma)$).
  - *Subpopulation Shift*: Strata filtering on high-amount transactions ($>90$th percentile).
  - *Prevalence Shift*: Synthetic resampling skewing fraud proportion from $0.17\%$ up to $2.5\%$.
  - *Temporal Drift*: Evaluation on the final temporal slice.

## Epic 2: Predictive Baseline Models
- **REQ-2.1 (Linear Baseline)**: Train and evaluate Logistic Regression with ElasticNet penalty.
- **REQ-2.2 (Tree Ensembles)**: Train and tune Random Forest and LightGBM models on source distribution $\mathcal{D}_{\text{ID}}^{\text{train}}$.
- **REQ-2.3 (Neural Baseline)**: Implement a PyTorch Tabular Multi-Layer Perceptron (MLP) with Monte Carlo Dropout.
- **REQ-2.4 (Evaluation Protocol)**: Baseline evaluation on clean ID test set reporting AUROC and AUPRC.

## Epic 3: Uncertainty Quantification & Calibration Engine
- **REQ-3.1 (Post-Hoc Probability Calibration)**: Implement Platt Scaling and Isotonic Regression fitted exclusively on the calibration set.
- **REQ-3.2 (Ensemble Epistemic Variance)**: Implement tree-level variance extraction for Random Forest and LightGBM:
  $$\mathcal{U}_{\text{ens}}(x) = \frac{1}{M} \sum_{m=1}^M (\hat{p}_m(x) - \bar{p}(x))^2$$
- **REQ-3.3 (Inductive Conformal Prediction)**: Compute non-conformity scores and generate distribution-free prediction sets $\Gamma^\alpha(x) \subseteq \{0, 1\}$ with bounded error rate $\le \alpha$.
- **REQ-3.4 (Calibration Metrics)**: Implement Expected Calibration Error (ECE), Brier Score, and Reliability Diagrams.

## Epic 4: Risk-Aware Selective Classification
- **REQ-4.1 (Risk Metric Formulation)**: Formulate normalized risk score $R(x)$ combining predicted confidence and epistemic disagreement.
- **REQ-4.2 (Gated Decision Engine)**: Implement decision policy:
  - If $R(x) \le \tau$: Accept prediction $\hat{y}$.
  - If $R(x) > \tau$: Abstain and route to human fraud triage queue.
- **REQ-4.3 (Pareto Optimization)**: Optimize threshold $\tau$ across target operational coverage levels $c \in [0.80, 0.95]$.
- **REQ-4.4 (Economic Utility Function)**: Implement cost-weighted loss:
  $$\text{Loss} = \$100 \cdot \text{FN}_{\text{accepted}} + \$5 \cdot \text{FP}_{\text{accepted}} + \$2 \cdot N_{\text{deferred}}$$

## Epic 5: Experimental Analysis & Reporting
- **REQ-5.1 (Shift Resilience Evaluation)**: Benchmark model error degradation across shift severities $\sigma \in \{0.2, 0.5, 1.0, 2.0\}$.
- **REQ-5.2 (Risk vs. Coverage Frontiers)**: Generate Risk-Coverage curves and calculate Area Under Risk-Coverage (AURC).
- **REQ-5.3 (Shift vs. Risk Disentanglement)**: Empirically measure the correlation between feature-space OOD distance (Isolation Forest) versus decision risk unreliability.
- **REQ-5.4 (Documentation & Deliverables)**: Generate final LaTeX report and single-page presentation poster (`poster.pdf`).
