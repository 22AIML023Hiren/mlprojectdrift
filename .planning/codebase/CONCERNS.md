# Technical Concerns, Risks, and Mitigations

## 1. Extreme Class Imbalance
- **Concern**: Only 492 out of 284,807 samples (0.172%) are positive (fraud). Standard classification accuracy will be misleadingly high (99.8%) even for a dummy zero-predictor.
- **Mitigation**:
  - Exclude standard accuracy from key evaluation metrics.
  - Rely strictly on Area Under the Precision-Recall Curve (AUPRC), cost-weighted loss, and selective coverage-risk frontiers.
  - Apply stratified sampling during validation/calibration splits to preserve exact fraud proportions.

## 2. Inappropriate Probability Calibration Under Imbalance
- **Concern**: Standard calibration methods (e.g. unweighted Platt scaling) can over-shrink fraud probabilities toward the prior mean ($0.17\%$).
- **Mitigation**:
  - Use Isotonic Regression on a dedicated validation slice with balanced or stratified subsampling.
  - Evaluate Temperature Scaling calibrated specifically with cost-sensitive logarithmic loss.

## 3. Disentangling Shift Detection from Predictive Risk
- **Concern**: Density-based out-of-distribution (OOD) detectors flag instances that are geographically distant in feature space $P(X)$, but feature novelty does not always equal decision unreliability.
- **Mitigation**:
  - Explicitly evaluate whether ensemble disagreement and conformal prediction sets correlate with misclassification error rather than just data distance.
  - Benchmark Mahalanobis distance / Isolation Forest directly against uncertainty-gated abstention.

## 4. Operational Abstention Collapse
- **Concern**: A risk-averse model could trivially achieve zero error by rejecting 90% of transactions, overwhelming human fraud analyst capacity.
- **Mitigation**:
  - Enforce a strict minimum throughput constraint ($\ge 85\%$ automated clearance).
  - Plot full Risk-Coverage curves to evaluate performance across realistic operational budgets (5%, 10%, 15% deferral).
