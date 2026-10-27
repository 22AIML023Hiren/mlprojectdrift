# Testing and Evaluation Protocols

## Testing Philosophy
Because this project centers on **Risk-Aware Machine Learning Under Distribution Shift**, testing spans two distinct layers:
1. **Software / Unit Verification**: Standard unit tests ensuring that data loaders, shift transformations, models, and metric functions behave correctly without runtime crashes.
2. **Experimental Validation Protocol**: Rigorous statistical evaluation across In-Distribution (ID) and Out-of-Distribution (OOD) test splits.

## Experimental Data Partitioning Protocol
- **Source Domain (In-Distribution, $\mathcal{D}_{ID}$)**:
  - First 70% chronological transactions ($\approx 199,365$ samples): **Training Set**
  - Next 15% chronological transactions ($\approx 42,721$ samples): **Calibration & Hyperparameter Set**
  - Final 15% chronological transactions ($\approx 42,721$ samples): **Clean In-Distribution Test Set**
- **Controlled Shift Testbeds ($\mathcal{D}_{OOD}$)**:
  - **Testbed A (Covariate Shift)**: Gaussian noise on features $V_1$–$V_4$ with perturbation scale $\epsilon \in \{0.2, 0.5, 1.0, 2.0\}$.
  - **Testbed B (Subpopulation Shift)**: High-value transaction tail ($>90$th percentile amount).
  - **Testbed C (Prevalence / Label Ratio Shift)**: Fraud rate altered from $0.17\%$ up to $2.5\%$.
  - **Testbed D (Temporal Concept Drift)**: Hold-out slice evaluated sequentially without retraining.

## Metric Verification Suite
- **Discriminative Metrics**:
  - Area Under Precision-Recall Curve (AUPRC) — primary metric due to 0.172% fraud prevalence.
  - Area Under ROC Curve (AUROC).
- **Calibration Metrics**:
  - Expected Calibration Error (ECE) across $K=15$ bins.
  - Brier Score and Negative Log-Likelihood (NLL).
- **Selective Classification Metrics**:
  - Coverage vs. Risk curve across coverage $c \in [0.5, 1.0]$.
  - Area Under Risk-Coverage curve (AURC).
  - Cost-Weighted Loss Function:
    $$L = \$100 \cdot \text{FN}_{\text{accepted}} + \$5 \cdot \text{FP}_{\text{accepted}} + \$2 \cdot N_{\text{deferred}}$$

## Automated Verification Gates
- Unit test coverage for:
  - Dimension consistency of synthetic shift generators.
  - Probability bounds: $0 \le \hat{p}(x) \le 1$ for all calibrated outputs.
  - Coverage monotonicity: Abstention rate strictly increases as threshold $\tau$ becomes more conservative.
