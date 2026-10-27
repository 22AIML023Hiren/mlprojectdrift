# Phase 1 Research: Data Pipeline & Temporal Partitioning

## 1. Domain Context: Kaggle ULB Credit Card Fraud Benchmark
- **Source**: Machine Learning Group (MLG) at Université Libre de Bruxelles (ULB) & Worldline.
- **Dataset Scale**: 284,807 transactions recorded across 2 days (48 hours = 172,800 seconds).
- **Target Skew**: 492 fraud cases (0.1727%) vs. 284,315 legitimate transactions (99.8273%).
- **Feature Set**:
  - `Time`: Monotonically non-decreasing integer representing seconds elapsed from first event.
  - `Amount`: Highly skewed transaction amounts (median ~ €22, mean ~ €88, max ~ €25,691).
  - `V1`–`V28`: PCA-transformed continuous components with zero mean and orthogonal variance.
  - `Class`: 0 (Legitimate), 1 (Fraud).

## 2. Critical Methodological Considerations

### 2.1 Why Random K-Fold Cross-Validation is Catastrophic
In sequential financial transactions, random shuffling leaks future information into past training instances. Fraud rings evolve rapidly; models trained on randomly leaked future fraud patterns achieve artificially high accuracy that collapses in production.
- **Decision**: Enforce strict chronological splitting by sorting on `Time`.

### 2.2 Preventing Preprocessing Data Leakage
Applying scalers or imputation across the entire dataset before splitting leaks distribution parameters ($\mu, \sigma$, quantiles) from the test set into the training pipeline.
- **Decision**: Fit `RobustScaler` strictly on `X_train` (`Time` and `Amount`). Apply `.transform()` to `X_calib` and `X_test`.

### 2.3 Partition Proportions
- **Training Set (70%)**: $\approx 199,365$ samples. Large enough for high-capacity gradient boosted trees and neural networks.
- **Calibration Split (15%)**: $\approx 42,721$ samples. Essential for post-hoc calibration (Isotonic Regression) and Inductive Conformal Prediction non-conformity quantiles without overfitting to train data.
- **In-Distribution Test Set (15%)**: $\approx 42,721$ samples. Pure holdout for benchmarking clean performance before stress-testing under distribution shifts.

### 2.4 Data Access Strategy
1. Primary: Local `data/raw/creditcard.csv` if downloaded via Kaggle CLI / web.
2. Automated Utility: Script `scripts/download_data.py` providing Kaggle CLI download or OpenML / Direct mirror download.
3. Fallback: High-fidelity synthetic benchmark generator (`generate_synthetic_benchmark()`) producing identical schema and statistical moments for immediate offline development.
