# Phase 3 Research: Baseline Classifiers Training

## 1. Model Family Analysis on Imbalanced Tabular Streams

Evaluating risk-aware machine learning under distribution shift requires diverse model families spanning linear, bagging, boosting, and deep neural paradigms. Each family exhibits distinct inductive biases and vulnerability profiles under shift:

### 1.1 Logistic Regression (Linear Paradigm)
- **Inductive Bias**: Linear decision hyperplane in feature space $\mathcal{X}$.
- **Loss Function**: Weighted negative log-likelihood (binary cross-entropy):
  $$\mathcal{L}_{\text{LR}}(\theta) = -\sum_{i=1}^N \left[ w_1 y_i \ln \sigma(\theta^T x_i) + w_0 (1 - y_i) \ln (1 - \sigma(\theta^T x_i)) \right]$$
- **Behavior Under Shift**: High bias, low variance. When distributions shift along linear axes, degradation is predictable. However, it fails to capture non-linear transaction feature interactions.
- **Role in Suite**: The baseline against which non-linear gains are measured.

### 1.2 Random Forest (Bagging Paradigm)
- **Inductive Bias**: Orthogonal axis-aligned hyperplanes averaged across $M$ decorrelated decision trees.
- **Uncertainty Property**: Directly exposes inter-tree variance:
  $$\hat{p}_m(x) = \text{Tree}_m(x), \quad \text{Var}_{\text{trees}}(x) = \frac{1}{M}\sum_{m=1}^M \left(\hat{p}_m(x) - \bar{p}(x)\right)^2$$
- **Behavior Under Shift**: Highly robust to feature noise. In regions devoid of training data, individual trees produce decorrelated extrapolations, causing tree disagreement to surge—a natural epistemic uncertainty signal.

### 1.3 LightGBM (Gradient Boosting Paradigm)
- **Inductive Bias**: Sequential gradient-boosted decision trees grown leaf-wise (best-first).
- **Optimization**: Second-order Taylor approximation of loss with histogram-based feature binning.
- **Behavior Under Shift**: Exceptional in-distribution discriminative power (high AUPRC), but notoriously prone to **overconfident extrapolation** out-of-distribution. Tree leaves assign extreme probabilities to distant points based on single threshold rules.
- **Role in Suite**: SOTA tabular predictor demonstrating why calibration and risk-gating are essential.

### 1.4 Tabular Multi-Layer Perceptron (Deep Learning Paradigm)
- **Inductive Bias**: Continuous non-linear feature representations with Layer Normalization and residual connections.
- **Uncertainty Property**: **Monte Carlo Dropout** (Gal & Ghahramani, 2016). Keeping dropout active during inference performs approximate Bayesian variational inference:
  $$p(y = 1 \mid x, \mathcal{D}) \approx \frac{1}{T} \sum_{t=1}^T \text{Softmax}\left(f^{\hat{W}_t}(x)\right)$$
- **Behavior Under Shift**: Continuous representation space provides smoother uncertainty degradation than step-function tree boundaries.

---

## 2. Managing Extreme Class Imbalance ($0.172\%$ Fraud)

- **Weight Scaling**: Unweighted training on 99.828% negatives causes models to converge to the trivial all-zero predictor. We scale positive class weight:
  $$w_1 = \min\left(\frac{N_{\text{neg}}}{N_{\text{pos}}}, 50.0\right)$$
  (Capped at $50.0$ to avoid excessive logit distortion that destabilizes post-hoc calibration).
- **Evaluation Priority**: Classification accuracy is discarded in favor of:
  - **AUPRC (Area Under Precision-Recall Curve)**: Measures precision across varying recall thresholds on the minority positive class.
  - **AUROC (Area Under ROC Curve)**: Measures ranking quality across discrimination thresholds.

---

## 3. Serialization & Persistence Standards

- Scikit-learn & LightGBM models: Serialized via `joblib` with compression:
  - `models/logistic_regression.joblib`
  - `models/random_forest.joblib`
  - `models/lightgbm.joblib`
- PyTorch Tabular MLP: Serialized model weights (`state_dict`) and architecture config:
  - `models/tabular_mlp.pt`
