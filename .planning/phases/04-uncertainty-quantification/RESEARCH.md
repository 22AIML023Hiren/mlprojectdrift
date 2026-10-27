# Phase 4 Research: Uncertainty Quantification & Calibration Engine

## 1. Theoretical Foundations of Predictive Uncertainty

In safety-critical financial systems, accuracy alone is insufficient. A model must possess reliable self-awareness of its own predictive error likelihood, especially under distribution shift. We decompose predictive uncertainty into two fundamental components:

### 1.1 Aleatoric Uncertainty (Data-Level Ambiguity)
- **Nature**: Irreducible noise inherent in the data-generating process (e.g. overlapping legitimate and fraudulent transaction attributes).
- **Measurement**: Entropy of calibrated predicted probabilities:
  $$\mathcal{H}(p) = -p \log_2(p) - (1 - p) \log_2(1 - p)$$
  Maximally uncertain when $p = 0.5$, minimally uncertain when $p \in \{0, 1\}$.

### 1.2 Epistemic Uncertainty (Model-Level Ignorance)
- **Nature**: Reducible uncertainty arising from lack of knowledge in regions of feature space sparsely sampled during training (e.g. novel transaction channels, shifted spending vectors).
- **Measurement**: Inter-estimator predictive variance across tree ensembles or stochastic neural network passes:
  $$\mathcal{U}_{\text{ens}}(x) = \frac{1}{M} \sum_{m=1}^M \left(\hat{p}_m(x) - \bar{p}(x)\right)^2$$
  where $\hat{p}_m(x)$ is the prediction of tree or dropout draw $m$, and $\bar{p}(x)$ is the ensemble mean.

---

## 2. Probability Calibration Methods

Modern complex classifiers (GBDTs, deep neural nets) do not output true Bayesian posterior probabilities; their raw leaf fractions or uncalibrated softmax logits over-estimate confidence, particularly under extreme class imbalance ($0.172\%$).

### 2.1 Platt Scaling (Sigmoid Fitting)
- Fits a 1D logistic regression over raw uncalibrated logits $f(x)$:
  $$\hat{P}_{\text{Platt}}(Y = 1 \mid x) = \frac{1}{1 + \exp(A \cdot f(x) + B)}$$
- Parameters $A, B$ are optimized via maximum likelihood on a dedicated calibration holdout split $\mathcal{D}_{\text{calib}}$.
- **Pros**: Parameter-efficient, robust against overfitting on small minority sample counts.
- **Cons**: Rigid parametric assumption (strictly sigmoidal).

### 2.2 Isotonic Regression (Non-Parametric Piecewise Monotonic Fit)
- Fits a non-parametric isotonic (monotonically non-decreasing) step function $m$:
  $$\min_m \sum_{i \in \mathcal{D}_{\text{calib}}} \left(y_i - m(f(x_i))\right)^2 \quad \text{subject to } m(f_i) \le m(f_j) \text{ whenever } f_i \le f_j$$
  Solved efficiently via the Pair Adjacent Violators (PAV) algorithm in $\mathcal{O}(N)$.
- **Pros**: Can correct arbitrary non-linear calibration distortions without distributional assumptions.
- **Cons**: Can overfit if the calibration set has too few positive fraud samples ($<15$).

---

## 3. Inductive Conformal Prediction (Distribution-Free Guarantees)

Conformal Prediction (Vovk et al., 2005; Angelopoulos & Bates, 2021) provides finite-sample statistical validity without assuming any parametric model distribution.

### 3.1 Mathematical Guarantee
Given an exchangeable calibration set and a new query sample $X_{n+1}$, conformal prediction constructs a prediction set $\Gamma^\alpha(X_{n+1}) \subseteq \{0, 1\}$ satisfying:
$$\mathbb{P}\left(Y_{n+1} \in \Gamma^\alpha(X_{n+1})\right) \ge 1 - \alpha$$
for any user-chosen significance level $\alpha \in (0, 1)$ (e.g. $\alpha = 0.05 \implies 95\%$ guaranteed coverage).

### 3.2 Non-Conformity Scoring
We define non-conformity score as the error likelihood under model prediction:
$$s_i = 1 - \hat{P}(Y = y_i \mid x_i)$$
On $\mathcal{D}_{\text{calib}}$ of size $N_{\text{calib}}$, we compute the empirical $(1 - \alpha)$ quantile with finite-sample correction:
$$\hat{q} = \text{Quantile}\left(\{s_1, \dots, s_{N_{\text{calib}}}\}, \frac{\lceil (N_{\text{calib}} + 1)(1 - \alpha) \rceil}{N_{\text{calib}}}\right)$$

### 3.3 Set Cardinality as Risk Indicator
- $|\Gamma^\alpha(x)| = 1$ ($\{0\}$ or $\{1\}$): Confident prediction $\implies$ Safe to automate.
- $|\Gamma^\alpha(x)| = 2$ ($\{0, 1\}$): High epistemic ambiguity (model cannot rule out fraud) $\implies$ **Trigger selective abstention & human review**.
- $|\Gamma^\alpha(x)| = 0$ ($\emptyset$): Extreme out-of-distribution anomaly $\implies$ Flag as anomalous event.
