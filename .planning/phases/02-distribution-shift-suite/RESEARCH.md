# Phase 2 Research: Controlled Distribution Shift Suite

## 1. Theoretical Formulations of Tabular Distribution Shift

Supervised learning assumes the joint distribution $\mathcal{P}(X, Y)$ is invariant between training and testing. In financial tabular streams, real-world distribution shift decomposes into four distinct structural modes:

### 1.1 Mode A: Covariate Shift
- **Definition**: $\mathcal{P}_{\text{test}}(X) \neq \mathcal{P}_{\text{train}}(X)$ while the conditional ground-truth mapping remains unchanged:
  $$\mathcal{P}_{\text{test}}(Y \mid X) = \mathcal{P}_{\text{train}}(Y \mid X)$$
- **Financial Manifestation**: Changes in consumer spending behavior, payment terminals, or seasonal transaction frequency shifts.
- **Mathematical Simulation**: Additive Gaussian perturbation on sensitive PCA components with variance scaling:
  $$X_{\text{shifted}} = X + \mathcal{N}\left(0, \sigma^2 \cdot \text{diag}(\text{Var}(X))\right)$$
  where $\sigma \in \{0.2, 0.5, 1.0, 2.0\}$ controls the perturbation severity.

### 1.2 Mode B: Subpopulation Shift
- **Definition**: The data-generating distribution is a mixture of latent sub-groups $\mathcal{Z}$. The relative mixing weights of these groups shift between train and test:
  $$\mathcal{P}_{\text{test}}(X, Y) = \sum_{z} \mathcal{P}_{\text{test}}(z) \mathcal{P}(X, Y \mid z)$$
- **Financial Manifestation**: High-value transactions ($>85$th percentile amount) where fraud rates, velocity, and merchant categories diverge sharply from standard micro-transactions.
- **Simulation**: Stratified quantile filtering on `Amount` ($q \ge 0.85$), measuring degradation on tail demographics.

### 1.3 Mode C: Label / Prevalence Shift
- **Definition**: The prior probability of fraud shifts while class-conditional feature distributions remain invariant:
  $$\mathcal{P}_{\text{test}}(Y) \neq \mathcal{P}_{\text{train}}(Y), \quad \mathcal{P}_{\text{test}}(X \mid Y) = \mathcal{P}_{\text{train}}(X \mid Y)$$
- **Financial Manifestation**: Sudden coordinated attack waves (e.g. holiday shopping blitzes, compromised credential credential stuffing).
- **Simulation**: Controlled resampling adjusting fraud incidence from baseline $0.172\%$ up to $2.5\%$ or $5.0\%$.

### 1.4 Mode D: Concept Drift
- **Definition**: The feature-to-label relationship changes:
  $$\mathcal{P}_{\text{test}}(Y \mid X) \neq \mathcal{P}_{\text{train}}(Y \mid X)$$
- **Financial Manifestation**: Adversaries actively evading detection models by rotating behavioral features in sensitive principal components ($V_1, V_2, V_10, V_14$).
- **Simulation**: Inverting feature signs and injecting targeted directional offsets specifically on positive class samples ($Y = 1$).

---

## 2. Quantitative Verification of Distribution Shift

To confirm that shifts are genuine, reproducible, and mathematically verified, we employ two standard statistical distance metrics:
1. **Wasserstein Distance (Earth Mover's Distance)**:
   $$W_1(u, v) = \int_{-\infty}^\infty |U(x) - V(x)| dx$$
   Measures the minimum work required to transform the in-distribution marginal $P_{\text{train}}(X_j)$ into the shifted marginal $P_{\text{test}}(X_j)$.
2. **Kolmogorov-Smirnov (KS) Test**:
   Two-sample KS statistic measuring maximum divergence between cumulative empirical distributions $F_1(x)$ and $F_2(x)$:
   $$D = \sup_x |F_{\text{train}}(x) - F_{\text{shifted}}(x)|$$
   $p$-value $< 0.01$ provides rigorous statistical rejection of the null hypothesis that distributions are identical.
3. **Population Stability Index (PSI)**: Standard banking risk metric:
   $$\text{PSI} = \sum_{b=1}^B (P_b - Q_b) \ln(P_b / Q_b)$$
   $\text{PSI} > 0.25$ indicates significant distributional change requiring model governance action.
