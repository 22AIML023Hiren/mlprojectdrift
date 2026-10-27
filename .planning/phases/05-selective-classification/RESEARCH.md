# Phase 5 Research: Risk-Aware Selective Classification

## 1. Decision-Theoretic Foundations of Selective Classification

In high-stakes automated decision systems (such as financial fraud detection, clinical diagnostics, and autonomous vehicles), an incorrect prediction often incurs orders of magnitude higher cost than deferring the decision to a qualified human operator. Selective classification (Chow, 1970; Geifman & El-Yaniv, 2017) formalizes this paradigm by equipping the classifier with an abstention option.

### 1.1 Formal Problem Definition
Let $(X, Y) \sim \mathcal{P}$ denote a transaction with features $X \in \mathbb{R}^D$ and binary label $Y \in \{0, 1\}$. A selective classifier is a pair $(f, g)$:
1. **Prediction model** $f: \mathcal{X} \to \{0, 1\}$ predicting the target label.
2. **Selection function** $g: \mathcal{X} \to \{0, 1\}$ deciding whether to accept ($g(x) = 1$) or abstain/defer ($g(x) = 0$).

The gated prediction rule is defined as:
$$(f, g)(x) = \begin{cases} f(x) & \text{if } g(x) = 1 \quad (\text{Automated Decision}) \\ \text{ABSTAIN} & \text{if } g(x) = 0 \quad (\text{Route to Human Analyst}) \end{cases}$$

### 1.2 Selection Mechanism via Risk Thresholding
In practice, the binary selection function $g(x)$ is parameterized by a continuous risk function $R: \mathcal{X} \to [0, 1]$ and an abstention threshold $\tau \in [0, 1]$:
$$g_\tau(x) = \mathbb{I}[R(x) \le \tau]$$
where instances with estimated decision risk $R(x) > \tau$ are deferred to the triage queue.

---

## 2. Risk Scoring Formulations

To make risk-aware decisions under distribution shift, we evaluate and contrast four distinct risk scoring formulations:

### 2.1 Confidence Margin Risk ($R_{\text{conf}}$)
Measures aleatoric ambiguity at the model's decision boundary:
$$R_{\text{conf}}(x) = 2 \cdot \left(1 - \max\left(\hat{p}(x), 1 - \hat{p}(x)\right)\right)$$
- If $\hat{p}(x) = 0.5$ (maximal ambiguity), $R_{\text{conf}}(x) = 1.0$.
- If $\hat{p}(x) \in \{0.0, 1.0\}$ (complete certainty), $R_{\text{conf}}(x) = 0.0$.
- *Limitation*: Can be falsely confident on out-of-distribution inputs when uncalibrated.

### 2.2 Epistemic Disagreement Risk ($R_{\text{var}}$)
Measures ensemble model uncertainty (epistemic ignorance) across $M$ estimators:
$$R_{\text{var}}(x) = \frac{\mathcal{U}_{\text{ens}}(x)}{0.25} = \frac{\frac{1}{M}\sum_{m=1}^M \left(\hat{p}_m(x) - \bar{p}(x)\right)^2}{0.25}$$
normalized by $0.25$, which is the maximum theoretical variance of a Bernoulli trial.

### 2.3 Feature-Space Out-Of-Distribution Anomaly ($R_{\text{ood}}$)
Measures density shift $P(X) \ne P_{\text{train}}(X)$ in the raw input space using an in-distribution fitted Isolation Forest:
$$R_{\text{ood}}(x) = \text{MinMaxNorm}\left(-\text{Score}_{\text{IForest}}(x)\right) \in [0, 1]$$
- Captures novel transaction patterns or shifted feature vectors regardless of label association.

### 2.4 Composite Predictive Decision Risk ($R_{\text{comp}}$)
Synthesizes calibrated probability margin and epistemic ensemble disagreement:
$$R_{\text{comp}}(x) = w_{\text{conf}} \cdot R_{\text{conf}}(x) + w_{\text{var}} \cdot R_{\text{var}}(x)$$
with empirical weights $w_{\text{conf}} = 0.60$ and $w_{\text{var}} = 0.40$.

---

## 3. Economic Cost Matrix & Asymmetric Utility

In financial fraud detection, classification errors have starkly asymmetric operational consequences:
- **False Negative ($\text{FN}$)**: A fraudulent transaction goes undetected, resulting in direct financial loss and chargeback fees ($C_{\text{FN}} = \$100$).
- **False Positive ($\text{FP}$)**: A legitimate cardholder's transaction is declined, causing customer friction, brand erosion, and support overhead ($C_{\text{FP}} = \$5$).
- **Deferral / Abstention ($\text{Defer}$)**: An ambiguous transaction is routed to a human fraud analyst for secondary verification ($C_{\text{defer}} = \$2$).

### 3.1 Total Expected Economic Loss
For a test population of $N$ transactions with selection threshold $\tau$:
$$\mathcal{L}(\tau) = C_{\text{FN}} \cdot \sum_{i: g_\tau(x_i)=1} \mathbb{I}[y_i = 1, \hat{y}_i = 0] + C_{\text{FP}} \cdot \sum_{i: g_\tau(x_i)=1} \mathbb{I}[y_i = 0, \hat{y}_i = 1] + C_{\text{defer}} \cdot \sum_{i=1}^N \mathbb{I}[g_\tau(x_i)=0]$$

### 3.2 Operational Constraints
Financial institutions cannot manually review an unlimited volume of transactions; triage analyst capacity imposes an operational coverage constraint:
$$\text{Coverage}(\tau) = \frac{1}{N} \sum_{i=1}^N g_\tau(x_i) \ge c_{\min} \quad (\text{typically } c_{\min} = 80\% \text{ to } 90\%)$$
The optimal threshold $\tau^*$ minimizes $\mathcal{L}(\tau)$ subject to $\text{Coverage}(\tau) \ge c_{\min}$.

---

## 4. Risk-Coverage Frontiers & Area Under Risk-Coverage (AURC)

To evaluate selective classification independent of an ad-hoc operating threshold, we evaluate the full Risk-Coverage trade-off curve across all coverage levels $c \in (0, 1]$:
1. **Coverage**: $c(\tau) = \frac{|\{i : R(x_i) \le \tau\}|}{N}$
2. **Selective Risk / Error**: $e(c) = \frac{\sum_{i: R(x_i) \le \tau} \mathbb{I}[\hat{y}_i \ne y_i]}{|\{i : R(x_i) \le \tau\}|}$
3. **Area Under Risk-Coverage (AURC)**:
   $$\text{AURC} = \int_0^1 e(c) \, dc$$
A lower AURC signifies a superior risk-awareness mechanism that concentrates errors in the highest risk percentiles so they can be pruned early with minimal abstention.

---

## 5. Shift vs. Decision Risk Disentanglement Hypothesis

A central scientific hypothesis of this project is that **feature-space novelty does not equal decision unreliability**:
- A transaction may have unusual feature values (e.g. high dollar amount during holiday season), yet the classifier's posterior certainty $P(Y=0 \mid X)$ remains overwhelmingly confident and accurate.
- Conversely, an in-distribution transaction located directly on the ambiguous boundary between legitimate and fraudulent spending patterns carries high decision risk despite zero feature OOD distance.

Therefore, our benchmark explicitly correlates:
1. $\rho(\text{Error}, R_{\text{comp}})$ (Decision Risk unreliability)
2. $\rho(\text{Error}, R_{\text{ood}})$ (Pure feature-space shift)
We hypothesize $\rho(\text{Error}, R_{\text{comp}}) \gg \rho(\text{Error}, R_{\text{ood}})$.
