# Phase 6 Research: Comprehensive Evaluation & Analysis

## 1. Multi-Dimensional Evaluation Under Distribution Shift

Evaluating machine learning systems under distribution shift requires multi-dimensional diagnostic criteria beyond traditional accuracy or in-distribution F1-score. A robust model must simultaneously maintain:
1. **Discriminative Ordering**: Separating fraud from non-fraud even when input features or class ratios shift.
2. **Calibration Fidelity**: Confidence estimates that reflect true posterior error likelihoods rather than over-confident distortions.
3. **Selective Efficiency**: Minimizing AURC by selectively deferring ambiguous transactions to human review.
4. **Economic Resilience**: Minimizing asymmetric operational losses under strict automation constraints.
5. **Disentanglement Clarity**: Distinguishing between benign feature novelty and genuine predictive risk.

---

## 2. Metric Specifications

### 2.1 Area Under the Precision-Recall Curve (AUPRC)
Under extreme class imbalance ($0.17\%$ fraud rate), AUROC can be deceptively optimistic because a large volume of true negatives inflates specificity ($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} \approx 0$). In contrast, Precision-Recall curves focus strictly on the minority class:
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
AUPRC is the benchmark metric for fraud detection ranking quality.

### 2.2 Expected Calibration Error (ECE) & Brier Score
Given $K=15$ empirical probability bins $B_k \subset [0, 1]$:
$$\text{ECE} = \sum_{k=1}^K \frac{|B_k|}{N} \left| \text{acc}(B_k) - \text{conf}(B_k) \right|$$
where $\text{acc}(B_k)$ is the empirical fraud frequency and $\text{conf}(B_k)$ is the mean predicted confidence in bin $k$.
The Brier score provides the mean squared probability error:
$$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$

### 2.3 Area Under Risk-Coverage (AURC)
Measures the selective classifier's average error rate across all straight-through processing rates $c \in (0, 1]$:
$$\text{AURC} = \int_0^1 e(c) \, dc$$
where $e(c)$ is the error rate on the accepted fraction $c$. Lower AURC indicates that the model's risk ranking isolates errors into the highest risk percentiles, allowing maximal throughput with minimal error.

### 2.4 Economic Cost-Weighted Loss Function
Incorporates direct dollar costs:
$$\mathcal{L} = \$100 \cdot \text{FN}_{\text{accepted}} + \$5 \cdot \text{FP}_{\text{accepted}} + \$2 \cdot N_{\text{deferred}}$$
This penalizes missed fraud 20x more heavily than false alarms and 50x more heavily than analyst review.

---

## 3. Disentanglement Hypothesis Formalization

A key thesis of risk-aware machine learning is that **feature-space novelty ($P(X)$ shift) does not equal decision risk ($P(Y|X)$ error)**:
- Consider a benign subpopulation shift where transaction amounts increase due to inflation or seasonal shopping. An Isolation Forest or Mahalanobis distance detector flags these samples as severe OOD outliers ($R_{\text{ood}} \to 1.0$).
- However, if the classifier's posterior certainty $P(Y=0 \mid X)$ remains overwhelming ($>0.999$), routing these transactions to human review is an operational waste ($N_{\text{def}} \uparrow$, costing $\$2$ each).
- Conversely, a transaction situated directly on the ambiguous decision boundary within the source distribution carries high predictive risk ($R_{\text{comp}} \to 1.0$), yet exhibits zero feature OOD distance ($R_{\text{ood}} \approx 0$).

We evaluate this hypothesis across all shift testbeds by computing Spearman rank correlations:
$$\rho_{\text{comp}} = \text{SpearmanCorr}(R_{\text{comp}}(x), \mathbb{I}[y \ne \hat{y}])$$
$$\rho_{\text{ood}} = \text{SpearmanCorr}(R_{\text{ood}}(x), \mathbb{I}[y \ne \hat{y}])$$
Expected finding: $\rho_{\text{comp}} \gg \rho_{\text{ood}}$ across all non-trivial shifts.
