# Project Charter: Risk-Aware Machine Learning Under Distribution Shift

## Executive Summary
This project investigates **Risk-Aware Machine Learning Under Distribution Shift** in financial tabular streams (credit card fraud detection and risk underwriting). Standard supervised ML models suffer from high-confidence, silent catastrophic errors when deployment data departs from the training distribution ($\mathcal{P}_{\text{train}} \neq \mathcal{P}_{\text{test}}$). We formulate, build, and benchmark the **Risk-Aware Selective Classification Pipeline (RASCP)**—an end-to-end framework that quantifies epistemic uncertainty, decouples distribution shift from decision risk, and selectively abstains from high-risk predictions by routing them to human investigators while maintaining high automated throughput ($\ge 85\%$).

## Problem Statement
In mission-critical financial applications, models face:
1. **Covariate Shift**: Changes in feature distributions $\mathcal{P}(X)$ while $\mathcal{P}(Y|X)$ remains invariant.
2. **Subpopulation Shift**: Disproportionate fluctuations in specific demographic or transaction strata.
3. **Prevalence Drift**: Surges in attack frequency altering base rate $\mathcal{P}(Y)$.
4. **Concept Drift**: Attacker evolution changing conditional relationship $\mathcal{P}(Y|X)$.

Current practices rely on static decision thresholds ($P > 0.5$) applied to uncalibrated probabilities, lacking self-awareness of epistemic uncertainty. Standard OOD detectors discard data in novel feature spaces without assessing whether the decision boundary itself is at risk.

## Project Scope
- **Domain**: Imbalanced tabular financial transaction data.
- **Primary Benchmark**: Kaggle / ULB Credit Card Fraud Detection dataset (284,807 transactions, 30 features, 0.172% fraud).
- **Core Algorithms**: Logistic Regression, Random Forest, LightGBM, Tabular Multi-Layer Perceptron (PyTorch).
- **Uncertainty Quantification**: Probability Calibration (Isotonic / Platt), Ensemble Variance / Disagreement, Inductive Conformal Prediction.
- **Decision Engine**: Selective classification with risk-gated abstention threshold $\tau$.
- **Deliverables**:
  1. Open, reproducible Python codebase (`src/`).
  2. Tabular distribution shift generation test suite.
  3. Experimental benchmark comparing calibration, ensemble disagreement, and conformal prediction.
  4. Final academic project report and four-quadrant presentation poster (`poster.pdf`).

## Non-Goals
- Real-time distributed stream processing infrastructure (e.g. Apache Kafka / Flink deployment).
- Unsupervised representation learning on unstructured modalities (images/text/audio).
- Production payment gateway integration or commercial deployment.
