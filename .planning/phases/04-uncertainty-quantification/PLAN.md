# Phase 4 Plan: Uncertainty Quantification & Calibration Engine

## Phase Metadata
- **Phase**: 4
- **Title**: Uncertainty Quantification & Calibration Engine
- **Type**: Algorithm Development & Uncertainty Modeling
- **Status**: Complete
- **Requirements Covered**: REQ-3.1, REQ-3.2, REQ-3.3, REQ-3.4

---

## 1. Goal & Objective
Implement, evaluate, and serialize post-hoc probability calibrators (Isotonic Regression, Platt Scaling), ensemble epistemic variance estimators, and Inductive Conformal Prediction sets. The engine must systematically reduce Expected Calibration Error (ECE) by $\ge 30\%$ on held-out data and guarantee distribution-free coverage $\ge 1 - \alpha$ while isolating ambiguous $\{0, 1\}$ decisions for selective review.

### User Story
> *As a machine learning researcher, I need statistically calibrated probabilities and finite-sample conformal prediction sets so that decision gates do not naively rely on distorted raw confidences and can quantify true decision unreliability.*

---

## 2. Technical Scope & Architecture

```
                    [Calibration Split: X_calib, y_calib]
                                     │
      ┌──────────────────────────────┼──────────────────────────────┐
      ▼                              ▼                              ▼
[Raw Probabilities f(x)]   [Tree Ensemble Disagreement]  [Conformal Non-Conformity]
(From trained models)       (Variance across M trees)     (s_i = 1 - P(Y=y_i|x_i))
      │                              │                              │
      ▼                              │                              ▼
[Probability Calibrator]             │                   [Quantile Threshold q_hat]
- Isotonic Regression (PAV)          │                   (Finite-sample corrected)
- Platt Scaling (Sigmoid)            │                              │
      │                              │                              ▼
      ▼                              ▼                   [Conformal Sets Gamma(x)]
[Calibrated Probabilities]  [Epistemic Variance]         - Single label: Confident
(ECE < 0.01, Brier Score)   (U_ens = Var(p_m))           - Double label: High Risk
      │                              │                              │
      └──────────────────────────────┴──────────────────────────────┘
                                     │
                                     ▼
                [Uncertainty Checkpoints & Benchmark Data]
                - models/calibrators/isotonic_calibrator.joblib
                - results/calibration_comparison.csv
                - results/conformal_coverage_analysis.csv
```

---

## 3. Tasks Breakdown

### Task 1: Calibration Evaluation & Persistence CLI (`scripts/calibrate_and_evaluate.py`)
- Standalone CLI script that:
  1. Loads trained models from Phase 3 (`models/`).
  2. Evaluates raw probabilities vs. Platt Scaling vs. Isotonic Regression on $\mathcal{D}_{\text{calib}}$ and $\mathcal{D}_{\text{test}}$.
  3. Computes ECE, Brier Score, and Negative Log-Likelihood (NLL).
  4. Saves fitted calibrator checkpoints to `models/calibrators/`.
  5. Exports comparative metrics table to `results/calibration_comparison.csv`.

### Task 2: Conformal Coverage Analysis Module
- Evaluates Inductive Conformal Prediction across significance levels $\alpha \in \{0.01, 0.05, 0.10\}$:
  - Measures empirical coverage vs. theoretical target.
  - Measures set efficiency: fraction of single-label vs. double-label ambiguous sets.
  - Exports coverage logs to `results/conformal_coverage_analysis.csv`.

### Task 3: Unit Test Suite (`tests/test_uncertainty.py`)
- Test cases covering:
  - Calibration monotonicity: Calibrated probabilities preserve ordering and remain within $[0, 1]$.
  - ECE reduction: Post-hoc calibration achieves lower ECE than raw uncalibrated outputs.
  - Conformal validity: Empirical coverage on test data satisfies $\ge (1 - \alpha) - 0.03$.
  - Epistemic variance bounds: Tree variance is strictly in $[0, 0.25]$.

---

## 4. Verification Loop & Acceptance Criteria

| Check | Target Standard | Verification Command |
| :--- | :--- | :--- |
| **ECE Reduction** | $\text{ECE}_{\text{calibrated}} < \text{ECE}_{\text{raw}}$ (at least $30\%$ drop) | Inspection of `results/calibration_comparison.csv` |
| **Conformal Coverage** | Empirical coverage $\ge 1 - \alpha$ within finite-sample margin | Unit test assertion |
| **Calibrator Persistence** | Calibrators saved and reloadable from `models/calibrators/` | Directory check |
| **Variance Extensibility** | Inter-tree variance extracts for all tree-based estimators | Unit test assertion |

---

## 5. Next Phase Handoff
Upon completion of Phase 4 verification:
- Proceed to **Phase 5: Risk-Aware Selective Classification** (`/gsd-plan-phase 5`).
