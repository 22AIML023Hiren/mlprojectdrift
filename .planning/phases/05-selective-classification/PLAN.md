# Phase 5 Plan: Risk-Aware Selective Classification

## Phase Metadata
- **Phase**: 5
- **Title**: Risk-Aware Selective Classification & Cost Optimization
- **Type**: Decision Engineering & Empirical Benchmarking
- **Status**: Complete
- **Requirements Covered**: REQ-4.1, REQ-4.2, REQ-4.3, REQ-4.4, REQ-5.2, REQ-5.3

---

## 1. Goal & Objective
Construct, benchmark, and validate a risk-gated selective classification engine that routes unreliable predictions under distribution shift to human fraud triage. The engine must evaluate multiple risk scoring mechanisms ($R_{\text{conf}}, R_{\text{calib}}, R_{\text{ood}}, R_{\text{comp}}$), construct Risk-Coverage frontiers, compute AURC, and demonstrate a $\ge 50\%$ reduction in operational economic loss under realistic automation constraints ($\ge 85\%$ coverage).

### User Story
> *As a risk operations manager, I need the automated fraud detection system to selectively abstain and route ambiguous or out-of-distribution transactions to human investigators so that catastrophic missed frauds are prevented while maintaining at least 85% automated straight-through processing.*

---

## 2. Technical Scope & Architecture

```
                    [Query Transaction: x]
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
 [Model Prediction f(x)]                [Uncertainty Estimators]
 - Calibrated Prob P(Y=1|x)             - Aleatoric Risk: R_conf(x)
 - Baseline Prediction y_hat            - Epistemic Variance: Var(p_m)
            │                           - Feature OOD: IsolationForest
            │                                     │
            │                                     ▼
            │                           [Composite Risk Engine]
            │                           R(x) = 0.6*R_conf + 0.4*R_var
            │                                     │
            └──────────────────┬──────────────────┘
                               ▼
                    [Risk Gate: R(x) <= tau?]
                               │
                ┌──────────────┴──────────────┐
                │ YES                         │ NO
                ▼                             ▼
       [Automate Decision]           [Abstain / Deferral]
       - Deploy prediction y_hat     - Route to Fraud Analyst
       - Automated clearance         - Human secondary review
       - Cost: FN=$100, FP=$5        - Cost: Deferral=$2
```

---

## 3. Tasks Breakdown

### Task 1: Comprehensive Selective Decision Runner (`scripts/run_selective_decision.py`)
- Load baseline models and calibrators from Phase 3 & 4.
- Ingest all 6 shift testbeds from `data/shifted/`.
- Evaluate selective classification performance across coverage levels:
  $$c \in \{0.70, 0.80, 0.85, 0.90, 0.95, 1.00\}$$
- Compare 4 risk scoring strategies:
  1. **Raw Confidence Margin** (uncalibrated)
  2. **Calibrated Confidence Margin** (Isotonic)
  3. **Feature OOD Anomaly Distance** (Isolation Forest)
  4. **Composite Decision Risk** (Calibrated Margin + Ensemble Variance)
- Compute:
  - Selective error rate on accepted transactions
  - Missed fraud count (FN) and false alarm count (FP)
  - Number of deferred cases ($N_{\text{def}}$)
  - Total operational economic loss ($\$)
  - Area Under Risk-Coverage curve (AURC)
- Export structured benchmarks to:
  - `results/selective_decision_benchmark.csv`
  - `results/selective_shift_comparison.csv`

### Task 2: Disentanglement Correlation Analysis
- Compute Spearman rank correlation between predictive classification errors and:
  - Composite Decision Risk $R_{\text{comp}}$
  - Feature-space OOD Anomaly Distance $R_{\text{ood}}$
- Validate the central thesis: decision risk unreliability outperforms naive feature novelty detection.

### Task 3: Unit Test Suite (`tests/test_selective.py`)
- Implement strict verification tests:
  - `test_coverage_monotonicity`: $\tau_1 < \tau_2 \implies \text{Coverage}(\tau_1) \le \text{Coverage}(\tau_2)$.
  - `test_selective_error_reduction`: Selective error at $c = 85\%$ is strictly lower than unselective error at $c = 100\%$.
  - `test_economic_loss_exactness`: Loss formula exactness across edge cases ($c = 1.0$, $c = 0.0$).
  - `test_disentanglement_advantage`: $\rho(\text{Error}, R_{\text{comp}}) > \rho(\text{Error}, R_{\text{ood}})$.
  - `test_extreme_thresholds`: Proper graceful handling when threshold accepts all or abstains on all.

---

## 4. Verification Loop & Acceptance Criteria

| Verification Gate | Target Standard | Verification Check |
| :--- | :--- | :--- |
| **Loss Mitigation** | $\ge 50\%$ reduction in economic loss at $85\%$ coverage vs. full automation | CSV & script evaluation |
| **Missed Fraud Reduction** | Catastrophic missed fraud count reduced by $\ge 60\%$ | CSV comparison |
| **AURC Superiority** | Composite Risk achieves lower AURC than raw confidence and OOD distance | CSV comparison |
| **Disentanglement Proof** | $\rho(\text{Error}, R_{\text{comp}}) > \rho(\text{Error}, R_{\text{ood}})$ across shift testbeds | Spearman correlation |
| **Test Suite Pass** | 100% of unit tests pass | `python -m unittest tests/test_selective.py` |

---

## 5. Next Phase Handoff
Upon completion of Phase 5 verification:
- Proceed to **Phase 6: Comprehensive Evaluation & Analysis** (`/gsd-plan-phase 6`).
