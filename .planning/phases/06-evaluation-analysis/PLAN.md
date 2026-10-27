# Phase 6 Plan: Comprehensive Evaluation & Analysis

## Phase Metadata
- **Phase**: 6
- **Title**: Comprehensive Multi-Model Evaluation & Shift Disentanglement Analysis
- **Type**: Empirical Evaluation & Scientific Diagnostics
- **Status**: Complete
- **Requirements Covered**: REQ-2.4, REQ-3.4, REQ-5.1, REQ-5.2, REQ-5.3

---

## 1. Goal & Objective
Execute comprehensive comparative benchmarks across all four model architectures (Logistic Regression, Random Forest, LightGBM, Tabular MLP) and across all six distribution shift testbeds. Produce cross-model comparative tables, quantify calibration degradation, compute AURC frontiers, prove the shift-risk disentanglement hypothesis, and generate publication-quality figures for the project proposal and presentation poster.

### User Story
> *As an applied machine learning scientist, I need an end-to-end evaluation harness comparing linear, tree ensemble, and deep tabular models across diverse distribution shifts so that the trade-offs in discriminative power, calibration error, selective abstention savings, and OOD disentanglement are rigorously substantiated.*

---

## 2. Technical Scope & Architecture

```
         ┌─────────────────────────────────────────────────────────────┐
         │              [Trained Model Checkpoints]                    │
         │ - Logistic Regression   - Random Forest                     │
         │ - LightGBM              - Tabular MLP (MC Dropout)          │
         └──────────────────────────────┬──────────────────────────────┘
                                        │
         ┌──────────────────────────────┴──────────────────────────────┐
         │              [6 Shifted Evaluation Testbeds]                │
         │ - Clean ID                  - Subpopulation HighAmount      │
         │ - Covariate Mild            - Prevalence Shift              │
         │ - Covariate Severe          - Concept Drift                 │
         └──────────────────────────────┬──────────────────────────────┘
                                        │
                                        ▼
                 [Comprehensive Evaluation Pipeline: evaluate_comprehensive.py]
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
   [Discriminative]              [Calibration]                 [Selective]
   - AUROC                       - Expected Calib Error (ECE)  - Selective Error @ 85%
   - AUPRC                       - Brier Score                 - Operational Loss ($)
   - F1, Precision, Recall       - Reliability Diagrams        - AURC Integration
           │                            │                            │
           └────────────────────────────┼────────────────────────────┘
                                        │
                                        ▼
             [Scientific Diagnostics: Shift Disentanglement]
             - Spearman Corr(Decision Risk, Error)
             - Spearman Corr(Feature OOD Distance, Error)
             - Comparative Scatter & Bar Plots
                                        │
                                        ▼
             [Output Artifacts: results/ & results/figures/]
             - comprehensive_model_comparison.csv
             - comprehensive_shift_resilience.csv
             - disentanglement_analysis.csv
             - 5 Publication-Grade PNG Figures
```

---

## 3. Tasks Breakdown

### Task 1: Comprehensive Multi-Model Benchmarking CLI (`scripts/evaluate_comprehensive.py`)
- Load all 4 model checkpoints from `models/`.
- Load calibrated probabilities and uncertainty calibrators from `models/calibrators/`.
- Ingest in-distribution test set and all 5 shifted test sets.
- Calculate:
  - Discriminative metrics: AUPRC, AUROC, F1, Recall.
  - Calibration metrics: ECE, Brier score.
  - Selective classification metrics: Selective error at $85\%$ coverage, missed fraud count, false alarm count.
  - Economic cost: Baseline full-automation loss vs. selective loss with abstention.
  - AURC (Area Under Risk-Coverage curve).
  - Disentanglement correlations: $\rho(\text{Error}, R_{\text{comp}})$ vs. $\rho(\text{Error}, R_{\text{ood}})$.
- Export results to:
  - `results/comprehensive_model_comparison.csv`
  - `results/comprehensive_shift_resilience.csv`
  - `results/disentanglement_analysis.csv`

### Task 2: Visual Analytics & Figure Generation Engine (`src/evaluation/plots.py`)
- Extend plotting module with:
  1. `plot_multi_model_aurc_comparison`: Bar chart comparing AURC across models and shifts.
  2. `plot_disentanglement_analysis`: Side-by-side bar chart showing Spearman correlation of Decision Risk vs. Feature OOD distance.
  3. `plot_risk_coverage_frontier`: Multi-method trade-off curves.
  4. `plot_calibration_diagram`: Multi-model reliability diagrams.
  5. `plot_shift_resilience_comparison`: Operational loss reduction across shift testbeds.
- Render all figures to `results/figures/` at 300 DPI.

### Task 3: Unit Test Suite (`tests/test_evaluation.py`)
- Verify metric calculations on synthetic boundary cases.
- Verify that Spearman rank correlation functions return values within $[-1, 1]$.
- Verify that figure plotting functions generate non-empty PNG files without errors.
- Confirm all 30+ tests in repository test suite pass.

---

## 4. Verification Loop & Acceptance Criteria

| Check | Target Standard | Verification Check |
| :--- | :--- | :--- |
| **Multi-Model Coverage** | All 4 models evaluated across all 6 testbeds | CSV output validation |
| **Loss Reduction** | Selective abstention reduces cost across all severe shifts | CSV inspection |
| **Disentanglement Proof** | $\rho(\text{Error}, R) > \rho(\text{Error}, \text{OOD})$ holds across shifts | `disentanglement_analysis.csv` |
| **Figure Generation** | 5 high-resolution PNGs generated in `results/figures/` | File existence & size check |
| **Test Suite Pass** | 100% of unit tests pass | `python -m unittest discover tests` |

---

## 5. Next Phase Handoff
Upon completion of Phase 6 verification:
- Proceed to **Phase 7: Final Project Packaging & Poster Delivery** (`/gsd-plan-phase 7`).
