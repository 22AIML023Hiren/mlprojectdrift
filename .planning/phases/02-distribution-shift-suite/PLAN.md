# Phase 2 Plan: Controlled Distribution Shift Suite

## Phase Metadata
- **Phase**: 2
- **Title**: Controlled Distribution Shift Suite
- **Type**: Data Engineering & Experimental Testbeds
- **Status**: In Progress
- **Requirements Covered**: REQ-1.3

---

## 1. Goal & Objective
Construct, mathematically verify, and serialize a multi-axis **Controlled Distribution Shift Benchmark Suite** for tabular fraud detection. The suite must generate 4 distinct shift paradigms (Covariate Shift, Subpopulation Shift, Prevalence Shift, and Concept Drift) with rigorous statistical distance proofs (Wasserstein Distance, KS-Test $p$-values, and PSI).

### User Story
> *As a machine learning researcher, I need parameterized, reproducible out-of-distribution test sets accompanied by statistical divergence proofs so that I can evaluate how model calibration, uncertainty estimation, and selective classification behave under controlled forms of real-world drift.*

---

## 2. Technical Scope & Architecture

```
                 [In-Distribution Test: X_test, y_test]
                                  │
                                  ▼
                     [ShiftGenerator Pipeline]
                                  │
       ┌──────────────────────────┼──────────────────────────┐
       ▼                          ▼                          ▼
[Covariate Shift]      [Subpopulation Shift]      [Prevalence / Label]
(Gaussian noise sigma) (Tail amount strata)        (Resample fraud rate)
       │                          │                          │
       └──────────────────────────┼──────────────────────────┘
                                  ▼
                           [Concept Drift]
                         (Adversarial evasion)
                                  │
                                  ▼
               [Statistical Divergence Verification]
               - Wasserstein Distance (W1)
               - Kolmogorov-Smirnov (KS-test D and p-val)
               - Population Stability Index (PSI)
                                  │
                                  ▼
                 [data/shifted/ Partition Export]
                 - Clean_ID.csv
                 - Covariate_Mild.csv & Covariate_Severe.csv
                 - Subpopulation_HighAmount.csv
                 - Prevalence_Shift.csv
                 - Concept_Drift.csv
                 - manifest.json (with metadata & divergence stats)
```

---

## 3. Tasks Breakdown

### Task 1: Refined Parametric Shift Engine (`src/data/shifts.py`)
- Enhance `ShiftGenerator` with configurable severity sweeps:
  - Multi-level covariate noise: $\sigma \in \{0.2, 0.5, 1.0, 2.0\}$.
  - Multi-threshold subpopulation filtering ($q \in \{0.80, 0.85, 0.90\}$).
  - Target prevalence rates ($1.0\%, 2.5\%, 5.0\%$).
  - Target concept drift coefficients.

### Task 2: Shift Distance & Divergence Metric Engine (`src/data/shift_metrics.py`)
- Implement `compute_distribution_divergence(X_ref, X_shift, target_cols)`:
  - Wasserstein Distance ($W_1$) per continuous feature.
  - Kolmogorov-Smirnov test statistic ($D$) and asymptotic $p$-value.
  - Population Stability Index (PSI) across 10 quantile bins.
  - Mean multivariate shift score.

### Task 3: Dataset Serialization CLI & Manifest (`scripts/generate_shifts.py`)
- CLI script to pre-generate and save all testbeds to `data/shifted/`:
  - Serializes CSV files for each shifted testbed.
  - Writes `data/shifted/manifest.json` containing sample counts, positive ratios, and statistical divergence metrics.

### Task 4: Unit Test Suite (`tests/test_shifts.py`)
- Test cases covering:
  - KS test confirms statistically significant shift ($p < 0.01$) on perturbed features.
  - Feature dimensions and column names remain strictly identical to input.
  - Subpopulation filtering retains correct upper quantile threshold.
  - Prevalence shift achieves exact target positive proportion within rounding tolerance.

---

## 4. Verification Loop & Acceptance Criteria

| Check | Target Standard | Verification Command |
| :--- | :--- | :--- |
| **Statistical Divergence** | $p$-value $< 10^{-5}$ on perturbed features for Covariate Shift | `python tests/test_shifts.py` |
| **Prevalence Precision** | Target fraud rate matches requested value ($\pm 0.1\%$) | Unit test assertion |
| **Schema Invariance** | Column names and feature counts strictly match clean test set | Unit test assertion |
| **Artifact Generation** | `data/shifted/` contains all CSVs and `manifest.json` | `python scripts/generate_shifts.py` |

---

## 5. Next Phase Handoff
Upon completion of Phase 2 verification:
- Proceed to **Phase 3: Baseline Classifiers Training** (`/gsd-plan-phase 3`).
