# Phase 3 Plan: Baseline Classifiers Training

## Phase Metadata
- **Phase**: 3
- **Title**: Baseline Classifiers Training & Model Serialization
- **Type**: Model Development & Benchmarking
- **Status**: In Progress
- **Requirements Covered**: REQ-2.1, REQ-2.2, REQ-2.3, REQ-2.4

---

## 1. Goal & Objective
Train, evaluate, and serialize four distinct baseline classifier families (Logistic Regression, Random Forest, LightGBM, and Tabular MLP) on the in-distribution training partition ($X_{\text{train}}, y_{\text{train}}$). Establish clean in-distribution performance benchmarks (AUROC, AUPRC, F1, Brier Score, and ECE) and serialize trained model checkpoints for downstream uncertainty calibration and selective classification.

### User Story
> *As a machine learning researcher, I need diverse, well-tuned, and persistent baseline models so that I can systematically compare how linear, bagging, boosting, and deep neural architectures respond to distribution shifts and uncertainty-guided abstention.*

---

## 2. Technical Scope & Architecture

```
       [Source Domain: X_train, y_train (70% chronological)]
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
 [Logistic Regression]   [Random Forest]           [LightGBM]
 (ElasticNet / Balanced) (100 Trees / Balanced)   (Leaf-wise GBDT)
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
                       [Tabular MLP (PyTorch)]
                       (LayerNorm, Residuals, MC-Dropout)
                                │
                                ▼
            [Checkpoint Persistence: models/ directory]
            - logistic_regression.joblib
            - random_forest.joblib
            - lightgbm.joblib
            - tabular_mlp.pt
                                │
                                ▼
         [Clean In-Distribution Evaluation (X_test, y_test)]
         - AUROC, AUPRC, F1-Score, Brier Score, Raw ECE
         - Output: results/in_distribution_baselines.csv
```

---

## 3. Tasks Breakdown

### Task 1: Checkpoint Serialization & Model Persistence
- Implement serialization and loading utilities in `src/models/`:
  - `save_model(model, path)` supporting both Scikit-learn/LightGBM (via `joblib`) and PyTorch (`torch.save`).
  - `load_model(model_name, path)`.

### Task 2: Dedicated Model Training CLI (`scripts/train_models.py`)
- Standalone CLI script that:
  1. Loads dataset and chronological splits.
  2. Trains all 4 model families with deterministic random seeds.
  3. Serializes checkpoints to `models/`.
  4. Evaluates clean in-distribution test performance.
  5. Exports comparative metrics table to `results/in_distribution_baselines.csv`.

### Task 3: Unit Test Suite (`tests/test_models.py`)
- Test cases covering:
  - Scikit-learn estimator API conformity (`fit`, `predict_proba`, `predict`).
  - Probability constraints: $P(Y=0) + P(Y=1) = 1.0$ and $0 \le P \le 1$.
  - Random Forest tree-level prediction extraction (`get_individual_tree_predictions` returns shape `[n_trees, n_samples]`).
  - Tabular MLP Monte Carlo Dropout produces non-zero epistemic variance on test samples.
  - Discriminative power: All models achieve in-distribution $\text{AUROC} > 0.85$ and $\text{AUPRC} > 0.50$.

---

## 4. Verification Loop & Acceptance Criteria

| Check | Target Standard | Verification Command |
| :--- | :--- | :--- |
| **Model Convergence** | All 4 models train without error or NaN gradients | `python scripts/train_models.py` |
| **Discriminative Baseline** | $\text{AUROC} \ge 0.85$ and $\text{AUPRC} \ge 0.50$ on clean ID test | Inspection of `results/in_distribution_baselines.csv` |
| **Tree Variance Interface** | `get_individual_tree_predictions` returns array with variance $> 0$ | Unit test assertion |
| **MC-Dropout Interface** | `mc_dropout_predict` returns variance $> 0$ | Unit test assertion |
| **Model Persistence** | All 4 checkpoints saved to `models/` directory | Directory check |

---

## 5. Next Phase Handoff
Upon completion of Phase 3 verification:
- Proceed to **Phase 4: Uncertainty Quantification & Calibration Engine** (`/gsd-plan-phase 4`).
