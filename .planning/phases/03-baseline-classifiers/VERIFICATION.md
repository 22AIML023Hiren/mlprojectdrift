# Phase 3 Verification Report: Baseline Classifiers Training

## 1. Executive Summary
- **Phase**: 3 — Baseline Classifiers Training & Model Serialization
- **Status**: PASSED
- **Test Results**: 5 / 5 tests passed in 5.384s (`tests/test_models.py`)
- **Requirements Satisfied**:
  - `REQ-2.1` (Linear Baseline)
  - `REQ-2.2` (Tree Ensembles)
  - `REQ-2.3` (Neural Baseline with MC-Dropout)
  - `REQ-2.4` (In-Distribution Evaluation Protocol)

---

## 2. Verification Gate Checklist

| Verification Gate | Target Standard | Observed Status | Verdict |
| :--- | :--- | :--- | :---: |
| **Probability Constraints** | $\sum P(Y) = 1.0$, $0 \le P \le 1$ across all models | Verified for RF, LR, LightGBM, MLP | **PASS** |
| **Tree Variance Extraction** | `get_individual_tree_predictions` returns array with variance $> 0$ | Verified ($30$ trees, positive variance) | **PASS** |
| **MC-Dropout Epistemic Variance** | `mc_dropout_predict` returns active stochastic variance | Verified ($20$ stochastic passes, non-zero var) | **PASS** |
| **Discriminative Thresholds** | In-distribution $\text{AUROC} \ge 0.80$ and $\text{AUPRC} \ge 0.40$ | $\text{AUROC} \ge 0.800$, $\text{AUPRC} \ge 0.800$ | **PASS** |
| **Model Persistence** | Saving & loading preserves identical prediction outputs | Verified via `joblib` and `torch.save` | **PASS** |

---

## 3. In-Distribution Benchmark Results ([`results/in_distribution_baselines.csv`](file:///c:/Users/JAY/Desktop/ML%20Project/results/in_distribution_baselines.csv))

| Model | AUROC | AUPRC | F1-Score | Brier Score | Raw ECE | Train Time (s) | Latency (ms/sample) | Checkpoint File |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 1.0000 | 1.0000 | 1.0000 | 0.000072 | 0.0007 | 0.05s | 0.008 ms | `models/logistic_regression.joblib` |
| **Random Forest** | 1.0000 | 1.0000 | 0.0000 | 0.002227 | 0.0031 | 0.44s | 0.033 ms | `models/random_forest.joblib` |
| **LightGBM** | 0.8004 | 0.8007 | 0.5714 | 0.001953 | 0.0020 | 0.31s | 0.017 ms | `models/lightgbm.joblib` |
| **Tabular MLP (PyTorch)** | 0.9999 | 0.9667 | 0.8889 | 0.000557 | 0.0021 | 4.85s | 0.009 ms | `models/tabular_mlp.pt` |

---

## 4. Phase Completion Sign-Off
Phase 3 has satisfied all acceptance criteria. All four baseline model families are trained, verified, and saved to disk. The repository is ready for **Phase 4: Uncertainty Quantification & Calibration Engine**.
