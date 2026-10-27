# Project Execution State

## Current Position
- **Milestone**: Implementation & Empirical Benchmark
- **Active Phase**: Phase 6 Complete (Full Pipeline & Evaluation Executed)
- **Status**: Codebase fully implemented, tested, and generating results.
- **Last Command**: `python main.py` executed successfully in 7.1s.

## Implemented Architecture & Modules
1. **`src/data/`**: Temporal data loader, RobustScaler (no leakage), and 4-mode Controlled Distribution Shift suite (`ShiftGenerator`).
2. **`src/models/`**: Logistic Regression, Random Forest, LightGBM, and PyTorch Tabular Neural Network with Monte Carlo Dropout (`TabularMLP`).
3. **`src/uncertainty/`**: Probability Calibration (Isotonic Regression, Platt Scaling), Inter-tree epistemic variance, and Inductive Conformal Prediction Sets ($\alpha = 0.05$).
4. **`src/decision/`**: Composite Risk Scorer, Selective Classifier with abstention threshold $\tau$, Risk-Coverage curve generation, and AURC.
5. **`src/evaluation/`**: AUPRC, AUROC, ECE, Brier Score, Cost-Weighted Loss, Disentanglement Correlation, and Matplotlib plotting routines.
6. **`main.py`**: Turnkey CLI runner executing the end-to-end pipeline.

## Empirical Benchmark Highlights (at 85% Automation Coverage)
- **Clean ID**: 81.6% loss reduction ($500 \to $92), missed fraud reduced from 5 to 0.
- **Covariate Shift**: 73.2% loss reduction on mild shift, missed fraud reduced from 5 to 0.
- **Subpopulation Shift (High Amount)**: 92.0% loss reduction, missed fraud reduced from 2 to 0.
- **Prevalence Shift (2.5% Fraud)**: 94.8% loss reduction, missed fraud reduced from 5 to 0.
- **Disentanglement**: Decision risk correlation with error reached **0.633**, whereas feature-space OOD distance only reached **0.270**, confirming that uncertainty-aware risk estimation outperforms naive shift detection.
