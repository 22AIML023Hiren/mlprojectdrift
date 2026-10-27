# Risk-Aware Machine Learning Under Distribution Shift

An end-to-end Machine Learning research system that investigates **epistemic uncertainty quantification**, **controlled distribution shifts**, and **risk-aware selective classification** in safety-critical tabular financial streams.

---

## 🚀 Key Features & Implementation Architecture

The system implements the **Risk-Aware Selective Classification Pipeline (RASCP)** across three tiers:

```
[Input Transaction x]
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Tier 1: Predictive Inference Engine                    │
│ - Logistic Regression (L1/L2 ElasticNet Baseline)      │
│ - Random Forest (Tree Ensemble)                        │
│ - LightGBM (Gradient Boosted Decision Trees)           │
│ - Tabular MLP (PyTorch Deep Learning with MC-Dropout)  │
└──────────────────────────┬─────────────────────────────┘
                           │ Raw Probabilities / Logits
                           ▼
┌────────────────────────────────────────────────────────┐
│ Tier 2: Uncertainty & Risk Quantifier                  │
│ - Post-Hoc Probability Calibration (Isotonic / Platt)  │
│ - Ensemble Epistemic Disagreement (Inter-Tree Variance)│
│ - Inductive Conformal Prediction Sets (1 - alpha)      │
└──────────────────────────┬─────────────────────────────┘
                           │ Calibrated Risk Metric R(x)
                           ▼
┌────────────────────────────────────────────────────────┐
│ Tier 3: Risk-Gated Decision Engine                     │
│ Evaluate: Risk(x) <= tau                               │
└────────────┬───────────────────────────────┬───────────┘
             │                               │
             ▼ (Low Risk)                    ▼ (High Risk)
  [Decision: ACCEPT]              [Decision: ABSTAIN]
  - Automated prediction ŷ        - Route to Human Fraud Analyst
  - High speed, high confidence   - Zero-tolerance safety net
```

---

## 📊 Controlled Distribution Shift Testbeds

The framework generates four controlled out-of-distribution (OOD) shift scenarios to systematically stress-test models:
1. **Clean In-Distribution (`Clean_ID`)**: Chronological holdout evaluation.
2. **Covariate Shift (`Covariate_Mild` & `Covariate_Severe`)**: Gaussian noise injection on principal components $V_1$–$V_4$ ($X_{\text{shifted}} = X + \mathcal{N}(0, \sigma^2 \Sigma)$).
3. **Subpopulation Shift (`Subpopulation_HighAmount`)**: High-value transaction tail strata ($>85$th percentile) where fraud dynamics diverge.
4. **Prevalence / Label Shift (`Prevalence_Shift`)**: Simulating coordinated attack waves where fraud rate increases to $2.5\%$ (vs $0.17\%$).
5. **Concept Drift (`Concept_Drift`)**: Adversarial feature perturbation mimicking fraudsters actively evading known patterns.

---

## ⚡ Quick Start: Running the Project

### 1. Run the Complete Pipeline
Execute the full training, calibration, shift benchmarking, and visualization suite with a single command:
```bash
python main.py
```

### 2. Output Artifacts
Execution automatically produces:
* **Benchmark Results Table**: `results/shift_benchmark_results.csv`
* **Risk-Coverage Curve**: `results/figures/risk_coverage_curve.png`
* **Calibration Reliability Diagram**: `results/figures/calibration_reliability_diagram.png`
* **Economic Loss Mitigation Chart**: `results/figures/shift_resilience_loss_comparison.png`

---

## 📁 Repository Structure

```
ML Project/
├── main.py                              # Master experiment execution pipeline
├── README.md                            # Project documentation
├── PROJECT_PROPOSAL.md                  # Comprehensive technical proposal writeup
├── poster.html / poster_academic.html   # Interactive 4-quadrant presentation posters
├── poster.pdf / poster_academic.pdf     # Single-page compiled presentation posters
├── data/
│   ├── raw/                             # Dataset storage (Kaggle or synthetic benchmark)
│   └── shifted/                         # Exported shifted test partitions
├── src/
│   ├── data/
│   │   ├── loader.py                    # Temporal dataset loader & RobustScaler
│   │   └── shifts.py                    # 4-mode distribution shift generator suite
│   ├── models/
│   │   ├── baselines.py                 # Logistic Regression & Random Forest
│   │   ├── gbm.py                       # LightGBM Classifier
│   │   └── mlp.py                       # PyTorch Tabular MLP with Monte Carlo Dropout
│   ├── uncertainty/
│   │   ├── calibration.py               # Isotonic regression, Platt scaling, & ECE
│   │   ├── ensemble_var.py              # Inter-tree variance & predictive entropy
│   │   └── conformal.py                 # Inductive Conformal Prediction Sets
│   ├── decision/
│   │   ├── risk_scorer.py               # Composite decision risk scoring
│   │   └── selector.py                  # Selective classification & threshold optimization
│   └── evaluation/
│       ├── metrics.py                   # AUPRC, AUROC, ECE, Brier, AURC, and Cost Loss
│       └── plots.py                     # Publication-grade visualization routines
└── results/
    ├── shift_benchmark_results.csv      # Empirical metrics across all testbeds
    └── figures/                         # Generated experimental figures
```

---

## 🔬 Key Research Findings & Benchmarks

From our empirical benchmark (Target Automation Coverage: **85%**):
* **Catastrophic Loss Mitigation**: The Risk-Aware Selective Classifier achieves **73.2% to 94.8% reduction in total operational loss** across shift testbeds compared to the vanilla non-selective baseline.
* **Elimination of Silent False Negatives**: Across mild covariate shift, subpopulation shift, and prevalence shift, routing the top $15\%$ most uncertain predictions to human review reduced missed frauds from $5 \to 0$.
* **Disentanglement (Risk vs. OOD Distance)**: Predictive decision risk achieves a **$0.507$ to $0.633$ rank correlation with actual misclassifications**, significantly outperforming standard feature-space anomaly detection (Isolation Forest, correlation $\approx 0.163$ to $0.270$). This proves that data-level OOD novelty does not imply decision unreliability.
