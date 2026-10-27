# Risk-Aware Machine Learning Under Distribution Shift

An end-to-end Machine Learning research system that investigates **epistemic uncertainty quantification**, **controlled distribution shifts**, and **risk-aware selective classification** in safety-critical tabular financial streams.

---

### Project Metadata & Team
* **Project Title:** Risk-Aware Machine Learning Under Distribution Shift: Quantifying Predictive Uncertainty and Selective Abstention in Financial Tabular Streams
* **Course:** Machine Learning Project
* **Team Members (3 Members):**
  1. **Jay Lakkad** — `m26cse005@university.edu` (Roll No: M26CSE005)
  2. **Hiren Modhvadia** — `m26csa012@university.edu` (Roll No: M26CSA012)
  3. **Dev Pandya** — `m26cse003@university.edu` (Roll No: M26CSE003)
* **Academic Deliverables:**
  * Interactive Presentation Poster: [poster.html](file:///c:/Users/JAY/Desktop/ML%20Project/poster.html) & [poster_academic.html](file:///c:/Users/JAY/Desktop/ML%20Project/poster_academic.html)
  * Verified Single-Page Printable PDFs: [poster.pdf](file:///c:/Users/JAY/Desktop/ML%20Project/poster.pdf) & [poster_academic.pdf](file:///c:/Users/JAY/Desktop/ML%20Project/poster_academic.pdf)
  * Formal Course Proposal & Specification: [PROJECT_PROPOSAL.md](file:///c:/Users/JAY/Desktop/ML%20Project/PROJECT_PROPOSAL.md) and [main.tex](file:///c:/Users/JAY/Desktop/ML%20Project/main.tex)

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

The framework generates controlled out-of-distribution (OOD) shift scenarios to systematically stress-test models:
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

### 2. Run Comprehensive Multi-Model Benchmarks
```bash
python scripts/evaluate_comprehensive.py
```

### 3. Run Selective Classification Decision Evaluation
```bash
python scripts/run_selective_decision.py
```

### 4. Run the Full Automated Unit Test Suite
Verify all 30 tests spanning data loaders, shift generators, model training, uncertainty calibrators, selective gates, and evaluation metrics:
```bash
python -m unittest discover tests
```

---

## 🔬 Verified Key Research Findings & Benchmarks

From our empirical benchmark across shifts (Target Automation Coverage: **85%**):
* **Catastrophic Loss Mitigation**: The Risk-Aware Selective Classifier achieves **72.0% to 90.0% reduction in total operational loss** across shift testbeds compared to the vanilla non-selective baseline.
* **Elimination of Silent False Negatives**: Across mild covariate shift, subpopulation shift, and prevalence shift, routing the top $15\%$ most uncertain predictions to human review reduced missed frauds to **zero** ($FN \to 0$).
* **Disentanglement (Risk vs. OOD Distance)**: Predictive decision risk achieves a **$0.144$ to $0.359$ rank correlation with actual misclassifications**, outperforming standard feature-space anomaly detection (Isolation Forest, correlation $\approx 0.039$ to $0.118$) by **2.3x to 3.2x**. This proves that data-level OOD novelty does not imply decision unreliability.
* **Calibration Improvement**: Isotonic regression reduced Expected Calibration Error (ECE) by **68.7%** ($0.00308 \to 0.00096$) and Brier score from $0.002227 \to 0.000420$.
* **Conformal Guarantees**: Inductive Conformal Prediction satisfied finite-sample coverage across all significance levels (e.g., $96.67\%$ empirical vs $95.0\%$ target for $\alpha = 0.05$).

---

## 📁 Repository Structure

```
ML Project/
├── main.py                               # Master end-to-end experiment pipeline
├── README.md                             # Project documentation
├── PROJECT_PROPOSAL.md                   # Comprehensive technical proposal writeup
├── main.tex / project_proposal.tex       # Academic LaTeX writeups
├── poster.html / poster_academic.html    # Interactive 4-quadrant presentation posters
├── poster.pdf / poster_academic.pdf      # Verified single-page printable PDF posters
├── data/
│   ├── raw/                              # Dataset storage (Kaggle or synthetic benchmark)
│   └── shifted/                          # Exported shifted test partitions with manifest.json
├── models/
│   ├── random_forest.joblib              # Serialized Random Forest checkpoint
│   ├── lightgbm.joblib                   # Serialized LightGBM checkpoint
│   ├── logistic_regression.joblib       # Serialized Logistic Regression checkpoint
│   ├── tabular_mlp.pt                    # Serialized PyTorch Tabular MLP checkpoint
│   └── calibrators/                      # Isotonic and Platt probability calibrators
├── scripts/
│   ├── calibrate_and_evaluate.py         # Phase 4 uncertainty calibration CLI
│   ├── run_selective_decision.py         # Phase 5 selective decision benchmark CLI
│   └── evaluate_comprehensive.py         # Phase 6 multi-model & disentanglement CLI
├── src/
│   ├── data/                             # Data loading, temporal splitting, and shift suite
│   ├── models/                           # Model wrappers and MC Dropout implementations
│   ├── uncertainty/                      # Probability calibration, ensemble variance, conformal sets
│   ├── decision/                         # Risk scoring and selective classification engine
│   └── evaluation/                       # Multi-dimensional evaluation metrics and Matplotlib figures
├── tests/                                # 30 automated unit tests (100% pass)
└── results/
    ├── comprehensive_model_comparison.csv
    ├── selective_shift_comparison.csv
    ├── disentanglement_analysis.csv
    └── figures/                          # 5 publication-grade 300 DPI figures
```
