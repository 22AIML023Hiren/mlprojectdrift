# Technology Stack

## Core Runtime & Language
- **Python**: 3.12 (Primary language for data processing, machine learning modeling, and risk evaluation)
- **Node.js**: v22.16.0 (Tooling, local web server, and GSD ecosystem runtime)

## Machine Learning & Data Processing Libraries
- **scikit-learn** (1.9.1): Base linear models (LogisticRegression), Random Forest (RandomForestClassifier), metrics, and probability calibration (`CalibratedClassifierCV`, `IsotonicRegression`)
- **LightGBM** (4.7.0): High-performance Gradient Boosted Decision Trees for tabular benchmarks
- **PyTorch** (`torch` 2.14.0, `torchvision` 0.29.0): Tabular Multi-Layer Perceptrons (MLPs), Monte Carlo Dropout, and Deep Ensembles
- **NumPy** (2.5.3) & **Pandas** (2.2.2): Numerical matrix operations, dataset slicing, and tabular feature manipulation
- **SciPy** (1.18.1): Statistical distributions, non-parametric tests, and distribution shift distance calculations (Wasserstein distance, KS test)

## Uncertainty Quantification & Conformal Prediction
- **Conformal Prediction**: Non-conformity scores and inductive conformal risk control (`nonconformist` / native conformal scoring implementation)
- **Calibration Engine**: Platt Scaling (sigmoid fit), Isotonic Regression, Temperature Scaling

## Visualization, Reporting & Delivery
- **Matplotlib** (3.11.1): Performance curves (Risk-Coverage, AUPRC, ROC, Reliability Diagrams / Calibration plots)
- **LaTeX / pdflatex**: Academic report and proposal compilation (`project_proposal.tex`, `main.tex`)
- **HTML5 / Vanilla CSS**: Interactive and print-ready scientific posters (`poster.html`, `poster_academic.html`)
- **Chromium / Microsoft Edge**: Headless PDF generation for single-page presentation posters
