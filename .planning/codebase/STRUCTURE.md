# Codebase Structure

## Directory Layout (Planned & Existing)

```
ML Project/
├── .git/                                # Git version control repository
├── .planning/                           # GSD framework planning artifacts
│   ├── config.json                      # GSD configuration & workflow flags
│   ├── PROJECT.md                       # Master project charter and context
│   ├── REQUIREMENTS.md                  # Detailed scoped requirements
│   ├── ROADMAP.md                       # Phase execution breakdown
│   ├── STATE.md                         # Active project execution state
│   ├── codebase/                        # Evidence-backed codebase map
│   │   ├── STACK.md                     # Technology stack
│   │   ├── ARCHITECTURE.md              # Pipeline architecture
│   │   ├── STRUCTURE.md                 # Project structure & layout
│   │   ├── CONVENTIONS.md               # Coding standards
│   │   ├── TESTING.md                   # Evaluation protocols
│   │   ├── INTEGRATIONS.md              # External benchmarks & datasets
│   │   └── CONCERNS.md                  # Risks, edge cases, and debt
│   └── onboarding/
│       └── SUMMARY.md                   # Onboarding summary & handoff
├── data/                                # Dataset storage (raw and processed)
│   ├── raw/                             # Original downloaded ULB dataset
│   └── shifted/                         # Generated synthetic shift test partitions
├── src/                                 # Implementation source code
│   ├── data/                            # Ingestion, validation, and shift generators
│   │   ├── loader.py                    # Temporal dataset loader
│   │   └── shifts.py                    # Covariate, subpopulation, & label shift injections
│   ├── models/                          # Base classifiers
│   │   ├── baselines.py                 # Logistic Regression & Random Forest
│   │   ├── gbm.py                       # LightGBM model wrapper
│   │   └── mlp.py                       # PyTorch Tabular Neural Network
│   ├── uncertainty/                     # Uncertainty estimation modules
│   │   ├── calibration.py               # Isotonic regression & Platt scaling
│   │   ├── ensemble_var.py              # Inter-model disagreement
│   │   └── conformal.py                 # Inductive conformal prediction
│   ├── decision/                        # Selective classification & risk gating
│   │   ├── risk_scorer.py               # Composite risk metrics
│   │   └── selector.py                  # Coverage-risk thresholding & abstention
│   └── evaluation/                      # Metrics and visualization
│       ├── metrics.py                   # AUPRC, AUROC, ECE, Brier, AURC, Cost Loss
│       └── plots.py                     # Risk-Coverage curves & reliability diagrams
├── results/                             # Experiment outputs, tables, and plots
├── PROJECT_PROPOSAL.md                  # Comprehensive technical proposal
├── project_proposal.tex / main.tex      # LaTeX course project proposal documents
├── project_proposal.pdf / main.pdf      # Compiled PDF project proposal
├── poster.html / poster_academic.html   # 4-quadrant interactive and printable posters
└── poster.pdf / poster_academic.pdf     # Single-page compiled presentation posters
```
