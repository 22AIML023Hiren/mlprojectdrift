# Code Conventions and Standards

## Python Coding Guidelines
- **PEP 8 Compliance**: 4 spaces per indentation level, maximum line length of 100 characters.
- **Type Annotations**: Use Python `typing` (`Union`, `Optional`, `Tuple`, `Dict`, `List`, `np.ndarray`, `pd.DataFrame`) on all public function signatures.
- **Docstrings**: Google or NumPy style docstrings for all classes and functions detailing parameters, return types, and exceptions.
- **Deterministic Experiments**: Always pass explicit `random_state` / `seed` arguments across all splits, model initializations, and perturbations to guarantee reproducibility.

## Modularity & Separation of Concerns
- **Data Isolation**: Never allow test or shifted partitions to leak into training, scaler fitting, or calibration set fitting.
- **Clean Interfaces**: All models must conform to the scikit-learn estimator interface (`fit(X, y)` and `predict_proba(X)`).
- **Stateless Evaluators**: Evaluation modules must compute metrics directly from true labels, raw predictions, calibrated probabilities, and abstention masks.

## Version Control & Commits
- Follow conventional commit style:
  - `feat(...)`: new feature or model
  - `fix(...)`: bug fix in data loading or metric calculation
  - `docs(...)`: documentation or planning updates
  - `test(...)`: unit or regression tests
  - `eval(...)`: experiment results and benchmark logs
