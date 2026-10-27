# Phase 7 Research: Final Project Packaging & Poster Delivery

## 1. Academic Presentation & Packaging Standards

Course project delivery requires synchronizing empirical evidence, source code reproducibility, and presentation artifacts into an integrated submission package.

### 1.1 Team Formatting Constraints
- **Team Size**: Exactly 3 members.
- **Member Details**:
  1. **Jay Lakkad** — `M26CSE005` (Email: `m26cse005@university.edu`)
  2. **Hiren Modhvadia** — `M26CSA012` (Email: `m26csa012@university.edu`)
  3. **Dev Pandya** — `M26CSE003` (Email: `m26cse003@university.edu`)
All documents, LaTeX headers, HTML templates, and Markdown readmes must reflect this 3-member composition with balanced 3-column layouts.

### 1.2 Single-Page PDF Print Constraints
Academic posters must be strictly **single-page printable documents** (A3 landscape, $420\text{mm} \times 297\text{mm}$).
To prevent accidental multi-page overflow when rendering through headless Edge or Chromium:
- Use fixed viewport container heights: `height: 100vh; overflow: hidden;`
- In `@page` CSS: `size: A3 landscape; margin: 8mm;`
- `@media print`: enforce `height: 100vh !important; page-break-inside: avoid;`
- Restrict quadrant internal content using flexbox and auto-scaling font clamps.
- Verify byte streams with PDF inspection regex (`rb'/Type\s*/Page\b'`) ensuring exactly count = 1.

### 1.3 Empirical Synchronization
The final deliverables must transition from speculative "proposed" metrics to actual empirical findings established during Phases 1–6:
1. **ECE Calibration Improvement**: 68.7% reduction ($0.00308 \to 0.00096$) via Isotonic Regression.
2. **Economic Loss Mitigation**: 72.0% to 90.0% loss reduction at 85% target automation coverage on severe shifts.
3. **Catastrophic Error Elimination**: Complete elimination of missed fraud cases ($FN \to 0$) under high-amount subpopulation and prevalence shifts.
4. **Disentanglement Proof**: Decision risk unreliability achieves 2.3x to 3.2x higher rank correlation with errors than feature-space OOD distance.
5. **Conformal Guarantees**: Finite-sample empirical coverage ($96.67\%$ vs $95.0\%$ target at $\alpha = 0.05$).

---

## 2. LaTeX Document Structure & Typography

The LaTeX paper (`main.tex` / `project_proposal.tex`) follows a 2-column format with:
- Formal problem formulation under covariate, subpopulation, label ratio, and concept drift.
- Detailed mathematical exposition of Isotonic calibration, conformal quantiles $\hat{q}$, and selective thresholding $\tau$.
- Cross-model benchmark table comparing Logistic Regression, Random Forest, LightGBM, and Tabular MLP.
- Economic cost matrix analysis ($C_{\text{FN}}=\$100, C_{\text{FP}}=\$5, C_{\text{defer}}=\$2$).
- Verified author block formatting for the 3 team members.
