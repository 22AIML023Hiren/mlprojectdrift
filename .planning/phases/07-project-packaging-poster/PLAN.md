# Phase 7 Plan: Final Project Packaging & Poster Delivery

## Phase Metadata
- **Phase**: 7
- **Title**: Final Project Packaging & Poster Delivery
- **Type**: Deliverable Synthesis & Release Engineering
- **Status**: Complete
- **Requirements Covered**: REQ-5.4, All Epics Sign-Off

---

## 1. Goal & Objective
Synthesize all empirical findings, models, evaluations, and visualizations into a polished submission package. Update the HTML presentation posters with verified experimental data, compile verified single-page presentation PDFs (`poster.pdf`, `poster_academic.pdf`), synchronize the LaTeX manuscript (`main.tex` / `project_proposal.tex`) and Markdown report (`PROJECT_PROPOSAL.md`) with the 3 team member credentials, and finalize comprehensive README documentation.

### User Story
> *As a course evaluator, I need a complete, reproducible submission package containing runnable code, unit tests, academic writeups, and a verified single-page presentation poster featuring the 3 team members and actual empirical findings.*

---

## 2. Technical Scope & Deliverables

```
                [Completed Phases 1-6 Artifacts & Benchmarks]
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            ▼                         ▼                         ▼
   [Presentation Posters]      [LaTeX Manuscript]      [Package Documentation]
   - poster.html               - main.tex              - README.md
   - poster_academic.html      - project_proposal.tex  - PROJECT_PROPOSAL.md
            │                         │                         │
            ▼                         ▼                         │
   [Headless PDF Render]       [pdflatex Build]                 │
   - poster.pdf (1-page)       - main.pdf                       │
   - poster_academic.pdf       - project_proposal.pdf           │
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      │
                                      ▼
                        [Automated Verification Suite]
                        - 30 / 30 Unit Tests Pass
                        - Single-Page PDF Integrity Checked
                        - 3-Member Header Compliance
```

---

## 3. Tasks Breakdown

### Task 1: Update Presentation Posters (`poster.html`, `poster_academic.html`)
- Update Quadrant 4 with empirical metrics established in Phases 4–6:
  - Isotonic calibration achieved **68.7% ECE drop** ($0.00308 \to 0.00096$).
  - Selective classification at 85% coverage yielded **72.0% to 90.0% loss reduction**.
  - Catastrophic missed fraud eliminated ($FN \to 0$) on 5 of 6 shift testbeds.
  - Shift-risk disentanglement confirmed ($\rho_R \gg \rho_{\text{OOD}}$ by up to 3.2x).
  - Embed the generated high-resolution figures from `results/figures/`.
- Maintain strict 3-member header layout:
  1. Jay Lakkad (M26CSE005)
  2. Hiren Modhvadia (M26CSA012)
  3. Dev Pandya (M26CSE003)

### Task 2: Compile & Verify Single-Page PDFs
- Render `poster.pdf` and `poster_academic.pdf` using headless Edge.
- Verify byte streams with page count inspection regex (`rb'/Type\s*/Page\b'`) ensuring exactly 1 single page.

### Task 3: Synchronize Academic Paper & Markdown Proposal
- Update `main.tex`, `project_proposal.tex`, and `PROJECT_PROPOSAL.md`:
  - Replace author placeholders with the 3 student names and IDs.
  - Insert empirical cross-model benchmark tables and disentanglement findings.
  - Compile `main.pdf` / `project_proposal.pdf`.

### Task 4: Complete README Documentation (`README.md`)
- Add project overview, system architecture diagram, directory layout, installation steps, reproducible execution commands, and test suite commands.

### Task 5: Final Sign-Off & Verification Report (`VERIFICATION.md`)
- Document all verified acceptance criteria, test execution logs, and PDF inspection results.

---

## 4. Verification Loop & Acceptance Criteria

| Check | Target Standard | Verification Check |
| :--- | :--- | :--- |
| **3-Member Formatting** | Exactly 3 members listed on poster and documents | Inspection of headers |
| **Single-Page Poster** | `poster.pdf` and `poster_academic.pdf` are exactly 1 page | Python byte count inspection |
| **Empirical Accuracy** | All tables reflect real metrics from `results/*.csv` | Data cross-check |
| **Test Suite Pass** | 100% of repository tests pass (30/30) | `python -m unittest discover tests` |
| **Turnkey Execution** | `python main.py` runs end-to-end without errors | Execution check |

---

## 5. Next Phase Handoff
Phase 7 is the final phase of the roadmap. Upon completion, the project will be fully delivered, documented, and committed to git.
