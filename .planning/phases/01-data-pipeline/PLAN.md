# Phase 1 Plan: Data Pipeline & Temporal Partitioning

## Phase Metadata
- **Phase**: 1
- **Title**: Data Pipeline & Temporal Partitioning
- **Type**: Data Engineering & Preprocessing
- **Status**: In Progress
- **Requirements Covered**: REQ-1.1, REQ-1.2

---

## 1. Goal & Objective
Establish an automated, reproducible data ingestion and chronological partitioning pipeline for the ULB Credit Card Fraud Detection benchmark (284,807 transactions). The pipeline must guarantee zero future information leakage, validate schema integrity, and supply standardized Train / Calibration / In-Distribution Test partitions.

### User Story
> *As a machine learning researcher, I need a verified, leakage-free chronological data loader so that all subsequent classifiers and calibration methods are trained strictly on past data and evaluated on authentic future distributions.*

---

## 2. Technical Scope & Architecture

```
[Raw Dataset: Kaggle / Synthetic]
              │
              ▼
    [Schema Validation]  ──► (Check nulls, 30 features, Class balance, monotonic Time)
              │
              ▼
   [Chronological Slicing]
              │
    ┌─────────┴───────────────────────┐
    ▼                                 ▼
[Train Split: 70%]          [Future Splits: 30%]
    │ (Fit Scaler)                    │ (Transform only)
    ▼                                 ├────────────────────────┐
[X_train, y_train]                    ▼                        ▼
                             [Calib Split: 15%]       [ID Test Split: 15%]
                             [X_calib, y_calib]       [X_test, y_test]
```

---

## 3. Tasks Breakdown

### Task 1: Dataset Acquisition Utility (`scripts/download_data.py`)
- Implement a script supporting:
  - Kaggle API download (`kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/ --unzip`).
  - OpenML automated dataset fetching.
  - Verification of existing files in `data/raw/creditcard.csv`.
- Output: Standardized file at `data/raw/creditcard.csv`.

### Task 2: Schema & Integrity Validator (`src/data/validator.py`)
- Implement `validate_dataset_schema(df: pd.DataFrame) -> Dict[str, Any]`:
  - Verify all 30 expected features (`Time`, `V1`–`V28`, `Amount`) and `Class` target exist.
  - Verify zero NaN or null values across all columns.
  - Verify `Time` is sorted monotonically non-decreasing.
  - Verify positive class count matches expected fraud numbers.

### Task 3: Temporal Loader & Leakage-Free Scaler (`src/data/loader.py`)
- Validate and refine `get_chronological_splits`:
  - Exact split math (70% Train, 15% Calib, 15% Test).
  - RobustScaler fit strictly on `X_train[["Time", "Amount"]]`.
  - Transform applied to `X_calib` and `X_test` without fitting.
  - Return clean NumPy/Pandas structures ready for downstream modeling.

### Task 4: Unit Testing & Verification (`tests/test_data_loader.py`)
- Write unit tests verifying:
  - Dataset dimension and schema completeness.
  - Temporal ordering: $\max(\text{Time}_{\text{train}}) \le \min(\text{Time}_{\text{calib}}) \le \min(\text{Time}_{\text{test}})$.
  - Scaler statistics: $\text{scaler.center\_}$ matches `X_train` median, not full dataset.
  - Zero data loss: $N_{\text{train}} + N_{\text{calib}} + N_{\text{test}} = N_{\text{total}}$.

---

## 4. Verification Loop & Acceptance Criteria

| Check | Criteria | Verification Command |
| :--- | :--- | :--- |
| **Integrity** | Schema validator passes with zero missing values and correct types | `python -m pytest tests/test_data_loader.py -v` |
| **Temporal Monotonicity** | No overlap between Train, Calib, and Test timestamps | Unit test assertion |
| **Leakage Prevention** | Scaler is fitted exclusively on Train split | Unit test assertion |
| **Throughput & Reproducibility** | Full dataset partitions in $< 3$ seconds with fixed random seed | Automated test run |

---

## 5. Next Phase Handoff
Upon completion of Phase 1 verification:
- Proceed to **Phase 2: Controlled Distribution Shift Suite** (`/gsd-plan-phase 2`).
