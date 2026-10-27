# External Integrations and Benchmarks

## Primary Benchmark Dataset
- **Name**: Kaggle Credit Card Fraud Detection Benchmark (ULB / Worldline)
- **URL**: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transaction records (approx. 143 MB uncompressed CSV)
- **Features**:
  - `Time`: Elapsed seconds from first transaction in dataset.
  - `Amount`: Transaction amount in EUR.
  - `V1`–`V28`: PCA principal components (anonymized to protect personal identification).
  - `Class`: 0 (Legitimate, 99.828%), 1 (Fraud, 0.172%).
- **License**: Open Database License (ODbL) — fully approved for educational and academic research.

## Secondary Benchmark (Validation / Extension)
- **Name**: IEEE-CIS Fraud Detection Dataset (Vesta Corporation)
- **URL**: [https://www.kaggle.com/c/ieee-fraud-detection](https://www.kaggle.com/c/ieee-fraud-detection)
- **Size**: 590,540 transactions with complex transaction metadata, identity features, and card types.
- **Role**: Secondary cross-dataset validation if time and compute permit.

## Tooling & External Utilities
- **Kaggle API** (`kaggle` CLI): Automated scriptable downloading of raw datasets:
  ```bash
  kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/ --unzip
  ```
- **Microsoft Edge / Google Chrome Headless**: Native rendering of single-page posters to PDF.
- **LaTeX Distribution (MiKTeX / TeX Live)**: PDF generation for academic project reports.
