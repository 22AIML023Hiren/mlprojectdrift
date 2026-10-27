"""
CLI script to train, benchmark, and serialize baseline classifiers on the source distribution.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dataset, get_chronological_splits
from src.models import get_model, save_model
from src.evaluation.metrics import compute_comprehensive_metrics


def main():
    print("=" * 80)
    print("  BASELINE CLASSIFIERS TRAINING & MODEL PERSISTENCE")
    print("=" * 80)

    models_dir = PROJECT_ROOT / "models"
    results_dir = PROJECT_ROOT / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest dataset & get chronological splits
    print("[1/4] Loading dataset and splitting chronologically...")
    df = load_dataset(n_synthetic_samples=40000, random_state=42)
    splits = get_chronological_splits(df, train_ratio=0.70, calib_ratio=0.15, test_ratio=0.15)

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    print(f"  - Training on {len(X_train):,} samples ({y_train.sum()} frauds, {y_train.mean():.3%})")
    print(f"  - Clean ID Evaluation on {len(X_test):,} samples ({y_test.sum()} frauds, {y_test.mean():.3%})")

    # 2. Define baseline models
    model_configs = {
        "Logistic_Regression": ("logistic_regression", {"C": 1.0, "random_state": 42}, "logistic_regression.joblib"),
        "Random_Forest": ("random_forest", {"n_estimators": 100, "max_depth": 10, "random_state": 42}, "random_forest.joblib"),
        "LightGBM": ("lightgbm", {"n_estimators": 120, "learning_rate": 0.05, "random_state": 42}, "lightgbm.joblib"),
        "Tabular_MLP": ("tabular_mlp", {"hidden_dim": 64, "dropout_rate": 0.20, "epochs": 12, "random_state": 42}, "tabular_mlp.pt")
    }

    # 3. Train and persist models
    print("\n[2/4] Training and serializing baseline models...")
    benchmark_records = []

    for name, (model_type, kwargs, checkpoint_filename) in model_configs.items():
        t0 = time.time()
        print(f"  - Training {name}...", end="", flush=True)
        model = get_model(model_type, **kwargs)
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        print(f" done ({train_time:.2f}s)")

        # Save checkpoint
        checkpoint_path = models_dir / checkpoint_filename
        save_model(model, checkpoint_path)

        # 4. In-distribution evaluation
        t_eval = time.time()
        probs = model.predict_proba(X_test)[:, 1]
        metrics = compute_comprehensive_metrics(y_test, probs)
        infer_time_ms = (time.time() - t_eval) * 1000 / len(X_test)

        benchmark_records.append({
            "Model": name,
            "AUROC": round(metrics["auroc"], 4),
            "AUPRC": round(metrics["auprc"], 4),
            "F1_Score": round(metrics["f1"], 4),
            "Brier_Score": round(metrics["brier_score"], 6),
            "Raw_ECE": round(metrics["ece"], 4),
            "Train_Time_s": round(train_time, 2),
            "Latency_ms_per_sample": round(infer_time_ms, 3),
            "Checkpoint": str(checkpoint_filename)
        })

    # Export benchmark table
    results_df = pd.DataFrame(benchmark_records)
    csv_path = results_dir / "in_distribution_baselines.csv"
    results_df.to_csv(csv_path, index=False)

    print("\n[3/4] In-Distribution Clean Baseline Benchmark Results:")
    print("=" * 105)
    print(results_df.to_string(index=False))
    print("=" * 105)

    print(f"\n[4/4] Successfully saved baseline results to {csv_path}")
    print(f"Checkpoints stored in {models_dir}")


if __name__ == "__main__":
    main()
