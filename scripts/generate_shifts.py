"""
CLI script for pre-generating, statistically profiling, and serializing
all controlled distribution shift testbeds to data/shifted/.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dataset, get_chronological_splits
from src.data.shifts import ShiftGenerator
from src.data.shift_metrics import compute_distribution_divergence


def main():
    print("=" * 75)
    print("  CONTROLLED DISTRIBUTION SHIFT SUITE GENERATOR")
    print("=" * 75)

    data_dir = PROJECT_ROOT / "data"
    shifted_dir = data_dir / "shifted"
    shifted_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset & get clean test split
    print("[1/4] Loading dataset and extracting chronological test partition...")
    df = load_dataset(n_synthetic_samples=40000, random_state=42)
    splits = get_chronological_splits(df, train_ratio=0.70, calib_ratio=0.15, test_ratio=0.15)
    X_test, y_test = splits["X_test"], splits["y_test"]

    # 2. Generate all shift testbeds
    print("[2/4] Generating 6 controlled distribution shift testbeds...")
    sg = ShiftGenerator(random_state=42)
    testbeds = sg.generate_all_shift_testbeds(X_test, y_test)

    # 3. Compute statistical divergence metrics & export CSVs
    print("[3/4] Quantifying statistical divergence and serializing CSV partitions...")
    manifest = {
        "generator_version": "1.0",
        "reference_benchmark": "Kaggle/ULB Credit Card Fraud Detection",
        "clean_id_samples": len(X_test),
        "testbeds": {}
    }

    ref_X = testbeds["Clean_ID"][0]

    for name, (X_s, y_s) in testbeds.items():
        # Export combined DataFrame
        df_export = X_s.copy()
        df_export["Class"] = y_s
        out_csv = shifted_dir / f"{name}.csv"
        df_export.to_csv(out_csv, index=False)

        # Compute divergence profile relative to Clean_ID
        divergence = compute_distribution_divergence(ref_X, X_s)

        manifest["testbeds"][name] = {
            "file": str(out_csv.name),
            "samples": len(X_s),
            "fraud_count": int(y_s.sum()),
            "fraud_rate": round(float(y_s.mean()), 5),
            "mean_wasserstein_distance": divergence["mean_wasserstein_distance"],
            "mean_psi": divergence["mean_psi"],
            "significant_ks_features_count": divergence["significant_ks_features_count"],
            "total_features": divergence["total_features_evaluated"]
        }

        print(f"  [OK] Exported '{name}.csv' ({len(X_s):,} samples, {y_s.sum()} frauds) | W1: {divergence['mean_wasserstein_distance']:.4f} | PSI: {divergence['mean_psi']:.4f}")

    # 4. Save manifest.json
    manifest_path = shifted_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[4/4] Successfully saved testbed manifest to {manifest_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
