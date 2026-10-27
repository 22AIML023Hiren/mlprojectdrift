"""
Dataset download and acquisition script.
Supports:
1. Kaggle CLI download (if kaggle is installed and configured)
2. Direct local check and instruction
3. High-fidelity synthetic fallback generation
"""

import sys
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import RAW_DATA_DIR, generate_synthetic_benchmark


def check_or_download_data():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_csv = RAW_DATA_DIR / "creditcard.csv"

    if target_csv.exists():
        size_mb = target_csv.stat().st_size / (1024 * 1024)
        print(f"[Dataset] Found existing creditcard.csv at {target_csv} ({size_mb:.1f} MB)")
        return True

    print("[Dataset] creditcard.csv not found in data/raw/.")
    print("[Dataset] Attempting download via Kaggle CLI...")

    try:
        cmd = [
            "kaggle", "datasets", "download",
            "-d", "mlg-ulb/creditcardfraud",
            "-p", str(RAW_DATA_DIR),
            "--unzip"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and target_csv.exists():
            print(f"[Dataset] Successfully downloaded via Kaggle CLI: {target_csv}")
            return True
        else:
            print(f"[Dataset] Kaggle CLI download returned code {result.returncode}.")
            if result.stderr:
                print(f"[Dataset] Error: {result.stderr.strip()}")
    except FileNotFoundError:
        print("[Dataset] Kaggle CLI not found on PATH.")

    # Fallback to high-fidelity synthetic benchmark
    synthetic_csv = RAW_DATA_DIR / "creditcard_synthetic.csv"
    if not synthetic_csv.exists():
        print("[Dataset] Generating high-fidelity synthetic benchmark for offline use...")
        df = generate_synthetic_benchmark(n_samples=50000, random_state=42)
        df.to_csv(synthetic_csv, index=False)
        print(f"[Dataset] Generated {len(df):,} synthetic records at {synthetic_csv}")
    else:
        print(f"[Dataset] Using existing synthetic benchmark at {synthetic_csv}")

    print("\n[Notice] To use the official Kaggle dataset:")
    print("  1. Download 'creditcard.csv' from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print(f"  2. Place it in: {RAW_DATA_DIR / 'creditcard.csv'}\n")
    return True


if __name__ == "__main__":
    check_or_download_data()
