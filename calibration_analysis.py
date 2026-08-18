"""
Analyze calibration experiment results: target vs actual accuracy.

Reads eval logs from calibration.py, extracts target accuracy from task name,
and plots actual accuracy vs target, with statistics on calibration quality.

Usage:
    python calibration_analysis.py --logs ./logs
    python calibration_analysis.py --logs ./logs --plot calibration.png
"""

import argparse
import re
from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from inspect_ai.analysis import evals_df


def extract_target_accuracy(task_name: str) -> int | None:
    """Extract target accuracy from task name like 'mmlu_calibration_50pct'."""
    match = re.search(r'mmlu_calibration_(\d+)pct', task_name)
    return int(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="logs", help="Path to Inspect log directory")
    parser.add_argument("--plot", default=None, help="Optional path to save plot PNG")
    args = parser.parse_args()

    df = evals_df(logs=args.logs)
    if df.empty:
        raise SystemExit(f"No eval logs found in '{args.logs}'.")

    # Filter to only calibration tasks
    df = df[df["task_name"].str.contains("mmlu_calibration", na=False)]
    if df.empty:
        raise SystemExit(
            f"No calibration tasks found. Available tasks: {sorted(df['task_name'].unique())}"
        )

    # Keep only the most recent run per task
    # Group by task_name and keep only the row with the maximum timestamp
    if "timestamp" in df.columns:
        df = df.loc[df.groupby("task_name")["timestamp"].idxmax()]
    else:
        # Fallback: if no timestamp, just take the last occurrence of each task
        df = df.drop_duplicates(subset=["task_name"], keep="last")

    # Extract target accuracy and prepare comparison table
    df["target_pct"] = df["task_name"].apply(extract_target_accuracy)
    # actual_accuracy is a decimal (0.0-1.0), convert to percentage
    df["actual_pct"] = df["score_headline_value"] * 100

    # Remove any rows where extraction failed
    df = df.dropna(subset=["target_pct"])

    results = (
        df[["target_pct", "actual_pct"]]
        .drop_duplicates()
        .sort_values("target_pct")
        .reset_index(drop=True)
    )

    print("=" * 70)
    print("CALIBRATION RESULTS: target vs actual accuracy (latest run only)")
    print("=" * 70)
    # Display as percentages
    display_df = results.copy()
    display_df["target_pct"] = display_df["target_pct"].astype(int)
    display_df["actual_pct"] = display_df["actual_pct"].round(1)
    print(display_df.to_string(index=False))

    # Compute statistics (both in percentage points now)
    results["delta_pct"] = results["actual_pct"] - results["target_pct"]
    mae = results["delta_pct"].abs().mean()
    rmse = (results["delta_pct"] ** 2).mean() ** 0.5
    corr = results["target_pct"].corr(results["actual_pct"])

    print()
    print("=" * 70)
    print("CALIBRATION QUALITY")
    print("=" * 70)
    print(f"Mean absolute error: {mae:.1f} percentage points")
    print(f"RMSE: {rmse:.1f} percentage points")
    print(f"Correlation (target vs actual): {corr:.3f}")
    print()
    print("Per-target delta (actual % - target %):")
    for _, row in results.iterrows():
        target = int(row["target_pct"])
        actual = row["actual_pct"]
        delta = row["delta_pct"]
        print(f"  {target:3d}% target: {actual:5.1f}% actual ({delta:+6.1f} ppt)")

    # Plot if matplotlib available
    if args.plot and HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 6))
        plt.scatter(results["target_pct"], results["actual_pct"], s=100, alpha=0.6, label="Model performance")
        
        # Reference line: perfect calibration (diagonal)
        min_val, max_val = 0, 100
        plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect calibration", alpha=0.5)
        
        plt.xlabel("Target Accuracy (%)", fontsize=12)
        plt.ylabel("Actual Accuracy (%)", fontsize=12)
        plt.title("Calibration: Can the model hit target accuracies?", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(-5, 105)
        plt.ylim(-5, 105)
        plt.savefig(args.plot, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {args.plot}")
    elif args.plot and not HAS_MATPLOTLIB:
        print("\nNote: matplotlib not installed, skipping plot. Install with: pip install matplotlib")


if __name__ == "__main__":
    main()