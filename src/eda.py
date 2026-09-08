"""Phase 1 - Exploratory Data Analysis (EDA).

Run this once to understand the data BEFORE modeling:
    python src/eda.py

It prints a summary to the console and saves a few charts to figures/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless: save figures, don't open windows
import matplotlib.pyplot as plt

from data import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    PROJECT_ROOT,
    TARGET,
    add_target,
    load_raw,
)

FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    df = load_raw()
    df = add_target(df)

    section("1) SHAPE & COLUMNS")
    print(f"Rows: {len(df):,}   Columns: {df.shape[1]}")
    print(df.dtypes)

    section("2) MISSING VALUES (per column)")
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss):
        for col, n in miss.items():
            print(f"  {col:<45} {n:>8,}  ({n / len(df):.1%})")
    else:
        print("  none")

    section("3) TARGET: delivery_duration_seconds (and minutes)")
    t = df[TARGET]
    print(f"  valid targets: {t.notna().sum():,} / {len(df):,}")
    print(f"  negative or zero: {(t <= 0).sum():,}")
    pct = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    print("\n  Percentiles (minutes):")
    for p in pct:
        print(f"    {p*100:6.1f}% : {t.quantile(p)/60:8.1f} min")
    print(f"\n  mean: {t.mean()/60:.1f} min   median: {t.median()/60:.1f} min")

    section("4) CATEGORICAL CARDINALITY (why we need embeddings)")
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            print(f"  {col:<25} unique values: {df[col].nunique():>7,}")
    print("\n  -> store_id has thousands of values: one-hot would explode.")
    print("     This is exactly why the DoorDash blog uses EMBEDDINGS.")

    section("5) CORRELATION of numeric features with target")
    valid = df[df[TARGET].between(60, 3 * 3600)]
    corr = (
        valid[NUMERIC_COLS + [TARGET]]
        .corr()[TARGET]
        .drop(TARGET)
        .sort_values(key=np.abs, ascending=False)
    )
    for col, c in corr.items():
        print(f"  {col:<48} {c:+.3f}")

    # ---- Figures ----
    valid = valid.copy()
    valid["hour"] = valid["created_at"].dt.hour
    valid["dur_min"] = valid[TARGET] / 60

    # a) target distribution
    plt.figure(figsize=(8, 5))
    plt.hist(valid["dur_min"], bins=80, color="#e24329")
    plt.xlabel("Delivery duration (minutes)")
    plt.ylabel("Number of orders")
    plt.title("Delivery duration distribution (note the long right tail)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_target_distribution.png", dpi=110)
    plt.close()

    # b) duration by hour of day
    by_hour = valid.groupby("hour")["dur_min"].mean()
    plt.figure(figsize=(8, 5))
    plt.plot(by_hour.index, by_hour.values, marker="o", color="#e24329")
    plt.xlabel("Hour of day (order placed)")
    plt.ylabel("Avg delivery duration (min)")
    plt.title("Delivery duration by time of day")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_duration_by_hour.png", dpi=110)
    plt.close()

    # c) busy-dasher ratio vs duration (the 'undersupply' story)
    d = valid[valid["total_onshift_dashers"] > 0].copy()
    d["busy_ratio"] = (d["total_busy_dashers"] / d["total_onshift_dashers"]).clip(0, 2)
    bins = pd.cut(d["busy_ratio"], bins=20)
    by_busy = d.groupby(bins, observed=True)["dur_min"].mean()
    centers = [iv.mid for iv in by_busy.index]
    plt.figure(figsize=(8, 5))
    plt.plot(centers, by_busy.values, marker="o", color="#e24329")
    plt.xlabel("busy dashers / onshift dashers  (higher = undersupply)")
    plt.ylabel("Avg delivery duration (min)")
    plt.title("Dasher undersupply drives longer deliveries")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_undersupply_vs_duration.png", dpi=110)
    plt.close()

    print(f"\nSaved 3 figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
