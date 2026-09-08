"""Data loading, target creation, and cleaning for the DoorDash ETA project.

The task (exactly as DoorDash framed it in their take-home / engineering blog):
    Predict the TOTAL DELIVERY DURATION in seconds, from the moment the
    customer places the order (`created_at`) to when it arrives
    (`actual_delivery_time`).

Everything else in the project builds on top of the functions here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Project paths (this file lives in <project>/src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "historical_data.csv"

# Columns that are categorical by nature (IDs / labels, not quantities)
CATEGORICAL_COLS = ["market_id", "store_id", "store_primary_category", "order_protocol"]

# Columns that are true numeric quantities
NUMERIC_COLS = [
    "total_items",
    "subtotal",
    "num_distinct_items",
    "min_item_price",
    "max_item_price",
    "total_onshift_dashers",
    "total_busy_dashers",
    "total_outstanding_orders",
    "estimated_order_place_duration",
    "estimated_store_to_consumer_driving_duration",
]

TARGET = "delivery_duration_seconds"


def load_raw(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV, parsing timestamps and treating 'NA' as missing."""
    df = pd.read_csv(
        path,
        parse_dates=["created_at", "actual_delivery_time"],
        na_values=["NA", "nan", ""],
    )
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the regression target: delivery duration in seconds."""
    df = df.copy()
    df[TARGET] = (df["actual_delivery_time"] - df["created_at"]).dt.total_seconds()
    return df


def clean(
    df: pd.DataFrame,
    min_seconds: int = 60,
    max_seconds: int = 3 * 3600,
) -> pd.DataFrame:
    """Drop rows with no target and remove impossible / extreme durations.

    Defaults keep deliveries between 1 minute and 3 hours. We picked these
    bounds after looking at the distribution in the EDA step (Phase 1).
    """
    df = df.copy()
    df = df.dropna(subset=[TARGET])
    mask = (df[TARGET] >= min_seconds) & (df[TARGET] <= max_seconds)
    df = df[mask]
    return df.reset_index(drop=True)


def load_clean(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Convenience: raw -> target -> cleaned, in one call."""
    return clean(add_target(load_raw(path)))
