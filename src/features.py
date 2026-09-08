"""Phase 2 - Feature engineering.

This is where we turn raw columns into signals a model can learn from.
Each block maps directly to an idea from DoorDash's ETA engineering blog:

  * time-of-day patterns        -> hour / day-of-week / cyclical encoding
  * order size affects prep     -> item counts, prices, derived ratios
  * DASHER UNDERSUPPLY (the big -> busy_ratio, orders_per_dasher, free_dashers
    driver the blog emphasizes)
  * "delivery = sum of stages"  -> combine DoorDash's own stage estimates

We only create *per-row* features here (no statistics computed across rows),
so there is no risk of leaking information from the test set. Categorical
encoding that depends on the target is done later, fit on the train split only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Final feature lists (used by the training scripts)
NUMERIC_FEATURES = [
    # time
    "hour",
    "dayofweek",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    # order composition
    "total_items",
    "subtotal",
    "num_distinct_items",
    "min_item_price",
    "max_item_price",
    "avg_item_price",
    "price_range",
    "items_per_distinct",
    # supply & demand (the "undersupply" story)
    "total_onshift_dashers",
    "total_busy_dashers",
    "total_outstanding_orders",
    "free_dashers",
    "busy_ratio",
    "orders_per_dasher",
    # DoorDash's own stage estimates
    "estimated_order_place_duration",
    "estimated_store_to_consumer_driving_duration",
    "estimated_total_duration",
]

CATEGORICAL_FEATURES = [
    "market_id",
    "store_id",
    "store_primary_category",
    "order_protocol",
]


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """a / b, but 0 or NaN in the denominator yields NaN (imputed later)."""
    return a / b.replace(0, np.nan)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered feature columns to a copy of df and return it."""
    df = df.copy()

    # --- Time features (blog: strong time-of-day / meal-time patterns) ---
    ca = df["created_at"]
    df["hour"] = ca.dt.hour
    df["dayofweek"] = ca.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    # Cyclical encoding so 23:00 and 00:00 are "close" (matters for linear/NN models)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # --- Order composition (bigger / pricier orders take longer to prep) ---
    df["avg_item_price"] = _safe_div(df["subtotal"], df["total_items"])
    df["price_range"] = df["max_item_price"] - df["min_item_price"]
    df["items_per_distinct"] = _safe_div(df["total_items"], df["num_distinct_items"])

    # --- Supply & demand: the key driver DoorDash highlights ---
    onshift = df["total_onshift_dashers"]
    df["free_dashers"] = df["total_onshift_dashers"] - df["total_busy_dashers"]
    df["busy_ratio"] = _safe_div(df["total_busy_dashers"], onshift)
    df["orders_per_dasher"] = _safe_div(df["total_outstanding_orders"], onshift)

    # --- DoorDash's own stage estimates (delivery = sum of stages) ---
    df["estimated_total_duration"] = (
        df["estimated_order_place_duration"].fillna(0)
        + df["estimated_store_to_consumer_driving_duration"].fillna(0)
    )

    return df


def get_feature_columns() -> tuple[list[str], list[str]]:
    """Return (numeric_features, categorical_features)."""
    return NUMERIC_FEATURES, CATEGORICAL_FEATURES
