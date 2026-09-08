"""Phase 3 - Baseline model: gradient boosting (XGBoost).

This mirrors DoorDash's "old" approach (tree-based models). It's a strong,
honest baseline. In Phase 4 we train a neural net and compare - exactly the
old-vs-new story from their engineering blog.

Key choices explained:
  * TIME-BASED SPLIT: the data is a time series, so we train on earlier orders
    and test on later ones. This is more honest than a random split (you can't
    use the future to predict the past).
  * TARGET ENCODING for categoricals: store_id has thousands of values.
    We replace each category with the (cross-fitted) average duration for that
    category. sklearn's TargetEncoder cross-fits internally to avoid leakage.
  * REFERENCE BASELINES: we compare against "just predict the average" and
    DoorDash's own partial time estimate, so our number has context.

Run:  python src/train_baseline.py
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBRegressor

from data import PROJECT_ROOT, TARGET, load_clean
from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, engineer_features

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Train on the earliest (1 - test_frac); test on the most recent test_frac."""
    df = df.sort_values("created_at")
    n_test = int(len(df) * test_frac)
    return df.iloc[:-n_test].copy(), df.iloc[-n_test:].copy()


def rmse_min(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)) / 60)


def mae_min(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred) / 60)


def main() -> None:
    print("Loading + cleaning data...")
    df = engineer_features(load_clean())
    train, test = time_split(df)
    print(f"  train rows: {len(train):,}   test rows: {len(test):,}")

    feats = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_tr, y_tr = train[feats], train[TARGET].values
    X_te, y_te = test[feats], test[TARGET].values

    # Preprocessing: median-impute numerics; target-encode categoricals.
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", TargetEncoder(target_type="continuous")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = Pipeline(
        [
            ("pre", pre),
            (
                "xgb",
                XGBRegressor(
                    n_estimators=600,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=5,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )

    print("Training XGBoost baseline...")
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    # Reference baselines for context
    mean_pred = np.full_like(y_te, y_tr.mean(), dtype=float)
    dd_estimate = test["estimated_total_duration"].values  # partial (place + drive only)

    print("\n" + "=" * 52)
    print(f"{'Model':<22}{'RMSE (min)':>14}{'MAE (min)':>14}")
    print("-" * 52)
    print(f"{'Predict the mean':<22}{rmse_min(y_te, mean_pred):>14.2f}{mae_min(y_te, mean_pred):>14.2f}")
    print(f"{'DoorDash partial est.':<22}{rmse_min(y_te, dd_estimate):>14.2f}{mae_min(y_te, dd_estimate):>14.2f}")
    print(f"{'XGBoost (ours)':<22}{rmse_min(y_te, pred):>14.2f}{mae_min(y_te, pred):>14.2f}")
    print("=" * 52)

    # Save model + metrics
    joblib.dump(model, MODEL_DIR / "baseline_xgb.joblib")
    metrics = {
        "model": "xgboost_baseline",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "rmse_minutes": rmse_min(y_te, pred),
        "mae_minutes": mae_min(y_te, pred),
        "baseline_mean_rmse_minutes": rmse_min(y_te, mean_pred),
    }
    with open(MODEL_DIR / "baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved model  -> {MODEL_DIR / 'baseline_xgb.joblib'}")
    print(f"Saved metrics-> {MODEL_DIR / 'baseline_metrics.json'}")


if __name__ == "__main__":
    main()
