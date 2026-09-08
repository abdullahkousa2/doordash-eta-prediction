"""Phase 5a - Probabilistic forecast (simplified version of the blog's idea).

DoorDash's blog argues a single-number ETA hides how UNCERTAIN it is, so they
predict a whole distribution (using a Weibull model). We do a much simpler but
honest version: train three gradient-boosted quantile models to predict the
10th, 50th, and 90th percentile of delivery duration.

Result: instead of "42 min", we can say "42 min, likely between 33 and 58 min".
We then check CALIBRATION: about 80% of real deliveries should land inside the
[p10, p90] band. That's the same spirit as the blog's calibration check.

Run:  python src/train_quantiles.py
"""
from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBRegressor

from data import PROJECT_ROOT, TARGET, load_clean
from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, engineer_features
from train_baseline import time_split

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}


def make_pipeline(alpha: float) -> Pipeline:
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
    return Pipeline(
        [
            ("pre", pre),
            (
                "xgb",
                XGBRegressor(
                    objective="reg:quantileerror",
                    quantile_alpha=alpha,
                    n_estimators=400,
                    max_depth=7,
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


def main() -> None:
    print("Loading data...")
    df = engineer_features(load_clean())
    train, test = time_split(df)
    feats = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_tr, y_tr = train[feats], train[TARGET].values
    X_te, y_te = test[feats], test[TARGET].values

    preds = {}
    for name, alpha in QUANTILES.items():
        print(f"Training quantile model {name} (alpha={alpha})...")
        pipe = make_pipeline(alpha)
        pipe.fit(X_tr, y_tr)
        preds[name] = pipe.predict(X_te)
        joblib.dump(pipe, MODEL_DIR / f"quantile_{name}.joblib")

    p10, p50, p90 = preds["p10"], preds["p50"], preds["p90"]

    # Calibration: fraction of actual durations inside the [p10, p90] band
    coverage = float(np.mean((y_te >= p10) & (y_te <= p90)))
    avg_width_min = float(np.mean(p90 - p10) / 60)

    print("\n" + "=" * 56)
    print("PROBABILISTIC FORECAST — test set")
    print("-" * 56)
    print(f"  Target coverage of [p10, p90] band : 80%")
    print(f"  Actual coverage (calibration)      : {coverage:.1%}")
    print(f"  Average band width                 : {avg_width_min:.1f} min")
    print("=" * 56)
    print("\nExample predictions (first 5 test orders):")
    print(f"  {'actual':>8} {'p10':>7} {'p50':>7} {'p90':>7}   (minutes)")
    for i in range(5):
        print(
            f"  {y_te[i]/60:8.1f} {p10[i]/60:7.1f} {p50[i]/60:7.1f} {p90[i]/60:7.1f}"
        )

    with open(MODEL_DIR / "quantile_metrics.json", "w") as f:
        json.dump(
            {"coverage_p10_p90": coverage, "avg_band_width_minutes": avg_width_min},
            f,
            indent=2,
        )
    print(f"\nSaved 3 quantile models + metrics to {MODEL_DIR}")


if __name__ == "__main__":
    main()
