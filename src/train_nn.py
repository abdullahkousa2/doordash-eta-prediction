"""Phase 4 - Neural network with embeddings (the DoorDash "new" approach).

The single most important idea from DoorDash's blog is using EMBEDDINGS for
high-cardinality categoricals (store_id, market, cuisine) instead of one-hot
encoding. This script builds a small tabular neural net that does exactly that:

    [ store_id embed ] ┐
    [ market   embed ] ├─ concat ─> MLP ─> predicted duration
    [ cuisine  embed ] ┘
    [ numeric features ]

It's intentionally much smaller than DoorDash's Mixture-of-Experts model, but
it teaches the same core concept and lets us compare "old" (XGBoost) vs "new".

Run:  python src/train_nn.py
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from data import PROJECT_ROOT, TARGET, load_clean
from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, engineer_features

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------- preprocessing -----------------------------
def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("created_at")
    n_test = int(len(df) * test_frac)
    return df.iloc[:-n_test].copy(), df.iloc[-n_test:].copy()


def build_category_vocabs(train: pd.DataFrame) -> dict[str, dict]:
    """Map each category value -> integer index (0 reserved for unknown/missing)."""
    vocabs = {}
    for col in CATEGORICAL_FEATURES:
        cats = train[col].dropna().unique().tolist()
        vocabs[col] = {v: i + 1 for i, v in enumerate(cats)}  # 0 = unknown
    return vocabs


def encode_categoricals(df: pd.DataFrame, vocabs: dict[str, dict]) -> np.ndarray:
    cols = []
    for col in CATEGORICAL_FEATURES:
        mapping = vocabs[col]
        cols.append(df[col].map(mapping).fillna(0).astype(np.int64).values)
    return np.stack(cols, axis=1)


# ------------------------------- the model -------------------------------
class ETANet(nn.Module):
    def __init__(self, cardinalities: list[int], n_numeric: int):
        super().__init__()
        # One embedding table per categorical. Rule of thumb for embed dim.
        self.embeddings = nn.ModuleList()
        emb_total = 0
        for card in cardinalities:
            dim = int(min(50, round(1.6 * card**0.56)))
            dim = max(dim, 2)
            self.embeddings.append(nn.Embedding(card + 1, dim))  # +1 for unknown idx 0
            emb_total += dim

        in_dim = emb_total + n_numeric
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x_cat, x_num):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [x_num], dim=1)
        return self.mlp(x).squeeze(1)


def rmse_min(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)) / 60)


def mae_min(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred) / 60)


# --------------------------------- train ---------------------------------
def main(epochs: int = 15, batch_size: int = 1024, lr: float = 1e-3) -> None:
    print(f"Device: {DEVICE}")
    df = engineer_features(load_clean())
    train, test = time_split(df)
    print(f"  train rows: {len(train):,}   test rows: {len(test):,}")

    # --- numerics: median impute (train stats) + standardize ---
    num_median = train[NUMERIC_FEATURES].median()
    num_mean = train[NUMERIC_FEATURES].fillna(num_median).mean()
    num_std = train[NUMERIC_FEATURES].fillna(num_median).std().replace(0, 1)

    def prep_numeric(d):
        z = (d[NUMERIC_FEATURES].fillna(num_median) - num_mean) / num_std
        return z.values.astype(np.float32)

    # --- categoricals: integer-encode via train vocab ---
    vocabs = build_category_vocabs(train)
    cardinalities = [len(vocabs[c]) for c in CATEGORICAL_FEATURES]
    print("  categorical cardinalities:", dict(zip(CATEGORICAL_FEATURES, cardinalities)))

    Xtr_num, Xte_num = prep_numeric(train), prep_numeric(test)
    Xtr_cat = encode_categoricals(train, vocabs)
    Xte_cat = encode_categoricals(test, vocabs)

    # --- target: standardize for stable training, invert for evaluation ---
    y_tr_raw = train[TARGET].values.astype(np.float32)
    y_te_raw = test[TARGET].values.astype(np.float32)
    y_mean, y_std = y_tr_raw.mean(), y_tr_raw.std()
    y_tr = (y_tr_raw - y_mean) / y_std

    train_ds = TensorDataset(
        torch.tensor(Xtr_cat), torch.tensor(Xtr_num), torch.tensor(y_tr)
    )
    # drop_last avoids a size-1 final batch, which would break BatchNorm
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = ETANet(cardinalities, len(NUMERIC_FEATURES)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    Xte_cat_t = torch.tensor(Xte_cat).to(DEVICE)
    Xte_num_t = torch.tensor(Xte_num).to(DEVICE)

    print("Training neural net...")
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb_cat, xb_num, yb in train_dl:
            xb_cat, xb_num, yb = xb_cat.to(DEVICE), xb_num.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb_cat, xb_num)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(yb)
        # eval on test
        model.eval()
        with torch.no_grad():
            pred_std = model(Xte_cat_t, Xte_num_t).cpu().numpy()
        pred = pred_std * y_std + y_mean
        print(
            f"  epoch {epoch:2d}  train_mse(std)={running/len(train_ds):.4f}"
            f"   test_RMSE={rmse_min(y_te_raw, pred):.2f} min"
            f"   test_MAE={mae_min(y_te_raw, pred):.2f} min"
        )

    # final metrics
    metrics = {
        "model": "embedding_mlp",
        "rmse_minutes": rmse_min(y_te_raw, pred),
        "mae_minutes": mae_min(y_te_raw, pred),
        "epochs": epochs,
    }
    with open(MODEL_DIR / "nn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # save weights + preprocessing so the demo app can reuse them
    torch.save(model.state_dict(), MODEL_DIR / "nn_model.pt")
    joblib.dump(
        {
            "vocabs": vocabs,
            "cardinalities": cardinalities,
            "num_median": num_median,
            "num_mean": num_mean,
            "num_std": num_std,
            "y_mean": float(y_mean),
            "y_std": float(y_std),
        },
        MODEL_DIR / "nn_preprocess.joblib",
    )
    print(f"\nSaved NN -> {MODEL_DIR / 'nn_model.pt'}")
    print(f"Saved metrics -> {MODEL_DIR / 'nn_metrics.json'}")


if __name__ == "__main__":
    main()
