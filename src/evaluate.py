"""Phase 5b - Consolidated evaluation / comparison.

Reads the metrics saved by the training scripts and prints one clean table:
the naive floor, the XGBoost baseline ("old"), and the neural net ("new").
Also saves a bar chart of RMSE by model for the README.

Run (after training):  python src/evaluate.py
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "models"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def _load(name: str) -> dict | None:
    path = MODEL_DIR / name
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main() -> None:
    baseline = _load("baseline_metrics.json")
    nn = _load("nn_metrics.json")
    quant = _load("quantile_metrics.json")

    rows = []  # (label, rmse, mae)
    if baseline:
        rows.append(("Predict the mean", baseline["baseline_mean_rmse_minutes"], None))
        rows.append(("XGBoost (baseline)", baseline["rmse_minutes"], baseline["mae_minutes"]))
    if nn:
        rows.append(("Neural net + embeddings", nn["rmse_minutes"], nn["mae_minutes"]))

    print("\n" + "=" * 56)
    print(f"{'Model':<26}{'RMSE (min)':>14}{'MAE (min)':>14}")
    print("-" * 56)
    for label, rmse, mae in rows:
        mae_s = f"{mae:>14.2f}" if mae is not None else f"{'--':>14}"
        print(f"{label:<26}{rmse:>14.2f}{mae_s}")
    print("=" * 56)

    if baseline and nn:
        b, n = baseline["rmse_minutes"], nn["rmse_minutes"]
        better = (b - n) / b * 100
        winner = "neural net" if n < b else "XGBoost baseline"
        print(f"\nBest model: {winner}")
        print(f"NN vs baseline RMSE change: {better:+.1f}%  "
              f"({'NN better' if n < b else 'baseline better'})")

    if quant:
        print(f"\nProbabilistic band [p10,p90] coverage: {quant['coverage_p10_p90']:.1%} "
              f"(target 80%), avg width {quant['avg_band_width_minutes']:.1f} min")

    # Bar chart of RMSE
    if rows:
        labels = [r[0] for r in rows]
        rmses = [r[1] for r in rows]
        colors = ["#9aa0a6", "#4285f4", "#e24329"][: len(rows)]
        plt.figure(figsize=(8, 4.5))
        bars = plt.bar(labels, rmses, color=colors)
        for bar, v in zip(bars, rmses):
            plt.text(bar.get_x() + bar.get_width() / 2, v + 0.15, f"{v:.1f}",
                     ha="center", fontsize=10)
        plt.ylabel("RMSE (minutes) — lower is better")
        plt.title("DoorDash ETA: model comparison")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "04_model_comparison.png", dpi=110)
        plt.close()
        print(f"\nSaved comparison chart -> {FIG_DIR / '04_model_comparison.png'}")


if __name__ == "__main__":
    main()
