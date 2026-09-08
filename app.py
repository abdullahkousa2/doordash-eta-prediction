"""Hugging Face Spaces demo (Gradio SDK).

Docker Spaces now require a paid HF plan, so the hosted demo runs on the free
Gradio SDK. The original FastAPI version (custom HTML/CSS UI) still lives in
app/ and can be run locally with `python run_app.py`.

Predictions come from the three calibrated XGBoost quantile models, so the app
returns a delivery-time RANGE (p10-p50-p90), not just a single number.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from features import engineer_features  # noqa: E402

MODEL_DIR = ROOT / "models"
META = json.loads((MODEL_DIR / "feature_defaults.json").read_text())
DEFAULTS = META["defaults"]
CUISINES = META["cuisines"]
MARKETS = [str(m) for m in META["markets"]]
MODELS = {q: joblib.load(MODEL_DIR / f"quantile_{q}.joblib") for q in ("p10", "p50", "p90")}

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# --- Hugging Face ZeroGPU compatibility --------------------------------------
# HF's free tier now offers only ZeroGPU for new Spaces, and a ZeroGPU Space
# refuses to start unless it detects a @spaces.GPU function ("No @spaces.GPU
# function detected during startup"). This model is pure CPU (XGBoost), so we
# register one tiny probe to satisfy that startup check and keep the real
# inference on CPU — no GPU is requested per prediction. Guarded in try/except
# so the app still runs locally, where the `spaces` package isn't installed.
try:  # pragma: no cover - platform-specific
    import spaces

    @spaces.GPU
    def _zerogpu_startup_probe():
        """Exists only so ZeroGPU Spaces pass their startup check."""
        return "ok"

except Exception:  # not on HF / package unavailable -> plain CPU app
    pass


def _build_row(hour, day, cuisine, market, total_items, subtotal, distinct_items, busyness):
    """Turn friendly UI inputs into one raw row the trained pipeline understands."""
    row = dict(DEFAULTS)
    ts = pd.Timestamp("2015-02-02") + pd.Timedelta(days=DAYS.index(day), hours=int(hour))
    row["created_at"] = ts
    row["actual_delivery_time"] = ts  # unused by features; present for schema
    row["store_primary_category"] = cuisine
    row["market_id"] = float(market)
    row["total_items"] = int(total_items)
    row["subtotal"] = float(subtotal) * 100.0  # dollars -> cents
    row["num_distinct_items"] = int(distinct_items)

    # One "busyness" slider drives the Dasher supply/demand signals
    onshift = float(DEFAULTS["total_onshift_dashers"])
    b = max(0.0, min(1.0, float(busyness) / 100.0))
    row["total_onshift_dashers"] = onshift
    row["total_busy_dashers"] = round(onshift * (0.5 + 0.5 * b))
    row["total_outstanding_orders"] = round(onshift * (0.4 + 1.6 * b))

    return engineer_features(pd.DataFrame([row]))


def _result_html(p10: float, p50: float, p90: float) -> str:
    return f"""
<div class="dd-result">
  <div class="dd-big">{p50:.1f}<span>min</span></div>
  <div class="dd-bar"><div class="dd-fill"></div></div>
  <div class="dd-ticks"><span>{p10:.1f} min</span><span>{p90:.1f} min</span></div>
  <div class="dd-range">Likely between <b>{p10:.1f}</b> and <b>{p90:.1f}</b> minutes
    <span class="dd-dim">(80% of orders)</span></div>
</div>"""


def predict(hour, day, cuisine, market, total_items, subtotal, distinct_items, busyness):
    df = _build_row(hour, day, cuisine, market, total_items, subtotal, distinct_items, busyness)
    # sort guards against quantile crossing
    p10, p50, p90 = sorted(float(m.predict(df)[0]) / 60 for m in MODELS.values())
    return _result_html(p10, p50, p90)


CSS = """
.gradio-container{max-width:900px !important;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
#dd-head{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:2px;}
#dd-head h1{font-size:26px;margin:0;letter-spacing:-.02em;}
#dd-head h1 span{color:#eb1700;}
.dd-pill{display:inline-flex;align-items:center;gap:7px;font-size:13px;color:#6b6b6b;
  border:1px solid #e5e5e5;border-radius:999px;padding:6px 12px;background:#fff;}
.dd-dot{width:9px;height:9px;border-radius:50%;background:#16a34a;box-shadow:0 0 0 3px rgba(22,163,74,.15);}
#dd-sub{color:#6b6b6b;font-size:14.5px;margin:4px 0 14px;}
.dd-result{text-align:center;padding:10px 0 4px;}
.dd-big{font-size:54px;font-weight:800;letter-spacing:-.03em;line-height:1;color:#191919;}
.dd-big span{font-size:20px;font-weight:600;color:#6b6b6b;margin-left:6px;}
.dd-bar{position:relative;height:12px;background:#f0eeec;border-radius:999px;margin:18px auto 6px;max-width:460px;}
.dd-fill{position:absolute;left:8%;width:84%;top:0;height:100%;
  background:linear-gradient(90deg,#ffb199,#eb1700);border-radius:999px;}
.dd-ticks{display:flex;justify-content:space-between;max-width:460px;margin:0 auto;color:#6b6b6b;font-size:12px;}
.dd-range{color:#444;font-size:15px;margin-top:10px;}
.dd-dim{color:#8a8a8a;}
#dd-foot{color:#8a8a8a;font-size:12.5px;text-align:center;margin-top:18px;line-height:1.6;}
#dd-foot a{color:#eb1700;text-decoration:none;}
"""

with gr.Blocks(title="DoorDash ETA Predictor") as demo:
    gr.HTML(
        '<div id="dd-head"><h1>Door<span>Dash</span> ETA Predictor</h1>'
        '<span class="dd-pill"><span class="dd-dot"></span>model online</span></div>'
        '<p id="dd-sub">Predicts total delivery duration <b>with an uncertainty range</b>, '
        'trained on 197k real DoorDash orders. Inspired by DoorDash\'s production ETA model.</p>'
    )

    with gr.Row():
        hour = gr.Slider(0, 23, value=19, step=1, label="Hour of day")
        day = gr.Dropdown(DAYS, value="Friday", label="Day of week")
    with gr.Row():
        cuisine = gr.Dropdown(CUISINES, value=CUISINES[0], label="Cuisine")
        market = gr.Dropdown(MARKETS, value=MARKETS[0], label="Market")
    with gr.Row():
        total_items = gr.Number(value=3, precision=0, label="Total items")
        subtotal = gr.Number(value=22, label="Subtotal ($)")
        distinct_items = gr.Number(value=2, precision=0, label="Distinct items")

    busyness = gr.Slider(
        0, 100, value=60, step=1,
        label="How busy is it right now? (%)  —  higher = Dasher undersupply",
    )

    btn = gr.Button("Predict delivery time", variant="primary", size="lg")
    out = gr.HTML(_result_html(27.0, 35.0, 48.0))

    inputs = [hour, day, cuisine, market, total_items, subtotal, distinct_items, busyness]
    btn.click(predict, inputs=inputs, outputs=out)

    gr.HTML(
        '<div id="dd-foot">XGBoost quantile regression · calibrated ~77% coverage<br>'
        'A learning project rebuilding '
        '<a href="https://careersatdoordash.com/blog/deep-learning-for-smarter-eta-predictions/" '
        'target="_blank" rel="noopener">DoorDash\'s ETA system</a> on public data · '
        '<a href="https://github.com/abdullahkousa2/doordash-eta-prediction" target="_blank" '
        'rel="noopener">source on GitHub</a></div>'
    )

if __name__ == "__main__":
    # Gradio 6 moved css/theme from the Blocks constructor to launch()
    demo.launch(css=CSS, theme=gr.themes.Soft())
