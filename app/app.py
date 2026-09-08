"""Phase 6 - Demo web app (FastAPI).

Serves a small custom UI where you enter an order and get back a predicted
delivery time WITH an uncertainty range (p10-p90), powered by the calibrated
XGBoost quantile models. Deliberately torch-free so the deployed image stays
slim (matches the 'minimal Docker image' rule).

Run locally:  uvicorn app:app --host 0.0.0.0 --port 8000   (from the app/ folder)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Make src/ importable (feature engineering lives there)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from features import engineer_features  # noqa: E402

MODEL_DIR = PROJECT_ROOT / "models"

# ---- load models + defaults once at startup ----
META = json.loads((MODEL_DIR / "feature_defaults.json").read_text())
DEFAULTS = META["defaults"]
CUISINES = META["cuisines"]
MARKETS = META["markets"]
MODELS = {q: joblib.load(MODEL_DIR / f"quantile_{q}.joblib") for q in ("p10", "p50", "p90")}

app = FastAPI(title="DoorDash ETA Predictor")


class OrderInput(BaseModel):
    hour: int = 19
    dayofweek: int = 4          # 0=Mon .. 6=Sun
    cuisine: str = "american"
    market_id: int = 2
    total_items: int = 3
    subtotal_dollars: float = 22.0
    num_distinct_items: int = 2
    busyness: float = 0.6       # 0 = quiet, 1 = very busy (Dasher undersupply)


def build_row(o: OrderInput) -> pd.DataFrame:
    """Turn user-friendly inputs into one raw row the pipeline understands."""
    row = dict(DEFAULTS)  # start from training medians
    # a fixed reference date; we only use hour + weekday downstream
    ts = pd.Timestamp("2015-02-02") + pd.Timedelta(days=int(o.dayofweek), hours=int(o.hour))
    row["created_at"] = ts
    row["actual_delivery_time"] = ts  # unused by features, present for schema
    row["store_primary_category"] = o.cuisine
    row["market_id"] = float(o.market_id)
    row["total_items"] = int(o.total_items)
    row["subtotal"] = float(o.subtotal_dollars) * 100.0  # dollars -> cents
    row["num_distinct_items"] = int(o.num_distinct_items)

    # Map the single "busyness" slider onto supply/demand signals
    onshift = float(DEFAULTS["total_onshift_dashers"])
    b = max(0.0, min(1.0, float(o.busyness)))
    row["total_onshift_dashers"] = onshift
    row["total_busy_dashers"] = round(onshift * (0.5 + 0.5 * b))
    row["total_outstanding_orders"] = round(onshift * (0.4 + 1.6 * b))

    df = pd.DataFrame([row])
    return engineer_features(df)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models": list(MODELS)}


@app.post("/predict")
def predict(o: OrderInput) -> JSONResponse:
    df = build_row(o)
    preds = {q: float(m.predict(df)[0]) for q, m in MODELS.items()}
    # guard against quantile crossing (p10<=p50<=p90)
    p10, p50, p90 = sorted(preds.values())
    return JSONResponse(
        {
            "p10_min": round(p10 / 60, 1),
            "p50_min": round(p50 / 60, 1),
            "p90_min": round(p90 / 60, 1),
        }
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    cuisine_opts = "".join(f'<option>{c}</option>' for c in CUISINES)
    market_opts = "".join(f'<option value="{m}">Market {m}</option>' for m in MARKETS)
    return PAGE.replace("{{CUISINES}}", cuisine_opts).replace("{{MARKETS}}", market_opts)


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DoorDash ETA Predictor</title>
<style>
  :root{ --red:#eb1700; --ink:#191919; --muted:#6b6b6b; --line:#ececec; --bg:#faf7f5; --card:#fff; }
  *{box-sizing:border-box} body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    background:var(--bg);color:var(--ink);}
  .wrap{max-width:860px;margin:0 auto;padding:28px 20px 60px;}
  header{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:6px;}
  h1{font-size:24px;margin:0;letter-spacing:-.02em;}
  h1 span{color:var(--red);}
  .status{display:flex;align-items:center;gap:7px;font-size:13px;color:var(--muted);
    background:var(--card);border:1px solid var(--line);border-radius:999px;padding:6px 12px;}
  .dot{width:9px;height:9px;border-radius:50%;background:#bbb;}
  .dot.on{background:#16a34a;box-shadow:0 0 0 3px rgba(22,163,74,.15);}
  .dot.off{background:#dc2626;box-shadow:0 0 0 3px rgba(220,38,38,.15);}
  p.sub{color:var(--muted);margin:.2em 0 22px;font-size:14.5px;}
  .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px;
    box-shadow:0 1px 2px rgba(0,0,0,.03);}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px 20px;}
  label{display:block;font-size:12.5px;font-weight:600;color:var(--muted);margin-bottom:6px;text-transform:uppercase;letter-spacing:.03em;}
  input,select{width:100%;padding:10px 12px;border:1px solid var(--line);border-radius:10px;font-size:15px;background:#fff;color:var(--ink);}
  input[type=range]{padding:0}
  .full{grid-column:1/-1}
  .busyval{font-weight:700;color:var(--red)}
  button{margin-top:20px;width:100%;background:var(--red);color:#fff;border:0;border-radius:12px;
    padding:14px;font-size:16px;font-weight:700;cursor:pointer;transition:filter .15s;}
  button:hover{filter:brightness(.94)}
  .result{margin-top:22px;text-align:center;opacity:0;transition:opacity .3s;}
  .result.show{opacity:1}
  .big{font-size:52px;font-weight:800;letter-spacing:-.03em;line-height:1;}
  .big small{font-size:20px;font-weight:600;color:var(--muted)}
  .range{color:var(--muted);font-size:15px;margin-top:8px}
  .bar{position:relative;height:12px;background:#f0eeec;border-radius:999px;margin:20px auto 4px;max-width:520px;}
  .fill{position:absolute;top:0;height:100%;background:linear-gradient(90deg,#ffb199,var(--red));border-radius:999px;}
  .ticks{display:flex;justify-content:space-between;max-width:520px;margin:0 auto;color:var(--muted);font-size:12px;}
  footer{color:var(--muted);font-size:12.5px;text-align:center;margin-top:26px;line-height:1.5}
  footer a{color:var(--red);text-decoration:none}
</style></head>
<body><div class="wrap">
  <header>
    <h1>Door<span>Dash</span> ETA Predictor</h1>
    <div class="status"><span id="dot" class="dot"></span><span id="statustxt">checking…</span></div>
  </header>
  <p class="sub">Predicts total delivery duration <b>with an uncertainty range</b>, trained on 197k real DoorDash orders. Inspired by DoorDash's production ETA model.</p>

  <div class="card">
    <div class="grid">
      <div><label>Hour of day</label><input id="hour" type="number" min="0" max="23" value="19"></div>
      <div><label>Day of week</label>
        <select id="dayofweek">
          <option value="0">Monday</option><option value="1">Tuesday</option><option value="2">Wednesday</option>
          <option value="3">Thursday</option><option value="4" selected>Friday</option>
          <option value="5">Saturday</option><option value="6">Sunday</option>
        </select></div>
      <div><label>Cuisine</label><select id="cuisine">{{CUISINES}}</select></div>
      <div><label>Market</label><select id="market_id">{{MARKETS}}</select></div>
      <div><label>Total items</label><input id="total_items" type="number" min="1" max="40" value="3"></div>
      <div><label>Subtotal ($)</label><input id="subtotal_dollars" type="number" min="1" step="0.5" value="22"></div>
      <div><label>Distinct items</label><input id="num_distinct_items" type="number" min="1" max="30" value="2"></div>
      <div class="full"><label>How busy is it right now — <span class="busyval" id="bval">60%</span> (Dasher undersupply)</label>
        <input id="busyness" type="range" min="0" max="100" value="60" oninput="document.getElementById('bval').textContent=this.value+'%'"></div>
    </div>
    <button onclick="predict()">Predict delivery time</button>

    <div class="result" id="result">
      <div class="big"><span id="p50">–</span><small> min</small></div>
      <div class="bar"><div class="fill" id="fill"></div></div>
      <div class="ticks"><span id="t10">–</span><span id="t90">–</span></div>
      <div class="range">Likely between <b id="lo">–</b> and <b id="hi">–</b> minutes (80% of orders)</div>
    </div>
  </div>

  <footer>
    XGBoost quantile regression · calibrated ~77% coverage<br>
    A learning project rebuilding <a href="https://careersatdoordash.com/blog/deep-learning-for-smarter-eta-predictions/" target="_blank" rel="noopener">DoorDash's ETA system</a> on public data.
  </footer>
</div>

<script>
async function ping(){
  try{ const r=await fetch('/health'); if(!r.ok) throw 0;
    document.getElementById('dot').className='dot on';
    document.getElementById('statustxt').textContent='model online';
  }catch(e){ document.getElementById('dot').className='dot off';
    document.getElementById('statustxt').textContent='model offline'; }
}
ping(); setInterval(ping, 15000);

async function predict(){
  const body={
    hour:+val('hour'), dayofweek:+val('dayofweek'), cuisine:val('cuisine'),
    market_id:+val('market_id'), total_items:+val('total_items'),
    subtotal_dollars:+val('subtotal_dollars'), num_distinct_items:+val('num_distinct_items'),
    busyness:(+val('busyness'))/100
  };
  const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const d=await r.json();
  document.getElementById('p50').textContent=d.p50_min;
  document.getElementById('lo').textContent=d.p10_min;
  document.getElementById('hi').textContent=d.p90_min;
  document.getElementById('t10').textContent=d.p10_min+' min';
  document.getElementById('t90').textContent=d.p90_min+' min';
  const span=Math.max(d.p90_min-d.p10_min,1);
  const left=Math.max(0,Math.min(85,(d.p10_min-Math.max(0,d.p10_min-8))/(span+16)*100));
  document.getElementById('fill').style.left='8%';
  document.getElementById('fill').style.width='84%';
  document.getElementById('result').classList.add('show');
}
function val(id){return document.getElementById(id).value}
</script>
</body></html>
"""
