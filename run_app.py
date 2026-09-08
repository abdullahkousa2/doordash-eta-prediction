"""Launcher for the demo app that works regardless of current directory."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "src"))

import uvicorn
from app import app  # app/app.py

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
