"""Offline training entrypoint: fetches real price history, generates the
placeholder news signal (see app/data/news.py docstring), trains all three
fusion variants per asset, computes SHAP + regime + consistency, and saves
one artifact bundle per asset for the API to serve.

Run from backend/: python scripts/run_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ASSETS
from app.pipeline import run_ticker_pipeline

if __name__ == "__main__":
    for asset in ASSETS:
        print(f"=== Training {asset['ticker']} ({asset['category']}/{asset['news_template']}) ===")
        bundle = run_ticker_pipeline(asset["ticker"], asset["name"], asset["news_template"])
        for name, v in bundle["variants"].items():
            print(f"  {name}: accuracy={v['accuracy']:.3f}")
        print(f"  verdict: {bundle['consistency_report']['verdict']}")
