from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# "category" groups assets for reporting/generalization claims (equity vs futures).
# "news_template" picks which headline-phrasing set news.py uses -- futures on an
# index (ES=F) read like index headlines, futures on a physical commodity
# (GC=F, CL=F) read like commodity headlines, even though both are "futures".
ASSETS = [
    {"ticker": "AAPL", "name": "Apple", "category": "equity", "news_template": "equity"},
    {"ticker": "JPM", "name": "JPMorgan", "category": "equity", "news_template": "equity"},
    {"ticker": "XOM", "name": "Exxon Mobil", "category": "equity", "news_template": "equity"},
    {"ticker": "NVDA", "name": "NVIDIA", "category": "equity", "news_template": "equity"},
    {"ticker": "TSLA", "name": "Tesla", "category": "equity", "news_template": "equity"},
    {"ticker": "MSFT", "name": "Microsoft", "category": "equity", "news_template": "equity"},
    {"ticker": "AMZN", "name": "Amazon", "category": "equity", "news_template": "equity"},
    {"ticker": "GC=F", "name": "Gold", "category": "futures", "news_template": "commodity"},
    {"ticker": "CL=F", "name": "Crude Oil", "category": "futures", "news_template": "commodity"},
    {"ticker": "ES=F", "name": "the E-mini S&P 500", "category": "futures", "news_template": "index"},
    {"ticker": "ZN=F", "name": "the 10-Year Treasury Note", "category": "futures", "news_template": "rates"},
    {"ticker": "ZB=F", "name": "the 30-Year Treasury Bond", "category": "futures", "news_template": "rates"},
]
TICKERS = [a["ticker"] for a in ASSETS]

HISTORY_PERIOD = "2y"
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
# remainder is test

VOL_WINDOW = 20
EMBARGO_DAYS = 3

FUSION_VARIANTS = ["concat", "weighted", "coattention"]

CONSISTENCY_THRESHOLD = 0.7

RANDOM_SEED = 42
