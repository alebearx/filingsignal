"""Central settings: paths, horizons, and constants used across the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Raw downloads (large, never committed to git)
CACHE_DIR = ROOT / "cache"
SEC_CACHE = CACHE_DIR / "sec"            # JSON API responses (gzipped)
SECTION_CACHE = CACHE_DIR / "sections"   # extracted filing sections (gzipped)
PRICE_CACHE = CACHE_DIR / "prices"

# Processed research data (small, committed to git so the hosted app has it)
DATA_DIR = ROOT / "data"
PROCESSED = DATA_DIR / "processed"
COMPANY_DIR = PROCESSED / "companies"    # one parquet per company (CIK)
PANEL_PATH = PROCESSED / "panel.parquet"
RESEARCH_PATH = PROCESSED / "research.json"
MODELS_PATH = PROCESSED / "models.joblib"
UNIVERSE_CSV = DATA_DIR / "universe.csv"

for _p in (SEC_CACHE, SECTION_CACHE, PRICE_CACHE, COMPANY_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# Research design
RETURN_HORIZON_DAYS = 63      # trading days after filing ~= one quarter ("next quarter")
SHORT_HORIZON_DAYS = 5        # immediate market reaction window
DISTRESS_HORIZON_DAYS = 365   # distress-onset look-ahead
LABEL_MATURITY_DAYS = {"excess_63": 100, "distress_onset_12m": 400}

# Altman Z'' (non-manufacturer / book-equity version, Altman 1995) zone cutoffs
Z_DISTRESS = 1.1
Z_SAFE = 2.6
ICR_DISTRESS = 1.0            # EBIT does not cover interest expense

FINANCIAL_SIC_PREFIXES = ("60", "61", "62", "63", "64", "67")  # Z'' not meaningful
