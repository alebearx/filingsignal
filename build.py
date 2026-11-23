"""Build the research universe and run the backtests (run on your own computer).

Examples
  python build.py --limit 5 --years 3      # quick test (~5 minutes)
  python build.py                          # full universe, 10 years (a few hours; resumable)
  python build.py --research-only          # re-run backtests/models on already-built data

The build is resumable: finished companies are saved in data/processed/companies/
and skipped next time (use --refresh to rebuild them).
"""
import argparse
import time

import pandas as pd
from dotenv import load_dotenv

from src.config import COMPANY_DIR, UNIVERSE_CSV
from src.edgar import Edgar
from src.peers import BUSINESS_VEC_PATH, fit_business_vectorizer
from src.pipeline import add_dynamics, build_company
from src.prices import closes
from src.research import run_research

ETFS = ["SPY", "XLK", "XLC", "XLY", "XLP", "XLV", "XLF", "XLE", "XLI", "XLB", "XLU", "XLRE"]


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="only the first N companies (for testing)")
    ap.add_argument("--tickers", nargs="+", default=None, help="only these universe tickers (for testing)")
    ap.add_argument("--refresh", action="store_true", help="rebuild companies already on disk")
    ap.add_argument("--research-only", action="store_true")
    ap.add_argument("--peer-corpus", action="store_true",
                    help="(re)fit the peer business-similarity reference corpus even on a partial build")
    args = ap.parse_args()

    uni = pd.read_csv(UNIVERSE_CSV, dtype=str).fillna("")
    if args.tickers:
        uni = uni.loc[uni["ticker"].isin([t.upper() for t in args.tickers])]
    if args.limit:
        uni = uni.head(args.limit)

    if not args.research_only:
        edgar = Edgar()
        price_syms = [r.price_ticker or r.ticker for r in uni.itertuples() if r.price_ticker != "-"]
        print(f"Downloading prices for {len(price_syms)} stocks + {len(ETFS)} ETFs ...", flush=True)
        prices = closes(price_syms + ETFS, start=f"{pd.Timestamp.today().year - args.years - 2}-01-01")
        t0 = time.time()
        for k, r in enumerate(uni.itertuples(), 1):
            print(f"[{k}/{len(uni)}] {r.ticker}  (elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
            try:
                build_company(edgar, ticker=r.ticker, cik=r.cik or None, years=args.years,
                              refresh=args.refresh, prices=prices, price_ticker=r.price_ticker or None)
            except Exception as exc:
                print(f"   ERROR {r.ticker}: {exc}", flush=True)

    full_build = not (args.tickers or args.limit)
    if args.peer_corpus or (full_build and not args.research_only) or (full_build and not BUSINESS_VEC_PATH.exists()):
        print("Fitting the peer business-similarity reference corpus (10-K Item 1 texts) ...", flush=True)
        edgar = Edgar()
        tmap = edgar.ticker_map()
        ciks = [int(r.cik) if r.cik else tmap.get(r.ticker.upper(), {}).get("cik") for r in uni.itertuples()]
        fit_business_vectorizer(edgar, [c for c in ciks if c], log=lambda m: print(m, flush=True))

    files = sorted(COMPANY_DIR.glob("*.parquet"))
    panel = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # all universe members already built, not just this run's --tickers/--limit subset, so a quick
    # partial run never shrinks the saved research results
    full_uni = pd.read_csv(UNIVERSE_CSV, dtype=str).fillna("")
    panel = panel.loc[panel["ticker"].isin(set(full_uni["ticker"].str.upper()))]
    if args.years:
        cutoff = (pd.Timestamp.today() - pd.Timedelta(days=int(365.25 * args.years))).strftime("%Y-%m-%d")
        panel = panel.loc[panel["filing_date"] >= cutoff]
    # recompute ratio dynamics + distress labels so definition changes apply without re-downloading
    panel = pd.concat([add_dynamics(g) for _, g in panel.groupby("cik")], ignore_index=True)
    print(f"\nPanel: {panel['cik'].nunique()} companies, {len(panel):,} filings. Running research ...", flush=True)
    _, res = run_research(panel)
    u = res["universe"]
    print(f"Filings with 63-day returns: {u['with_returns']:,}; distress-labelled: {u['with_distress_label']:,} "
          f"({u['distress_events']} onsets)")
    bt = res["backtest_fcs"]
    if bt.get("status") == "ok" and bt.get("spread_mean") is not None and bt.get("spread_t") is not None:
        print("Change-score quintiles (mean 63d excess return):",
              ", ".join(f"Q{x['q']}: {x['mean']:+.2%}" for x in bt["quintiles"]),
              f"| Q5-Q1 {bt['spread_mean']:+.2%}/qtr, t={bt['spread_t']:.2f}")
    for name, m in res["return_models"].items():
        if m.get("status") == "ok" and m.get("oos_r2") is not None:
            print(f"Return model [{name}]: OOS R2 {m['oos_r2']:+.3f}, mean IC {m['mean_ic']:+.3f}")
    for name, m in res["distress_models"].items():
        if m.get("status") == "ok" and m.get("auc") is not None:
            print(f"Distress model [{name}]: AUC {m['auc']:.3f}, Brier {m['brier']:.4f} (base {m['brier_base']:.4f})")
    print("Saved data/processed/panel.parquet, research.json, models.joblib (+ business_tfidf.joblib)")


if __name__ == "__main__":
    main()
