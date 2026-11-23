"""Prices (Yahoo Finance via yfinance, free/unofficial) and benchmark-adjusted returns.

"Over/underpriced" is measured as EXCESS return: the stock's return minus its
sector ETF's return over the same window, starting the trading day AFTER the
filing date (so the filing is public before the hypothetical trade).
"""
import time

import numpy as np
import pandas as pd

from src.config import PRICE_CACHE

# SPDR sector ETFs. XLRE starts Oct-2015 and XLC Jun-2018; earlier dates fall back to SPY.
ETF_NAMES = {"XLK": "Technology", "XLC": "Communication Services", "XLY": "Consumer Discretionary",
             "XLP": "Consumer Staples", "XLV": "Health Care", "XLF": "Financials", "XLE": "Energy",
             "XLI": "Industrials", "XLB": "Materials", "XLU": "Utilities", "XLRE": "Real Estate",
             "SPY": "S&P 500 (broad market)"}


def sector_etf(sic):
    """Map a 4-digit SIC code to a SPDR sector ETF (rule-based, documented)."""
    try:
        s = int(str(sic)[:4])
    except ValueError:
        return "SPY"
    two, three = s // 100, s // 10
    if two == 13 or two == 29:
        return "XLE"
    if three == 283 or three in (384, 385) or two == 80 or s == 5122:
        return "XLV"
    if three == 284 or two in (1, 2, 20, 21) or two == 54 or s == 5912:
        return "XLP"
    if three in (357, 367, 366) or (three == 737) or three == 382:
        return "XLK"
    if two == 48 or two == 78 or two == 27:
        return "XLC"
    if two == 49:
        return "XLU"
    if s == 6798 or two == 65:
        return "XLRE"
    if 60 <= two <= 67:
        return "XLF"
    if three == 371 or two in (22, 23, 25, 31, 39, 70, 79, 82) or 52 <= two <= 59:
        return "XLY"
    if two in (10, 12, 14, 24, 26, 28, 30, 32, 33):
        return "XLB"
    if two in (15, 16, 17, 34, 35, 36, 37, 38, 50, 51, 87) or 40 <= two <= 47 or 72 <= two <= 76:
        return "XLI"
    return "SPY"


def _download(tickers, start):
    import yfinance as yf
    for attempt in range(3):
        try:
            data = yf.download(tickers, start=start, auto_adjust=True, progress=False,
                               threads=True, group_by="column")
            if data is not None and not data.empty:
                close = data["Close"]
                if isinstance(close, pd.Series):
                    close = close.to_frame(tickers[0] if isinstance(tickers, list) else tickers)
                return close
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    return pd.DataFrame()


def closes(tickers, start="2012-01-01", max_age_days=1):
    """Adjusted close prices, one column per ticker, cached per ticker on disk."""
    tickers = sorted({t.upper().replace(".", "-") for t in tickers if t})
    out, need = {}, []
    for t in tickers:
        p = PRICE_CACHE / f"{t}.csv"
        if p.exists() and (time.time() - p.stat().st_mtime) / 86400 < max_age_days:
            s = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
            if len(s) and s.index.min() <= pd.Timestamp(start) + pd.Timedelta(days=10):
                out[t] = s
                continue
        need.append(t)
    for i in range(0, len(need), 50):
        batch = need[i:i + 50]
        data = _download(batch, start)
        for t in batch:
            if t in data.columns:
                s = data[t].dropna()
                if len(s):
                    s.to_frame("close").to_csv(PRICE_CACHE / f"{t}.csv")
                    out[t] = s
    return pd.DataFrame(out).sort_index()


def _ret(series, i0, i1):
    if i0 < 0 or i1 >= len(series) or i1 <= i0:
        return np.nan
    a, b = series.iloc[i0], series.iloc[i1]
    return float(b / a - 1) if np.isfinite(a) and np.isfinite(b) and a > 0 else np.nan


def event_returns(stock, bench, spy, filed, etf_start=None, horizons=(5, 63)):
    """Excess returns after a filing date plus pre-filing momentum (point-in-time control)."""
    out = {f"excess_{h}": np.nan for h in horizons}
    out.update(ret_63=np.nan, pre_excess_63=np.nan)
    if stock is None or stock.dropna().empty:
        return out
    stock = stock.dropna()
    d = pd.Timestamp(filed)
    i0 = int(stock.index.searchsorted(d + pd.Timedelta(days=1)))   # first session AFTER filing day
    if i0 >= len(stock) or (stock.index[i0] - d).days > 7:
        return out
    use_bench = bench if (bench is not None and len(bench.dropna()) and
                          (etf_start is None or d >= etf_start)) else spy
    use_bench = use_bench.reindex(stock.index).ffill()
    for h in horizons:
        i1 = i0 + h
        if i1 < len(stock) and (stock.index[i1] - stock.index[i0]).days <= h * 1.6 + 10:
            r, b = _ret(stock, i0, i1), _ret(use_bench, i0, i1)
            out[f"excess_{h}"] = r - b if np.isfinite(r) and np.isfinite(b) else np.nan
            if h == 63:
                out["ret_63"] = r
    r, b = _ret(stock, i0 - 64, i0 - 1), _ret(use_bench, i0 - 64, i0 - 1)
    out["pre_excess_63"] = r - b if np.isfinite(r) and np.isfinite(b) else np.nan
    return out
