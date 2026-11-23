"""Build the per-filing research panel for one company.

One row = one 10-K or 10-Q. Each filing is compared with its proper predecessor:
  10-K -> previous year's 10-K
  10-Q -> SAME fiscal quarter one year earlier (avoids seasonal wording swings)
"""
import json

import numpy as np
import pandas as pd

from src.config import COMPANY_DIR, DISTRESS_HORIZON_DAYS
from src.fundamentals import _index, distress_state, ratios
from src.lexicon import TONE_CATEGORIES, THEMES, tone
from src.prices import closes, event_returns, sector_etf
from src.sections import get_sections
from src.textdiff import compare_section, safe_mean

ETF_START = {"XLRE": pd.Timestamp("2015-10-08"), "XLC": pd.Timestamp("2018-06-19")}
SECTIONS = ("risk", "mdna")


def _predecessor(filings, i):
    """Index of the comparison filing for filings[i] (same form, ~1 year earlier)."""
    cur = filings[i]
    try:
        cur_end = pd.Timestamp(cur.get("reportDate") or cur["filingDate"])
    except ValueError:
        return None
    best, best_gap = None, None
    for j in range(i - 1, -1, -1):
        p = filings[j]
        if p["form"] != cur["form"]:
            continue
        try:
            gap = (cur_end - pd.Timestamp(p.get("reportDate") or p["filingDate"])).days
        except ValueError:
            continue
        if 320 <= gap <= 410 and (best_gap is None or abs(gap - 365) < abs(best_gap - 365)):
            best, best_gap = j, gap
        if gap > 410:
            break
    return best


def analyze_filing_pair(cur_sec, prev_sec):
    """Language features for a filing vs its comparison filing."""
    feats, evidence = {}, {"added": [], "removed": []}
    theme_totals, sims, added_shares, n_sent, risk_w = {}, [], [], 0, 0.0
    d_tone = {c: [] for c in TONE_CATEGORIES}
    for s in SECTIONS:
        cur_text, prev_text = cur_sec.get(s, ""), (prev_sec or {}).get(s, "")
        t = tone(cur_text) if cur_text else {f"tone_{c}": np.nan for c in TONE_CATEGORIES}
        for c in TONE_CATEGORIES:
            feats[f"{s}_tone_{c}"] = t[f"tone_{c}"]
        if cur_text and prev_text:
            tp = tone(prev_text)
            for c in TONE_CATEGORIES:
                d_tone[c].append(t[f"tone_{c}"] - tp[f"tone_{c}"])
        feats[f"{s}_words"] = t.get("words", 0) if cur_text else 0
        if cur_text and prev_text:
            c = compare_section(cur_text, prev_text)
            feats[f"{s}_cos"], feats[f"{s}_jac"] = c["cos"], c["jac"]
            feats[f"{s}_len_logratio"] = c["len_logratio"]
            feats[f"{s}_added_share"] = c["added_share"]
            feats[f"{s}_removed_share"] = c["removed_share"]
            feats[f"{s}_n_added"] = c["n_added"]
            sims.append(c["cos"])
            added_shares.append(c["added_share"])
            n_sent += c["n_sent"]
            risk_w += c["risk_weighted_added"]
            for k, v in c["theme_counts"].items():
                theme_totals[k] = theme_totals.get(k, 0) + v
            evidence["added"] += [dict(e, section=s) for e in c["evidence_added"]]
            evidence["removed"] += [dict(e, section=s) for e in c["evidence_removed"]]
        else:
            for k in ("cos", "jac", "len_logratio", "added_share", "removed_share", "n_added"):
                feats[f"{s}_{k}"] = np.nan
    has_pair = bool(sims)
    feats["sim_cos"] = safe_mean(sims)
    feats["change"] = 1 - feats["sim_cos"] if has_pair else np.nan
    feats["added_share"] = safe_mean(added_shares)
    feats["risk_added_per100"] = 100 * risk_w / n_sent if has_pair and n_sent else np.nan
    for k in THEMES:
        feats[f"theme_{k}"] = 100 * theme_totals.get(k, 0) / n_sent if has_pair and n_sent else np.nan
        feats[f"n_theme_{k}"] = theme_totals.get(k, 0) if has_pair else np.nan
    for c in TONE_CATEGORIES:
        feats[f"tone_{c}"] = safe_mean([feats[f"{s}_tone_{c}"] for s in SECTIONS])
        feats[f"d_tone_{c}"] = safe_mean(d_tone[c])
    evidence["added"].sort(key=lambda e: -e["weight"])
    evidence["added"] = evidence["added"][:20]
    return feats, evidence


def build_company(edgar, ticker=None, cik=None, years=10, refresh=False, log=print, prices=None,
                  price_ticker=None):
    """Return a DataFrame of filings with language, accounting and return features.

    price_ticker: Yahoo symbol for prices; "-" disables prices (delisted firms whose old
    ticker now belongs to a different company)."""
    cik, ticker = edgar.resolve(ticker, cik)
    out_path = COMPANY_DIR / f"{cik}.parquet"
    if out_path.exists() and not refresh:
        df = pd.read_parquet(out_path)
        if len(df) and df["years_requested"].iloc[0] >= years:
            cutoff = (pd.Timestamp.today() - pd.Timedelta(days=int(365.25 * years))).strftime("%Y-%m-%d")
            return df.loc[df["filing_date"] >= cutoff].reset_index(drop=True)
    meta, filings = edgar.periodic_filings(cik, years=years + 1.1)  # +1y so early filings have a predecessor
    if not ticker and meta["tickers"]:
        ticker = meta["tickers"][0]
    log(f"{ticker or cik}: {meta['name']} | SIC {meta['sic']} | {len(filings)} periodic filings found")
    facts = _index(edgar.company_facts(cik))
    etf = sector_etf(meta["sic"])

    # Prices: stock, sector ETF, SPY
    psym = (ticker or "") if price_ticker is None else price_ticker
    psym = "" if psym == "-" else psym.upper().replace(".", "-")
    if prices is None:
        prices = closes([t for t in (psym, etf, "SPY") if t], start="2012-01-01")
    stock = prices.get(psym) if psym and psym in prices else None
    bench, spy = prices.get(etf), prices.get("SPY")

    sections, rows = {}, []
    cutoff = (pd.Timestamp.today() - pd.Timedelta(days=int(365.25 * years))).strftime("%Y-%m-%d")
    for i, f in enumerate(filings):
        try:
            sections[i] = get_sections(edgar, f)
        except Exception as exc:  # one bad document should not kill the company
            log(f"  skip {f['form']} {f['filingDate']}: {exc}")
            sections[i] = None
    for i, f in enumerate(filings):
        if f["filingDate"] < cutoff or sections[i] is None:
            continue
        j = _predecessor(filings, i)
        prev = filings[j] if j is not None else None
        feats, evidence = analyze_filing_pair(sections[i], sections.get(j) if j is not None else None)
        row = {
            "ticker": ticker, "cik": cik, "company": meta["name"], "sic": meta["sic"],
            "industry": meta["sic_description"], "benchmark": etf,
            "form": f["form"], "filing_date": f["filingDate"], "report_date": f.get("reportDate", ""),
            "accession": f["accessionNumber"], "url": f["url"],
            "cmp_accession": prev["accessionNumber"] if prev else "",
            "cmp_url": prev["url"] if prev else "",
            "cmp_filing_date": prev["filingDate"] if prev else "",
            "cmp_basis": ("Prior-year 10-K" if f["form"] == "10-K" else "Same quarter, prior year")
            if prev else "No comparable prior filing",
            "has_risk": bool(sections[i].get("risk")), "has_mdna": bool(sections[i].get("mdna")),
            "risk_no_change": bool(sections[i].get("risk_no_change")),
            "used_fallback": bool(sections[i].get("fallback")),
            "evidence": json.dumps(evidence),
            "years_requested": years,
        }
        row.update(feats)
        row.update(ratios(facts, f, meta["sic"]))
        row.update(event_returns(stock, bench, spy, f["filingDate"], ETF_START.get(etf)))
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = add_dynamics(df)
    df.to_parquet(out_path, index=False)
    return df


def add_dynamics(df):
    """Year-over-year ratio changes and the forward distress-onset label."""
    df = df.sort_values("filing_date").reset_index(drop=True)
    dates = pd.to_datetime(df["filing_date"])
    df["distress_now"] = df.apply(distress_state, axis=1)
    df["dz_yoy"] = np.nan
    df["distress_onset_12m"] = np.nan
    df["zscore_next_12m_min"] = np.nan
    for i in range(len(df)):
        d = dates.iloc[i]
        past = df.loc[(dates >= d - pd.Timedelta(days=400)) & (dates <= d - pd.Timedelta(days=330))
                      & (df["form"] == df.at[i, "form"])]
        if len(past) and np.isfinite(df.at[i, "zscore"]):
            df.at[i, "dz_yoy"] = df.at[i, "zscore"] - past["zscore"].iloc[-1]
        fut = df.loc[(dates > d + pd.Timedelta(days=30)) & (dates <= d + pd.Timedelta(days=DISTRESS_HORIZON_DAYS + 30))]
        states = fut["distress_now"].dropna()
        if df.at[i, "distress_now"] == 0 and len(states) >= 2:
            # persistence rule: at least two distressed filings, so one bad quarter isn't an "onset"
            df.at[i, "distress_onset_12m"] = float(states.sum() >= 2)
        if fut["zscore"].notna().any():
            df.at[i, "zscore_next_12m_min"] = fut["zscore"].min()
    return df
