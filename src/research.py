"""Research layer: change score, backtests, out-of-sample models, lead-lag test.

Everything here is point-in-time: scores use only filings already public, and
models are evaluated with expanding windows whose training labels had fully
matured before the test year began.
"""
import json
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (FINANCIAL_SIC_PREFIXES, LABEL_MATURITY_DAYS, MODELS_PATH, PANEL_PATH,
                        RESEARCH_PATH)
from src.lexicon import THEMES

# ----------------------------------------------------------------------------- feature sets
SCORE_COMPONENTS = {"change": "Overall text change (1 - cosine similarity)",
                    "d_tone_negative": "Change in negative-word share",
                    "risk_added_per100": "Severity-weighted new risk sentences per 100 sentences"}
LANGUAGE = (["change", "risk_cos", "mdna_cos", "added_share", "risk_added_per100", "fcs",
             "d_tone_negative", "d_tone_uncertainty", "d_tone_litigious", "d_tone_constraining",
             "risk_len_logratio", "mdna_len_logratio"] + [f"theme_{k}" for k in THEMES])
RETURN_CONTROLS = ["log_assets", "pre_excess_63", "leverage", "zscore", "icr_log", "is_10k"]
DISTRESS_RATIOS = ["zscore", "icr_log", "leverage", "cash_assets", "current_ratio", "op_margin", "dz_yoy"]
MIN_TRAIN = 250


def add_tone_deltas(panel):
    """Tone deltas are computed in the pipeline; here only add simple flags."""
    p = panel.copy()
    p["is_10k"] = (p["form"] == "10-K").astype(float)
    return p


def _pct_rank(value, ref):
    ref = ref[np.isfinite(ref)]
    if not np.isfinite(value) or len(ref) < 20:
        return np.nan
    return 100.0 * (np.sum(ref < value) + 0.5 * np.sum(ref == value)) / len(ref)


def change_score(rows, reference):
    """Filing Change Score (0-100): mean percentile of the score components, where each
    percentile is vs. same-form filings in the reference set filed in the prior 365 days.
    Pre-specified equal weights (nothing fitted to returns)."""
    ref = reference.copy()
    ref["_d"] = pd.to_datetime(ref["filing_date"])
    out = pd.DataFrame(index=rows.index)
    for comp in SCORE_COMPONENTS:
        out[f"pct_{comp}"] = np.nan
    for idx, r in rows.iterrows():
        d = pd.Timestamp(r["filing_date"])
        win = ref.loc[(ref["form"] == r["form"]) & (ref["_d"] < d) & (ref["_d"] >= d - pd.Timedelta(days=365))]
        if len(win) < 40:   # thin year: widen to the 200 most recent EARLIER same-form filings (never future ones)
            win = ref.loc[(ref["form"] == r["form"]) & (ref["_d"] < d)].sort_values("_d").tail(200)
        for comp in SCORE_COMPONENTS:
            out.at[idx, f"pct_{comp}"] = _pct_rank(r.get(comp, np.nan), win[comp].to_numpy(dtype=float))
    out["fcs"] = out[[f"pct_{c}" for c in SCORE_COMPONENTS]].mean(axis=1, skipna=True)
    out.loc[out[[f"pct_{c}" for c in SCORE_COMPONENTS]].notna().sum(axis=1) < 2, "fcs"] = np.nan
    return out


def prepare_panel(panel):
    p = add_tone_deltas(panel)
    p = p.drop(columns=[c for c in p.columns if c.startswith("pct_") or c == "fcs"], errors="ignore")
    p = p.join(change_score(p, p))
    p["quarter"] = pd.to_datetime(p["filing_date"]).dt.to_period("Q").astype(str)
    p["is_financial"] = p["sic"].astype(str).str[:2].isin(FINANCIAL_SIC_PREFIXES)
    for col in ["excess_63", "excess_5", "pre_excess_63"]:
        lo, hi = p[col].quantile([0.01, 0.99])
        p[f"{col}_w"] = p[col].clip(lo, hi)
    return p


# ----------------------------------------------------------------------------- backtests
def newey_west_t(x, lags=2):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 6:
        return np.nan, n
    e = x - x.mean()
    var = e @ e / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        var += 2 * w * (e[L:] @ e[:-L]) / n
    se = math.sqrt(max(var, 1e-12) / n)
    return float(x.mean() / se), n


def quintile_backtest(p, score="fcs", target="excess_63_w"):
    d = p.loc[p[score].notna() & p[target].notna()].copy()
    if len(d) < 100:
        return {"status": "insufficient data", "n": int(len(d))}
    if score == "fcs":   # already a point-in-time percentile -> fixed cutoffs
        d["q"] = pd.cut(d[score], [-0.1, 20, 40, 60, 80, 100.1], labels=[1, 2, 3, 4, 5]).astype(int)
    else:                # raw measure -> sort into quintiles within each filing quarter (Lazy Prices style)
        d = d.groupby("quarter").filter(lambda g: len(g) >= 10)
        d["q"] = d.groupby("quarter")[score].transform(
            lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) + 1).astype(int)
        if score == "sim_cos":   # high similarity = LOW change; flip so Q5 = most change
            d["q"] = 6 - d["q"]
    by_q = d.groupby("q")[target].agg(["mean", "count"]).reset_index()
    cells = d.groupby(["quarter", "q"])[target].agg(["mean", "count"])
    per_quarter = cells["mean"].where(cells["count"] >= 3).unstack()   # ignore thin cells
    spread = (per_quarter.get(5) - per_quarter.get(1)).dropna()
    t, n_q = newey_west_t(spread.values)
    return {
        "status": "ok", "n": int(len(d)), "score": score,
        "quintiles": [{"q": int(r.q), "mean": float(r["mean"]), "n": int(r["count"])} for _, r in by_q.iterrows()],
        "spread_mean": float(spread.mean()) if len(spread) else None,
        "spread_t": t, "spread_quarters": n_q,
        "spread_series": [{"quarter": k, "spread": float(v)} for k, v in spread.items()],
    }


def _year_splits(p, target, min_train=MIN_TRAIN):
    d = p.loc[p[target].notna()].copy()
    d["_d"] = pd.to_datetime(d["filing_date"])
    for year in sorted(d["_d"].dt.year.unique()):
        start = pd.Timestamp(f"{year}-01-01")
        train = d.loc[d["_d"] + pd.Timedelta(days=LABEL_MATURITY_DAYS[target]) < start]
        test = d.loc[d["_d"].dt.year == year]
        if len(train) >= min_train and len(test) >= 20:
            yield year, train, test


def _ridge():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=50.0))


def _logit():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         LogisticRegression(C=0.2, max_iter=3000))


def evaluate_return_models(p):
    target = "excess_63"
    sets = {"Controls only": RETURN_CONTROLS, "Language only": LANGUAGE, "Controls + language": RETURN_CONTROLS + LANGUAGE}
    p = p.assign(excess_63=p["excess_63_w"])
    results, oos = {}, {k: [] for k in sets}
    for year, train, test in _year_splits(p, target):
        for name, cols in sets.items():
            m = _ridge().fit(train[cols], train[target])
            pred = m.predict(test[cols])
            oos[name].append(pd.DataFrame({"year": year, "pred": pred, "y": test[target].to_numpy(),
                                           "train_mean": train[target].mean()}))
    for name, frames in oos.items():
        if not frames:
            results[name] = {"status": "insufficient data"}
            continue
        o = pd.concat(frames)
        sse, sst = ((o.y - o.pred) ** 2).sum(), ((o.y - o.train_mean) ** 2).sum()
        ics = o.groupby("year").apply(lambda g: g["pred"].corr(g["y"], method="spearman"), include_groups=False)
        results[name] = {"status": "ok", "n_oos": int(len(o)), "years": [int(y) for y in ics.index],
                         "oos_r2": float(1 - sse / sst), "mean_ic": float(ics.mean()),
                         "ic_by_year": [float(v) for v in ics.values],
                         "hit_rate": float(np.mean(np.sign(o.pred - o.train_mean) == np.sign(o.y - o.train_mean))),
                         "resid_q10": float(np.quantile(o.y - o.pred, 0.10)),
                         "resid_q90": float(np.quantile(o.y - o.pred, 0.90))}
    return results


def evaluate_distress_models(p):
    target = "distress_onset_12m"
    d = p.loc[~p["is_financial"]]
    sets = {"Ratios only": DISTRESS_RATIOS, "Language only": LANGUAGE, "Ratios + language": DISTRESS_RATIOS + LANGUAGE}
    results, oos = {}, {k: [] for k in sets}
    for year, train, test in _year_splits(d, target):
        if train[target].sum() < 8:
            continue
        for name, cols in sets.items():
            m = _logit().fit(train[cols], train[target].astype(int))
            oos[name].append(pd.DataFrame({"year": year, "prob": m.predict_proba(test[cols])[:, 1],
                                           "y": test[target].astype(int).to_numpy(),
                                           "base": train[target].mean()}))
    for name, frames in oos.items():
        if not frames:
            results[name] = {"status": "insufficient events"}
            continue
        o = pd.concat(frames)
        if o.y.nunique() < 2:
            results[name] = {"status": "no events in test years"}
            continue
        results[name] = {"status": "ok", "n_oos": int(len(o)), "events_oos": int(o.y.sum()),
                         "auc": float(roc_auc_score(o.y, o.prob)), "brier": float(brier_score_loss(o.y, o.prob)),
                         "brier_base": float(brier_score_loss(o.y, o.base)), "base_rate": float(o.y.mean())}
    return results


def lead_lag(p, window=(-8, 2)):
    """Event study around distress ONSET: does the language score move before Z''?"""
    d = p.loc[~p["is_financial"]].sort_values(["cik", "filing_date"])
    events, rows = [], []
    for cik, g in d.groupby("cik"):
        g = g.reset_index(drop=True)
        s = g["distress_now"]
        for i in range(4, len(g) - 1):
            if s.iloc[i] == 1 and s.iloc[i + 1] == 1 and (s.iloc[i - 4:i] == 0).all():
                events.append((cik, g.at[i, "ticker"], g.at[i, "filing_date"]))
                for k in range(window[0], window[1] + 1):
                    if 0 <= i + k < len(g):
                        rows.append({"k": k, "fcs": g.at[i + k, "fcs"], "zscore": g.at[i + k, "zscore"]})
                break   # first onset per company only
    if not rows:
        return {"status": "no distress onsets in universe", "n_events": 0}
    r = pd.DataFrame(rows)
    agg = r.groupby("k").agg(fcs_mean=("fcs", "mean"), fcs_se=("fcs", lambda x: x.std() / max(np.sqrt(x.count()), 1)),
                             z_median=("zscore", "median"), n=("fcs", "count")).reset_index()
    pre = r.loc[r.k.between(-4, -1)]
    return {"status": "ok", "n_events": len(events),
            "events": [{"cik": int(c), "ticker": t, "onset_filing": f} for c, t, f in events],
            "path": agg.replace({np.nan: None}).to_dict("records"),
            "pre_event_fcs_mean": float(pre["fcs"].mean()) if len(pre) else None,
            "pre_event_flag_rate": float((pre["fcs"] >= 80).mean()) if len(pre) else None,
            "baseline_flag_rate": float((p["fcs"] >= 80).mean())}


# ----------------------------------------------------------------------------- fit & persist
def fit_final_models(p):
    models = {}
    r = p.loc[p["excess_63_w"].notna()]
    if len(r) >= MIN_TRAIN:
        cols = RETURN_CONTROLS + LANGUAGE
        models["return"] = {"model": _ridge().fit(r[cols], r["excess_63_w"]), "cols": cols,
                            "n": int(len(r)), "last_label": str(r["filing_date"].max())}
    d = p.loc[~p["is_financial"] & p["distress_onset_12m"].notna()]
    if len(d) >= MIN_TRAIN and d["distress_onset_12m"].sum() >= 8:
        cols = DISTRESS_RATIOS + LANGUAGE
        models["distress"] = {"model": _logit().fit(d[cols], d["distress_onset_12m"].astype(int)), "cols": cols,
                              "n": int(len(d)), "events": int(d["distress_onset_12m"].sum()),
                              "base_rate": float(d["distress_onset_12m"].mean())}
        ratio_cols = DISTRESS_RATIOS
        models["distress_ratios"] = {"model": _logit().fit(d[ratio_cols], d["distress_onset_12m"].astype(int)),
                                     "cols": ratio_cols}
    return models


def run_research(panel):
    p = prepare_panel(panel)
    p.to_parquet(PANEL_PATH, index=False)
    res = {
        "universe": {"companies": int(p["cik"].nunique()), "filings": int(len(p)),
                     "first": str(p["filing_date"].min()), "last": str(p["filing_date"].max()),
                     "with_returns": int(p["excess_63"].notna().sum()),
                     "with_distress_label": int(p["distress_onset_12m"].notna().sum()),
                     "distress_events": int(p["distress_onset_12m"].sum())},
        "backtest_fcs": quintile_backtest(p, "fcs"),
        "backtest_similarity": quintile_backtest(p, "sim_cos"),
        "backtest_fcs_5d": quintile_backtest(p, "fcs", "excess_5_w"),
        "return_models": evaluate_return_models(p),
        "distress_models": evaluate_distress_models(p),
        "lead_lag": lead_lag(p),
    }
    joblib.dump(fit_final_models(p), MODELS_PATH)
    RESEARCH_PATH.write_text(json.dumps(res, indent=2, default=lambda x: None if x is None else float(x)))
    return p, res


def load_research():
    panel = pd.read_parquet(PANEL_PATH) if PANEL_PATH.exists() else pd.DataFrame()
    res = json.loads(RESEARCH_PATH.read_text()) if RESEARCH_PATH.exists() else {}
    models = joblib.load(MODELS_PATH) if MODELS_PATH.exists() else {}
    return panel, res, models
