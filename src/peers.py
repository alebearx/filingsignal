"""Comparable-company selection: transparent and reproducible.

1. Candidates = currently listed (NYSE/Nasdaq) SEC registrants with the SAME
   4-digit SIC industry code.
2. If fewer than needed, widen to the same 3-digit SIC group.
3. Pre-rank by closeness in size (|log total assets difference|) using one SEC
   XBRL "frames" call that returns total assets for every filer.
4. Re-rank a candidate pool by BUSINESS SIMILARITY: TF-IDF cosine similarity
   between each candidate's latest 10-K Item 1 (Business) and the target's,
   with word weights fitted on the research universe's business descriptions.
   This fixes coarse SIC codes (e.g. a fitness-hardware maker vs furniture makers)
   for free and reproducibly, with no AI model involved.

   peer_score = business_similarity - 0.05 * |ln(size ratio)| - 0.03 * SIC tier
   (tier 0 = same 4-digit, 1 = same 3-digit, 2 = same 2-digit)
Every peer row records WHY it was chosen.
"""
import math
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import PROCESSED
from src.sections import get_business

BUSINESS_VEC_PATH = PROCESSED / "business_tfidf.joblib"   # fitted by build.py, committed with the data

SIZE_PENALTY = 0.05
TIER_PENALTY = 0.03

LISTED_EXCHANGES = {"NYSE", "Nasdaq", "NYSE American", "NYSE MKT", "CBOE"}


def latest_business_text(edgar, cik):
    """Item 1 text of the company's most recent 10-K, or '' if unavailable (e.g. 20-F filers)."""
    try:
        _, filings = edgar.periodic_filings(cik, years=2)
    except Exception:
        return ""
    tenks = [f for f in filings if f["form"] == "10-K"]
    if not tenks:
        return ""
    try:
        return get_business(edgar, tenks[-1])
    except Exception:
        return ""


def _vectorizer(max_df):
    return TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 1), min_df=1,
                           max_df=max_df, max_features=30_000, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")


def fit_business_vectorizer(edgar, ciks, log=print):
    """Fit word weights (IDF) on the research universe's 10-K business descriptions and save them.

    Why: fitted on only the ~15 candidates, IDF cannot tell industry words ("airline", "apparel")
    from 10-K boilerplate ("employees", "regulation"); a broad reference corpus can. Tested on real
    filings: same-industry peers then score 2-3x higher than unrelated firms."""
    texts = []
    for k, c in enumerate(ciks, 1):
        if k % 25 == 0:
            log(f"   business descriptions {k}/{len(ciks)}")
        t = latest_business_text(edgar, c)
        if t:
            texts.append(t)
    if len(texts) < 20:
        log(f"   only {len(texts)} business descriptions; reference corpus not saved")
        return None
    vec = _vectorizer(0.5).fit(texts)
    joblib.dump({"vectorizer": vec, "n_docs": len(texts)}, BUSINESS_VEC_PATH)
    log(f"   reference corpus: {len(texts)} 10-K business descriptions, {len(vec.vocabulary_):,} words")
    return vec


@lru_cache(maxsize=1)
def _reference_vectorizer():
    if BUSINESS_VEC_PATH.exists():
        try:
            return joblib.load(BUSINESS_VEC_PATH)["vectorizer"]
        except Exception:
            return None
    return None


def business_similarity(target_text, texts, vectorizer="auto"):
    """Cosine similarity of TF-IDF vectors (target vs each text); NaN where text is empty.
    Uses the saved research-universe reference corpus when available, else fits on these texts."""
    sims = np.full(len(texts), np.nan)
    ok = [i for i, t in enumerate(texts) if t]
    if not target_text or not ok:
        return sims
    docs = [target_text] + [texts[i] for i in ok]
    vec = _reference_vectorizer() if vectorizer == "auto" else vectorizer
    m = vec.transform(docs) if vec is not None else _vectorizer(0.9).fit_transform(docs)
    sims[ok] = (m[1:] @ m[0].T).toarray().ravel()
    return sims


def find_peers(edgar, cik, sic, n=5, exclude=(), known_sics=(), rerank=True, pool=None, log=None):
    """known_sics: SIC codes seen in the research universe, used for the 2-digit fallback tier.
    rerank: re-rank a pool of the closest-size candidates by 10-K business-description similarity."""
    log = log or (lambda m: None)
    tmap = edgar.ticker_map()
    by_cik = {}
    for t, rec in tmap.items():
        if rec["exchange"] in LISTED_EXCHANGES:
            by_cik.setdefault(rec["cik"], t)   # first (primary) ticker per company
    assets, asof = edgar.latest_assets()
    target_assets = assets.get(int(cik))

    def candidates(sics, level):
        rows = []
        for code in sics:
            for c in edgar.companies_in_sic(code):
                if c["cik"] == int(cik) or c["cik"] not in by_cik or by_cik[c["cik"]] in exclude:
                    continue
                a = assets.get(c["cik"])
                rows.append({"ticker": by_cik[c["cik"]], "cik": c["cik"], "name": c["name"],
                             "sic": str(c.get("sic") or code), "match": level, "assets": a})
        return rows

    sic = str(sic)
    rows = candidates([sic], "Same 4-digit SIC")
    usable = {r["cik"] for r in rows if r["assets"] or not target_assets}
    if len(usable) < n and len(sic) == 4:
        siblings = [f"{sic[:3]}{d}" for d in range(10) if f"{sic[:3]}{d}" != sic]
        rows += candidates(siblings, "Same 3-digit SIC group")
    usable = {r["cik"] for r in rows if r["assets"] or not target_assets}
    if len(usable) < n and len(sic) == 4:
        two = sorted({str(s) for s in known_sics if str(s)[:2] == sic[:2] and str(s)[:3] != sic[:3]})
        rows += candidates(two, "Same 2-digit SIC division")
    if not rows:
        return pd.DataFrame(), {"asof": asof, "target_assets": target_assets}
    df = pd.DataFrame(rows).drop_duplicates("cik")
    if target_assets:
        df["size_distance"] = df["assets"].apply(
            lambda a: abs(math.log(a) - math.log(target_assets)) if a and a > 0 else 99.0)
        df["size_ratio"] = df["assets"] / target_assets
    else:
        df["size_distance"], df["size_ratio"] = 0.0, float("nan")
    df["level_rank"] = df["match"].map({"Same 4-digit SIC": 0, "Same 3-digit SIC group": 1,
                                        "Same 2-digit SIC division": 2})
    df = df.loc[df["assets"].notna() | (target_assets is None)]
    df = df.sort_values(["level_rank", "size_distance"]).reset_index(drop=True)
    info = {"asof": asof, "target_assets": target_assets, "method": "SIC + size"}
    df["business_similarity"] = np.nan
    if rerank:
        pool = pool or min(max(3 * n, 10), 15)
        df = df.head(pool).copy()
        log(f"Reading the latest 10-K business description of the target and {len(df)} candidates")
        target_text = latest_business_text(edgar, cik)
        texts = []
        for t, c in zip(df["ticker"], df["cik"]):
            log(f"  business description: {t}")
            texts.append(latest_business_text(edgar, c))
        df["business_similarity"] = business_similarity(target_text, texts)
        if df["business_similarity"].notna().any():
            size = df["size_distance"].where(df["size_distance"] < 99, 5.0)
            df["peer_score"] = (df["business_similarity"].fillna(0) - SIZE_PENALTY * size
                                - TIER_PENALTY * df["level_rank"])
            df = df.sort_values("peer_score", ascending=False).reset_index(drop=True)
            info.update(method="SIC + size, re-ranked by business similarity", pool=len(df))
        else:
            log("  business descriptions unavailable; keeping SIC + size ranking")
    df = df.head(n).reset_index(drop=True)

    def why(r):
        parts = [f"{r['match']} ({r['sic']})",
                 "assets " + (f"{r['size_ratio']:.2f}x target" if pd.notna(r["size_ratio"]) else "n/a")]
        if pd.notna(r["business_similarity"]):
            parts.append(f"business-description similarity {r['business_similarity']:.2f}")
        elif rerank:
            parts.append("no 10-K business text (similarity n/a)")
        return "; ".join(parts)

    df["reason"] = df.apply(why, axis=1)
    return df.drop(columns=["level_rank"]), info
