"""The analysis agent: a fixed, auditable tool chain run for one ticker.

Steps (each logged with what it used):
  1. Resolve ticker -> SEC CIK, industry (SIC), sector benchmark
  2. Collect the company's 10-K/10-Q history and analyze each filing vs its predecessor
  3. Choose comparable companies (same SIC, closest size, re-ranked by 10-K business-
     description similarity) and analyze their filings
  4. Score every filing against the research universe (point-in-time percentiles)
  5. Apply the out-of-sample-validated models (expected excess return, distress onset)
  6. Write takeaways from the measured numbers

Why not a free-form LLM agent? Free LLMs can hallucinate tickers, filings and
numbers. Here every peer choice, comparison filing and number has a recorded,
reproducible source. The step structure leaves room to add an LLM narrator later.
"""
import time

import numpy as np
import pandas as pd

from src.config import FINANCIAL_SIC_PREFIXES
from src.peers import find_peers
from src.pipeline import build_company
from src.prices import ETF_NAMES
from src.research import add_tone_deltas, change_score
from src.takeaways import build_takeaways, pricing_label


class AnalysisAgent:
    def __init__(self, edgar, universe, research, models, progress=None):
        self.edgar, self.universe, self.research, self.models = edgar, universe, research, models
        self.progress = progress or (lambda frac, msg: None)
        self.log = []

    def step(self, msg, frac=None):
        self.log.append({"time": time.strftime("%H:%M:%S"), "step": msg})
        if frac is not None:
            self.progress(frac, msg)

    def _score(self, df):
        if df.empty:
            return df
        df = add_tone_deltas(df)
        ref = self.universe if len(self.universe) >= 200 else pd.concat([self.universe, df])
        scored = change_score(df, ref)
        return df.drop(columns=[c for c in scored.columns if c in df.columns]).join(scored)

    def _predict(self, row):
        out = {"expected": None, "interval": None, "distress": None, "distress_ratios": None}
        X = pd.DataFrame([row])
        rm = self.models.get("return")
        if rm:
            out["expected"] = float(rm["model"].predict(X.reindex(columns=rm["cols"]))[0])
            ev = self.research.get("return_models", {}).get("Controls + language", {})
            if ev.get("status") == "ok":
                out["interval"] = (out["expected"] + ev["resid_q10"], out["expected"] + ev["resid_q90"])
        financial = str(row.get("sic", ""))[:2] in FINANCIAL_SIC_PREFIXES
        if not financial and row.get("distress_now") != 1:
            dm, dr = self.models.get("distress"), self.models.get("distress_ratios")
            if dm:
                out["distress"] = float(dm["model"].predict_proba(X.reindex(columns=dm["cols"]))[0, 1])
            if dr:
                out["distress_ratios"] = float(dr["model"].predict_proba(X.reindex(columns=dr["cols"]))[0, 1])
        return out

    def run(self, ticker, years=10, n_peers=5, peer_years=3, extra_peers=()):
        self.step(f"Resolving {ticker} with SEC EDGAR", 0.02)
        cik, ticker = self.edgar.resolve(ticker)
        meta, _ = self.edgar.periodic_filings(cik, years=1)
        self.step(f"{meta['name']} — CIK {cik}, SIC {meta['sic']} ({meta['sic_description']})", 0.05)

        self.step(f"Collecting and analyzing {years} years of 10-K/10-Q filings (first run downloads from SEC)", 0.08)
        company = build_company(self.edgar, ticker=ticker, cik=cik, years=years,
                                log=lambda m: self.step(m))
        if company.empty:
            raise ValueError(f"No 10-K/10-Q filings found for {ticker} in the last {years} years.")
        company = self._score(company)
        self.step(f"Analyzed {len(company)} filings; {int(company['cmp_accession'].astype(bool).sum())} "
                  f"had a matching prior filing to compare against", 0.45)

        self.step("Selecting comparable companies (same SIC industry, closest total assets, "
                  "re-ranked by 10-K business-description similarity)", 0.5)
        known = self.universe["sic"].astype(str).unique() if len(self.universe) else ()
        peers, info = find_peers(self.edgar, cik, meta["sic"], n=n_peers, known_sics=known,
                                 log=lambda m: self.step(m))
        for t in extra_peers:
            t = t.upper().strip()
            if t and t != ticker and (peers.empty or t not in set(peers["ticker"])):
                peers = pd.concat([peers, pd.DataFrame([{"ticker": t, "match": "Added by user",
                                                         "reason": "Added manually by user"}])], ignore_index=True)
        self.step(f"Peers: {', '.join(peers['ticker']) if len(peers) else 'none found'} "
                  f"(method: {info.get('method')}; sizes from SEC XBRL frame {info.get('asof')})", 0.55)

        peer_frames = []
        for k, (_, p) in enumerate(peers.iterrows()):
            self.step(f"Analyzing peer {p['ticker']} ({peer_years}y of filings)",
                      0.55 + 0.35 * k / max(len(peers), 1))
            try:
                pf = build_company(self.edgar, ticker=p["ticker"], years=peer_years, log=lambda m: None)
                if len(pf):
                    peer_frames.append(self._score(pf).assign(peer_reason=p.get("reason", "")))
            except Exception as exc:
                self.step(f"  peer {p['ticker']} skipped: {exc}")
        peer_panel = pd.concat(peer_frames, ignore_index=True) if peer_frames else pd.DataFrame()

        latest = company.sort_values("filing_date").iloc[-1].to_dict()
        peers_latest = (peer_panel.sort_values("filing_date").groupby("ticker").tail(1)
                        if len(peer_panel) else pd.DataFrame())
        self.step("Applying out-of-sample-validated models", 0.93)
        pred = self._predict(latest)
        ev = self.research.get("return_models", {}).get("Controls + language", {})
        model_ok = ev.get("status") == "ok" and ev.get("mean_ic", 0) > 0 and ev.get("oos_r2", -1) > -0.02
        label, tone = pricing_label(pred["expected"], model_ok)
        takeaways = build_takeaways(latest, peers_latest, self.research, pred["expected"],
                                    pred["distress"], pred["distress_ratios"])
        self.step("Done", 1.0)
        return {"ticker": ticker, "cik": cik, "meta": meta, "company": company, "latest": latest,
                "peers": peers, "peer_panel": peer_panel, "peers_latest": peers_latest,
                "prediction": pred, "pricing_label": label, "pricing_tone": tone, "model_ok": model_ok,
                "benchmark_name": ETF_NAMES.get(latest.get("benchmark"), ""),
                "takeaways": takeaways, "log": self.log}
