"""Official SEC EDGAR access: free, no API key, rate-limited and cached.

SEC fair-access rules: identify yourself with a User-Agent containing a contact
email, and stay under 10 requests/second. See
https://www.sec.gov/about/webmaster-frequently-asked-questions
"""
import gzip
import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from urllib.parse import quote

import requests

from src.config import SEC_CACHE

DATA = "https://data.sec.gov"
WWW = "https://www.sec.gov"


class Edgar:
    def __init__(self, user_agent=None, delay=0.12):
        self.agent = (user_agent or os.getenv("SEC_USER_AGENT", "")).strip()
        if "@" not in self.agent or "example.com" in self.agent:
            raise ValueError(
                "SEC_USER_AGENT is not set. Put e.g. SEC_USER_AGENT=\"Your Name your.email@school.edu\" "
                "in the .env file (locally) or in Streamlit Secrets (hosted)."
            )
        self.delay = max(float(delay), 0.11)  # <= ~9 requests/second
        self._last = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.agent, "Accept-Encoding": "gzip, deflate"})

    # ------------------------------------------------------------------ transport
    def fetch(self, url, cache=True, max_age_days=None):
        """GET with polite rate limiting, retries, and an optional gzip disk cache."""
        path = SEC_CACHE / (hashlib.sha256(url.encode()).hexdigest() + ".gz")
        if cache and path.exists():
            age = (time.time() - path.stat().st_mtime) / 86400
            if max_age_days is None or age < max_age_days:
                return gzip.decompress(path.read_bytes())
        last_error = None
        for attempt in range(5):
            wait = self.delay - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                r = self.session.get(url, timeout=60)
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(2 ** attempt)
                continue
            finally:
                self._last = time.monotonic()
            if r.status_code == 404:
                raise FileNotFoundError(url)
            if r.status_code in (403, 429, 500, 502, 503, 504):
                last_error = f"HTTP {r.status_code}"
                wait = 2 ** attempt + 1 if r.status_code >= 500 else 10 * (attempt + 1)
                print(f"   SEC returned {r.status_code} (throttled or busy); waiting {wait}s. "
                      f"Avoid running two SEC jobs at once.", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            if cache:
                path.write_bytes(gzip.compress(r.content))
            return r.content
        raise ConnectionError(f"SEC request failed ({last_error}): {url}")

    def json(self, url, max_age_days=None):
        return json.loads(self.fetch(url, max_age_days=max_age_days).decode("utf-8"))

    # ------------------------------------------------------------------ lookups
    def ticker_map(self):
        """{TICKER: {cik, name, exchange}} for currently listed SEC registrants."""
        raw = self.json(f"{WWW}/files/company_tickers_exchange.json", max_age_days=7)
        cols = raw["fields"]
        out = {}
        for row in raw["data"]:
            rec = dict(zip(cols, row))
            if rec.get("ticker"):
                out[rec["ticker"].upper()] = {"cik": int(rec["cik"]), "name": rec["name"],
                                              "exchange": rec.get("exchange") or ""}
        return out

    def resolve(self, ticker=None, cik=None):
        """Return (cik, ticker) from either. Delisted firms need an explicit CIK."""
        if cik:
            return int(cik), (ticker or "").upper()
        tmap = self.ticker_map()
        t = (ticker or "").upper().strip().replace(".", "-")
        if t not in tmap:
            raise ValueError(f"Ticker '{ticker}' not found in SEC's list of currently listed companies.")
        return tmap[t]["cik"], t

    def submissions(self, cik, max_age_days=2):
        """Company metadata + full filing index (recent and archived pages)."""
        data = self.json(f"{DATA}/submissions/CIK{int(cik):010d}.json", max_age_days=max_age_days)
        recent = data["filings"]["recent"]
        rows = [dict(zip(recent, vals)) for vals in zip(*recent.values())]
        for page in data["filings"].get("files", []):
            older = self.json(f"{DATA}/submissions/{page['name']}")  # archived pages never change
            rows.extend(dict(zip(older, vals)) for vals in zip(*older.values()))
        return data, rows

    def company_facts(self, cik, max_age_days=7):
        try:
            return self.json(f"{DATA}/api/xbrl/companyfacts/CIK{int(cik):010d}.json", max_age_days=max_age_days)
        except FileNotFoundError:
            return {"facts": {}}

    def periodic_filings(self, cik, years=10):
        """10-K and 10-Q filings (no amendments) filed in the last `years` years, oldest first."""
        info, rows = self.submissions(cik)
        cutoff = (date.today() - timedelta(days=int(365.25 * years))).isoformat()
        keep = []
        for r in rows:
            if r.get("form") not in ("10-K", "10-Q"):
                continue
            if r.get("filingDate", "") < cutoff or not r.get("primaryDocument"):
                continue
            acc = r["accessionNumber"].replace("-", "")
            r["url"] = f"{WWW}/Archives/edgar/data/{int(cik)}/{acc}/{quote(r['primaryDocument'], safe='/-_.')}"
            keep.append(r)
        keep.sort(key=lambda r: (r["filingDate"], r["accessionNumber"]))
        meta = {"cik": int(cik), "name": info.get("name", ""), "sic": str(info.get("sic") or ""),
                "sic_description": info.get("sicDescription", ""),
                "tickers": info.get("tickers", []), "exchanges": info.get("exchanges", [])}
        return meta, keep

    def document(self, url):
        """Raw filing HTML. Not disk-cached (large); extracted sections are cached instead."""
        return self.fetch(url, cache=False)

    # ------------------------------------------------------------------ peer discovery helpers
    def companies_in_sic(self, sic, max_pages=40):
        """All SEC registrants assigned a 4-digit SIC code (includes inactive ones)."""
        out = []
        for page in range(max_pages):
            url = (f"{WWW}/cgi-bin/browse-edgar?action=getcompany&SIC={sic}&type=10-K"
                   f"&dateb=&owner=include&start={page * 100}&count=100&output=atom")
            try:
                xml = self.fetch(url, max_age_days=30)
            except FileNotFoundError:
                break
            xml = re.sub(rb'\sxmlns="[^"]+"', b"", xml, count=1)
            try:
                root = ET.fromstring(xml)
            except ET.ParseError:
                break
            entries = root.findall("entry")
            for e in entries:
                cik = e.findtext("content/company-info/cik")
                if cik:
                    out.append({"cik": int(cik), "name": e.findtext("content/company-info/name") or "",
                                "sic": e.findtext("content/company-info/sic") or str(sic)})
            if len(entries) < 100:
                break
        return out

    def frame(self, tag, period, unit="USD"):
        """One XBRL value per company for a period, e.g. Assets at CY2024Q4I."""
        try:
            return self.json(f"{DATA}/api/xbrl/frames/us-gaap/{tag}/{unit}/{period}.json", max_age_days=30)
        except FileNotFoundError:
            return None

    def latest_assets(self):
        """{cik: most recent total assets} merged from the last four quarterly frames.

        Merging several frames covers companies with non-calendar fiscal years (e.g. retailers),
        whose balance-sheet dates don't fall in every calendar-quarter frame."""
        today = date.today()
        y, q = today.year, (today.month - 1) // 3 + 1
        out, used = {}, []
        for _ in range(7):
            q -= 1
            if q == 0:
                y, q = y - 1, 4
            data = self.frame("Assets", f"CY{y}Q{q}I")
            if not data or len(data.get("data", [])) < 3000:
                continue       # frame for a very recent quarter not populated yet
            used.append(f"CY{y}Q{q}")
            for d in data["data"]:
                out.setdefault(int(d["cik"]), float(d["val"]))   # newest frame wins
            if len(used) == 4:
                break
        return out, (f"{used[-1]}–{used[0]}" if used else None)
