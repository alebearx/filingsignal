# FilingSignal

**Does unusual change in 10-K / 10-Q language flag mispricing and financial distress before the ratios move?**

Type in a ticker and the dashboard:

1. downloads the company's last 10 years of 10-K and 10-Q filings from SEC EDGAR (free, official),
2. compares each filing's **Risk Factors** and **MD&A** with the matching filing one year earlier
   (10-K vs prior 10-K; 10-Q vs the *same quarter* last year),
3. finds the exact **new sentences** and tags them by risk theme (liquidity, debt/covenants, going concern, ...),
4. picks **comparable companies** (same SEC industry code, closest in size) and analyzes their filings too,
5. scores how unusual the change is (**Filing Change Score**, 0–100) against a research universe of ~150 companies,
6. reports what that historically meant: **expected next-quarter excess return vs. the sector** ("over/underpriced")
   and **12-month distress-onset probability**, next to the out-of-sample track record of those estimates.

Everything shown links back to the source filing on sec.gov. All data sources are free.

---

## 1. Setup (Windows, one time, ~10 minutes)

You need **Python 3.10–3.12** (check with `python --version`).

Open **PowerShell** in this folder (in File Explorer: open the folder, click the address bar, type `powershell`, Enter):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
copy .env.example .env
notepad .env                           # put YOUR name and email, save
```

The SEC requires every automated request to identify a contact. Use your real email and never commit `.env`
(it is already in `.gitignore`).

*(Optional, recommended for the final paper)* Download the free **Loughran-McDonald Master Dictionary** CSV from
https://sraf.nd.edu/loughranmcdonald-master-dictionary/ and save it in `data/`. The code picks it up automatically.
Without it, a smaller built-in word list is used, and the app says so.

## 2. Build the research universe (run once on your laptop)

```powershell
python build.py --tickers MSFT KSS AMC BBBY --years 3     # 5-minute smoke test
python build.py                                           # full: ~150 companies x 10 years
```

The full build downloads about 6,000 filings and takes **3–6 hours**, so run it overnight. It is **resumable**:
re-running skips finished companies. When it finishes it prints the backtest results and saves:

| File | What |
|---|---|
| `data/processed/companies/*.parquet` | per-company filing features |
| `data/processed/panel.parquet` | the combined research panel, scored |
| `data/processed/research.json` | backtests and out-of-sample model evaluations |
| `data/processed/models.joblib` | the fitted return and distress models |

To change the universe, edit `data/universe.csv`. Delisted or bankrupt firms need their SEC **CIK** and
`price_ticker` set to `-` (their old ticker may now belong to a different company).

`python build.py --research-only` re-runs the statistics without downloading anything.

## 3. Run the dashboard locally

```powershell
streamlit run app.py
```

It opens at **http://localhost:8501**. Only you can see a localhost page. To share it, deploy (next step).

## 4. Put it online for free (Streamlit Community Cloud)

You need a free GitHub account and a free Streamlit account.

1. **Create the GitHub repo.** This folder is already a git repository (branch `main`). The easiest route is
   **GitHub Desktop** (https://desktop.github.com): *File → Add local repository* → choose this folder →
   commit → *Publish repository* (make it **Public**). `.env` and `cache/` are git-ignored and never published.
   The `data/processed/` folder **is** published on purpose: the hosted app reads the pre-built universe
   (and the peer-similarity corpus `business_tfidf.joblib`) from it.
2. Go to **https://share.streamlit.io**, sign in with GitHub, click **Create app**, pick your repo,
   branch `main`, main file `app.py`.
3. Under **Advanced settings**, choose **Python 3.12** (the pinned packages and saved models are built on it), and
   under **Secrets** paste (a dedicated project email is fine; visitors never see it):
   ```toml
   SEC_USER_AGENT = "FilingSignal research project.email@example.org"
   ```
4. Click **Deploy**. You'll get a public link like `https://filingsignal.streamlit.app` to share with your group
   and professor.

Notes: the free tier sleeps after inactivity (the first visit wakes it in ~30 s) and has ~1 GB RAM, which is why
the heavy universe build runs on your laptop, not the server. New tickers typed into the hosted app are analyzed
live (1–5 min the first time). To update the research results, rebuild locally and push again.

**Alternative host:** Hugging Face Spaces (https://huggingface.co/spaces): create a Space with the *Streamlit*
SDK, upload the same files, and add `SEC_USER_AGENT` under *Settings → Secrets*.

---

## 5. How it works

```
app.py                 Streamlit dashboard (5 tabs)
build.py               Builds the universe, runs backtests, fits models
src/agent.py           The analysis agent: fixed, logged tool chain for one ticker
src/edgar.py           SEC EDGAR client (rate-limited, cached)
src/sections.py        Extracts Item 1A Risk Factors + MD&A from filing HTML
src/textdiff.py        Similarity (Lazy Prices), sentence-level diff, theme tagging
src/lexicon.py         Loughran-McDonald tone + risk-theme keyword rules
src/fundamentals.py    Point-in-time XBRL ratios, Altman Z'', interest coverage
src/prices.py          yfinance prices, sector-ETF mapping, excess returns
src/peers.py           Comparable-company selection (SIC code + size + 10-K business-text similarity)
src/research.py        Change score, quintile backtests, OOS models, lead-lag study
src/takeaways.py       Plain-English takeaways (templated from measured numbers)
tests/                 Offline unit tests:  python -m pytest -q
```

### Why a fixed "agent" instead of a free-form LLM agent
The agent in `src/agent.py` runs the same six logged steps for every ticker (resolve → collect → peers → score →
model → explain), and the dashboard shows its audit trail. A free LLM agent would choose peers and filings
non-reproducibly and could hallucinate numbers. Here every peer, every comparison filing and every number has a
recorded source, which is what a PM (or a grader) needs to trust it. An LLM summarizer could be added later as a
final step that only rewrites the computed takeaways.

### Research design
| Piece | Choice | Why |
|---|---|---|
| Unit | one 10-K or 10-Q filing | the moment new language becomes public |
| Comparison | prior-year 10-K; same quarter prior-year 10-Q | removes seasonal wording |
| Change measures | 1 − cosine similarity; Δ negative tone (LM); severity-weighted new risk sentences | Cohen, Malloy & Nguyen (2020); Loughran & McDonald (2011) |
| Filing Change Score | equal-weight mean of point-in-time percentiles (prior 365 days, same form) | no fitting means no overfitting, and the same score applies to new tickers |
| Return target | 63-trading-day return minus sector SPDR ETF, entry the day *after* filing | "over/underpriced vs. its sector" |
| Distress target | within 12 months: interest coverage < 1.0, or Z'' < 1.1 with coverage < 2.0; firms not currently distressed | ratio-only label, so no language leakage; the coverage condition avoids flagging solvent but levered giants |
| Validation | quintile sorts with Newey-West t-stats; expanding-window OOS by year with label-maturity purge | standard asset-pricing practice |
| Key comparison | controls/ratios only vs. + language | tests the *incremental* value of language |
| Lead-lag | change score and Z'' in the 8 filings before distress onset | "does language move first?" |

### Limitations (put these in your write-up)
- **Survivorship:** free price data excludes delisted stocks. Bankrupt firms are kept for the distress tests
  but have no returns.
- **Sample size:** ~150 firms. Treat t-stats as suggestive, and report the confidence intervals.
- **SIC codes are coarse:** peer search goes from the same 4-digit code, to the 3-digit group, to 2-digit codes
  present in the research universe, then re-ranks up to 15 size-closest candidates by TF-IDF similarity of their
  latest 10-K *Item 1 Business* text (`peer_score = similarity − 0.05·|ln size ratio| − 0.03·SIC tier`).
  Bag-of-words similarity can still miss paraphrased business models, so the manual-peer box exists.
  Foreign filers (20-F) have no 10-K text and rank on size only.
- **Distress label quirks:** consolidated figures for firms with captive finance arms (e.g. Ford) show low
  interest coverage, which can trigger the label without real distress.
- **Keyword themes** miss paraphrases; a later extension could use FinBERT or sentence embeddings offline.
- **yfinance** is unofficial and can rate-limit. No transaction costs are modeled.

Research prototype. Not investment advice.
