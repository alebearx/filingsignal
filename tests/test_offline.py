"""Offline unit tests (no network): python -m pytest -q"""
import numpy as np
import pandas as pd

from src.fundamentals import distress_state
from src.prices import event_returns, sector_etf
from src.research import _pct_rank, newey_west_t, quintile_backtest
from src.peers import business_similarity
from src.sections import extract_business, extract_sections
from src.textdiff import compare_section

TENK = b"""<html><body>
<div>Table of Contents</div><div>Item 1A. Risk Factors 12</div><div>Item 1B. Unresolved Staff Comments 30</div>
<div>Item 1A. Risk</div><div><span>Fac</span><span>tors</span></div>
<p>""" + b"Our business faces many risks. We depend on a small number of customers for much of our revenue. " * 5 + b"""
Our credit facility contains covenants that restrict our ability to borrow.</p>
<div>Item 1B. Unresolved Staff Comments</div><p>None.</p>
<div>Item 7. Management's Discussion and Analysis</div>
<p>""" + b"Revenue grew because demand for our products increased in all regions. " * 40 + b"""</p>
<div>Item 8. Financial Statements and Supplementary Data</div></body></html>"""


def test_sections_found_and_split_spans_joined():
    s = extract_sections(TENK, "10-K")
    assert "covenants" in s["risk"] and "Table of Contents" not in s["risk"]
    assert s["mdna"].count("Revenue grew") == 40


def test_new_sentence_detection_and_themes():
    prev = "We sell widgets to many customers worldwide in stable markets. Our results depend on general economic conditions."
    cur = prev + " There is substantial doubt about our ability to continue as a going concern over the next twelve months."
    c = compare_section(cur, prev)
    assert c["n_added"] == 1
    assert "going_concern" in c["evidence_added"][0]["themes"]
    assert c["cos"] < 1


def test_numbers_do_not_count_as_changes():
    prev = "Revenue increased 5% to $1,200 million in fiscal 2023 compared with the prior year period."
    cur = "Revenue increased 7% to $1,350 million in fiscal 2024 compared with the prior year period."
    assert compare_section(cur, prev)["n_added"] == 0


def test_event_returns_start_after_filing():
    idx = pd.bdate_range("2024-01-01", periods=200)
    stock = pd.Series(np.linspace(100, 160, 200), idx)
    bench = pd.Series(100.0, idx)
    r = event_returns(stock, bench, bench, "2024-04-15")
    assert r["excess_63"] > 0 and np.isfinite(r["pre_excess_63"])


def test_sector_map_and_distress():
    assert sector_etf("7372") == "XLK" and sector_etf("2834") == "XLV" and sector_etf("6022") == "XLF"
    assert distress_state({"zscore": 0.5, "icr": 1.5}) == 1
    assert distress_state({"zscore": 0.5, "icr": 8}) == 0      # low Z'' but ample coverage
    assert distress_state({"zscore": 3.0, "icr": 0.5}) == 1
    assert distress_state({"zscore": 3.0, "icr": 5}) == 0
    assert np.isnan(distress_state({"zscore": np.nan, "icr": np.nan}))


def test_stats_helpers():
    assert _pct_rank(5, np.arange(100.0)) == 5.5
    t, n = newey_west_t(np.r_[np.full(20, 0.01), np.full(20, 0.03)])
    assert n == 40 and t > 3
    rng = np.random.default_rng(0)
    p = pd.DataFrame({"fcs": rng.uniform(0, 100, 600), "quarter": np.repeat([f"2020Q{i}" for i in range(1, 5)], 150)})
    p["excess_63_w"] = -0.0005 * p["fcs"] + rng.normal(0, 0.01, 600)
    bt = quintile_backtest(p)
    assert bt["status"] == "ok" and bt["spread_mean"] < 0


def test_business_section_and_similarity():
    html = (b"<html><body><div>Item 1. Business 3</div><div>Item 1A. Risk Factors 9</div>"
            b"<div>Item 1. Business</div><p>" + b"We operate discount department stores selling apparel. " * 30
            + b"</p><div>Item 1A. Risk Factors</div><p>Risks.</p></body></html>")
    biz = extract_business(html)
    assert "discount department stores" in biz and "Risks." not in biz
    sims = business_similarity(biz, ["We run department stores selling apparel and home goods. " * 20,
                                     "We design semiconductors for data center accelerators. " * 20, ""])
    assert sims[0] > sims[1] and np.isnan(sims[2])
