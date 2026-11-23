"""Point-in-time accounting ratios from SEC XBRL company facts.

Every value is taken from the SAME filing (accession number) that it is paired
with, so later restatements cannot leak backwards in time.

Altman Z'' (Altman 1995, non-manufacturer / book-equity version, usable across
industries without market data):
    Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
    X1 = working capital / total assets
    X2 = retained earnings / total assets
    X3 = EBIT / total assets         (10-Q: quarterly EBIT x 4)
    X4 = book equity / total liabilities
Zones: < 1.1 distress, 1.1-2.6 grey, > 2.6 safe.
"""
import math

import numpy as np
import pandas as pd

from src.config import FINANCIAL_SIC_PREFIXES, ICR_DISTRESS, Z_DISTRESS

TAGS = {
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "liab_and_equity": ["LiabilitiesAndStockholdersEquity"],
    "equity": ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "StockholdersEquity"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "Cash"],
    "lt_debt": ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
}
FLOW_TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "ebit": ["OperatingIncomeLoss"],
    "pretax": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
               "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "interest": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt",
                 "InterestAndDebtExpense"],
    "net_income": ["NetIncomeLoss"],
}
RATIO_COLS = ["log_assets", "leverage", "cash_assets", "current_ratio", "op_margin",
              "zscore", "icr", "icr_log", "z_x1", "z_x2", "z_x3", "z_x4"]


def _index(facts):
    """{tag: [fact entries in USD]} for us-gaap facts."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    return {tag: body.get("units", {}).get("USD", []) for tag, body in gaap.items()}


def _instant(idx, tags, accn, end):
    for tag in tags:
        for x in idx.get(tag, []):
            if x.get("accn") == accn and x.get("end") == end and "start" not in x:
                return float(x["val"])
    return np.nan


def _flow(idx, tags, accn, end, annual):
    lo, hi = (330, 380) if annual else (80, 100)
    for tag in tags:
        for x in idx.get(tag, []):
            if x.get("accn") != accn or x.get("end") != end or not x.get("start"):
                continue
            days = (pd.Timestamp(x["end"]) - pd.Timestamp(x["start"])).days
            if lo <= days <= hi:
                return float(x["val"])
    return np.nan


def _div(a, b):
    return a / b if np.isfinite(a) and np.isfinite(b) and b != 0 else np.nan


def ratios(facts_index, filing, sic=""):
    accn, end, annual = filing["accessionNumber"], filing.get("reportDate", ""), filing["form"] == "10-K"
    v = {k: _instant(facts_index, t, accn, end) for k, t in TAGS.items()}
    f = {k: _flow(facts_index, t, accn, end, annual) for k, t in FLOW_TAGS.items()}
    if not np.isfinite(f["interest"]):   # some filers only tag NET interest (negative = net expense)
        net = _flow(facts_index, ["InterestIncomeExpenseNet", "InterestIncomeExpenseNonoperatingNet"], accn, end, annual)
        if np.isfinite(net) and net < 0:
            f["interest"] = -net
    if not np.isfinite(v["liabilities"]) and np.isfinite(v["liab_and_equity"]) and np.isfinite(v["equity"]):
        v["liabilities"] = v["liab_and_equity"] - v["equity"]
    scale = 1.0 if annual else 4.0
    ebit = f["ebit"]
    if not np.isfinite(ebit) and np.isfinite(f["pretax"]) and np.isfinite(f["interest"]):
        ebit = f["pretax"] + f["interest"]
    ta = v["assets"]
    out = {
        "assets": ta,
        "log_assets": math.log(ta) if np.isfinite(ta) and ta > 0 else np.nan,
        "leverage": _div(v["liabilities"], ta),
        "cash_assets": _div(v["cash"], ta),
        "current_ratio": _div(v["current_assets"], v["current_liabilities"]),
        "op_margin": _div(ebit, f["revenue"]),
        "revenue": f["revenue"] * scale if np.isfinite(f["revenue"]) else np.nan,
        "icr": _div(ebit, abs(f["interest"])) if np.isfinite(f["interest"]) and f["interest"] != 0 else np.nan,
    }
    out["icr_log"] = (np.sign(out["icr"]) * math.log1p(min(abs(out["icr"]), 1000))
                      if np.isfinite(out["icr"]) else np.nan)
    x1 = _div(v["current_assets"] - v["current_liabilities"], ta)
    x2 = _div(v["retained_earnings"], ta)
    x3 = _div(ebit * scale, ta)
    x4 = _div(v["equity"], v["liabilities"])
    out.update(z_x1=x1, z_x2=x2, z_x3=x3, z_x4=x4)
    financial = str(sic)[:2] in FINANCIAL_SIC_PREFIXES
    parts = [x1, x2, x3, x4]
    out["zscore"] = (6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
                     if not financial and all(np.isfinite(p) for p in parts) else np.nan)
    return out


def distress_state(row):
    """1 = in distress, 0 = not, NaN = cannot tell. Language-free definition (ratios only):
    EBIT does not cover interest (ICR < 1), OR Z'' is in the distress zone AND coverage is thin (ICR < 2).
    Requiring thin coverage alongside a low Z'' avoids flagging large, highly levered but comfortably
    solvent firms (e.g. captive-finance or negative-working-capital models) as distressed."""
    z, icr = row.get("zscore", np.nan), row.get("icr", np.nan)
    if np.isfinite(icr):
        if icr < ICR_DISTRESS:
            return 1.0
        if np.isfinite(z):
            return float(z < Z_DISTRESS and icr < 2.0)
        return 0.0
    if np.isfinite(z):   # no interest expense reported -> Z'' alone (rarely binding: little debt)
        return float(z < Z_DISTRESS)
    return np.nan
