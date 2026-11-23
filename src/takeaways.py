"""Plain-English takeaways built from measured numbers only (no generative model).

Every sentence is a template filled with a computed statistic, so each claim
can be traced back to a number shown elsewhere in the dashboard.
"""
import json

import numpy as np

from src.config import Z_DISTRESS, Z_SAFE
from src.lexicon import THEME_LABEL, THEMES


def fmt_pct(x, signed=True, digits=1):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x * 100:+.{digits}f}%" if signed else f"{x * 100:.{digits}f}%"


def z_zone(z):
    if z is None or not np.isfinite(z):
        return "n/a"
    return "distress zone" if z < Z_DISTRESS else ("grey zone" if z < Z_SAFE else "safe zone")


def pricing_label(expected, model_ok):
    if expected is None or not np.isfinite(expected):
        return "Insufficient data", "neutral"
    if not model_ok:
        return "No reliable signal", "neutral"
    if expected <= -0.015:
        return "Potentially overpriced", "negative"
    if expected >= 0.015:
        return "Potentially underpriced", "positive"
    return "No clear mispricing", "neutral"


def theme_counts(row):
    counts = {k: row.get(f"n_theme_{k}", 0) or 0 for k in THEMES}
    return {k: int(v) for k, v in sorted(counts.items(), key=lambda kv: -kv[1]) if v and v > 0}


def build_takeaways(row, peers_latest, research, expected, distress_p, distress_ratio_p):
    t = []
    fcs = row.get("fcs", np.nan)
    if row.get("cmp_accession"):
        level = ("unusually large" if fcs >= 80 else "above-typical" if fcs >= 60
                 else "typical" if fcs >= 40 else "below-typical") if np.isfinite(fcs) else "unscored"
        t.append(f"**Change vs. {row['cmp_basis'].lower()}:** the {row['form']} filed {row['filing_date']} shows "
                 f"{level} language change (Filing Change Score {fcs:.0f}/100, i.e. more change than "
                 f"{fcs:.0f}% of same-type filings in the prior year)." if np.isfinite(fcs) else
                 f"**Change vs. {row['cmp_basis'].lower()}:** score unavailable (not enough reference filings).")
    else:
        t.append("**No comparable prior filing** was found, so change metrics are undefined for this filing.")

    parts = []
    for s, name in (("risk", "Risk Factors"), ("mdna", "MD&A")):
        cos, add = row.get(f"{s}_cos", np.nan), row.get(f"{s}_n_added", np.nan)
        if np.isfinite(cos):
            parts.append(f"{name} similarity {cos:.2f} with {int(add)} new sentences" if np.isfinite(add) else
                         f"{name} similarity {cos:.2f}")
    if row.get("risk_no_change"):
        parts.append("Risk Factors states no material changes")
    if parts:
        t.append("**Where it changed:** " + "; ".join(parts) + ".")

    tc = theme_counts(row)
    if tc:
        top = ", ".join(f"{THEME_LABEL[k]} ({v})" for k, v in list(tc.items())[:3])
        t.append(f"**What the new language is about:** {top} new sentences.")
    if tc.get("going_concern"):
        t.append("⚠️ **New solvency language** (going concern / bankruptcy / substantial doubt) appears in this "
                 "filing. Read the excerpts in the *What changed* tab.")

    dneg = row.get("d_tone_negative", np.nan)
    if np.isfinite(dneg):
        t.append(f"**Tone:** negative-word share {'rose' if dneg > 0 else 'fell'} by {abs(dneg):.2f} percentage "
                 f"points vs. the comparison filing.")

    z, dz, icr = row.get("zscore", np.nan), row.get("dz_yoy", np.nan), row.get("icr", np.nan)
    if np.isfinite(z):
        msg = f"**Ratios:** Altman Z'' = {z:.2f} ({z_zone(z)})"
        if np.isfinite(dz):
            msg += f", {dz:+.2f} vs. a year earlier"
        if np.isfinite(icr):
            msg += f"; interest coverage {icr:.1f}x"
        t.append(msg + ".")
        if np.isfinite(fcs) and fcs >= 80 and z >= Z_SAFE and (not np.isfinite(dz) or dz > -0.3):
            t.append("🔎 **Language is moving before the ratios:** text change is in the top quintile while "
                     "Z'' remains in the safe zone and has not fallen. This is the pattern the project tests for.")

    if peers_latest is not None and len(peers_latest) and np.isfinite(fcs):
        pf = peers_latest["fcs"].dropna()
        if len(pf):
            t.append(f"**Versus peers:** higher change score than {int((pf < fcs).sum())} of {len(pf)} peers' "
                     f"latest comparable filings (peer median {pf.median():.0f}).")

    bt = research.get("backtest_fcs", {})
    if bt.get("status") == "ok" and np.isfinite(fcs):
        q = min(int(fcs // 20) + 1, 5)
        qrow = next((x for x in bt["quintiles"] if x["q"] == q), None)
        if qrow:
            msg = (f"**Historical base rate:** filings in change-quintile {q} averaged {fmt_pct(qrow['mean'])} "
                   f"vs. sector over the next ~quarter (n={qrow['n']:,}).")
            st_ = bt.get("spread_t")
            if bt.get("spread_mean") is not None and st_ is not None and np.isfinite(st_):
                msg += (f" Top-minus-bottom quintile spread {fmt_pct(bt['spread_mean'])} per quarter, "
                        f"t = {st_:.2f}" + (" (not statistically significant)." if abs(st_) < 2 else "."))
            if qrow["n"] < 100:
                msg += " Small sample: treat as indicative only."
            t.append(msg)
    if expected is not None and np.isfinite(expected):
        t.append(f"**Model estimate:** expected excess return over the next 63 trading days {fmt_pct(expected)} "
                 f"vs. {row.get('benchmark', 'sector ETF')}.")
    if distress_p is not None:
        msg = f"**Distress onset (12m):** {fmt_pct(distress_p, signed=False)} probability"
        if distress_ratio_p is not None:
            msg += f" ({fmt_pct(distress_ratio_p, signed=False)} using ratios alone)"
        t.append(msg + ".")
    return t


def evidence_of(row):
    try:
        return json.loads(row.get("evidence") or "{}")
    except (TypeError, ValueError):
        return {"added": [], "removed": []}
