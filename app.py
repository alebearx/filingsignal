"""FilingSignal — Streamlit dashboard.

Run locally:   streamlit run app.py
"""
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.config import Z_DISTRESS, Z_SAFE
from src.lexicon import THEME_LABEL, dictionary_name
from src.research import SCORE_COMPONENTS, load_research
from src.takeaways import evidence_of, fmt_pct, z_zone

load_dotenv()
try:
    if "SEC_USER_AGENT" in st.secrets:
        os.environ["SEC_USER_AGENT"] = st.secrets["SEC_USER_AGENT"]
except Exception:
    pass

st.set_page_config(page_title="FilingSignal", page_icon="📄", layout="wide")

# Validated categorical order (blue, orange, aqua, ...) + neutral ink for chart chrome.
C_TARGET, C_PEER, C_THIRD, C_MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
C_NEG, C_POS = "#d03b3b", "#0ca30c"
st.markdown("""<style>
.block-container{padding-top:1.6rem;max-width:1360px}
[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.25);border-radius:10px;padding:12px 14px}
.signal{font-size:1.05rem;font-weight:600}
</style>""", unsafe_allow_html=True)


def base_layout(fig, title, ytitle, height=320, pct=False):
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=height, margin=dict(l=10, r=10, t=48, b=10),
                      hovermode="x unified", legend=dict(orientation="h", y=-0.18),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False, linecolor="rgba(128,128,128,.4)")
    fig.update_yaxes(title=ytitle, gridcolor="rgba(128,128,128,.15)", zerolinecolor="rgba(128,128,128,.4)",
                     tickformat=".0%" if pct else None)
    return fig


@st.cache_resource(show_spinner=False)
def research_bundle():
    return load_research()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def run_analysis(ticker, years, n_peers, peer_years, extra):
    from src.agent import AnalysisAgent
    from src.edgar import Edgar
    universe, research, models = research_bundle()
    bar = st.progress(0.0, text="Starting analysis agent ...")
    agent = AnalysisAgent(Edgar(), universe, research, models,
                          progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
    out = agent.run(ticker, years=years, n_peers=n_peers, peer_years=peer_years, extra_peers=extra)
    bar.empty()
    return out


universe, research, models = research_bundle()

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("FilingSignal")
    st.caption("Do changes in 10-K/10-Q language predict mispricing and distress before the ratios move?")
    with st.form("controls"):
        ticker = st.text_input("Ticker", value=st.session_state.get("ticker", "KSS")).upper().strip()
        years = st.slider("Company history (years)", 3, 10, 10)
        n_peers = st.slider("Number of comparable companies", 2, 8, 5)
        peer_years = st.slider("Peer history (years)", 1, 5, 3, help="Shorter = faster first run")
        extra = st.text_input("Add peers manually (optional)", placeholder="e.g. M, JWN")
        go_btn = st.form_submit_button("Analyze", type="primary", width="stretch")
    st.caption("First analysis of a new company downloads its filings from SEC EDGAR (1–5 min). "
               "Repeat views are cached.")
    st.divider()
    u = research.get("universe", {})
    if u:
        st.caption(f"**Research universe:** {u['companies']} companies, {u['filings']:,} filings "
                   f"({u['first'][:4]}–{u['last'][:4]}).")
    else:
        st.warning("No research universe built yet: run `python build.py`. Backtests and model "
                   "estimates are hidden until then.")
    st.caption("Data: SEC EDGAR (filings, XBRL financials) · Yahoo Finance via yfinance (prices). "
               "Free sources only. Research prototype — not investment advice.")

if go_btn:
    st.session_state["ticker"] = ticker
    st.session_state["params"] = (ticker, years, n_peers, peer_years,
                                  tuple(t.strip().upper() for t in extra.split(",") if t.strip()))

if "params" not in st.session_state:
    st.title("FilingSignal")
    st.markdown("""
Enter a ticker on the left and press **Analyze**. The agent will:

1. pull the company's 10-K and 10-Q filings from SEC EDGAR,
2. compare each filing's **Risk Factors** and **MD&A** with the matching filing a year earlier,
3. pick comparable companies (same industry code, closest size) and do the same for them,
4. score how unusual the changes are, and
5. report what that historically meant for **next-quarter returns vs. the sector** and **12-month distress risk**,
   with the exact filings, sentences and peers it used.
""")
    st.stop()

try:
    R = run_analysis(*st.session_state["params"])
except Exception as exc:
    st.error(f"Analysis failed: {exc}")
    if "SEC_USER_AGENT" in str(exc):
        st.info("Add your contact email as SEC_USER_AGENT (see README → Setup).")
    st.stop()

company, latest, pred = R["company"], R["latest"], R["prediction"]

# ----------------------------------------------------------------------------- header + KPI strip
st.title(f"{R['meta']['name']} ({R['ticker']})")
st.caption(f"SEC industry: {R['meta']['sic_description']} (SIC {R['meta']['sic']}) · Benchmark: "
           f"{latest['benchmark']} – {R['benchmark_name']} · Latest filing: {latest['form']} filed "
           f"{latest['filing_date']} · [Open on SEC]({latest['url']})")

k1, k2, k3, k4 = st.columns(4)
exp = pred.get("expected")
interval = pred.get("interval")
k1.metric("Pricing signal (next ~quarter)", R["pricing_label"],
          delta=f"{fmt_pct(exp)} expected vs sector" if exp is not None else None,
          delta_color="off" if not R["model_ok"] else "normal")
if interval:
    k1.caption(f"80% range {fmt_pct(interval[0])} to {fmt_pct(interval[1])}"
               + ("" if R["model_ok"] else " · model lacks out-of-sample skill"))
dm = models.get("distress", {})
if pred.get("distress") is not None:
    k2.metric("Distress onset, 12 months", fmt_pct(pred["distress"], signed=False),
              delta=f"base rate {fmt_pct(dm.get('base_rate'), signed=False)}", delta_color="off")
else:
    k2.metric("Distress onset, 12 months", "n/a")
    k2.caption("Already in distress, a financial firm (Z'' not meaningful), or model not trained.")
fcs = latest.get("fcs", np.nan)
k3.metric("Filing Change Score", f"{fcs:.0f} / 100" if np.isfinite(fcs) else "n/a",
          delta=("top quintile" if fcs >= 80 else "") if np.isfinite(fcs) else None, delta_color="off")
k3.caption("Percentile vs same-type filings in the prior year")
z = latest.get("zscore", np.nan)
k4.metric("Altman Z''", f"{z:.2f}" if np.isfinite(z) else "n/a",
          delta=f"{latest.get('dz_yoy'):+.2f} YoY" if np.isfinite(latest.get("dz_yoy", np.nan)) else None)
k4.caption(z_zone(z))

st.subheader("Takeaways")
for t in R["takeaways"]:
    st.markdown(f"- {t}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["History", "What changed", "Peers & comparisons",
                                        "Evidence (backtests)", "Method & audit trail"])

# ----------------------------------------------------------------------------- tab 1: history
with tab1:
    c = company.copy()
    c["date"] = pd.to_datetime(c["filing_date"])
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_hrect(y0=80, y1=100, fillcolor=C_NEG, opacity=0.06, line_width=0)
        fig.add_trace(go.Scatter(x=c["date"], y=c["fcs"], mode="lines+markers", name=R["ticker"],
                                 line=dict(color=C_TARGET, width=2), marker=dict(size=8),
                                 customdata=c[["form"]], hovertemplate="%{customdata[0]}: %{y:.0f}<extra></extra>"))
        pp = R["peer_panel"]
        if len(pp) and "fcs" in pp:
            med = pp.assign(q=pd.to_datetime(pp["filing_date"]).dt.to_period("Q").dt.start_time) \
                    .groupby("q")["fcs"].median().reset_index()
            fig.add_trace(go.Scatter(x=med["q"], y=med["fcs"], mode="lines", name="Peer median",
                                     line=dict(color=C_PEER, width=2, dash="dot"),
                                     hovertemplate="Peer median: %{y:.0f}<extra></extra>"))
        st.plotly_chart(base_layout(fig, "Filing Change Score (shaded = top quintile)", "Percentile"),
                        width="stretch")
    with right:
        fig = go.Figure()
        fig.add_hrect(y0=min(c["zscore"].min(skipna=True) if c["zscore"].notna().any() else 0, 0) - 1,
                      y1=Z_DISTRESS, fillcolor=C_NEG, opacity=0.07, line_width=0)
        fig.add_trace(go.Scatter(x=c["date"], y=c["zscore"], mode="lines+markers", name="Altman Z''",
                                 line=dict(color=C_TARGET, width=2), marker=dict(size=8),
                                 hovertemplate="Z'': %{y:.2f}<extra></extra>"))
        fig.add_hline(y=Z_SAFE, line=dict(color=C_MUTED, dash="dash", width=1),
                      annotation_text="safe ≥ 2.6", annotation_position="top left")
        st.plotly_chart(base_layout(fig, "Altman Z'' (shaded = distress zone < 1.1)", "Z''"),
                        width="stretch")
    if c["excess_63"].notna().any():
        fig = go.Figure(go.Bar(x=c["date"], y=c["excess_63"],
                               marker_color=np.where(c["excess_63"] >= 0, C_TARGET, C_PEER),
                               customdata=np.stack([c["form"], c["fcs"].round(0)], axis=1),
                               hovertemplate="%{customdata[0]} · score %{customdata[1]}<br>"
                                             "Excess return: %{y:.1%}<extra></extra>"))
        st.plotly_chart(base_layout(fig, f"Realized excess return vs {latest['benchmark']}, 63 trading days "
                                         "after each filing", "Excess return", pct=True, height=280),
                        width="stretch")
    st.markdown("**All analyzed filings** (each compared with the filing in *Compared against*)")
    show = c.sort_values("filing_date", ascending=False)[
        ["filing_date", "form", "fcs", "change", "d_tone_negative", "risk_n_added", "mdna_n_added",
         "zscore", "icr", "excess_63", "url", "cmp_basis", "cmp_url"]]
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "fcs": st.column_config.NumberColumn("Change score", format="%.0f"),
        "change": st.column_config.NumberColumn("1 − similarity", format="%.4f"),
        "d_tone_negative": st.column_config.NumberColumn("Δ negative tone (pp)", format="%.2f"),
        "risk_n_added": st.column_config.NumberColumn("New sentences: Risk", format="%d"),
        "mdna_n_added": st.column_config.NumberColumn("New sentences: MD&A", format="%d"),
        "zscore": st.column_config.NumberColumn("Z''", format="%.2f"),
        "icr": st.column_config.NumberColumn("Int. coverage", format="%.1f"),
        "excess_63": st.column_config.NumberColumn("63d excess return", format="percent"),
        "url": st.column_config.LinkColumn("Filing", display_text="SEC"),
        "cmp_basis": "Compared against",
        "cmp_url": st.column_config.LinkColumn("Comparison filing", display_text="SEC")})

# ----------------------------------------------------------------------------- tab 2: what changed
with tab2:
    opts = company.sort_values("filing_date", ascending=False).reset_index(drop=True)
    i = st.selectbox("Filing", range(len(opts)),
                     format_func=lambda j: f"{opts.at[j, 'filing_date']} · {opts.at[j, 'form']} · "
                                           f"score {opts.at[j, 'fcs']:.0f}" if np.isfinite(opts.at[j, 'fcs'])
                     else f"{opts.at[j, 'filing_date']} · {opts.at[j, 'form']}")
    row = opts.iloc[i].to_dict()
    a, b = st.columns(2)
    a.markdown(f"**This filing:** {row['form']} filed {row['filing_date']} (period {row['report_date']}) "
               f"— [SEC]({row['url']})")
    if row["cmp_url"]:
        b.markdown(f"**Compared against:** {row['cmp_basis']}, filed {row['cmp_filing_date']} — [SEC]({row['cmp_url']})")
    else:
        b.warning("No comparable filing one year earlier, so change metrics are undefined.")
    m = pd.DataFrame({
        "Section": ["Risk Factors", "MD&A"],
        "Similarity (cosine)": [row.get("risk_cos"), row.get("mdna_cos")],
        "Jaccard": [row.get("risk_jac"), row.get("mdna_jac")],
        "Length change": [row.get("risk_len_logratio"), row.get("mdna_len_logratio")],
        "New sentences": [row.get("risk_n_added"), row.get("mdna_n_added")],
        "Share new": [row.get("risk_added_share"), row.get("mdna_added_share")],
        "Negative tone %": [row.get("risk_tone_negative"), row.get("mdna_tone_negative")],
    })
    st.dataframe(m, hide_index=True, width="stretch", column_config={
        "Similarity (cosine)": st.column_config.NumberColumn(format="%.4f"),
        "Jaccard": st.column_config.NumberColumn(format="%.3f"),
        "Length change": st.column_config.NumberColumn(help="log(current words / prior words)", format="%+.3f"),
        "New sentences": st.column_config.NumberColumn(format="%d"),
        "Share new": st.column_config.NumberColumn(format="percent"),
        "Negative tone %": st.column_config.NumberColumn(format="%.2f")})
    if row.get("risk_no_change"):
        st.info("This 10-Q's Risk Factors item says there were no material changes from the last 10-K.")
    if row.get("used_fallback"):
        st.warning("Section headings could not be located, so the document start was used. Treat this filing's metrics with caution.")
    ev = evidence_of(row)
    st.markdown("#### New risk-relevant sentences (verbatim, highest severity first)")
    st.caption("A sentence counts as *new* when no sentence in the comparison filing is a close match "
               "(TF-IDF similarity < 0.75, numbers ignored). Themes are assigned by transparent keyword rules.")
    if not ev.get("added"):
        st.write("No new risk-themed sentences found.")
    for e in ev.get("added", []):
        labels = " · ".join(THEME_LABEL[t] for t in e["themes"])
        sec = "Risk Factors" if e["section"] == "risk" else "MD&A"
        with st.expander(f"{labels}  —  {sec}"):
            st.write(e["text"])
            st.caption(f"Closest sentence in comparison filing: similarity {e['closest_prior_sim']:.2f}")
    if ev.get("removed"):
        st.markdown("#### Removed risk-relevant sentences")
        for e in ev["removed"]:
            st.markdown(f"- ~~{e['text'][:400]}~~  \n  _{' · '.join(THEME_LABEL[t] for t in e['themes'])}_")

# ----------------------------------------------------------------------------- tab 3: peers
with tab3:
    peers = R["peers"]
    st.markdown("**Comparable companies and why they were chosen**")
    if peers.empty:
        st.warning("No listed companies found with the same SIC code. Add peers manually in the sidebar.")
    else:
        pt = peers.copy()
        pl = R["peers_latest"]
        if len(pl):
            pt = pt.merge(pl[["ticker", "form", "filing_date", "fcs", "zscore", "url", "cmp_url"]],
                          on="ticker", how="left")
        st.caption("Method: candidates share the target's SEC industry code (widening to 3-, then 2-digit), "
                   "are pre-ranked by size, then re-ranked by TF-IDF similarity of their latest 10-K "
                   "*Item 1 Business* text to the target's. Peer score = similarity − 0.05·|ln size ratio| "
                   "− 0.03·SIC tier. Free and reproducible; no AI model.")
        st.dataframe(pt.drop(columns=[c for c in ("cik", "size_distance", "match") if c in pt]),
                     hide_index=True, width="stretch", column_config={
                         "assets": st.column_config.NumberColumn("Total assets ($)", format="compact"),
                         "size_ratio": st.column_config.NumberColumn("Size vs target", format="%.2fx"),
                         "business_similarity": st.column_config.NumberColumn(
                             "Business similarity", format="%.2f", help="TF-IDF cosine, 10-K Item 1 vs target"),
                         "peer_score": st.column_config.NumberColumn("Peer score", format="%.3f"),
                         "fcs": st.column_config.NumberColumn("Latest change score", format="%.0f"),
                         "zscore": st.column_config.NumberColumn("Z''", format="%.2f"),
                         "url": st.column_config.LinkColumn("Latest filing", display_text="SEC"),
                         "cmp_url": st.column_config.LinkColumn("Compared against", display_text="SEC")})
    if len(R["peers_latest"]):
        pl = R["peers_latest"].dropna(subset=["fcs"])
        bars = pd.concat([pd.DataFrame([{"ticker": R["ticker"], "fcs": fcs, "form": latest["form"]}]),
                          pl[["ticker", "fcs", "form"]]]).sort_values("fcs")
        fig = go.Figure(go.Bar(x=bars["fcs"], y=bars["ticker"], orientation="h",
                               marker_color=[C_TARGET if t == R["ticker"] else C_MUTED for t in bars["ticker"]],
                               customdata=bars[["form"]], hovertemplate="%{y} (%{customdata[0]}): %{x:.0f}<extra></extra>"))
        fig.add_vline(x=80, line=dict(color=C_NEG, dash="dash", width=1), annotation_text="top quintile")
        st.plotly_chart(base_layout(fig, "Latest Filing Change Score: target vs peers", "", height=300)
                        .update_xaxes(range=[0, 100], title="Percentile"), width="stretch")
    st.markdown("**Every filing used in the comparison set**")
    pp = R["peer_panel"]
    if len(pp):
        st.dataframe(pp.sort_values(["ticker", "filing_date"], ascending=[True, False])[
            ["ticker", "form", "filing_date", "fcs", "excess_63", "url", "cmp_basis", "cmp_url"]],
            hide_index=True, width="stretch", column_config={
                "fcs": st.column_config.NumberColumn("Change score", format="%.0f"),
                "excess_63": st.column_config.NumberColumn("63d excess", format="percent"),
                "url": st.column_config.LinkColumn("Filing", display_text="SEC"),
                "cmp_url": st.column_config.LinkColumn("Comparison", display_text="SEC")})

# ----------------------------------------------------------------------------- tab 4: evidence
with tab4:
    if not research:
        st.warning("Run `python build.py` to build the research universe and backtests.")
    else:
        u = research["universe"]
        st.markdown(f"Universe: **{u['companies']} companies, {u['filings']:,} filings** ({u['first']} to {u['last']}); "
                    f"{u['with_returns']:,} with 63-day returns; {u['distress_events']} distress onsets among "
                    f"{u['with_distress_label']:,} labelled filings.")
        st.markdown("#### 1 · Do high-change filings underperform their sector next quarter?")
        a, b = st.columns(2)
        for col, key, title in ((a, "backtest_fcs", "By Filing Change Score"),
                                (b, "backtest_similarity", "By similarity alone (Lazy Prices replication)")):
            bt = research.get(key, {})
            with col:
                if bt.get("status") != "ok":
                    st.info(f"{title}: {bt.get('status', 'not run')}")
                    continue
                q = pd.DataFrame(bt["quintiles"])
                fig = go.Figure(go.Bar(x=[f"Q{x}" for x in q["q"]], y=q["mean"], marker_color=C_TARGET,
                                       text=[fmt_pct(v) for v in q["mean"]], textposition="outside",
                                       customdata=q["n"], hovertemplate="%{x}: %{y:.2%} (n=%{customdata})<extra></extra>"))
                st.plotly_chart(base_layout(fig, f"{title}<br><sup>Q1 = least change, Q5 = most · "
                                                 f"Q5−Q1 {fmt_pct(bt['spread_mean'])}/qtr, t = {bt['spread_t']:.2f} "
                                                 f"({bt['spread_quarters']} quarters)</sup>",
                                            "Mean 63d excess return", pct=True, height=330), width="stretch")
        bt = research.get("backtest_fcs", {})
        if bt.get("status") == "ok" and bt["spread_series"]:
            s = pd.DataFrame(bt["spread_series"])
            s["cum"] = (1 + s["spread"]).cumprod() - 1
            fig = go.Figure(go.Scatter(x=s["quarter"], y=s["cum"], mode="lines+markers",
                                       line=dict(color=C_TARGET, width=2), name="Q5 − Q1"))
            st.plotly_chart(base_layout(fig, "Cumulative Q5 − Q1 spread (most-changed minus least-changed filings)",
                                        "Cumulative", pct=True, height=280), width="stretch")
            st.caption("Negative = high-change filers underperform, the Lazy Prices prediction. Equal-weighted, "
                       "excess vs sector ETF, entry the trading day after filing. Overlapping 63-day windows "
                       "are why t-stats use Newey-West standard errors. No transaction costs.")

        st.markdown("#### 2 · Does language add predictive power beyond standard controls? (out-of-sample)")
        rm = research.get("return_models", {})
        rows = [{"Model": k, "OOS R²": v.get("oos_r2"), "Mean rank IC": v.get("mean_ic"),
                 "Direction hit rate": v.get("hit_rate"), "OOS filings": v.get("n_oos")}
                for k, v in rm.items() if v.get("status") == "ok"]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
                "OOS R²": st.column_config.NumberColumn(format="%.4f"),
                "Mean rank IC": st.column_config.NumberColumn(format="%.3f"),
                "Direction hit rate": st.column_config.NumberColumn(format="percent")})
            st.caption("Ridge regression, expanding window by calendar year; each test year uses only filings "
                       "whose returns had fully played out before that year began. IC = Spearman correlation of "
                       "predicted and realized excess return. Return predictability is small even in published "
                       "work: an IC of 0.02–0.05 is meaningful.")
        dmods = research.get("distress_models", {})
        rows = [{"Model": k, "AUC": v.get("auc"), "Brier": v.get("brier"), "Brier (base rate)": v.get("brier_base"),
                 "OOS filings": v.get("n_oos"), "Events": v.get("events_oos")}
                for k, v in dmods.items() if v.get("status") == "ok"]
        st.markdown("#### 3 · Does language predict distress onset beyond the ratios? (out-of-sample)")
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
                "AUC": st.column_config.NumberColumn(format="%.3f"),
                "Brier": st.column_config.NumberColumn(format="%.4f"),
                "Brier (base rate)": st.column_config.NumberColumn(format="%.4f")})
            st.caption("Distress onset = within 12 months, interest coverage falls below 1.0, or Z'' falls "
                       "below 1.1 while coverage is under 2.0, for a firm not in distress today. Defined from ratios only, so language cannot leak into "
                       "the label. Financial firms are excluded.")
        else:
            st.info("Not enough distress events for an out-of-sample test. Expand the universe.")

        st.markdown("#### 4 · Lead-lag: does the language move before Z''?")
        ll = research.get("lead_lag", {})
        if ll.get("status") == "ok":
            path = pd.DataFrame(ll["path"])
            a, b = st.columns(2)
            fig = go.Figure(go.Scatter(x=path["k"], y=path["fcs_mean"], mode="lines+markers",
                                       line=dict(color=C_TARGET, width=2),
                                       error_y=dict(type="data", array=1.96 * path["fcs_se"].fillna(0), color=C_MUTED),
                                       customdata=path["n"], hovertemplate="filing %{x}: %{y:.0f} (n=%{customdata})<extra></extra>"))
            fig.add_hline(y=50, line=dict(color=C_MUTED, dash="dash", width=1), annotation_text="average filing")
            fig.add_vline(x=0, line=dict(color=C_NEG, width=1))
            a.plotly_chart(base_layout(fig, "Filing Change Score around distress onset", "Mean percentile")
                           .update_xaxes(title="Filings relative to onset (0 = first distressed filing)"),
                           width="stretch")
            fig = go.Figure(go.Scatter(x=path["k"], y=path["z_median"], mode="lines+markers",
                                       line=dict(color=C_PEER, width=2)))
            fig.add_vline(x=0, line=dict(color=C_NEG, width=1))
            fig.add_hline(y=Z_DISTRESS, line=dict(color=C_MUTED, dash="dash", width=1), annotation_text="distress line")
            b.plotly_chart(base_layout(fig, "Median Altman Z'' around distress onset", "Z''")
                           .update_xaxes(title="Filings relative to onset"), width="stretch")
            st.caption(f"{ll['n_events']} first-time distress onsets. In the 4 filings before onset, "
                       f"{fmt_pct(ll['pre_event_flag_rate'], signed=False)} were top-quintile change filings, vs "
                       f"{fmt_pct(ll['baseline_flag_rate'], signed=False)} of all filings.")
            with st.expander("Which companies are the onset events?"):
                st.dataframe(pd.DataFrame(ll["events"]), hide_index=True)
        else:
            st.info(ll.get("status", "Lead-lag analysis not run."))

# ----------------------------------------------------------------------------- tab 5: method
with tab5:
    st.markdown(f"""
#### How the numbers are produced
- **Comparison basis:** 10-K vs the prior-year 10-K; 10-Q vs the **same fiscal quarter one year earlier**.
  Sections: Item 1A Risk Factors and MD&A (Item 7 / Part I Item 2).
- **Similarity:** cosine similarity of word-frequency vectors and Jaccard similarity (Cohen, Malloy & Nguyen 2020).
- **New sentences:** sentences with no close match (TF-IDF cosine < 0.75, numbers ignored) in the comparison filing.
  Themes are assigned by keyword rules with fixed severity weights.
- **Tone:** {dictionary_name()}.
- **Filing Change Score:** the equal-weighted average percentile of {', '.join(SCORE_COMPONENTS.values()).lower()},
  each ranked against same-type filings from the research universe in the 365 days **before** the filing.
  Nothing in the score is fitted to returns.
- **Excess return:** stock return minus its SPDR sector ETF (mapped from SIC) over 63 trading days, starting the
  trading day after the filing date.
- **Ratios:** XBRL values from the *same filing* (accession-matched), so later restatements can't leak in.
  Altman Z'' = 6.56·WC/TA + 3.26·RE/TA + 6.72·EBIT/TA + 1.05·BookEquity/TL.
- **Pricing label:** "Potentially over/underpriced" when the validated model expects ≤ −1.5% / ≥ +1.5% excess
  return and has positive out-of-sample IC; otherwise "No reliable signal".
- **Peers:** listed companies with the same 4-digit SIC (widened to the 3-digit group if needed), ranked by
  closeness in total assets from one SEC XBRL frame.

#### Known limitations
Survivorship (delisted firms have no free price history, though they stay in the distress tests); a universe of
roughly 150 companies; SIC codes are coarse; keyword themes miss paraphrase; yfinance is unofficial; no
transaction costs. Not investment advice.
""")
    st.markdown("#### Agent audit trail (this run)")
    st.dataframe(pd.DataFrame(R["log"]), hide_index=True, width="stretch")
    st.download_button("Download this company's filing-level data (CSV)",
                       company.drop(columns=["evidence"], errors="ignore").to_csv(index=False).encode(),
                       file_name=f"{R['ticker']}_filingsignal.csv", mime="text/csv")
