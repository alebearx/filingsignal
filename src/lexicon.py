"""Tone dictionaries and risk-theme rules.

Tone: Loughran & McDonald (2011) finance sentiment categories. If the free
master dictionary CSV is placed in data/ (download from
https://sraf.nd.edu/loughranmcdonald-master-dictionary/), the full word lists
are used. Otherwise a compact built-in stem list drawn from the most frequent
LM words is used, and the app labels it as such.

Risk themes: transparent regular-expression rules used to categorize sentences
that are NEW relative to the comparison filing.
"""
import re
from functools import lru_cache

import pandas as pd

from src.config import DATA_DIR

TONE_CATEGORIES = ["negative", "positive", "uncertainty", "litigious", "constraining"]

SEED_STEMS = {
    "negative": ["advers", "against", "bankrupt", "breach", "burden", "claim", "closing", "concern", "conflict",
                 "costly", "crisis", "critical", "curtail", "damag", "decline", "decreas", "default", "deficien",
                 "delay", "delinquen", "deteriorat", "difficult", "diminish", "disappoint", "disput", "disrupt",
                 "distress", "doubt", "downgrad", "downturn", "erode", "erosion", "fail", "fraud", "harm",
                 "impair", "inability", "inadequa", "ineffective", "insolven", "interrupt", "investigat", "lawsuit",
                 "layoff", "liquidat", "litigat", "loss", "losses", "lost", "negative", "negatively", "obsolete",
                 "penalt", "poor", "problem", "recall", "restat", "restructur", "severe", "shortage", "shortfall",
                 "shutdown", "slowdown", "suspend", "terminat", "unable", "unfavorabl", "unprofitab", "volatil",
                 "weak", "worse", "writedown", "write-down", "writeoff"],
    "positive": ["achiev", "attain", "benefit", "best", "better", "enhanc", "excellen", "favorabl", "gain",
                 "great", "improv", "innovat", "opportunit", "outperform", "positive", "profitab", "progress",
                 "rebound", "record", "strength", "strong", "succeed", "success", "superior"],
    "uncertainty": ["anticipat", "apparent", "approximat", "assum", "believ", "contingen", "could", "depend",
                    "fluctuat", "indefinite", "may", "maybe", "might", "nearly", "pending", "perhaps", "possib",
                    "predict", "preliminar", "probab", "risk", "roughly", "seldom", "sometime", "somewhat",
                    "suggest", "tentative", "uncertain", "unclear", "unknown", "unpredictab", "unproven", "variab"],
    "litigious": ["adjudicat", "allegation", "amend", "appeal", "attorney", "claimant", "complaint", "contract",
                  "counterclaim", "court", "defendant", "indemn", "injunct", "judg", "jurisdiction", "legal",
                  "legislat", "litigat", "plaintiff", "regulat", "settlement", "statut", "subpoena", "testimony",
                  "tribunal", "violat"],
    "constraining": ["bound", "commit", "compel", "constrain", "covenant", "encumb", "entail", "forbid",
                     "impose", "limit", "mandat", "must", "necessitat", "obligat", "prerequisite", "prevent",
                     "prohibit", "require", "restrict", "unavailab"],
}

TOKEN_RE = re.compile(r"[a-z][a-z\-]+")


@lru_cache(maxsize=1)
def _lm_sets():
    files = sorted(DATA_DIR.glob("Loughran*McDonald*MasterDictionary*.csv"))
    if not files:
        return None
    df = pd.read_csv(files[-1])
    df.columns = [c.strip().lower() for c in df.columns]
    df["word"] = df["word"].astype(str).str.lower()
    sets = {}
    for cat in TONE_CATEGORIES:
        if cat in df.columns:
            sets[cat] = frozenset(df.loc[df[cat] != 0, "word"])
    return sets


def dictionary_name():
    return "Loughran-McDonald master dictionary (full)" if _lm_sets() else "Built-in LM-style seed stems (download full LM CSV for research use)"


@lru_cache(maxsize=1)
def _seed_res():
    return {c: re.compile("^(?:" + "|".join(map(re.escape, s)) + ")") for c, s in SEED_STEMS.items()}


def tone(text: str) -> dict:
    """Share of words (in %) in each tone category, plus word count."""
    tokens = TOKEN_RE.findall(text.lower())
    n = len(tokens)
    out = {"words": n}
    if n == 0:
        return {**out, **{f"tone_{c}": float("nan") for c in TONE_CATEGORIES}}
    counts = pd.Series(tokens).value_counts()
    lm = _lm_sets()
    for cat in TONE_CATEGORIES:
        if lm and cat in lm:
            hits = counts[counts.index.isin(lm[cat])].sum()
        else:
            rx = _seed_res()[cat]
            hits = sum(v for w, v in counts.items() if rx.match(w))
        out[f"tone_{cat}"] = 100.0 * hits / n
    return out


# ----------------------------------------------------------------------------- risk themes
# (label, severity weight, regex). Weight reflects how directly the theme relates to
# credit / solvency risk; it is a fixed, pre-specified choice (not fitted to returns).
THEMES = {
    "going_concern": ("Going concern / solvency", 5.0,
                      r"going concern|substantial doubt|bankruptcy|chapter 11|insolven|restructuring support agreement"),
    "liquidity": ("Liquidity & cash", 3.0,
                  r"liquidity|cash (position|burn|runway|on hand)|working capital|sufficient (cash|funds|liquidity)"
                  r"|ability to (fund|meet our)|access to (capital|financing)|capital markets"),
    "debt": ("Debt, covenants & refinancing", 3.0,
             r"covenant|refinanc|indebtedness|credit (facility|agreement|rating)|revolving|borrowing|maturit"
             r"|downgrad|senior notes|default|leverage ratio|waiver"),
    "controls": ("Accounting, controls & impairment", 2.0,
                 r"material weakness|restat|internal control|impairment|write-?down|goodwill|valuation allowance"),
    "legal": ("Legal & regulatory", 1.5,
              r"litigation|lawsuit|class action|investigation|subpoena|settlement|antitrust|enforcement"
              r"|department of justice|consent decree|fine[sd]? "),
    "restructuring": ("Restructuring & workforce", 1.5,
                      r"restructur|layoff|reduction in force|workforce reduction|headcount|facility closure"
                      r"|strike|labor (shortage|dispute)|union"),
    "demand": ("Demand, customers & competition", 1.0,
               r"demand|customer|backlog|order(s)? |pricing pressure|competit|market share|lower (sales|volume)"),
    "costs_supply": ("Costs, inflation & supply chain", 1.0,
                     r"supply chain|supplier|shortage|inflation|raw material|tariff|commodity|freight|cost increase"),
    "macro": ("Macro, rates & FX", 0.5,
              r"interest rate|recession|macroeconomic|economic (downturn|conditions|slowdown)|foreign (currency|exchange)"
              r"|geopolit|war |sanction"),
    "tech_cyber": ("Cybersecurity & technology", 0.5,
                   r"cyber|data breach|ransomware|security incident|artificial intelligence|outage"),
}
THEME_RES = {k: re.compile(v[2], re.I) for k, v in THEMES.items()}
THEME_LABEL = {k: v[0] for k, v in THEMES.items()}
THEME_WEIGHT = {k: v[1] for k, v in THEMES.items()}


def themes_of(sentence: str) -> list:
    return [k for k, rx in THEME_RES.items() if rx.search(sentence)]
