"""Compare a filing section with the matching section of the comparison filing.

Similarity measures follow Cohen, Malloy & Nguyen (2020, "Lazy Prices"):
cosine similarity of term-frequency vectors and Jaccard similarity of word sets.
On top of that we diff at the sentence level to surface the exact sentences
that were ADDED or REMOVED, and tag added sentences with risk themes.
"""
import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.lexicon import THEME_WEIGHT, TOKEN_RE, themes_of

SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(\"“•])|\n+")
MAX_SENTENCES = 3000
MODIFIED_THRESHOLD = 0.75   # TF-IDF cosine at/above which a sentence counts as a light edit, not new


def sentences(text):
    out = []
    for s in SENT_SPLIT.split(text or ""):
        s = s.strip(" •\t-–—")
        if len(s) >= 40 and sum(c.isalpha() for c in s) / len(s) > 0.6:
            out.append(s[:1500])
    return out[:MAX_SENTENCES]


def _norm(s):
    s = s.lower()
    s = re.sub(r"\d[\d,\.]*", "#", s)       # numbers change every period; ignore them
    return re.sub(r"[^a-z#]+", " ", s).strip()


def cosine_tf(a, b):
    ca, cb = Counter(TOKEN_RE.findall(a.lower())), Counter(TOKEN_RE.findall(b.lower()))
    if not ca or not cb:
        return float("nan")
    dot = sum(v * cb.get(k, 0) for k, v in ca.items())
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb)


def jaccard(a, b):
    sa, sb = set(TOKEN_RE.findall(a.lower())), set(TOKEN_RE.findall(b.lower()))
    if not sa or not sb:
        return float("nan")
    return len(sa & sb) / len(sa | sb)


def sentence_diff(current, previous):
    """Classify current sentences as unchanged / modified / added; find removed ones."""
    cur, prev = sentences(current), sentences(previous)
    if not cur or not prev:
        return {"n_cur": len(cur), "n_prev": len(prev), "added": [], "removed": [], "n_modified": 0}
    cur_n, prev_n = [_norm(s) for s in cur], [_norm(s) for s in prev]
    prev_set, cur_set = set(prev_n), set(cur_n)
    cand_cur = [i for i, s in enumerate(cur_n) if s not in prev_set]
    cand_prev = [j for j, s in enumerate(prev_n) if s not in cur_set]
    added, removed, modified = [], [], 0
    if cand_cur or cand_prev:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        vec.fit(cur_n + prev_n)
        if cand_cur:
            sims = (vec.transform([cur_n[i] for i in cand_cur]) @ vec.transform(prev_n).T).max(axis=1).toarray().ravel()
            for i, s in zip(cand_cur, sims):
                if s >= MODIFIED_THRESHOLD:
                    modified += 1
                else:
                    added.append((cur[i], float(s)))
        if cand_prev:
            sims = (vec.transform([prev_n[j] for j in cand_prev]) @ vec.transform(cur_n).T).max(axis=1).toarray().ravel()
            removed = [(prev[j], float(s)) for j, s in zip(cand_prev, sims) if s < MODIFIED_THRESHOLD]
    return {"n_cur": len(cur), "n_prev": len(prev), "added": added, "removed": removed, "n_modified": modified}


def compare_section(current, previous):
    """All change metrics for one section pair, plus tagged evidence sentences."""
    out = {"cos": cosine_tf(current, previous), "jac": jaccard(current, previous)}
    n_cur_words = len(TOKEN_RE.findall((current or "").lower()))
    n_prev_words = len(TOKEN_RE.findall((previous or "").lower()))
    out["len_logratio"] = (math.log((n_cur_words + 1) / (n_prev_words + 1))
                           if current and previous else float("nan"))
    d = sentence_diff(current, previous)
    n = max(d["n_cur"], 1)
    out["n_sent"] = d["n_cur"]
    out["added_share"] = len(d["added"]) / n if d["n_prev"] else float("nan")
    out["removed_share"] = len(d["removed"]) / max(d["n_prev"], 1) if d["n_prev"] else float("nan")
    theme_counts = Counter()
    evidence = []
    for sent, sim in d["added"]:
        th = themes_of(sent)
        theme_counts.update(th)
        if th:
            weight = sum(THEME_WEIGHT[t] for t in th)
            evidence.append({"text": sent, "themes": th, "weight": weight, "closest_prior_sim": round(sim, 2)})
    out["theme_counts"] = dict(theme_counts)
    out["risk_weighted_added"] = sum(e["weight"] for e in evidence)
    evidence.sort(key=lambda e: -e["weight"])
    out["evidence_added"] = evidence[:15]
    removed_risk = [{"text": s, "themes": themes_of(s)} for s, _ in d["removed"] if themes_of(s)]
    out["evidence_removed"] = removed_risk[:8]
    out["n_added"], out["n_removed"], out["n_modified"] = len(d["added"]), len(d["removed"]), d["n_modified"]
    return out


def safe_mean(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")
