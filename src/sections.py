"""Extract the narrative sections that matter from 10-K / 10-Q HTML.

10-K: Item 1A (Risk Factors) and Item 7 (MD&A)
10-Q: Part II Item 1A (Risk Factors updates) and Part I Item 2 (MD&A)

Approach: flatten HTML to text (dropping numeric tables but keeping tables that
only hold headings), then find every "Item X" heading match and keep the
longest span that ends at the next item heading. The longest span is almost
always the real section rather than the table-of-contents entry.
"""
import gzip
import json
import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from src.config import SECTION_CACHE

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

SEP = r"\s*[\.\-–—:]?\s*"
PATTERNS = {
    "10-K": {
        "risk": (rf"item\s*1a{SEP}risk\s+factors",
                 rf"item\s*1b{SEP}unresolved|item\s*1c{SEP}cybersecurity|item\s*2{SEP}(description\s+of\s+)?propert"),
        "mdna": (rf"item\s*7{SEP}management.{{0,3}}s\s+discussion",
                 rf"item\s*7a{SEP}quantitative|item\s*8{SEP}(consolidated\s+)?financial\s+statements"),
    },
    "10-Q": {
        "mdna": (rf"item\s*2{SEP}management.{{0,3}}s\s+discussion",
                 rf"item\s*3{SEP}quantitative|item\s*4{SEP}controls"),
        "risk": (rf"item\s*1a{SEP}risk\s+factors",
                 rf"item\s*2{SEP}unregistered|item\s*3{SEP}defaults|item\s*4{SEP}(mine|submission)"
                 rf"|item\s*5{SEP}other\s+information|item\s*6{SEP}exhibits"),
    },
}
MIN_SECTION_CHARS = {"risk": 300, "mdna": 1500}
MAX_SECTION_CHARS = 400_000
NO_CHANGE_RE = re.compile(r"(no|not been any)\s+material\s+changes?", re.I)


def html_to_text(html: bytes) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    for tag in soup.find_all(re.compile(r"^ix:header$")):
        tag.decompose()
    for table in soup.find_all("table"):
        txt = table.get_text(" ", strip=True)
        digits = sum(c.isdigit() for c in txt)
        # numeric tables -> drop; layout tables holding headings/prose -> keep
        if not txt or digits / max(len(txt), 1) > 0.08:
            table.decompose()
    # Block elements end a line; table cells get a space. Inline elements (span, font, a)
    # are joined WITHOUT a separator, because filings often split words across spans.
    for el in soup.find_all(["br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6"]):
        el.append("\n")
    for el in soup.find_all(["td", "th"]):
        el.append(" ")
    text = soup.get_text("")
    text = text.replace("\xa0", " ").replace("​", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text


LINE = r"(?im)^[ \t]*(?:part\s+i{1,2}\s*[\.,\-–—:]?\s*)?"   # heading must start a line


def _longest_span(text, start_re, end_re):
    start_c = re.compile(LINE + "(?:" + start_re + ")")
    end_c = re.compile(LINE + "(?:" + end_re + ")")
    best, fallback = "", ""
    for m in start_c.finditer(text):
        s = m.start()
        e = end_c.search(text, m.end() + 40)
        if e is None:   # no closing heading: keep only as a last resort
            if not fallback:
                fallback = text[s:s + 150_000]
            continue
        if e.start() - s > len(best):
            best = text[s:e.start()]
    return (best or fallback)[:MAX_SECTION_CHARS]


def extract_sections(html: bytes, form: str) -> dict:
    text = html_to_text(html)
    out = {"full_chars": len(text)}
    for name, (start_re, end_re) in PATTERNS[form].items():
        sec = _longest_span(text, start_re, end_re)
        out[name] = sec if len(sec) >= MIN_SECTION_CHARS[name] else ""
    # 10-Q risk sections often just say "no material changes" -> keep, but flag
    out["risk_no_change"] = bool(form == "10-Q" and out["risk"] and len(out["risk"]) < 3000
                                 and NO_CHANGE_RE.search(out["risk"]))
    if not out["risk"] and not out["mdna"]:
        out["fallback"] = text[:250_000]  # sections not found -> use document start, flagged
    return out


BUSINESS_RE = (rf"item\s*1{SEP}(description\s+of\s+)?business",
               rf"item\s*1a{SEP}risk\s+factors|item\s*2{SEP}(description\s+of\s+)?propert")
BUSINESS_MAX_CHARS = 60_000


def extract_business(html: bytes) -> str:
    """10-K Item 1 (Business) text, truncated: what the company actually sells, to whom."""
    sec = _longest_span(html_to_text(html), *BUSINESS_RE)
    return sec[:BUSINESS_MAX_CHARS] if len(sec) >= 1000 else ""


def get_business(edgar, filing: dict) -> str:
    """Cached Item 1 text for one 10-K filing record (used for peer similarity)."""
    path = SECTION_CACHE / f"{filing['accessionNumber']}.business.txt.gz"
    if path.exists():
        return gzip.decompress(path.read_bytes()).decode("utf-8")
    text = extract_business(edgar.document(filing["url"]))
    path.write_bytes(gzip.compress(text.encode("utf-8")))
    return text


def get_sections(edgar, filing: dict) -> dict:
    """Cached section extraction for one filing record."""
    path = SECTION_CACHE / f"{filing['accessionNumber']}.json.gz"
    if path.exists():
        return json.loads(gzip.decompress(path.read_bytes()))
    html = edgar.document(filing["url"])
    sections = extract_sections(html, filing["form"])
    path.write_bytes(gzip.compress(json.dumps(sections).encode()))
    return sections
