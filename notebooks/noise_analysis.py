"""Noise analysis of matched training pairs (NonStandard.md T18), TRAIN DATA ONLY.

Samples ~50k matched (S1, S2/S3) pairs per country (half S2, half S3) and ~2.5k
distractor records per country, then measures:
  - token-level alignment of names and addresses (exact / accent / number format /
    typo / abbreviation / merge / substitution / dropped / added) and the most
    frequent substitution pairs: the generator's normalisation dictionary
  - typo edit types, name-level events (legal suffixes, brackets, junk, domains,
    script, case, reordering), address-level events (house number, postcode,
    state, city, landmarks, component reordering, missing)
  - how distractors differ from matched records (nearest train S1 via char
    3-gram TF-IDF fitted on train S1 only)

Run:  python notebooks/noise_analysis.py
Writes experiments/noise_analysis.md and experiments/noise_substitutions.csv.
"""

from __future__ import annotations

import gc
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import psutil
from rapidfuzz import fuzz
from rapidfuzz.distance import OSA, Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code" / "business_entity_resolution" / "src"))
from data import load_ground_truth_pairs, scan_source  # noqa: E402

SEED = 42
N_PAIRS_PER_COUNTRY = 50_000
N_DISTRACTORS_PER_COUNTRY = 2_500
N_S1_SURFACE = 50_000
OUT_MD = REPO / "experiments" / "noise_analysis.md"
OUT_CSV = REPO / "experiments" / "noise_substitutions.csv"
S1 = "source1_entity_id"

pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
pl.Config.set_tbl_hide_column_data_types(True)
pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(400)
pl.Config.set_fmt_str_lengths(120)
pl.Config.set_float_precision(3)

MD: list[str] = []
_T0 = time.perf_counter()


def emit(*parts: object) -> None:
    text = "\n".join(str(p) for p in parts)
    MD.append(text)
    print(text, flush=True)


def section(title: str) -> None:
    rss = psutil.Process().memory_info().rss / 2**30
    print(f"[{time.perf_counter() - _T0:7.1f}s  RSS {rss:.2f} GiB] {title}", file=sys.stderr, flush=True)
    emit(f"\n## {title}\n")


# --------------------------------------------------------------------------
# Reference lists written from general knowledge (not mined from any data)
# --------------------------------------------------------------------------
US_STATES = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas", "CA": "california",
    "CO": "colorado", "CT": "connecticut", "DE": "delaware", "FL": "florida", "GA": "georgia",
    "HI": "hawaii", "ID": "idaho", "IL": "illinois", "IN": "indiana", "IA": "iowa",
    "KS": "kansas", "KY": "kentucky", "LA": "louisiana", "ME": "maine", "MD": "maryland",
    "MA": "massachusetts", "MI": "michigan", "MN": "minnesota", "MS": "mississippi",
    "MO": "missouri", "MT": "montana", "NE": "nebraska", "NV": "nevada", "NH": "new hampshire",
    "NJ": "new jersey", "NM": "new mexico", "NY": "new york", "NC": "north carolina",
    "ND": "north dakota", "OH": "ohio", "OK": "oklahoma", "OR": "oregon", "PA": "pennsylvania",
    "RI": "rhode island", "SC": "south carolina", "SD": "south dakota", "TN": "tennessee",
    "TX": "texas", "UT": "utah", "VT": "vermont", "VA": "virginia", "WA": "washington",
    "WV": "west virginia", "WI": "wisconsin", "WY": "wyoming", "DC": "district of columbia",
    "PR": "puerto rico",
}
IN_STATES = {
    "AP": "andhra pradesh", "AR": "arunachal pradesh", "AS": "assam", "BR": "bihar",
    "CG": "chhattisgarh", "CT": "chhattisgarh", "GA": "goa", "GJ": "gujarat", "HR": "haryana",
    "HP": "himachal pradesh", "JH": "jharkhand", "KA": "karnataka", "KL": "kerala",
    "MP": "madhya pradesh", "MH": "maharashtra", "MN": "manipur", "ML": "meghalaya",
    "MZ": "mizoram", "NL": "nagaland", "OD": "odisha", "OR": "odisha", "PB": "punjab",
    "RJ": "rajasthan", "SK": "sikkim", "TN": "tamil nadu", "TS": "telangana", "TG": "telangana",
    "TR": "tripura", "UP": "uttar pradesh", "UK": "uttarakhand", "UA": "uttarakhand",
    "WB": "west bengal", "AN": "andaman and nicobar islands", "CH": "chandigarh",
    "DN": "dadra and nagar haveli and daman and diu", "DD": "dadra and nagar haveli and daman and diu",
    "DL": "delhi", "JK": "jammu and kashmir", "LA": "ladakh", "LD": "lakshadweep", "PY": "puducherry",
}
IN_STATE_ALIASES = {"orissa": "odisha", "pondicherry": "puducherry", "uttaranchal": "uttarakhand",
                    "nct of delhi": "delhi", "new delhi": None}
STATES = {"US": US_STATES, "India": IN_STATES}

LEGAL = {
    "private", "pvt", "limited", "ltd", "llp", "llc", "inc", "incorporated", "corp",
    "corporation", "co", "company", "pc", "plc", "lp", "pllc", "pty", "gmbh",
    "sarl", "sas", "sasu", "sa", "eurl", "snc", "sci",
}
LANDMARK_RE = re.compile(r"\b(near|nr|opp|opposite|behind|beside|next to|adjacent|adj|in front of|facing)\b", re.I)
DOMAIN_RE = re.compile(r"\.(com|in|net|org|co|biz|info|us)\s*$", re.I)
JUNK_PREFIX_RE = re.compile(r"""^[^\w\s"'(\[]+""")
JUNK_SUFFIX_RE = re.compile(r"[^\w\s.)\]]+$")
BRACKET_RE = re.compile(r"[\[\](){}]")
NULL_RE = re.compile(r"<\s*null\s*>|\bnull\b|\bn/?a\b|\bnone\b", re.I)
TOKEN_RE = re.compile(r"[\ẁ-ͯऀ-෿]+")  # keep Indic vowel signs / combining marks
NUM_RE = re.compile(r"\d+")

SCRIPT_RANGES = [
    (0x0900, 0x097F, "Devanagari"), (0x0980, 0x09FF, "Bengali"), (0x0A00, 0x0A7F, "Gurmukhi"),
    (0x0A80, 0x0AFF, "Gujarati"), (0x0B00, 0x0B7F, "Oriya"), (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"), (0x0C80, 0x0CFF, "Kannada"), (0x0D00, 0x0D7F, "Malayalam"),
    (0x0600, 0x06FF, "Arabic"), (0x0400, 0x04FF, "Cyrillic"), (0x4E00, 0x9FFF, "CJK"),
]
QWERTY = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
_KEYPOS = {c: (r, i) for r, row in enumerate(QWERTY) for i, c in enumerate(row)}


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
def fold(s: str) -> str:
    """NFKD, drop combining marks, casefold."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).casefold()


def base(s: str) -> str:
    return unicodedata.normalize("NFKC", s).casefold()


def tokens(s: str) -> list[str]:
    return TOKEN_RE.findall(base(s))


def script_of(s: str) -> str | None:
    """First non-Latin script found in s, else None."""
    for ch in s:
        o = ord(ch)
        if o < 0x0370:
            continue
        for lo, hi, name in SCRIPT_RANGES:
            if lo <= o <= hi:
                return name
    return None


def has_accent(s: str) -> bool:
    return any(unicodedata.combining(c) for c in unicodedata.normalize("NFKD", s) if ord(c) < 0x0900)


def case_style(s: str) -> str:
    letters = [c for c in s if c.isalpha() and ord(c) < 0x0250]
    if not letters:
        return "no_latin"
    if all(c.isupper() for c in letters):
        return "upper"
    if all(c.islower() for c in letters):
        return "lower"
    words = [w for w in s.split() if w[:1].isalpha()]
    if words and all(w[0].isupper() for w in words):
        return "title"
    return "mixed"


def is_subseq(short: str, long: str) -> bool:
    it = iter(long)
    return all(c in it for c in short)


def keyboard_adjacent(a: str, b: str) -> bool:
    pa, pb = _KEYPOS.get(a), _KEYPOS.get(b)
    return bool(pa and pb and abs(pa[0] - pb[0]) <= 1 and abs(pa[1] - pb[1]) <= 1)


# --------------------------------------------------------------------------
# Token alignment
# --------------------------------------------------------------------------
def classify_pair(a: str, b: str) -> tuple[str, float] | None:
    """Relation of S1 token a to other-record token b (a != b), with a cost (lower = better)."""
    if fold(a) == fold(b):
        return "accent", 0.0
    if a.isdigit() and b.isdigit():
        if a.lstrip("0") == b.lstrip("0"):
            return "number_format", 0.1
        if Levenshtein.distance(a, b) == 1:
            return "number_edit", 0.5
        return None
    fa, fb = fold(a), fold(b)
    short, long_ = (fb, fa) if len(fb) < len(fa) else (fa, fb)
    if (short and short[0] == long_[0] and len(long_) - len(short) >= 2
            and len(short) <= 0.75 * len(long_) and is_subseq(short, long_) and not short.isdigit()):
        return ("abbreviation" if len(fb) < len(fa) else "expansion"), 0.15
    if len(fa) >= 3 and len(fb) >= 3:
        d = OSA.distance(fa, fb)  # adjacent transposition = 1 edit
        n = max(len(fa), len(fb))
        if d <= (1 if n <= 4 else 2 if n <= 7 else 3) and fuzz.ratio(fa, fb) >= 60:
            return "typo", 0.2 + d * 0.1
        if fa[0] == fb[0] and fuzz.ratio(fa, fb) >= 50:
            return "heavy_typo", 0.8
    return None


def align(ta: list[str], tb: list[str]) -> list[tuple[str, str | None, str | None]]:
    """Align S1 tokens ta with other tokens tb. Returns (class, a, b) events."""
    ca, cb = Counter(ta), Counter(tb)
    common = ca & cb
    events: list[tuple[str, str | None, str | None]] = [("exact", t, t) for t in common.elements()]
    la = list((ca - common).elements())
    lb = list((cb - common).elements())
    # merges / splits: "meerut kutir" -> "meerutkutir"
    for i in range(len(la)):
        for j in range(len(la)):
            if i != j and la[i] and la[j] and la[i] + la[j] in lb:
                merged = la[i] + la[j]
                events.append(("merge", f"{la[i]} {la[j]}", merged))
                lb.remove(merged)
                la[i] = la[j] = ""
    la = [t for t in la if t]
    # greedy best-cost matching of leftovers
    cands = []
    for i, a in enumerate(la):
        for j, b in enumerate(lb):
            r = classify_pair(a, b)
            if r:
                cands.append((r[1], i, j, r[0]))
    used_a, used_b = set(), set()
    for _, i, j, cls in sorted(cands):
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        events.append((cls, la[i], lb[j]))
    ra = [a for i, a in enumerate(la) if i not in used_a]
    rb = [b for j, b in enumerate(lb) if j not in used_b]
    if len(ra) == 1 and len(rb) == 1:
        events.append(("substitution", ra[0], rb[0]))
    else:
        events.extend(("dropped", a, None) for a in ra)
        events.extend(("added", None, b) for b in rb)
    return events


def typo_ops(a: str, b: str) -> list[str]:
    a, b = fold(a), fold(b)
    n_transp = Levenshtein.distance(a, b) - OSA.distance(a, b)
    if n_transp > 0:  # each adjacent swap saves one edit under OSA
        return ["transposition"] * n_transp + ["other_edit_with_transposition"] * (OSA.distance(a, b) - n_transp)
    ops = Levenshtein.editops(a, b)
    out = []
    for o in ops:
        if o.tag == "replace":
            kb = keyboard_adjacent(a[o.src_pos], b[o.dest_pos])
            out.append("replace_keyboard_adjacent" if kb else "replace_other")
        else:
            out.append(o.tag)
    return out


# --------------------------------------------------------------------------
# Address structure
# --------------------------------------------------------------------------
def components(addr: str) -> list[str]:
    return [c.strip() for c in addr.split(",") if c.strip()]


def find_state(comps: list[str], country: str) -> tuple[str | None, str | None, int | None]:
    """(canonical state, form, component index) for the first component that is a state."""
    table = STATES.get(country, {})
    names = {v: v for v in table.values()}
    for i, c in enumerate(comps):
        if script_of(c):
            if len(c.split()) <= 3:
                return None, "native_script", i
            continue
        f = fold(c)
        if c.strip().upper() in table and len(c.strip()) == 2:
            return table[c.strip().upper()], "code", i
        if f in names:
            return f, "full_name", i
        if country == "India" and f in IN_STATE_ALIASES and IN_STATE_ALIASES[f]:
            return IN_STATE_ALIASES[f], "alias", i
    return None, None, None


def postcode(addr: str, country: str) -> str | None:
    comps = components(addr)
    if country == "India":
        m = re.search(r"(?<!\d)[1-9]\d{5}(?!\d)", addr)
        return m.group(0) if m else None
    # US-like: a later component that is/ends with a 5-digit (+4) code
    for c in comps[1:]:
        m = re.search(r"(?:^|\s)(\d{5})(?:-\d{4})?$", c)
        if m:
            return m.group(1)
    return None


def number_relation(n1: list[str], n2: list[str]) -> str:
    if not n1:
        return "s1_none_other_has" if n2 else "both_none"
    if not n2:
        return "dropped_all"
    h = n1[0]
    if h in n2:
        return "exact"
    if any(x.lstrip("0") == h.lstrip("0") for x in n2):
        return "zero_padded"
    if any(Levenshtein.distance(h, x) == 1 for x in n2):
        return "one_digit_edit"
    return "missing_or_changed"


def component_order(c1: list[str], c2: list[str]) -> str:
    if len(c1) < 2 or len(c2) < 2:
        return "n/a"
    idx = []
    for a in c1:
        fa = fold(a)
        best, bj = 0, None
        for j, b in enumerate(c2):
            s = fuzz.ratio(fa, fold(b))
            if s > best:
                best, bj = s, j
        if best >= 70:
            idx.append(bj)
    if len(idx) < 2:
        return "unmatched"
    return "same_order" if all(x < y for x, y in zip(idx, idx[1:])) else "reordered"


# --------------------------------------------------------------------------
# Per-record surface stats (used for S1, matched, distractors)
# --------------------------------------------------------------------------
def surface(name: str, addr: str) -> dict:
    return dict(
        name_len=len(name),
        name_tokens=len(tokens(name)),
        name_case=case_style(name),
        name_script=script_of(name) or "Latin",
        name_accent=has_accent(name),
        name_junk_prefix=bool(JUNK_PREFIX_RE.search(name)),
        name_junk_suffix=bool(JUNK_SUFFIX_RE.search(name)),
        name_bracket=bool(BRACKET_RE.search(name)),
        name_domain=bool(DOMAIN_RE.search(name)),
        name_double_space="  " in name,
        name_has_legal=any(t in LEGAL for t in tokens(name)),
        addr_missing=addr.strip() == "",
        addr_null_token=bool(NULL_RE.search(addr)),
        addr_case=case_style(addr) if addr.strip() else "missing",
        addr_script=(script_of(addr) or "Latin") if addr.strip() else "missing",
        addr_components=len(components(addr)),
        addr_landmark=bool(LANDMARK_RE.search(addr)),
        addr_hash_prefix=addr.lstrip().startswith("#"),
    )


SURFACE_AGG = [
    pl.len().alias("n"),
    pl.col("name_len").mean().alias("name_len"),
    pl.col("name_tokens").mean().alias("name_tok"),
    (pl.col("name_case") == "upper").mean().alias("name_upper"),
    (pl.col("name_case") == "lower").mean().alias("name_lower"),
    (pl.col("name_script") != "Latin").mean().alias("name_nonlatin"),
    pl.col("name_accent").mean().alias("name_accent"),
    pl.col("name_junk_prefix").mean().alias("junk_pre"),
    pl.col("name_junk_suffix").mean().alias("junk_suf"),
    pl.col("name_bracket").mean().alias("bracket"),
    pl.col("name_domain").mean().alias("domain"),
    pl.col("name_double_space").mean().alias("dbl_space"),
    pl.col("name_has_legal").mean().alias("has_legal"),
    pl.col("addr_missing").mean().alias("addr_miss"),
    pl.col("addr_null_token").mean().alias("addr_null"),
    (pl.col("addr_case") == "upper").mean().alias("addr_upper"),
    (pl.col("addr_script").is_in(["Latin", "missing"]).not_()).mean().alias("addr_nonlatin"),
    pl.col("addr_components").mean().alias("addr_comps"),
    pl.col("addr_landmark").mean().alias("landmark"),
    pl.col("addr_hash_prefix").mean().alias("addr_hash"),
]


# --------------------------------------------------------------------------
# Pair analysis
# --------------------------------------------------------------------------
def analyse_pair(r: dict) -> tuple[dict, list[tuple]]:
    n1, n2 = r["business_name"], r["o_name"]
    a1, a2 = r["business_address"], r["o_addr"]
    country = r["country"]
    t1, t2 = tokens(n1), tokens(n2)
    ta1, ta2 = tokens(a1), tokens(a2)
    ev_name = align(t1, t2)
    ev_addr = align(ta1, ta2) if a2.strip() else []

    legal1 = [t for t in t1 if t in LEGAL]
    legal2 = [t for t in t2 if t in LEGAL]
    common = [t for t in t1 if t in set(t2)]
    common2 = [t for t in t2 if t in set(t1)]
    dedup = lambda xs: list(dict.fromkeys(xs))  # noqa: E731

    c1, c2 = components(a1), components(a2)
    st1 = find_state(c1, country)
    st2 = find_state(c2, country) if a2.strip() else (None, None, None)
    city = c1[st1[2] - 1] if st1[2] else None
    if not a2.strip():
        city_rel = "addr_missing"
    elif city is None:
        city_rel = "n/a"
    elif any(fold(city) == fold(c) for c in c2):
        city_rel = "exact_component"
    elif fuzz.partial_ratio(fold(city), fold(a2)) >= 90:
        city_rel = "variant"
    else:
        city_rel = "missing"
    pc1, pc2 = postcode(a1, country), postcode(a2, country)
    lm1, lm2 = bool(LANDMARK_RE.search(a1)), bool(LANDMARK_RE.search(a2))

    feats = dict(
        name_identical=n1 == n2,
        name_identical_casefold=base(n1) == base(n2),
        name_tsr=fuzz.token_set_ratio(fold(n1), fold(n2)),
        name_ratio=fuzz.ratio(fold(n1), fold(n2)),
        addr_tsr=fuzz.token_set_ratio(fold(a1), fold(a2)) if a2.strip() else None,
        addr_identical=a1 == a2,
        name_script=script_of(n2) or "Latin",
        name_case=case_style(n2),
        name_accent_injected=has_accent(n2) and not has_accent(n1),
        name_bracket_added=bool(BRACKET_RE.search(n2)) and not BRACKET_RE.search(n1),
        name_junk_added=(bool(JUNK_PREFIX_RE.search(n2)) and not JUNK_PREFIX_RE.search(n1))
        or (bool(JUNK_SUFFIX_RE.search(n2)) and not JUNK_SUFFIX_RE.search(n1)),
        name_domain=bool(DOMAIN_RE.search(n2)) and not DOMAIN_RE.search(n1),
        name_double_space="  " in n2 and "  " not in n1,
        name_dup_token=any(x == y for x, y in zip(t2, t2[1:])),
        name_word_order_changed=len(common) >= 2 and dedup(common) != dedup(common2),
        name_unrelated=fuzz.token_set_ratio(fold(n1), fold(n2)) < 40 and not script_of(n2),
        legal_s1=bool(legal1),
        legal_dropped=bool(legal1) and not legal2,
        legal_changed=bool(legal1) and bool(legal2) and legal1 != legal2,
        legal_added=not legal1 and bool(legal2),
        legal_moved_front=bool(legal2) and bool(t2) and t2[0] in LEGAL and not (t1 and t1[0] in LEGAL),
        amp_s1="&" in n1,
        amp_to_and="&" in n1 and "&" not in n2 and "and" in t2,
        amp_to_plus="&" in n1 and "+" in n2,
        addr_missing=not a2.strip(),
        addr_null_token=bool(NULL_RE.search(a2)) and not NULL_RE.search(a1),
        addr_case=case_style(a2) if a2.strip() else "missing",
        addr_script_state=st2[1] == "native_script",
        addr_order=component_order(c1, c2) if a2.strip() else "addr_missing",
        addr_n_comp_diff=len(c2) - len(c1) if a2.strip() else None,
        house_number=number_relation(NUM_RE.findall(a1), NUM_RE.findall(a2)) if a2.strip() else "addr_missing",
        addr_hash_added=a2.lstrip().startswith("#") and not a1.lstrip().startswith("#"),
        state_form_s1=st1[1] or "none",
        state_form_other=(st2[1] or "none") if a2.strip() else "addr_missing",
        state_changed=bool(st1[0] and st2[0] and st1[0] != st2[0]),
        state_moved_front=st2[2] == 0 and st1[2] not in (0, None),
        city_rel=city_rel,
        postcode_s1=pc1 is not None,
        postcode_other=pc2 is not None,
        postcode_rel=("both_none" if not pc1 and not pc2 else "dropped" if pc1 and not pc2
                      else "added" if pc2 and not pc1 else "same" if pc1 == pc2 else "changed"),
        landmark_s1=lm1,
        landmark_other=lm2,
        landmark_rel="none" if not (lm1 or lm2) else "both" if lm1 and lm2 else "dropped" if lm1 else "added",
        n_name_tok_s1=len(t1),
        n_addr_tok_s1=len(ta1),
    )
    for field, ev in (("name", ev_name), ("addr", ev_addr)):
        cnt = Counter(e[0] for e in ev)
        for cls in ("exact", "accent", "number_format", "number_edit", "typo", "heavy_typo", "abbreviation",
                    "expansion", "merge", "substitution", "dropped", "added"):
            feats[f"{field}_{cls}"] = cnt.get(cls, 0)
    events = [("name", *e) for e in ev_name] + [("addr", *e) for e in ev_addr]
    return feats, events


# --------------------------------------------------------------------------
# Data loading (train only)
# --------------------------------------------------------------------------
def load_samples() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    gt = load_ground_truth_pairs()
    s1_all = scan_source("train", 1).collect()
    pos = (
        gt.filter(pl.col("match_id").is_not_null())
        .join(s1_all.select(pl.col("entity_id").alias(S1), "country"), on=S1)
        .with_columns(src=pl.col("match_id").str.slice(0, 2))
    )
    parts = []
    for (country, src), g in sorted(pos.group_by("country", "src"), key=lambda x: x[0]):
        parts.append(g.sample(min(N_PAIRS_PER_COUNTRY // 2, g.height), seed=SEED))
    sample = pl.concat(parts)

    matched_ids = pos.select("match_id")
    s1_card = gt.group_by(S1).agg(pl.col("match_id").drop_nulls().len().alias("n_true"))
    del gt, pos
    gc.collect()

    need = sample["match_id"].implode()
    other_parts, distr_parts = [], []
    for s in (2, 3):
        lf = scan_source("train", s)
        other_parts.append(lf.filter(pl.col("entity_id").is_in(need)).collect())
        # distractors: records matched to no S1
        ids = lf.select("entity_id", "country").collect()
        unmatched = ids.join(matched_ids, left_on="entity_id", right_on="match_id", how="anti")
        distr_parts.append(unmatched)
        del ids
    others = pl.concat(other_parts).rename(
        {"entity_id": "match_id", "business_name": "o_name", "business_address": "o_addr", "country": "o_country"}
    )
    sample = (
        sample.join(s1_all.rename({"entity_id": S1}).drop("country"), on=S1)
        .join(others, on="match_id")
    )
    unmatched = pl.concat(distr_parts)
    dsample = pl.concat([
        g.sample(min(N_DISTRACTORS_PER_COUNTRY, g.height), seed=SEED)
        for _, g in sorted(unmatched.group_by("country"), key=lambda x: x[0])
    ])
    dneed = dsample["entity_id"].implode()
    distractors = pl.concat([
        scan_source("train", s).filter(pl.col("entity_id").is_in(dneed)).collect() for s in (2, 3)
    ]).with_columns(src=pl.col("entity_id").str.slice(0, 2))
    s1_all = s1_all.join(s1_card, left_on="entity_id", right_on=S1, how="left")
    return sample, distractors, s1_all


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------
def pick(df: pl.DataFrame, n: int, seed: int = SEED) -> pl.DataFrame:
    return df.sample(min(n, df.height), seed=seed)


def rate_table(df: pl.DataFrame, cols: list[str], by=("country", "src")) -> pl.DataFrame:
    return (
        df.group_by(*by)
        .agg(pl.len().alias("pairs"), *[pl.col(c).cast(pl.Float64).mean().alias(c) for c in cols])
        .sort(*by)
    )


def dist_table(df: pl.DataFrame, col: str, by=("country", "src")) -> pl.DataFrame:
    return (
        df.group_by(*by, col).len()
        .with_columns((pl.col("len") / pl.col("len").sum().over(*by)).alias("share"))
        .pivot(on=col, index=list(by), values="share")
        .fill_null(0.0)
        .sort(*by)
    )


def top_pairs(ev: pl.DataFrame, field: str, classes: list[str], n: int = 40) -> pl.DataFrame:
    return (
        ev.filter((pl.col("field") == field) & pl.col("cls").is_in(classes))
        .group_by("a", "b")
        .agg(pl.len().alias("count"), pl.col("country").unique().sort().str.join("+").alias("countries"),
             pl.col("src").unique().sort().str.join("+").alias("sources"))
        .sort("count", descending=True)
        .head(n)
    )


def top_tokens(ev: pl.DataFrame, field: str, cls: str, col: str, n: int = 30) -> pl.DataFrame:
    return (
        ev.filter((pl.col("field") == field) & (pl.col("cls") == cls))
        .group_by("country", col).len("count")
        .sort("count", descending=True)
        .group_by("country", maintain_order=True).head(n)
        .with_columns(pl.int_range(pl.len()).over("country").alias("rank"))
        .pivot(on="country", index="rank", values=[col, "count"])
        .sort("rank")
    )


# --------------------------------------------------------------------------
# Distractor analysis
# --------------------------------------------------------------------------
def nearest_s1(queries: pl.DataFrame, s1c: pl.DataFrame) -> pl.DataFrame:
    """Top-1 train S1 (same country) by char 3-gram TF-IDF on name + address.

    Vectoriser fitted on train S1 of that country only. Common trigrams
    (max_df) are dropped to keep the sparse products small; EDA only.
    """
    text = lambda df, n, a: (df[n].map_elements(fold, return_dtype=pl.String) + " | "  # noqa: E731
                             + df[a].map_elements(fold, return_dtype=pl.String)).to_list()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=0.02,
                          sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(text(s1c, "business_name", "business_address"))
    XT = X.T.tocsr()
    Q = vec.transform(text(queries, "business_name", "business_address"))
    best_idx = np.empty(Q.shape[0], dtype=np.int64)
    best_sim = np.empty(Q.shape[0], dtype=np.float32)
    for lo in range(0, Q.shape[0], 50):
        S = (Q[lo:lo + 50] @ XT).tocsr()
        best_idx[lo:lo + 50] = np.asarray(S.argmax(axis=1)).ravel()
        best_sim[lo:lo + 50] = S.max(axis=1).toarray().ravel()
        del S
    del X, XT, Q
    gc.collect()
    nn = s1c[best_idx.tolist()].select(
        pl.col("entity_id").alias("nn_s1"), pl.col("business_name").alias("nn_name"),
        pl.col("business_address").alias("nn_addr"), pl.col("n_true").alias("nn_n_true"))
    return pl.concat([queries, nn], how="horizontal").with_columns(nn_sim=pl.Series(best_sim))


def nn_bucket(r: dict) -> dict:
    ns = fuzz.token_set_ratio(fold(r["business_name"]), fold(r["nn_name"]))
    a = r["business_address"]
    as_ = fuzz.token_set_ratio(fold(a), fold(r["nn_addr"])) if a.strip() else None
    if as_ is None:
        b = "addr_missing"
    elif ns >= 80 and as_ >= 80:
        b = "near_copy"
    elif ns >= 80 and as_ < 60:
        b = "same_name_diff_addr"
    elif as_ >= 80 and ns < 60:
        b = "same_addr_diff_name"
    elif ns < 60 and as_ < 60:
        b = "unrelated"
    else:
        b = "partial"
    return dict(nn_name_tsr=ns, nn_addr_tsr=as_, nn_bucket=b)


# --------------------------------------------------------------------------
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()
    emit("# Noise analysis of training pairs (NonStandard T18)\n",
         f"Train data only. Seed {SEED}. {N_PAIRS_PER_COUNTRY:,} matched (S1, S2/S3) pairs per country "
         f"(half S2, half S3), {N_DISTRACTORS_PER_COUNTRY:,} distractors per country, "
         f"{N_S1_SURFACE:,} S1 records for surface stats. Generated by `notebooks/noise_analysis.py`.",
         "\nToken classes (S1 token → other-record token): exact; accent (equal after accent folding); "
         "number_format (equal after stripping leading zeros); number_edit (one digit edited); typo "
         "(OSA distance ≤1/≤2/≤3 for tokens of ≤4/≤7/8+ chars and fuzz ratio ≥60; an adjacent swap is 1 edit); "
         "abbreviation (other side is an in-order subsequence with the same first letter, ≥2 chars shorter "
         "and ≤75% of the length, e.g. road→rd); expansion (the reverse); merge "
         "(two S1 tokens glued together); substitution (the single leftover token on each side); "
         "heavy_typo (leftover pair with the same first letter and fuzz ratio ≥50, e.g. equipment→euspmeent); dropped (in S1 only); added (in the other record only). "
         "Tokens are `\\w+` after NFKC + casefold. Names in non-Latin script are excluded from token stats.")

    sample, distractors, s1_all = load_samples()
    print(f"loaded samples in {time.perf_counter() - t0:.1f}s: pairs={sample.height} "
          f"distractors={distractors.height}", file=sys.stderr)

    # ---- per-pair analysis -------------------------------------------------
    feats, events = [], []
    for r in sample.iter_rows(named=True):
        f, ev = analyse_pair(r)
        feats.append(f)
        latin_name = f["name_script"] == "Latin"
        for field, cls, a, b in ev:
            if field == "name" and not latin_name:
                continue
            events.append((field, cls, a, b, r["country"], r["src"]))
    P = pl.concat([sample.select(S1, "match_id", "country", "src", "business_name", "o_name",
                                 "business_address", "o_addr"),
                   pl.DataFrame(feats, infer_schema_length=None)], how="horizontal")
    EV = pl.DataFrame(events, schema=["field", "cls", "a", "b", "country", "src"], orient="row")
    del feats, events
    gc.collect()
    latin = P.filter(pl.col("name_script") == "Latin")

    # ---- headline similarity ---------------------------------------------
    section("1. How similar are true matches? (token_set_ratio percentiles, 0-100)")
    q = lambda c, p: pl.col(c).quantile(p).alias(f"{c}_p{int(p*100):02d}")  # noqa: E731
    emit(P.group_by("country", "src").agg(
        pl.len().alias("pairs"),
        pl.col("name_identical").mean().alias("name_identical"),
        pl.col("name_identical_casefold").mean().alias("name_ident_cf"),
        q("name_tsr", 0.05), q("name_tsr", 0.25), q("name_tsr", 0.5),
        pl.col("addr_identical").mean().alias("addr_identical"),
        q("addr_tsr", 0.05), q("addr_tsr", 0.25), q("addr_tsr", 0.5),
        (pl.col("name_tsr") < 60).mean().alias("name_tsr_lt60"),
        ((pl.col("addr_tsr") < 60) | pl.col("addr_tsr").is_null()).mean().alias("addr_tsr_lt60_or_miss"),
        ((pl.col("name_tsr") < 60) & ((pl.col("addr_tsr") < 60) | pl.col("addr_tsr").is_null()))
        .mean().alias("both_lt60"),
    ).sort("country", "src"))

    # ---- token alignment summary ----------------------------------------
    section("2. Token alignment: share of S1 tokens by class")
    for field in ("name", "addr"):
        src_df = latin if field == "name" else P.filter(~pl.col("addr_missing"))
        cls_cols = [c for c in src_df.columns if c.startswith(f"{field}_") and c.split("_", 1)[1] in
                    ("exact", "accent", "number_format", "number_edit", "typo", "heavy_typo", "abbreviation", "expansion",
                     "merge", "substitution", "dropped", "added")]
        denom = pl.col(f"n_{field}_tok_s1").sum()
        emit(f"\n**{field}** (per S1 token; `added` is per S1 token too, so it can exceed 0 without an S1 counterpart)\n")
        emit(src_df.group_by("country", "src").agg(
            [(pl.col(c).sum() / denom).alias(c.split("_", 1)[1]) for c in cls_cols]
        ).sort("country", "src"))
        emit(f"\nShare of **pairs** with at least one event of each class ({field}):\n")
        emit(src_df.group_by("country", "src").agg(
            [(pl.col(c) > 0).mean().alias(c.split("_", 1)[1]) for c in cls_cols if not c.endswith("exact")]
        ).sort("country", "src"))

    # ---- mined dictionary --------------------------------------------------
    section("3. Mined substitution dictionary (top pairs S1 token → other token)")
    for field in ("name", "addr"):
        emit(f"\n**{field}: abbreviations / expansions**\n")
        emit(top_pairs(EV, field, ["abbreviation", "expansion"], 40))
        emit(f"\n**{field}: 1:1 substitutions (unexplained leftovers)**\n")
        emit(top_pairs(EV, field, ["substitution"], 30))
    emit("\n**name: merges (two S1 tokens glued)** — top 10\n")
    emit(top_pairs(EV, "name", ["merge"], 10))
    emit("\n**Number-format changes in addresses** (top 15)\n")
    emit(top_pairs(EV, "addr", ["number_format", "number_edit"], 15))

    section("4. Tokens dropped from S1 / added in S2/S3 (top 30 per country)")
    for field in ("name", "addr"):
        emit(f"\n**{field}: dropped (present in S1, missing in match)**\n")
        emit(top_tokens(EV, field, "dropped", "a"))
        emit(f"\n**{field}: added (absent in S1, present in match)**\n")
        emit(top_tokens(EV, field, "added", "b"))

    # ---- typos -------------------------------------------------------------
    section("5. Typos: edit types and rates")
    typo_rows = EV.filter(pl.col("cls") == "typo").select("field", "a", "b", "country", "src")
    ops = []
    for field, a, b, country, src in typo_rows.iter_rows():
        for op in typo_ops(a, b):
            ops.append((field, op, country, src))
    OPS = pl.DataFrame(ops, schema=["field", "op", "country", "src"], orient="row")
    emit("Edit operations inside typo tokens (share of ops):\n")
    emit(dist_table(OPS, "op", by=("field", "country", "src")))
    emit("\nEdits per typo token:\n")
    emit(typo_rows.with_columns(
        d=pl.struct("a", "b").map_elements(lambda s: OSA.distance(fold(s["a"]), fold(s["b"])),
                                          return_dtype=pl.Int64))
         .group_by("field", "d").len().sort("field", "d"))
    emit("\nTypo-token rate (typo tokens / S1 tokens) and share of pairs with ≥1 typo:\n")
    emit(latin.group_by("country", "src").agg(
        (pl.col("name_typo").sum() / pl.col("n_name_tok_s1").sum()).alias("name_typo_per_tok"),
        (pl.col("name_typo") > 0).mean().alias("name_pairs_with_typo"),
        (pl.col("addr_typo").sum() / pl.col("n_addr_tok_s1").sum()).alias("addr_typo_per_tok"),
        (pl.col("addr_typo") > 0).mean().alias("addr_pairs_with_typo"),
    ).sort("country", "src"))

    # ---- name events -------------------------------------------------------
    section("6. Name-level events (share of pairs)")
    name_cols = ["name_accent_injected", "name_bracket_added", "name_junk_added", "name_domain",
                 "name_double_space", "name_dup_token", "name_word_order_changed", "name_unrelated",
                 "legal_s1", "legal_dropped", "legal_changed", "legal_added", "legal_moved_front",
                 "amp_s1", "amp_to_and", "amp_to_plus"]
    emit(rate_table(P, name_cols))
    emit("\nScript of the matched name:\n")
    emit(dist_table(P, "name_script"))
    emit("\nCase style of the matched name:\n")
    emit(dist_table(P, "name_case"))
    emit("\nExamples of `name_unrelated` (Latin name, token_set_ratio < 40):\n")
    emit(pick(P.filter(pl.col("name_unrelated")), 12)
         .select("country", "src", "business_name", "o_name", "business_address", "o_addr"))
    emit("\nJunk prefixes/suffixes added to names (top 15):\n")
    emit(P.filter(pl.col("name_junk_added")).select(
        pl.concat_str(pl.col("o_name").str.extract(r"""^([^\w\s"'(\[]+)""").fill_null(""), pl.lit(" … "),
                      pl.col("o_name").str.extract(r"([^\w\s.)\]]+)$").fill_null("")).alias("junk"))
        .group_by("junk").len().sort("len", descending=True).head(15))

    # ---- address events ----------------------------------------------------
    section("7. Address-level events")
    emit("Rates (share of pairs):\n")
    emit(rate_table(P, ["addr_missing", "addr_null_token", "addr_hash_added", "addr_script_state",
                        "state_changed", "state_moved_front", "postcode_s1", "postcode_other",
                        "landmark_s1", "landmark_other"]))
    for col, title in [("house_number", "First number of S1 address vs numbers in match"),
                       ("addr_order", "Component order (S1 components fuzzy-mapped into the match)"),
                       ("state_form_other", "State form in the match"),
                       ("city_rel", "S1 city (component before the state) in the match"),
                       ("postcode_rel", "Postcode relation"),
                       ("landmark_rel", "Landmark phrase relation"),
                       ("addr_case", "Address case style in the match")]:
        emit(f"\n{title}:\n")
        emit(dist_table(P, col))
    emit("\nState form in S1:\n")
    emit(dist_table(P.unique(S1), "state_form_s1", by=("country",)))
    emit("\nComponent-count difference (match − S1), non-missing addresses:\n")
    emit(dist_table(P.filter(pl.col("addr_n_comp_diff").is_not_null())
                    .with_columns(pl.col("addr_n_comp_diff").clip(-3, 3).cast(pl.String)), "addr_n_comp_diff"))
    emit("\nLandmark phrases in matched addresses (top 15 lead words):\n")
    emit(P.select("country", pl.col("o_addr").str.extract_all(LANDMARK_RE.pattern.replace(r"\b(", r"(?i)\b("))
                  .alias("lm")).explode("lm").drop_nulls()
         .with_columns(pl.col("lm").str.to_lowercase()).group_by("country", "lm").len()
         .sort("len", descending=True).head(15))

    # ---- worked examples -----------------------------------------------------
    section("8. Worked alignment examples (spot check)")
    ex = pick(P.filter(pl.col("name_script") == "Latin"), 16, SEED + 1)
    for r in ex.iter_rows(named=True):
        _, ev = analyse_pair(r)
        emit(f"- [{r['country']} {r['src']}] `{r['business_name']}` → `{r['o_name']}`  \n"
             f"  `{r['business_address']}` → `{r['o_addr']}`  \n"
             "  " + "; ".join(f"{fld}:{c}({a}→{b})" for fld, c, a, b in ev if c != "exact"))

    # ---- CSV dictionary ------------------------------------------------------
    (EV.filter(pl.col("cls").is_in(["abbreviation", "expansion", "substitution", "typo", "heavy_typo", "accent",
                                    "number_format", "number_edit", "merge"]))
     .group_by("field", "cls", "a", "b", "country").len("count")
     .filter(pl.col("count") >= 3)
     .sort("count", descending=True)
     .write_csv(OUT_CSV))

    # ---- S1 cleanliness ------------------------------------------------------
    section("9. Surface stats: S1 vs matched S2/S3 vs distractors")
    s1s = s1_all.sample(N_S1_SURFACE, seed=SEED)
    surf_s1 = pl.DataFrame([surface(n, a) for n, a in s1s.select("business_name", "business_address").iter_rows()]) \
        .with_columns(group=pl.lit("S1"), country=s1s["country"])
    ctrl = P.unique("match_id").group_by("country", "src").head(N_DISTRACTORS_PER_COUNTRY // 2)
    surf_m = pl.DataFrame([surface(n, a) for n, a in ctrl.select("o_name", "o_addr").iter_rows()]) \
        .with_columns(group=pl.lit("matched ") + ctrl["src"], country=ctrl["country"])
    surf_d = pl.DataFrame([surface(n, a) for n, a in distractors.select("business_name", "business_address").iter_rows()]) \
        .with_columns(group=pl.lit("distractor ") + distractors["src"], country=distractors["country"])
    emit(pl.concat([surf_s1, surf_m, surf_d]).group_by("country", "group").agg(SURFACE_AGG).sort("country", "group"))

    # ---- distractors vs matched controls: nearest S1 ----------------------------
    section("10. Distractors vs matched records: nearest train S1 (char 3-gram TF-IDF, same country)")
    res = []
    for country in sorted(distractors["country"].unique()):
        s1c = s1_all.filter(pl.col("country") == country)
        dq = distractors.filter(pl.col("country") == country).select(
            "entity_id", "business_name", "business_address", "country", "src",
            pl.lit("distractor").alias("kind"), pl.lit(None, pl.String).alias("true_s1"))
        cq = ctrl.filter(pl.col("country") == country).select(
            pl.col("match_id").alias("entity_id"), pl.col("o_name").alias("business_name"),
            pl.col("o_addr").alias("business_address"), "country", "src",
            pl.lit("matched").alias("kind"), pl.col(S1).alias("true_s1"))
        res.append(nearest_s1(pl.concat([dq, cq]), s1c))
        del s1c
        gc.collect()
    NN = pl.concat(res)
    NN = pl.concat([NN, pl.DataFrame([nn_bucket(r) for r in NN.iter_rows(named=True)])], how="horizontal")
    emit("Nearest-S1 similarity (TF-IDF cosine) and fuzzy name/address agreement with that S1:\n")
    emit(NN.group_by("country", "kind").agg(
        pl.len().alias("n"),
        pl.col("nn_sim").quantile(0.1).alias("cos_p10"), pl.col("nn_sim").median().alias("cos_p50"),
        pl.col("nn_sim").quantile(0.9).alias("cos_p90"),
        pl.col("nn_name_tsr").median().alias("name_tsr_p50"),
        pl.col("nn_addr_tsr").median().alias("addr_tsr_p50"),
        (pl.col("nn_s1") == pl.col("true_s1")).mean().alias("top1_is_true_s1"),
        (pl.col("nn_n_true") == 0).mean().alias("nn_is_singleton"),
    ).sort("country", "kind"))
    emit("\nBucket of the nearest S1 (name/address token_set_ratio: near_copy both ≥80; "
         "same_name_diff_addr name ≥80 & addr <60; same_addr_diff_name addr ≥80 & name <60; unrelated both <60):\n")
    emit(dist_table(NN, "nn_bucket", by=("country", "kind")))
    emit("\nDistractor examples with their nearest S1:\n")
    emit(pick(NN.filter(pl.col("kind") == "distractor"), 16)
         .select("country", "src", "nn_bucket", "business_name", "nn_name", "business_address", "nn_addr"))

    emit(f"\n_Runtime {time.perf_counter() - t0:.0f}s, peak RSS observed at end "
         f"{psutil.Process().memory_info().rss / 2**30:.2f} GiB._")
    OUT_MD.write_text("\n".join(MD) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD} and {OUT_CSV}", file=sys.stderr)


if __name__ == "__main__":
    main()
