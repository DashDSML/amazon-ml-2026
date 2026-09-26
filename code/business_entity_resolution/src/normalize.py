"""Multi-view normalisation of business names and addresses (NonStandard T14-T17).

Views per record (T14):
  * ``*_nfkc``  raw-ish: NFKC + lowercase + whitespace collapsed
  * ``*_fold``  char view: accents folded, junk / brackets / URLs / null components
                removed, dotted acronyms collapsed, native Indic script transliterated
                to Latin (mined lexicon first, ``indic_transliteration`` fallback)
  * tokens      canonical tokens: mined abbreviation table applied, numbers without
                leading zeros, duplicated tokens removed, legal forms / honorifics
                split off (names), states / street types / markers canonicalised and
                order dropped (addresses)

Every dictionary comes from ``resources/``, which ``mine_resources.py`` rebuilds from
TRAINING pairs only (NonStandard G1). The few hand-written lists below come from
general knowledge (English / French conventions); none of them was written from
test records. Country is an open set: a country without its own mined table gets
the country-independent rows (key ``*``), and never crashes or gets dropped.

All heavy lifting is vectorised polars (string expressions, explode / join /
group_by on token-long frames). Python only runs on *unique* native-script tokens
(transliteration fallback), cached across chunks.

CLI: ``python normalize.py --split train`` writes normalised parquet caches to
``dataset/cache/<split>/norm_source{1,2,3}.parquet`` chunk by chunk.
"""

from __future__ import annotations

import json
import time
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import polars as pl

SRC_DIR = Path(__file__).resolve().parent
PKG_DIR = SRC_DIR.parent
RESOURCES_DIR = PKG_DIR / "resources"
ALL = "*"  # country key for country-independent rows / unseen countries

# --------------------------------------------------------------------------
# General-knowledge lists (not mined; English and French conventions)
# --------------------------------------------------------------------------
NULL_COMPONENTS = ["null", "<null>", "n/a", "na", "none", "nil", "-", "--", "n a", "not available"]
TLDS = r"co\.in|com|net|org|biz|info|co|in|us|fr|io"
URL_RE = r"\|?\s*(?:https?://)?www\.[^\s|,]+"
DOMAIN_RE = rf"(?:^|\s)(?:www\.)?([\p{{L}}\p{{N}}-]+)\.(?:{TLDS})(?:$|\s)"
MS_RE = r"\bm\s*/\s*s\b\.?"
# "aka" is deliberately not a marker: it occurs inside real S1 names.
ALIAS_MARKER = (r"(?:\bd\.?\s?/?b\.?\s?/?a\b\.?|\bdoing business as\b|\bformerly known as\b"
                r"|\bformerly\b|\bf\.?/?k\.?/?a\b\.?|\btrading as\b|\banciennement\b)")
ALIAS_RE = rf"^(?P<pre>.*?\S)\s*{ALIAS_MARKER}\s*:?\s*(?P<post>\S.*)$"
LANDMARK_WORDS = ["near", "nr", "opp", "opposite", "behind", "beside", "next to", "adjacent to",
                  "adjacent", "adj", "in front of", "facing", "pres de", "en face de", "face a",
                  "a cote de", "derriere"]
LANDMARK_RE = r"\b(?:" + "|".join(w.replace(" ", r"\s+") for w in LANDMARK_WORDS) + r")\b.*$"
# tokens after which a number is a unit / box / floor, not a house number
UNIT_WORDS = {"unit", "apt", "apartment", "suite", "ste", "fl", "floor", "box", "pmb", "room", "rm",
              "bldg", "building", "etage", "bureau", "bp", "cedex", "cs"}
ORDINAL_WORDS = {
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th", "sixth": "6th",
    "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th", "eleventh": "11th",
    "twelfth": "12th", "thirteenth": "13th", "fourteenth": "14th", "fifteenth": "15th",
    "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th", "nineteenth": "19th",
    "twentieth": "20th",
}
FRENCH_LEGAL = {"sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "scop", "scp", "scm", "selarl",
                "gie", "ets", "cie", "societe", "ste"}
FRENCH_NAME_EQUIV = {"etablissements": "ets", "etablissement": "ets", "compagnie": "cie", "et": "and"}
ARTICLES = {"the", "le", "la", "les", "l"}
FRENCH_ADDR_EQUIV = {  # variant -> canonical (short form)
    "avenue": "ave", "av": "ave", "boulevard": "blvd", "bd": "blvd", "bld": "blvd", "place": "pl",
    "chemin": "ch", "impasse": "imp", "route": "rte", "allee": "all", "quai": "qu",
    "faubourg": "fg", "fbg": "fg", "square": "sq", "cours": "crs", "residence": "res",
    "saint": "st", "sainte": "ste", "batiment": "bat", "lieu dit": "ld", "lieudit": "ld",
}
FRENCH_STREET_TYPES = {"rue", "ave", "blvd", "pl", "ch", "imp", "rte", "all", "qu", "fg", "sq",
                       "crs", "res", "passage", "sentier", "voie", "cite", "hameau"}
FRENCH_ADDR_MARKERS = {"bis", "ter"}  # house-number suffixes, kept out of house_no

INDIC_BLOCKS = [  # (start, scheme); every Brahmic block is 0x80 wide with a parallel layout
    (0x0900, "devanagari"), (0x0980, "bengali"), (0x0A00, "gurmukhi"), (0x0A80, "gujarati"),
    (0x0B00, "oriya"), (0x0B80, "tamil"), (0x0C00, "telugu"), (0x0C80, "kannada"),
    (0x0D00, "malayalam"),
]
INDIC_RE = r"[ऀ-ൿ]"
SCRIPT_NAMES = ["Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Oriya", "Tamil", "Telugu",
                "Kannada", "Malayalam", "Arabic", "Cyrillic", "Han", "Greek", "Hebrew", "Thai"]


# --------------------------------------------------------------------------
# Resources
# --------------------------------------------------------------------------
@dataclass
class Resources:
    """Tables mined from training pairs (see mine_resources.py)."""

    name_equiv: pl.DataFrame      # ckey, variant, canonical
    addr_equiv: pl.DataFrame      # ckey, variant, canonical
    states: pl.DataFrame          # ckey, variant, state_key
    cities: pl.DataFrame          # ckey, city_key, s1_count
    city_suffixes: pl.DataFrame   # ckey, suffix
    legal: frozenset[str]
    prefixes: frozenset[str]
    addr_markers: frozenset[str]
    street_types: frozenset[str]
    lexicon: dict[str, str]
    gib_grams: pl.DataFrame       # gram, logodds
    gib_threshold: float
    known_ckeys: frozenset[str]


def _expand_by_country(df: pl.DataFrame, known: set[str], key: str = "variant") -> pl.DataFrame:
    """Rows for every known ckey = its own rows plus the ``*`` rows it does not override."""
    general = df.filter(pl.col("ckey") == ALL)
    parts = [general]
    for c in sorted(known - {ALL}):
        own = df.filter(pl.col("ckey") == c)
        parts.append(own)
        parts.append(general.join(own.select(key), on=key, how="anti").with_columns(ckey=pl.lit(c)))
    return pl.concat(parts).unique(["ckey", key], keep="first", maintain_order=True)


def _read_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema=False, encoding="utf8")


@lru_cache(maxsize=4)
def load_resources(resources_dir: Path = RESOURCES_DIR) -> Resources:
    d = Path(resources_dir)
    ne = _read_csv(d / "name_equiv.csv").select("ckey", "variant", "canonical")
    ae = _read_csv(d / "addr_equiv.csv").select("ckey", "variant", "canonical")
    st = _read_csv(d / "states.csv").select("ckey", "variant", "state_key")
    ci = _read_csv(d / "cities.csv").select("ckey", "city_key", pl.col("s1_count").cast(pl.Int64))
    cs = _read_csv(d / "city_suffixes.csv").select("ckey", "suffix")
    known = set(ne["ckey"]) | set(ae["ckey"]) | set(st["ckey"]) | set(ci["ckey"]) | {ALL}
    legal = set(_read_csv(d / "legal_forms.csv")["token"])
    prefixes = set(_read_csv(d / "name_prefixes.csv")["token"])
    markers = set(_read_csv(d / "addr_markers.csv")["token"])
    street = set(_read_csv(d / "street_types.csv")["token"])
    lex = _read_csv(d / "indic_lexicon.csv")
    gib = _read_csv(d / "gibberish_trigrams.csv").select("gram", pl.col("logodds").cast(pl.Float64))
    meta = json.loads((d / "gibberish_meta.json").read_text(encoding="utf-8"))
    return Resources(
        name_equiv=_expand_by_country(ne, known),
        addr_equiv=_expand_by_country(ae, known),
        states=_expand_by_country(st, known),
        cities=ci,
        city_suffixes=cs,
        legal=frozenset(legal),
        prefixes=frozenset(prefixes),
        addr_markers=frozenset(markers),
        street_types=frozenset(street),
        lexicon=dict(zip(lex["native"], lex["latin"])),
        gib_grams=gib,
        gib_threshold=float(meta["threshold"]),
        known_ckeys=frozenset(known),
    )


# --------------------------------------------------------------------------
# String-level building blocks (resource-free; also used by the miners)
# --------------------------------------------------------------------------
def nfkc_expr(e: pl.Expr) -> pl.Expr:
    return (e.str.normalize("NFKC").str.to_lowercase()
            .str.replace_all(r"\s+", " ").str.strip_chars())


def fold_expr(e: pl.Expr) -> pl.Expr:
    """NFKD, strip Latin combining accents (not Indic vowel signs), NFC, lowercase."""
    return (e.str.normalize("NFKD")
            .str.replace_all(r"[̀-ͯ​-‍⁠﻿]", "")
            .str.normalize("NFC").str.to_lowercase()
            .str.replace_many(["ß", "æ", "œ", "ø", "ł", "đ", "ð", "þ", "ı"],
                              ["ss", "ae", "oe", "o", "l", "d", "d", "th", "i"]))


def clean_expr(e: pl.Expr) -> pl.Expr:
    """Punctuation cleanup on folded text: keeps letters/marks/digits of any script."""
    return (e.str.replace_all(r"['’`´]", "")
            .str.replace_all(r"^[^\p{L}\p{N}]*[&+]|[&+][^\p{L}\p{N}]*$", " ")  # junk "& ..." / "... &"
            .str.replace_all(r"[&+]", " and ")
            .str.replace_all(r"\b(\p{L})\.", "$1 ")  # dotted initials -> letters, merged later
            .str.replace_all(r"[^\p{L}\p{M}\p{N}]+", " ")
            .str.strip_chars())


def merge_letters_expr(e: pl.Expr) -> pl.Expr:
    """Glue runs of single Latin letters ("l l c" -> "llc", "g i d c" -> "gidc").

    Rust regex has no lookaround, so: double the spaces, mark each single letter
    with , then repeatedly glue a marked token onto the preceding marked token.
    """
    e = pl.lit(" ") + e.str.replace_all(" ", "  ") + pl.lit(" ")
    e = e.str.replace_all(r" ([a-z]) ", " ${1} ")
    for _ in range(4):
        e = e.str.replace_all(r"  ([a-z]+)", "${1}")
    return e.str.replace_all(r"", "").str.replace_all(" +", " ").str.strip_chars()


def script_expr(e: pl.Expr) -> pl.Expr:
    """First non-Latin script found in the string, else null."""
    out = pl.lit(None, pl.String)
    for name in reversed(SCRIPT_NAMES):
        out = pl.when(e.str.contains(rf"\p{{{name}}}")).then(pl.lit(name)).otherwise(out)
    return out


def tokens_long(df: pl.DataFrame, col: str, rid: str = "rid") -> pl.DataFrame:
    """(rid, pos, tok) for a column of cleaned space-separated strings; empty rows vanish."""
    return (df.select(rid, pl.col(col).str.split(" ").alias("tok"))
            .with_columns(pos=pl.int_ranges(0, pl.col("tok").list.len(), dtype=pl.Int32))
            .explode("tok", "pos")
            .filter(pl.col("tok").is_not_null() & (pl.col("tok") != "")))


def strip_zeros_expr(e: pl.Expr) -> pl.Expr:
    return e.str.replace(r"^0+(\d)", "$1")


def translit_key_expr(e: pl.Expr) -> pl.Expr:
    """T17 transliteration-folding key (Agarwal/Aggarwal, Laxmi/Lakshmi, Srinivas/Shrinivas)."""
    e = (e.str.replace_all("x", "ks").str.replace_all("ee", "i").str.replace_all("oo", "u")
         .str.replace_many(["ph", "bh", "dh", "th", "kh", "gh", "sh", "w"],
                           ["f", "b", "d", "t", "k", "g", "s", "v"]))
    for c in "abcdefghijklmnopqrstuvwxyz":
        e = e.str.replace_all(c + c + "+", c)
    return e.str.replace(r"^(\p{L}{3,}?)a$", "$1")


# --------------------------------------------------------------------------
# Native-script transliteration
# --------------------------------------------------------------------------
_TRANSLIT_CACHE: dict[str, str] = {}


def _indic_scheme(tok: str) -> tuple[str, int] | None:
    for ch in tok:
        o = ord(ch)
        for start, scheme in INDIC_BLOCKS:
            if start <= o < start + 0x80:
                return scheme, start
    return None


def transliterate_token(tok: str) -> str:
    """Fallback Indic -> Latin via indic_transliteration (MIT), IAST then accent-folded.

    Drops the inherent final 'a' after a bare consonant (schwa deletion), so
    लिमिटेड -> "limited", not "limiteda".
    """
    hit = _TRANSLIT_CACHE.get(tok)
    if hit is not None:
        return hit
    found = _indic_scheme(tok)
    if found is None:
        return tok
    from indic_transliteration import sanscript

    scheme, start = found
    iast = sanscript.transliterate(tok, scheme, sanscript.IAST)
    latin = "".join(c for c in unicodedata.normalize("NFKD", iast) if not unicodedata.combining(c))
    latin = "".join(c for c in latin.lower() if c.isascii() and c.isalnum())
    last = ord(tok[-1]) - start
    if latin.endswith("a") and len(latin) > 2 and (0x15 <= last <= 0x39 or 0x58 <= last <= 0x5F):
        latin = latin[:-1]
    _TRANSLIT_CACHE[tok] = latin or tok
    return _TRANSLIT_CACHE[tok]


def transliterate_long(long: pl.DataFrame, lexicon: dict[str, str] | None) -> pl.DataFrame:
    """Replace native Indic tokens: mined lexicon first, library fallback second."""
    is_native = pl.col("tok").str.contains(INDIC_RE)
    native = long.filter(is_native).select("tok").unique()
    if native.height == 0:
        return long.with_columns(native=pl.lit(False))
    lex = lexicon or {}
    toks = native["tok"].to_list()
    mapped = pl.DataFrame({"tok": toks, "_lat": [lex.get(t) or transliterate_token(t) for t in toks]},
                          schema={"tok": pl.String, "_lat": pl.String})
    return (long.join(mapped, on="tok", how="left")
            .with_columns(native=pl.col("_lat").is_not_null(),
                          tok=pl.coalesce("_lat", "tok"))
            .drop("_lat")
            .filter(pl.col("tok") != ""))


def _map_tokens(long: pl.DataFrame, table: pl.DataFrame, col: str = "tok") -> pl.DataFrame:
    """Country-aware token replacement: join on (ckey, variant)."""
    return (long.join(table, left_on=["ckey", col], right_on=["ckey", "variant"], how="left")
            .with_columns(pl.coalesce("canonical", col).alias(col)).drop("canonical"))


def ckey_expr(country: pl.Expr, known: frozenset[str]) -> pl.Expr:
    c = fold_expr(country).str.strip_chars()
    return pl.when(c.is_in(list(known))).then(c).otherwise(pl.lit(ALL))


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------
def name_prepare(df: pl.DataFrame, col: str = "business_name") -> pl.DataFrame:
    """String-level name views (resource-free). Needs a ``rid`` column."""
    f = fold_expr(pl.col(col))
    return (df.with_columns(
                name_nfkc=nfkc_expr(pl.col(col)),
                name_script=script_expr(pl.col(col)),
                _f=f)
            .with_columns(name_has_url=pl.col("_f").str.contains(r"www\.|https?://"),
                          _f=pl.col("_f").str.replace_all(URL_RE, " "))
            .with_columns(name_is_domain=pl.col("_f").str.contains(DOMAIN_RE),
                          _f=pl.col("_f").str.replace_all(DOMAIN_RE, " $1 ")
                          .str.replace_all(MS_RE, " ms "))
            .with_columns(_g=pl.col("_f").str.extract_groups(ALIAS_RE))
            .with_columns(name_has_alias=pl.col("_g").struct.field("post").is_not_null(),
                          name_alias=clean_expr(pl.col("_g").struct.field("pre")),
                          _main=merge_letters_expr(clean_expr(
                              pl.coalesce(pl.col("_g").struct.field("post"), pl.col("_f")))))
            .drop("_f", "_g"))


def normalize_names(df: pl.DataFrame, res: Resources, col: str = "business_name",
                    country_col: str = "country") -> pl.DataFrame:
    """Adds name_* columns (row-aligned with ``df``)."""
    base = df.with_row_index("rid").with_columns(ckey=ckey_expr(pl.col(country_col), res.known_ckeys))
    base = name_prepare(base, col)
    long = tokens_long(base, "_main")
    long = transliterate_long(long, res.lexicon)
    long = long.join(base.select("rid", "ckey"), on="rid")

    fold_view = long.sort("rid", "pos").group_by("rid").agg(pl.col("tok").str.join(" ").alias("name_fold"))

    long = _map_tokens(long.with_columns(tok=strip_zeros_expr(pl.col("tok"))), res.name_equiv).sort("rid", "pos")
    same_rid = pl.col("rid") == pl.col("rid").shift(1)
    long = long.filter(~(same_rid & (pl.col("tok") == pl.col("tok").shift(1))).fill_null(False))  # "allied allied"
    legal, pref = list(res.legal), list(res.prefixes | ARTICLES)
    is_legal = pl.col("tok").is_in(legal)
    lead_ok = is_legal | pl.col("tok").is_in(pref) | (pl.col("tok") == "and")
    trail_ok = is_legal | (pl.col("tok") == "and")
    # a token is in the leading (trailing) run if no non-strippable token of the same
    # record precedes (follows) it: global cumulative sums minus the record's offset
    long = long.with_columns(_nl=(~lead_ok).cast(pl.Int32), _nt=(~trail_ok).cast(pl.Int32))
    long = long.with_columns(_cl=pl.col("_nl").cum_sum(), _ct=pl.col("_nt").cum_sum(reverse=True))
    long = long.with_columns(
        _lead=(pl.col("_cl") - pl.col("_cl").first().over("rid") + pl.col("_nl").first().over("rid")) == 0,
        _trail=(pl.col("_ct") - pl.col("_ct").last().over("rid") + pl.col("_nt").last().over("rid")) == 0,
    ).with_columns(_edge=pl.col("_lead") | pl.col("_trail"))
    long = long.with_columns(
        _core=~pl.col("_edge"),
        _legal=pl.col("_edge") & is_legal,
    )
    # gibberish score per unique Latin token (min over core tokens of len >= 4)
    utok = long.filter(pl.col("_core") & pl.col("tok").str.contains(r"^[a-z]{4,}$")).select("tok").unique()
    gib = gibberish_scores(utok["tok"], res)
    long = long.join(gib, on="tok", how="left")

    agg = long.group_by("rid").agg(
        pl.col("tok").alias("name_tokens"),
        pl.col("tok").filter(pl.col("_core")).alias("_core_toks"),
        pl.col("tok").filter(~pl.col("tok").is_in(pref) & (pl.col("tok") != "and")).alias("_nopref"),
        pl.col("tok").filter(pl.col("_legal")).unique().sort().str.join(" ").alias("name_legal"),
        pl.col("_gib").min().alias("name_gibberish"),
        pl.col("native").any().alias("_native"),
    ).with_columns(
        _core_toks=pl.when(pl.col("_core_toks").list.len() > 0).then("_core_toks")
        .otherwise("_nopref"))
    ukey = agg.select(pl.col("_core_toks").alias("tok")).explode("tok").drop_nulls().unique()
    ukey = ukey.with_columns(key=translit_key_expr(pl.col("tok")))
    tkey = (agg.select("rid", pl.col("_core_toks").alias("tok"))
            .with_columns(pos=pl.int_ranges(0, pl.col("tok").list.len()))
            .explode("tok", "pos").drop_nulls()
            .join(ukey, on="tok", how="left").sort("rid", "pos")
            .group_by("rid").agg(pl.col("key").str.join(" ").alias("name_tkey")))

    out = (base.select("rid", "name_nfkc", "name_script", "name_has_url", "name_is_domain",
                       "name_has_alias", "name_alias")
           .join(fold_view, on="rid", how="left")
           .join(agg, on="rid", how="left")
           .join(tkey, on="rid", how="left")
           .sort("rid")
           .with_columns(
               name_fold=pl.col("name_fold").fill_null(""),
               name_tokens=pl.col("name_tokens").fill_null([]),
               name_core=pl.col("_core_toks").list.join(" ").fill_null(""),
               name_compact=pl.col("_core_toks").list.join("").fill_null(""),
               name_legal=pl.col("name_legal").fill_null(""),
               name_tkey=pl.col("name_tkey").fill_null(""),
               name_alias=pl.when(pl.col("name_has_alias")).then("name_alias"),
               name_is_gibberish=(pl.col("name_gibberish") > res.gib_threshold).fill_null(False),
               name_translit=pl.col("_native").fill_null(False),
           )
           .drop("rid", "_core_toks", "_nopref", "_native"))
    return pl.concat([df, out], how="horizontal")


def gibberish_scores(tokens: pl.Series, res: Resources) -> pl.DataFrame:
    """Mean char-trigram log-odds (generator gibberish vs S1 names) per token."""
    t = pl.DataFrame({"tok": tokens}).with_columns(_p=pl.lit("^") + pl.col("tok") + pl.lit("$"))
    grams = (t.with_columns(_i=pl.int_ranges(0, pl.col("_p").str.len_chars() - 2))
             .explode("_i")
             .with_columns(gram=pl.col("_p").str.slice(pl.col("_i"), 3))
             .join(res.gib_grams, on="gram", how="left"))
    return grams.group_by("tok").agg(pl.col("logodds").fill_null(0.0).mean().alias("_gib"))


# --------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------
def components_long(df: pl.DataFrame, col: str = "business_address", rid: str = "rid") -> pl.DataFrame:
    """(rid, cpos, comp) with comp folded + cleaned; null placeholders dropped (resource-free)."""
    return (df.select(rid, fold_expr(pl.col(col)).str.split(",").alias("comp"))
            .explode("comp")
            .with_columns(pl.col("comp").str.replace_all(r"\s+", " ").str.strip_chars(" .;:-_"))
            .filter(pl.col("comp").is_not_null() & (pl.col("comp") != "")
                    & ~pl.col("comp").str.replace_all(r"[<>()\[\]]", "").str.strip_chars()
                    .is_in(NULL_COMPONENTS))
            .with_columns(
                landmark=pl.col("comp").str.extract(rf"({LANDMARK_RE})", 1),
                comp=merge_letters_expr(clean_expr(pl.col("comp"))),
            )
            .with_columns(landmark=clean_expr(pl.col("landmark")))
            .filter(pl.col("comp") != "")
            .with_columns(cpos=(pl.col(rid).cum_count().over(rid) - 1).cast(pl.Int32)))


def normalize_addresses(df: pl.DataFrame, res: Resources, col: str = "business_address",
                        country_col: str = "country") -> pl.DataFrame:
    """Adds addr_* columns (row-aligned with ``df``)."""
    base = df.with_row_index("rid").with_columns(
        ckey=ckey_expr(pl.col(country_col), res.known_ckeys),
        addr_nfkc=nfkc_expr(pl.col(col)))
    comps = components_long(base, col).join(base.select("rid", "ckey"), on="rid")

    # --- state: whole component equal to a mined variant (code / full / native / alias)
    comps = comps.join(res.states, left_on=["ckey", "comp"], right_on=["ckey", "variant"], how="left")
    first_state = comps.filter(pl.col("state_key").is_not_null()).group_by("rid").agg(
        pl.col("state_key").sort_by("cpos").first().alias("addr_state"),
        pl.col("cpos").sort_by("cpos").first().alias("_state_cpos"))
    comps = comps.join(first_state, on="rid", how="left").with_columns(
        is_state=(pl.col("cpos") == pl.col("_state_cpos")).fill_null(False))

    # --- tokens (state component replaced by its key), transliterated and canonicalised.
    # rc = one integer key per (record, component); rows stay sorted by (rc, pos).
    comps = comps.with_columns(rc=pl.col("rid").cast(pl.UInt64) * 1024 + pl.col("cpos"))
    toks = (comps.select("rid", "rc", "ckey",
                         pl.when(pl.col("is_state")).then("state_key").otherwise("comp")
                         .str.split(" ").alias("tok"))
            .with_columns(pos=pl.int_ranges(0, pl.col("tok").list.len(), dtype=pl.Int32))
            .explode("tok", "pos")
            .filter(pl.col("tok") != ""))
    toks = transliterate_long(toks, res.lexicon)
    toks = (toks.with_columns(tok=strip_zeros_expr(pl.col("tok")).replace(ORDINAL_WORDS))
            .pipe(_map_tokens, res.addr_equiv)
            .sort("rc", "pos"))
    is_num = pl.col("tok").str.contains(r"\d") & ~pl.col("tok").str.contains(r"^\d+(?:st|nd|rd|th)$")
    after_unit = (pl.col("rc") == pl.col("rc").shift(1)) & pl.col("tok").shift(1).is_in(list(UNIT_WORDS))
    toks = toks.with_columns(
        num=pl.when(is_num).then(strip_zeros_expr(pl.col("tok").str.extract(r"(\d+)", 1))),
        _unit_num=after_unit.fill_null(False),
        _marker=pl.col("tok").is_in(list(res.addr_markers | FRENCH_ADDR_MARKERS)),
    )

    # component text rebuilt (for landmark removal, city lookup and the fold view);
    # group_by keeps the row order inside each group
    comp_txt = (toks.group_by("rc")
                .agg(pl.col("tok").str.join(" ").alias("ctext"),
                     pl.col("num").drop_nulls().len().alias("_nnum"),
                     pl.col("tok").is_in(list(res.street_types)).any().alias("_street"))
                .join(comps.select("rc", "rid", "cpos", "ckey", "is_state", "landmark"), on="rc"))
    comp_txt = comp_txt.with_columns(
        ctext_nolm=pl.when(pl.col("landmark").is_not_null())
        .then(pl.col("ctext").str.replace(LANDMARK_RE, "").str.strip_chars())
        .otherwise("ctext"))

    # --- city: mined vocabulary (suffix-stripped, de-spaced), else a generic fallback
    ck = comp_txt.filter(~pl.col("is_state") & (pl.col("_nnum") == 0)).select(
        "rid", "cpos", "ckey", "ctext", "landmark", "_street")
    ck = ck.join(res.city_suffixes, on="ckey", how="left").with_columns(
        _stripped=pl.when(pl.col("suffix").is_not_null()
                          & pl.col("ctext").str.ends_with(pl.lit(" ") + pl.col("suffix")))
        .then(pl.col("ctext").str.slice(0, pl.col("ctext").str.len_chars().cast(pl.Int64)
                                        - pl.col("suffix").str.len_chars().cast(pl.Int64) - 1))
    ).group_by("rid", "cpos", "ckey", "ctext", "landmark", "_street").agg(
        pl.coalesce(pl.col("_stripped").drop_nulls().first(), pl.col("ctext").first()).alias("_base"))
    ck = ck.with_columns(city_key=pl.col("_base").str.replace_all(" ", ""))
    vocab_hit = (ck.join(res.cities, on=["ckey", "city_key"], how="inner")
                 .sort("s1_count", descending=True).group_by("rid").agg(
                     pl.col("city_key").first().alias("_city_vocab")))
    fallback = (ck.filter(pl.col("landmark").is_null() & ~pl.col("_street")
                          & (pl.col("ctext").str.count_matches(" ") <= 2))
                .sort("rid", "cpos").group_by("rid").agg(pl.col("city_key").last().alias("_city_fb")))

    agg = toks.group_by("rid").agg(
        pl.col("tok").filter(~pl.col("_marker")).unique().sort().alias("addr_tokens"),
        pl.col("num").filter(pl.col("num").is_not_null()).alias("addr_numbers"),
        pl.col("num").filter(pl.col("num").is_not_null() & ~pl.col("_unit_num")).first().alias("addr_house_no"),
        pl.col("native").any().alias("addr_translit"),
    )
    views = comp_txt.sort("rid", "cpos").group_by("rid").agg(
        pl.col("ctext").str.join(", ").alias("addr_fold"),
        pl.col("ctext_nolm").filter(pl.col("ctext_nolm") != "").str.join(", ").alias("addr_nolm"),
        pl.col("landmark").drop_nulls().str.join(" | ").alias("addr_landmark"),
        pl.len().alias("addr_n_comp"),
    )
    out = (base.select("rid", "addr_nfkc")
           .join(views, on="rid", how="left")
           .join(first_state.select("rid", "addr_state"), on="rid", how="left")
           .join(vocab_hit, on="rid", how="left")
           .join(fallback, on="rid", how="left")
           .join(agg, on="rid", how="left")
           .sort("rid")
           .with_columns(
               addr_missing=pl.col("addr_n_comp").is_null(),
               addr_fold=pl.col("addr_fold").fill_null(""),
               addr_nolm=pl.col("addr_nolm").fill_null(""),
               addr_landmark=pl.when(pl.col("addr_landmark") != "").then("addr_landmark"),
               addr_city=pl.coalesce("_city_vocab", "_city_fb"),
               addr_tokens=pl.col("addr_tokens").fill_null([]),
               addr_numbers=pl.col("addr_numbers").fill_null([]),
               addr_n_comp=pl.col("addr_n_comp").fill_null(0).cast(pl.Int32),
               addr_translit=pl.col("addr_translit").fill_null(False),
           )
           .drop("rid", "_city_vocab", "_city_fb"))
    return pl.concat([df, out], how="horizontal")


def normalize_records(df: pl.DataFrame, res: Resources | None = None) -> pl.DataFrame:
    """Name and address views for a frame with business_name, business_address, country."""
    res = res or load_resources()
    return normalize_addresses(normalize_names(df, res), res)


# --------------------------------------------------------------------------
# Parquet cache
# --------------------------------------------------------------------------
def norm_cache_path(split: str, source: int) -> Path:
    from data import DATASET_DIR
    return DATASET_DIR / "cache" / split / f"norm_source{source}.parquet"


def _inputs_mtime() -> float:
    files = [Path(__file__), *RESOURCES_DIR.glob("*")]
    return max(f.stat().st_mtime for f in files)


def build_norm_cache(split: str, source: int, chunk_rows: int = 400_000, force: bool = False) -> Path:
    """Normalise one source file chunk by chunk into a parquet cache (memory-bounded)."""
    import pyarrow.parquet as pq

    from data import build_cache, source_path, SOURCE_COLUMNS

    out = norm_cache_path(split, source)
    raw = build_cache(source_path(split, source), SOURCE_COLUMNS)
    if not force and out.exists() and out.stat().st_mtime >= max(_inputs_mtime(), raw.stat().st_mtime):
        return out
    res = load_resources()
    n = pl.scan_parquet(raw).select(pl.len()).collect().item()
    tmp = out.with_suffix(".parquet.tmp")
    writer = None
    t0 = time.perf_counter()
    try:
        for off in range(0, n, chunk_rows):
            chunk = pl.scan_parquet(raw).slice(off, chunk_rows).collect()
            normed = normalize_records(chunk, res).to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(tmp, normed.schema, compression="zstd")
            writer.write_table(normed)
            print(f"  {split} S{source}: {min(off + chunk_rows, n):,}/{n:,} rows "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(out)
    return out


def scan_normalized(split: str, source: int) -> pl.LazyFrame:
    return pl.scan_parquet(build_norm_cache(split, source))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--sources", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for s in args.sources:
        print(build_norm_cache(args.split, s, force=args.force))
