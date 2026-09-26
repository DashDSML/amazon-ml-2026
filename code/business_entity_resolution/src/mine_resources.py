"""Mine normalisation dictionaries from TRAINING pairs only (NonStandard T18, G1).

Writes to ``code/business_entity_resolution/resources/``:

  states.csv            country key, variant (code / full name / native script / alias) -> state key
  cities.csv            place vocabulary per country (non-state S1 address components)
  city_suffixes.csv     words the noise adds after a city ("township", "twp", "cdp" ...)
  addr_equiv.csv        address abbreviation table (street -> st, road -> rd ...) + ordinals
  street_types.csv      canonical street-type tokens (for the city fallback)
  addr_markers.csv      house-number markers the noise adds (h, no, door, plot ...)
  name_equiv.csv        name abbreviation table (limited -> ltd, private -> pvt ...)
  legal_forms.csv       legal-form tokens (mined + French legal forms from general knowledge)
  name_prefixes.csv     honorifics / leading words the noise adds (ms, sri, shri, smt, dr ...)
  indic_lexicon.csv     native-script token -> Latin token (positional alignment of names)
  gibberish_trigrams.csv, gibberish_meta.json   char-trigram log-odds of generator gibberish
  mining_summary.md     what was mined, with counts

Country keys are ``fold(country)``; rows seen in every training country get key ``*``
so unseen countries (France) only receive country-independent rules plus the
general-knowledge French rows. No test file is opened here.

Run:  python code/business_entity_resolution/src/mine_resources.py
"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_ground_truth_pairs, scan_source  # noqa: E402
from normalize import (  # noqa: E402
    ALL, ARTICLES, FRENCH_ADDR_EQUIV, FRENCH_LEGAL, FRENCH_NAME_EQUIV, FRENCH_STREET_TYPES,
    INDIC_RE, ORDINAL_WORDS, RESOURCES_DIR, components_long, fold_expr, merge_letter_runs,
    name_prepare, strip_zeros_expr, tokens_long,
)

SEED = 42
N_PAIRS_PER_COUNTRY = 250_000
S1 = "source1_entity_id"
LOG: list[str] = []
_T0 = time.perf_counter()


def log(*parts: object) -> None:
    text = " ".join(str(p) for p in parts)
    LOG.append(text)
    print(f"[{time.perf_counter() - _T0:6.0f}s] {text}", flush=True)


def ckey(country: str) -> str:
    return pl.select(fold_expr(pl.lit(country))).item().strip()


# --------------------------------------------------------------------------
# Loading (train only)
# --------------------------------------------------------------------------
def load_pairs() -> tuple[pl.DataFrame, pl.DataFrame]:
    """(S1 train with ckey, sample of matched pairs with both sides' fields)."""
    s1 = scan_source("train", 1).collect()
    s1 = s1.with_columns(ckey=fold_expr(pl.col("country")).str.strip_chars())
    gt = load_ground_truth_pairs().filter(pl.col("match_id").is_not_null())
    pos = (gt.join(s1.select(pl.col("entity_id").alias(S1), "ckey"), on=S1)
           .with_columns(src=pl.col("match_id").str.slice(0, 2)))
    del gt
    parts = []
    for (_, _), g in sorted(pos.group_by("ckey", "src"), key=lambda x: x[0]):
        parts.append(g.sample(min(N_PAIRS_PER_COUNTRY // 2, g.height), seed=SEED))
    sample = pl.concat(parts)
    del pos
    need = sample["match_id"].implode()
    others = pl.concat([scan_source("train", s).filter(pl.col("entity_id").is_in(need)).collect()
                        for s in (2, 3)])
    pairs = (sample
             .join(s1.select(pl.col("entity_id").alias(S1), pl.col("business_name").alias("s1_name"),
                             pl.col("business_address").alias("s1_addr")), on=S1)
             .join(others.select(pl.col("entity_id").alias("match_id"), pl.col("business_name").alias("o_name"),
                                 pl.col("business_address").alias("o_addr")), on="match_id")
             .sort(S1, "match_id")
             .with_row_index("pid"))
    gc.collect()
    return s1, pairs


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def name_long(df: pl.DataFrame, col: str, rid: str) -> pl.DataFrame:
    """Base name tokens (no dictionaries): (rid, pos, tok)."""
    prep = name_prepare(df.select(pl.col(rid).alias("rid"), col), col)
    long = merge_letter_runs(tokens_long(prep, "_main"))
    return long.with_columns(tok=strip_zeros_expr(pl.col("tok"))).rename({"rid": rid})


def comps(df: pl.DataFrame, col: str, rid: str) -> pl.DataFrame:
    return components_long(df.select(pl.col(rid).alias("rid"), col), col).rename({"rid": rid})


def is_abbrev(short: str, long: str) -> bool:
    """short is an in-order subsequence of long, same first letter, clearly shorter."""
    if len(short) < 2 or short[0] != long[0] or len(short) > 0.75 * len(long) or len(long) - len(short) < 2:
        return False
    it = iter(long)
    return all(c in it for c in short)


def leftovers(a: pl.DataFrame, b: pl.DataFrame, key: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Tokens of a not in b (per key) and vice versa."""
    a_u = a.select(key, "ckey", "tok").unique()
    b_u = b.select(key, "ckey", "tok").unique()
    return (a_u.join(b_u.select(key, "tok"), on=[key, "tok"], how="anti"),
            b_u.join(a_u.select(key, "tok"), on=[key, "tok"], how="anti"))


def mine_abbreviations(la: pl.DataFrame, lb: pl.DataFrame, key: str, n_pairs: int,
                       min_rate: float, min_share: float) -> pl.DataFrame:
    """Abbreviation / expansion pairs among leftover tokens -> (ckey, long, short, count, share)."""
    x = (la.rename({"tok": "a"}).join(lb.rename({"tok": "b"}).drop("ckey"), on=key)
         .filter(pl.col("a").str.contains(r"^[a-z]+$") & pl.col("b").str.contains(r"^[a-z]+$")
                 & (pl.col("a").str.slice(0, 1) == pl.col("b").str.slice(0, 1))
                 & (pl.col("a") != pl.col("b")))
         .with_columns(long=pl.when(pl.col("a").str.len_chars() >= pl.col("b").str.len_chars())
                       .then("a").otherwise("b"),
                       short=pl.when(pl.col("a").str.len_chars() >= pl.col("b").str.len_chars())
                       .then("b").otherwise("a")))
    cnt = x.group_by("ckey", "long", "short").len("count")
    cnt = cnt.filter(pl.col("count") >= max(10, min_rate * n_pairs))
    cnt = cnt.filter(pl.struct("short", "long").map_elements(
        lambda s: is_abbrev(s["short"], s["long"]), return_dtype=pl.Boolean))
    # share: how much of the rarer leftover this pair explains (short forms such as
    # "inc" are also dropped/added on their own, so their total alone is too noisy)
    tot = pl.concat([la, lb]).group_by("ckey", "tok").len("total")
    cnt = (cnt.join(tot.rename({"tok": "short", "total": "short_total"}), on=["ckey", "short"], how="left")
           .join(tot.rename({"tok": "long", "total": "long_total"}), on=["ckey", "long"], how="left")
           .with_columns(share=pl.col("count") / pl.min_horizontal("short_total", "long_total")))
    cnt = cnt.filter(pl.col("share") >= min_share)
    # one canonical per variant: the strongest pair wins
    return cnt.sort("count", descending=True).unique(["ckey", "long"], keep="first")


def generalise(df: pl.DataFrame, countries: list[str], keys: list[str]) -> pl.DataFrame:
    """Rows present in every training country become ``*`` rows."""
    n = len(countries)
    everywhere = df.group_by(keys).agg(pl.col("ckey").n_unique().alias("_n")).filter(pl.col("_n") == n)
    gen = df.join(everywhere.select(keys), on=keys).group_by(keys).agg(
        pl.col("count").sum()).with_columns(ckey=pl.lit(ALL))
    rest = df.join(everywhere.select(keys), on=keys, how="anti")
    return pl.concat([gen.select(rest.columns), rest]).with_columns(pl.col("count").cast(pl.Int64)).sort("ckey", "count", descending=[False, True])


# --------------------------------------------------------------------------
# States and cities
# --------------------------------------------------------------------------
def mine_states(s1c: pl.DataFrame, pairs: pl.DataFrame, oc: pl.DataFrame, countries: list[str]):
    """Greedy set cover of S1 records by short digit-free components = the state list.

    S1 always carries a state; the component that covers the most not-yet-covered
    records is the next state. When a city covers the same records as its state
    (every GA record is in Atlanta, every DC record in Washington) the tie goes to the
    component that is most often absent from the matched record: the noise rewrites
    the state form (TX -> Texas) far more often than it drops a city.
    Variants are other-side components that replace an absent S1 state often (>=3%
    of its absences) and point to it specifically (>=90%).
    """
    picks_all = {}
    s1_pair = pairs.select("pid", S1).join(s1c.select(pl.col("rid").alias(S1), "comp"), on=S1).select("pid", "comp").unique()
    absent_rate = (s1_pair.join(oc.select("pid", "comp").unique().with_columns(_hit=pl.lit(True)),
                                on=["pid", "comp"], how="left")
                   .group_by("comp").agg(pl.col("_hit").is_null().mean().alias("absent_rate")))
    cand_all = s1c.filter(pl.col("comp").str.contains(r"^[a-z]+(?: [a-z]+){0,2}$")).select(
        "rid", "ckey", "comp").unique()
    for c in countries:
        cand = cand_all.filter(pl.col("ckey") == c)
        n_total = s1c.filter(pl.col("ckey") == c)["rid"].n_unique()
        remaining, picks, covered = cand, [], 0
        while True:
            top = remaining.group_by("comp").len().sort(["len", "comp"], descending=[True, False])
            if top.height == 0:
                break
            top = (top.filter(pl.col("len") >= 0.95 * top["len"].max()).join(absent_rate, on="comp", how="left")
                   .sort(["absent_rate", "len", "comp"], descending=[True, True, False], nulls_last=True).head(1))
            if top.height == 0 or top["len"][0] < max(30, 0.0003 * n_total):
                break
            comp, k = top["comp"][0], top["len"][0]
            picks.append((comp, k))
            covered += k
            hit = remaining.filter(pl.col("comp") == comp)["rid"].implode()
            remaining = remaining.filter(~pl.col("rid").is_in(hit))
        picks_all[c] = [p for p, _ in picks]
        log(f"states {c}: {len(picks)} picked, cover {covered / n_total:.4f} of {n_total:,} S1; "
            f"smallest {picks[-1] if picks else None}")
    states = pl.DataFrame([(c, s) for c, ps in picks_all.items() for s in ps],
                          schema={"ckey": pl.String, "state": pl.String}, orient="row")

    # S1 state of each pair (first component that is a pick)
    p_state = (s1c.join(states.rename({"state": "comp"}), on=["ckey", "comp"])
               .sort("rid", "cpos").group_by("rid").agg(pl.col("comp").first().alias("state"),
                                                        pl.col("ckey").first()))
    ps = pairs.select("pid", S1).join(p_state.rename({"rid": S1}), on=S1)
    ocs = oc.select("pid", "comp").unique()
    s1set = pairs.select("pid", S1).join(s1c.select(pl.col("rid").alias(S1), "comp"), on=S1).select("pid", "comp")
    present = ps.join(ocs, left_on=["pid", "state"], right_on=["pid", "comp"], how="semi")
    absent = ps.join(present.select("pid"), on="pid", how="anti")
    v = (absent.join(ocs, on="pid").join(s1set, on=["pid", "comp"], how="anti")
         .filter(pl.col("comp").str.count_matches(" ") <= 3))
    sub = v.group_by("ckey", "state", "comp").len("count")
    n_abs = absent.group_by("ckey", "state").len("n_absent")
    tot = (ps.join(ocs, on="pid").join(s1set, on=["pid", "comp"], how="anti")
           .group_by("ckey", "comp").len("comp_total"))
    sub = (sub.join(n_abs, on=["ckey", "state"]).join(tot, on=["ckey", "comp"])
           .with_columns(rate=pl.col("count") / pl.col("n_absent"),
                         spec=pl.col("count") / pl.col("comp_total")))
    var = sub.filter((pl.col("rate") >= 0.03) & (pl.col("spec") >= 0.9) & (pl.col("count") >= 10))
    var = var.join(states.rename({"state": "comp"}), on=["ckey", "comp"], how="anti")  # not another state
    kind = (pl.when(pl.col("variant").str.contains(INDIC_RE)).then(pl.lit("native"))
            .when(pl.col("variant").str.len_chars() <= 3).then(pl.lit("code"))
            .otherwise(pl.lit("name")))
    own = states.join(p_state.group_by("ckey", pl.col("state")).len("count"), on=["ckey", "state"])
    out = pl.concat([
        own.select("ckey", pl.col("state").alias("variant"), pl.col("state").alias("state_key"),
                   pl.lit("s1_form").alias("kind"), "count", pl.lit(1.0).alias("rate"), pl.lit(1.0).alias("spec")),
        var.rename({"comp": "variant", "state": "state_key"}).with_columns(kind=kind)
        .select("ckey", "variant", "state_key", "kind", "count", "rate", "spec"),
    ]).sort("ckey", "state_key", "count", descending=[False, False, True])
    out = out.unique(["ckey", "variant"], keep="first", maintain_order=True)
    for c in countries:
        k = out.filter(pl.col("ckey") == c)
        log(f"state variants {c}: " + ", ".join(f"{kd}={n}" for kd, n in
                                                 k.group_by("kind").len().sort("kind").iter_rows()))
    return out, states


def mine_cities(s1c: pl.DataFrame, states: pl.DataFrame, street_variants: set[str],
                pairs: pl.DataFrame, oc: pl.DataFrame):
    cand = (s1c.filter(pl.col("comp").str.contains(r"^[a-z]+(?: [a-z]+){0,3}$") & pl.col("landmark").is_null())
            .join(states.rename({"state": "comp"}), on=["ckey", "comp"], how="anti")
            .filter(~pl.col("comp").str.extract(r"(\S+)$", 1).is_in(list(street_variants))))
    cities = (cand.select("rid", "ckey", "comp").unique().group_by("ckey", "comp").len("s1_count")
              .filter(pl.col("s1_count") >= 20)
              .with_columns(city_key=pl.col("comp").str.replace_all(" ", "")))
    # suffixes added after the S1 city on the other side ("houston township")
    s1city = (pairs.select("pid", S1).join(s1c.select(pl.col("rid").alias(S1), "comp", "ckey"), on=S1)
              .join(cities.select("ckey", "comp"), on=["ckey", "comp"]))
    oc_u = oc.select("pid", pl.col("comp").alias("ocomp")).unique()
    suf = (s1city.join(oc_u, on="pid")
           .filter(pl.col("ocomp").str.starts_with(pl.col("comp") + pl.lit(" ")))
           .with_columns(suffix=pl.col("ocomp").str.slice(pl.col("comp").str.len_chars() + 1))
           .filter(pl.col("suffix").str.contains(r"^[a-z]+$"))
           .group_by("ckey", "suffix").len("count"))
    n_by_c = pairs.group_by("ckey").len("n")
    suf = (suf.join(n_by_c, on="ckey").filter(pl.col("count") >= 0.001 * pl.col("n"))
           .drop("n").sort("ckey", "count", descending=[False, True]))
    agg = (cities.sort("s1_count", descending=True)
           .group_by("ckey", "city_key").agg(pl.col("s1_count").sum(), pl.col("comp").first().alias("example")))
    return agg.sort("ckey", "s1_count", descending=[False, True]), suf


# --------------------------------------------------------------------------
# Gibberish model
# --------------------------------------------------------------------------
def trigram_counts(tokens: pl.Series) -> pl.DataFrame:
    t = pl.DataFrame({"tok": tokens}).with_columns(_p=pl.lit("^") + pl.col("tok") + pl.lit("$"))
    return (t.with_columns(_i=pl.int_ranges(0, pl.col("_p").str.len_chars() - 2)).explode("_i")
            .select(gram=pl.col("_p").str.slice(pl.col("_i"), 3)).group_by("gram").len("n"))


def score_tokens(tokens: pl.DataFrame, grams: pl.DataFrame) -> pl.DataFrame:
    t = tokens.select("tok").unique().with_columns(_p=pl.lit("^") + pl.col("tok") + pl.lit("$"))
    return (t.with_columns(_i=pl.int_ranges(0, pl.col("_p").str.len_chars() - 2)).explode("_i")
            .with_columns(gram=pl.col("_p").str.slice(pl.col("_i"), 3))
            .join(grams, on="gram", how="left")
            .group_by("tok").agg(pl.col("logodds").fill_null(0.0).mean().alias("score")))


def mine_gibberish(s1_long: pl.DataFrame, pairs: pl.DataFrame, o_long: pl.DataFrame,
                   s1_ignore: set[str]):
    """Char-trigram log-odds: generator gibberish vs S1 name tokens.

    Positives: tokens of the pre-marker part of "X dba <real name>" (the post part
    matches S1 in 99.98% of cases, so the pre part is the generator's made-up alias).
    Names sharing no token with their S1 are only used for evaluation: they also hold
    heavy typos of real words, which would teach the model that English is gibberish.
    Negatives: S1 name tokens.
    Split by record (20% held out) so the reported rates are on unseen names; the
    threshold flags 1% of held-out S1 names.
    """
    word = pl.col("tok").str.contains(r"^[a-z]{4,}$")
    held_pid = (pl.col("pid").hash(SEED) % 5) == 0
    held_rid = (pl.col("rid").hash(SEED) % 5) == 0
    prep = name_prepare(pairs.select(pl.col("pid").alias("rid"), "o_name"), "o_name").rename({"rid": "pid"})
    alias = tokens_long(prep.filter(pl.col("name_has_alias")).rename({"pid": "rid"}), "name_alias").rename(
        {"rid": "pid"}).filter(word)
    s1_by_pid = pairs.select("pid", S1).join(s1_long.select(pl.col("rid").alias(S1), "tok"), on=S1)
    overlap = (o_long.select("pid", "tok").unique()
               .join(s1_by_pid.select("pid", "tok"), on=["pid", "tok"], how="semi").select("pid").unique())
    unrelated = (o_long.join(overlap, on="pid", how="anti")
                 .join(prep.filter(~pl.col("name_is_domain") & ~pl.col("name_has_alias")).select("pid"),
                       on="pid", how="semi")
                 .filter(word & ~pl.col("tok").is_in(list(s1_ignore))))
    s1_words = s1_long.filter(word & ~pl.col("tok").is_in(list(s1_ignore)))
    neg_tr = s1_words.filter(~held_rid).select("tok").unique()
    pos_tr = alias.filter(~held_pid).select("tok").unique().join(neg_tr, on="tok", how="anti")
    cp, cn = trigram_counts(pos_tr["tok"]), trigram_counts(neg_tr["tok"])
    g = cp.join(cn, on="gram", how="full", coalesce=True, suffix="_neg").fill_null(0)
    v, np_, nn_ = g.height, g["n"].sum(), g["n_neg"].sum()
    g = g.with_columns(logodds=((pl.col("n") + 1) / (np_ + v)).log() - ((pl.col("n_neg") + 1) / (nn_ + v)).log())
    grams = g.select("gram", "logodds")

    def name_scores(long: pl.DataFrame, key: str) -> pl.Series:
        sc = score_tokens(long.select("tok"), grams)
        return long.join(sc, on="tok").group_by(key).agg(pl.col("score").min())["score"]

    s1_held = name_scores(s1_words.filter(held_rid), "rid")
    thr = float(s1_held.quantile(0.99))
    alias_rate = float((name_scores(alias.filter(held_pid), "pid") > thr).mean())
    unrel_rate = float((name_scores(unrelated.filter(held_pid), "pid") > thr).mean())
    log(f"gibberish: trained on {pos_tr.height:,} positive / {neg_tr.height:,} negative token types; "
        f"threshold {thr:.3f} flags 1% of held-out S1 names, {alias_rate:.3f} of held-out dba/formerly "
        f"aliases and {unrel_rate:.3f} of held-out no-overlap names")
    meta = {"threshold": thr, "heldout_s1_flag_rate": 0.01, "heldout_alias_flag_rate": alias_rate,
            "heldout_no_overlap_flag_rate": unrel_rate}
    return grams.filter(pl.col("logodds").abs() > 0.05), meta


# --------------------------------------------------------------------------
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    out = RESOURCES_DIR
    out.mkdir(parents=True, exist_ok=True)
    s1, pairs = load_pairs()
    countries = sorted(s1["ckey"].unique().to_list())
    n_pairs = {c: n for c, n in pairs.group_by("ckey").len().iter_rows()}
    log(f"train S1 {s1.height:,}; pair sample {pairs.height:,} {n_pairs}")

    # ---------------- addresses ----------------
    s1c = comps(s1.rename({"entity_id": "rid"}), "business_address", "rid").join(
        s1.select(pl.col("entity_id").alias("rid"), "ckey"), on="rid")
    oc = comps(pairs, "o_addr", "pid").join(pairs.select("pid", "ckey"), on="pid")
    state_tbl, states = mine_states(s1c, pairs, oc, countries)

    state_variants = state_tbl.select("ckey", pl.col("variant").alias("comp"))
    s1_state_comps = s1c.join(state_variants, on=["ckey", "comp"], how="semi").select("rid", "cpos")

    def addr_tokens(c: pl.DataFrame, key: str) -> pl.DataFrame:
        c = c.join(state_variants, on=["ckey", "comp"], how="anti")
        t = (c.select(key, "cpos", "ckey", pl.col("comp").str.split(" ").alias("tok")).explode("tok", empty_as_null=True)
             .with_columns(pos=pl.int_range(pl.len()).over(key, "cpos").cast(pl.Int32)))
        t = merge_letter_runs(t.rename({key: "rid"}), by=("cpos",)).rename({"rid": key})
        t = t.join(c.select(key, "cpos", "ckey").unique(), on=[key, "cpos"])
        return t.with_columns(tok=strip_zeros_expr(pl.col("tok")).replace(ORDINAL_WORDS))

    pair_s1c = pairs.select("pid", S1).join(s1c.rename({"rid": S1}), on=S1).drop(S1)
    a_tok, b_tok = addr_tokens(pair_s1c, "pid"), addr_tokens(oc, "pid")
    la, lb = leftovers(a_tok, b_tok, "pid")
    ab = mine_abbreviations(la, lb, "pid", sum(n_pairs.values()) // len(n_pairs),
                            min_rate=0.0002, min_share=0.3)
    ab = ab.filter(~pl.col("long").is_in(state_tbl["variant"].to_list()) & ~pl.col("short").is_in(state_tbl["variant"].to_list()))
    addr_equiv = generalise(ab.select("ckey", pl.col("long").alias("variant"), pl.col("short").alias("canonical"),
                                      "count"), countries, ["variant", "canonical"])
    gk = pl.DataFrame({"ckey": ["france"] * len(FRENCH_ADDR_EQUIV), "variant": list(FRENCH_ADDR_EQUIV),
                       "canonical": list(FRENCH_ADDR_EQUIV.values()), "count": [0] * len(FRENCH_ADDR_EQUIV)}, schema_overrides={"count": pl.Int64})
    addr_equiv = pl.concat([addr_equiv.with_columns(source=pl.lit("mined")),
                            gk.with_columns(source=pl.lit("general_knowledge"))])
    log("addr_equiv:", addr_equiv.height, "rows;", ", ".join(
        f"{r[1]}->{r[2]}({r[0]})" for r in addr_equiv.head(40).iter_rows()))

    # street types: canonical forms whose long variant usually ends a numbered S1 component
    last_tok = s1c.filter(pl.col("comp").str.contains(r"^\d")).select(
        pl.col("comp").str.extract(r"(\S+)$", 1).alias("tok")).group_by("tok").len("n_last")
    all_tok = s1c.select(pl.col("comp").str.split(" ").alias("tok")).explode("tok", empty_as_null=True).group_by("tok").len("n_all")
    st_rate = last_tok.join(all_tok, on="tok").with_columns(r=pl.col("n_last") / pl.col("n_all"))
    street = (addr_equiv.filter(pl.col("source") == "mined")
              .join(st_rate, left_on="variant", right_on="tok").filter(pl.col("r") >= 0.3)
              .select(pl.col("canonical").alias("token")).unique())
    street = pl.concat([street, pl.DataFrame({"token": sorted(FRENCH_STREET_TYPES)})]).unique().sort("token")
    street_variants = set(addr_equiv["variant"]) | set(addr_equiv["canonical"])

    # house-number markers: added tokens right before a number on the other side
    nxt = pl.col("tok").shift(-1).over("pid", "cpos")
    mk = (b_tok.sort("pid", "cpos", "pos")
          .with_columns(_next_num=(nxt.str.contains(r"^\d") | nxt.is_in(["no", "nos"])).fill_null(False))
          .join(lb.select("pid", "tok"), on=["pid", "tok"], how="semi")
          .filter(pl.col("_next_num") & pl.col("tok").str.contains(r"^[a-z]{2,}$"))
          .group_by("tok").len("count"))
    mk_tot = b_tok.group_by("tok").len("total")
    markers = (mk.join(mk_tot, on="tok").with_columns(share=pl.col("count") / pl.col("total"))
               .filter((pl.col("count") >= 0.001 * pairs.height) & (pl.col("share") >= 0.2))
               .sort("count", descending=True))
    log("addr markers:", markers.select("tok", "count", "share").rows())

    city_tbl, city_suf = mine_cities(s1c, states, street_variants, pairs, oc)
    log("cities:", {c: n for c, n in city_tbl.group_by("ckey").len().iter_rows()},
        "suffixes:", city_suf.rows())
    del a_tok, b_tok, la, lb
    gc.collect()

    # ---------------- names ----------------
    s1n = name_long(s1.select(pl.col("entity_id").alias("rid"), "business_name"), "business_name", "rid").join(
        s1.select(pl.col("entity_id").alias("rid"), "ckey"), on="rid")
    pa = pairs.select("pid", S1).join(s1n.rename({"rid": S1}), on=S1).drop(S1)
    pb = name_long(pairs, "o_name", "pid").join(pairs.select("pid", "ckey"), on="pid")
    latin_pids = pb.group_by("pid").agg(pl.col("tok").str.contains(INDIC_RE).any().alias("nat")).filter(~pl.col("nat"))
    la, lb = leftovers(pa.join(latin_pids, on="pid", how="semi"), pb.join(latin_pids, on="pid", how="semi"), "pid")
    nab = mine_abbreviations(la, lb, "pid", sum(n_pairs.values()) // len(n_pairs), min_rate=0.0005, min_share=0.15)
    name_equiv = generalise(nab.select("ckey", pl.col("long").alias("variant"), pl.col("short").alias("canonical"),
                                       "count"), countries, ["variant", "canonical"])
    gk = pl.DataFrame({"ckey": [ALL] * len(FRENCH_NAME_EQUIV), "variant": list(FRENCH_NAME_EQUIV),
                       "canonical": list(FRENCH_NAME_EQUIV.values()), "count": [0] * len(FRENCH_NAME_EQUIV)}, schema_overrides={"count": pl.Int64})
    gk = gk.with_columns(ckey=pl.when(pl.col("variant") == "et").then(pl.lit("france")).otherwise("ckey"))
    name_equiv = pl.concat([name_equiv.with_columns(source=pl.lit("mined")),
                            gk.with_columns(source=pl.lit("general_knowledge"))])
    log("name_equiv:", name_equiv.rows())
    eq = {v: c for _, v, c, *_ in name_equiv.filter(pl.col("ckey") != "france").iter_rows()}

    # legal forms: frequent in S1, sit at the end of S1 names, dropped/changed often in matches
    def canon(e: pl.Expr) -> pl.Expr:
        return e.replace(eq)

    s1c_tok = s1n.with_columns(tok=canon(pl.col("tok")), n=pl.len().over("rid"))
    stat = s1c_tok.group_by("tok").agg(
        pl.col("rid").n_unique().alias("s1_count"),
        (pl.col("pos") >= pl.col("n") - 2).mean().alias("end_rate"))
    pa_c = pa.join(latin_pids, on="pid", how="semi").with_columns(tok=canon(pl.col("tok")))
    pb_c = pb.join(latin_pids, on="pid", how="semi").with_columns(tok=canon(pl.col("tok")))
    dropped = pa_c.select("pid", "tok").unique().join(pb_c.select("pid", "tok").unique(), on=["pid", "tok"], how="anti")
    drop_rate = (pa_c.select("pid", "tok").unique().group_by("tok").len("n_pairs")
                 .join(dropped.group_by("tok").len("n_drop"), on="tok", how="left")
                 .with_columns(drop_rate=pl.col("n_drop").fill_null(0) / pl.col("n_pairs")))
    stat = stat.join(drop_rate, on="tok", how="left")
    legal_cand = stat.filter((pl.col("s1_count") >= 0.0005 * s1.height) & (pl.col("end_rate") >= 0.85)).sort(
        "s1_count", descending=True)
    log("legal candidates (s1_count, end_rate, drop_rate):",
        legal_cand.select("tok", "s1_count", "end_rate", "drop_rate").head(40).rows())
    # Generic trailing words (group, center, trust, society ...) are also dropped by the
    # noise, so position + drop rate is not enough. Legal forms are acronym-like
    # (llc, pc, lp, pvt ...) or the long form of a mined abbreviation pair (limited ...).
    long_forms = set(name_equiv.filter(pl.col("source") == "mined")["variant"])
    # (private is never swapped with pvt by the generator, so it enters through end_rate >= 0.99)
    legal = legal_cand.filter((pl.col("s1_count") >= 0.005 * s1.height) & (pl.col("end_rate") >= 0.97)
                              & (pl.col("drop_rate") >= 0.1)
                              & ((pl.col("tok").str.len_chars() <= 4) | pl.col("tok").is_in(list(long_forms))
                                 | (pl.col("end_rate") >= 0.99))
                              & ~pl.col("tok").str.contains(r"^[ivx]+$"))
    legal = pl.concat([legal.select(pl.col("tok").alias("token"), "s1_count", "end_rate", "drop_rate",
                                    pl.lit("mined").alias("source")),
                       pl.DataFrame({"token": sorted(FRENCH_LEGAL)}).with_columns(
                           s1_count=pl.lit(None, pl.UInt32), end_rate=pl.lit(None, pl.Float64),
                           drop_rate=pl.lit(None, pl.Float64), source=pl.lit("general_knowledge"))
                       ]).unique("token", keep="first", maintain_order=True)
    log("legal forms:", legal["token"].to_list())

    # prefixes: leading tokens added on the other side (honorifics, "the")
    first = (pb_c.filter(pl.col("pos") == 0).join(pa_c.select("pid", "tok").unique(), on=["pid", "tok"], how="anti")
             .group_by("tok").len("count"))
    first_tot = pb_c.group_by("tok").len("total")
    prefixes = (first.join(first_tot, on="tok").with_columns(first_share=pl.col("count") / pl.col("total"))
                .filter((pl.col("count") >= 0.002 * pairs.height) & (pl.col("first_share") >= 0.6)
                        & ~pl.col("tok").is_in(legal["token"].to_list()))
                .sort("count", descending=True))
    log("name prefixes:", prefixes.rows())

    # native-script lexicon: positional alignment when token counts agree
    nat_pids = pb.group_by("pid").agg(pl.col("tok").str.contains(INDIC_RE).any().alias("nat")).filter("nat")
    na_ = pa.join(nat_pids, on="pid", how="semi").select("pid", "pos", pl.col("tok").alias("latin"))
    nb_ = pb.join(nat_pids, on="pid", how="semi").select("pid", "pos", pl.col("tok").alias("native"))
    same_len = (na_.group_by("pid").len("la").join(nb_.group_by("pid").len("lb"), on="pid")
                .filter(pl.col("la") == pl.col("lb")).select("pid"))
    al = (nb_.join(same_len, on="pid", how="semi").join(na_, on=["pid", "pos"])
          .filter(pl.col("native").str.contains(INDIC_RE) & pl.col("latin").str.contains(r"^[a-z0-9]+$")))
    # state names in native script (address side) join the lexicon token by token
    st_nat = state_tbl.filter(pl.col("kind") == "native").with_columns(
        native=pl.col("variant").str.split(" "), latin=pl.col("state_key").str.split(" "))
    st_nat = st_nat.filter(pl.col("native").list.len() == pl.col("latin").list.len()).explode(
        "native", "latin").select("native", "latin", pl.col("count").cast(pl.UInt32))
    lex = pl.concat([al.group_by("native", "latin").len("count"), st_nat]).group_by("native", "latin").agg(
        pl.col("count").sum())
    lex = (lex.with_columns(prob=pl.col("count") / pl.col("count").sum().over("native"))
           .sort("count", descending=True).unique("native", keep="first")
           .filter((pl.col("count") >= 2) & (pl.col("prob") >= 0.5)).sort("count", descending=True))
    log(f"indic lexicon: {lex.height:,} native tokens from {same_len.height:,} aligned names; top:",
        lex.head(15).select("native", "latin", "count").rows())

    s1_ignore = set(legal["token"]) | set(prefixes["tok"]) | ARTICLES | set(name_equiv["variant"])
    grams, gmeta = mine_gibberish(s1n, pairs, pb, s1_ignore)

    # ---------------- write ----------------
    state_tbl.write_csv(out / "states.csv")
    city_tbl.write_csv(out / "cities.csv")
    city_suf.write_csv(out / "city_suffixes.csv")
    addr_equiv.write_csv(out / "addr_equiv.csv")
    street.write_csv(out / "street_types.csv")
    markers.select(pl.col("tok").alias("token"), "count", "share").write_csv(out / "addr_markers.csv")
    name_equiv.write_csv(out / "name_equiv.csv")
    legal.write_csv(out / "legal_forms.csv")
    prefixes.select(pl.col("tok").alias("token"), "count", "first_share").write_csv(out / "name_prefixes.csv")
    lex.write_csv(out / "indic_lexicon.csv")
    grams.with_columns(pl.col("logodds").round(4)).sort("gram").write_csv(out / "gibberish_trigrams.csv")
    (out / "gibberish_meta.json").write_text(json.dumps(gmeta, indent=2), encoding="utf-8")
    (out / "mining_summary.md").write_text(
        "# Mined normalisation resources (train only)\n\n"
        f"Generated by `src/mine_resources.py`, seed {SEED}, {pairs.height:,} matched pairs "
        f"({N_PAIRS_PER_COUNTRY:,} per country, half S2 / half S3) and all {s1.height:,} train S1 records.\n\n"
        "```\n" + "\n".join(LOG) + "\n```\n", encoding="utf-8")
    log(f"wrote resources to {out}")


if __name__ == "__main__":
    main()
