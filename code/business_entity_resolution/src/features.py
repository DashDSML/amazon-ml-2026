"""Pairwise features for the LightGBM matcher (CLAUDE.md D3), one row per candidate
pair from blocking v1's ``candidates.parquet`` (src, qi, s1, score, name_s, addr_s, rank).

Every input is a column normalize.py already produced (name_core, name_compact,
name_tokens, name_legal, name_alias, addr_house_no, addr_city, addr_state,
addr_numbers, addr_missing, name_is_gibberish, name_translit, addr_translit,
name_script). Features are pure per-pair comparisons -- nothing is fit here, so
the same code runs unchanged on train and test (country is deliberately excluded:
France is unseen in training, G1).

Feature groups (see FEATURE_COLS for the exact final list):
  blocking   score, name_s, addr_s, rank, best (record's top score), gap_best
             (best - score), gap_top2 (record's best - its 2nd-best, 0 if only
             one candidate), n_cand_record (candidates for this S2/S3 record)
  s1_side    rank_s1 (this pair's rank among candidates competing for the same
             S1, by score), n_cand_s1 (candidates competing for this S1)
  name       jaro_winkler(name_compact), token_sort_ratio(name_core),
             partial_ratio(name_core), name_token_jaccard, name_len_diff,
             legal_match, alias_fwd (record's alias vs S1's name),
             alias_bwd (S1's alias vs record's name)
  address    house_eq / house_diff / house_missing, city_eq, state_eq,
             addr_num_jaccard, addr_missing_either
  record     name_is_gibberish, name_translit, addr_translit, name_script_nonlatin
             (S2/S3 side only -- S1 is a clean deduplicated reference)

Rapidfuzz string comparisons run in chunks of ~500k pairs (cpdist, workers=-1) so
peak memory stays bounded on the full 12.8M-pair train set.

    python features.py --split train --smoke 20000   # dev sample
    python features.py --split train                 # full (run yourself; slow)
    python features.py --split test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

import cv_harness as ev
from blocking import out_dir
from data import DATASET_DIR

SEED = 42
CHUNK = 500_000

# columns read from norm_source1 (suffixed _s1 on join) and norm_source{2,3} (record)
CMP_COLS = [
    "name_core", "name_compact", "name_tokens", "name_legal", "name_alias",
    "addr_house_no", "addr_city", "addr_state", "addr_numbers", "addr_missing",
]
STR_COLS = ["name_core", "name_compact", "name_legal", "addr_house_no", "addr_city", "addr_state"]
LIST_COLS = ["name_tokens", "addr_numbers"]
REC_FLAG_COLS = ["name_is_gibberish", "name_translit", "addr_translit", "name_script"]

KEY_COLS = ["src", "qi", "s1"]
FEATURE_COLS = [
    "score", "name_s", "addr_s", "rank", "best", "gap_best", "gap_top2", "n_cand_record",
    "rank_s1", "n_cand_s1",
    "jaro_winkler", "token_sort", "partial_ratio", "name_jaccard", "name_len_diff",
    "legal_match", "alias_fwd", "alias_bwd",
    "house_eq", "house_diff", "house_missing", "city_eq", "state_eq",
    "addr_num_jaccard", "addr_missing_either",
    "name_is_gibberish", "name_translit", "addr_translit", "name_script_nonlatin",
]


def _load_side(split: str, source: int, cols: list[str], id_col: str) -> pl.DataFrame:
    df = (
        pl.scan_parquet(ev.norm_path(split, source))
        .select(cols)
        .with_row_index(id_col)
        .with_columns(
            *[pl.col(c).fill_null("") for c in STR_COLS if c in cols],
            *[pl.col(c).fill_null(pl.lit([], dtype=pl.List(pl.String))) for c in LIST_COLS if c in cols],
        )
        .collect()
    )
    if "addr_missing" in cols:
        df = df.with_columns(pl.col("addr_missing").fill_null(False))
    return df


def _rec_side(split: str, source: int) -> pl.DataFrame:
    return _load_side(split, source, CMP_COLS + REC_FLAG_COLS, "qi").with_columns(
        pl.col("name_is_gibberish").fill_null(False),
        pl.col("name_translit").fill_null(False),
        pl.col("addr_translit").fill_null(False),
    )


def _s1_side(split: str) -> pl.DataFrame:
    df = _load_side(split, 1, CMP_COLS, "s1")
    return df.rename({c: f"{c}_s1" for c in CMP_COLS})


def _fuzzy_features(df: pl.DataFrame) -> pl.DataFrame:
    """rapidfuzz comparisons, chunked (~CHUNK pairs) to bound memory."""
    n = df.height
    jw = np.empty(n, np.float32)
    tsr = np.empty(n, np.float32)
    pr = np.empty(n, np.float32)
    afwd = np.zeros(n, np.float32)
    abwd = np.zeros(n, np.float32)

    name_compact, name_compact_s1 = df["name_compact"], df["name_compact_s1"]
    name_core, name_core_s1 = df["name_core"], df["name_core_s1"]
    alias, alias_s1 = df["name_alias"], df["name_alias_s1"]
    has_alias = alias.is_not_null().to_numpy()
    has_alias_s1 = alias_s1.is_not_null().to_numpy()

    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        jw[a:b] = cpdist(
            name_compact[a:b].to_list(), name_compact_s1[a:b].to_list(),
            scorer=JaroWinkler.normalized_similarity, workers=-1, dtype=np.float32,
        )
        tsr[a:b] = cpdist(
            name_core[a:b].to_list(), name_core_s1[a:b].to_list(),
            scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32,
        ) / 100.0
        pr[a:b] = cpdist(
            name_core[a:b].to_list(), name_core_s1[a:b].to_list(),
            scorer=fuzz.partial_ratio, workers=-1, dtype=np.float32,
        ) / 100.0

        sel = np.flatnonzero(has_alias[a:b]) + a
        if sel.size:
            afwd[sel] = cpdist(
                alias.gather(sel).to_list(), name_core_s1.gather(sel).to_list(),
                scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32,
            ) / 100.0
        sel = np.flatnonzero(has_alias_s1[a:b]) + a
        if sel.size:
            abwd[sel] = cpdist(
                alias_s1.gather(sel).to_list(), name_core.gather(sel).to_list(),
                scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32,
            ) / 100.0

    return df.with_columns(
        pl.Series("jaro_winkler", jw), pl.Series("token_sort", tsr), pl.Series("partial_ratio", pr),
        pl.Series("alias_fwd", afwd), pl.Series("alias_bwd", abwd),
    )


def build_features(split: str, smoke: int | None = None) -> Path:
    t0 = time.perf_counter()
    cands = pl.read_parquet(out_dir(split, None) / "candidates.parquet")
    if smoke:
        rng = np.random.default_rng(SEED)
        uniq_s1 = cands["s1"].unique().to_numpy()
        pick = rng.choice(uniq_s1, size=min(smoke, len(uniq_s1)), replace=False)
        cands = cands.filter(pl.col("s1").is_in(pl.Series(pick)))

    if split == "train":
        truth = ev.load_truth_idx()
        cands = cands.join(
            truth.with_columns(pl.lit(True).alias("is_true")), on=KEY_COLS, how="left"
        ).with_columns(pl.col("is_true").fill_null(False))

    cands = cands.with_columns(
        best=pl.col("score").max().over(["src", "qi"]),
        n_cand_record=pl.len().over(["src", "qi"]),
        rank_s1=(pl.col("score").rank(method="ordinal", descending=True).over("s1") - 1).cast(pl.Int32),
        n_cand_s1=pl.len().over("s1"),
    ).with_columns(gap_best=pl.col("best") - pl.col("score"))

    second = cands.filter(pl.col("rank") == 1).select("src", "qi", pl.col("score").alias("_second"))
    cands = (
        cands.join(second, on=["src", "qi"], how="left")
        .with_columns(gap_top2=pl.col("best") - pl.col("_second").fill_null(0.0))
        .drop("_second")
    )

    parts = []
    for src in (2, 3):
        rec = _rec_side(split, src)
        part = cands.filter(pl.col("src") == src).join(rec, on="qi", how="left")
        parts.append(part)
    cands = pl.concat(parts, how="vertical_relaxed")
    cands = cands.join(_s1_side(split), on="s1", how="left")

    cands = cands.with_columns(
        _inter_n=pl.col("name_tokens").list.set_intersection("name_tokens_s1").list.len(),
        _union_n=pl.col("name_tokens").list.set_union("name_tokens_s1").list.len(),
        _inter_a=pl.col("addr_numbers").list.set_intersection("addr_numbers_s1").list.len(),
        _union_a=pl.col("addr_numbers").list.set_union("addr_numbers_s1").list.len(),
    ).with_columns(
        name_jaccard=pl.when(pl.col("_union_n") > 0).then(pl.col("_inter_n") / pl.col("_union_n")).otherwise(0.0),
        addr_num_jaccard=pl.when(pl.col("_union_a") > 0).then(pl.col("_inter_a") / pl.col("_union_a")).otherwise(0.0),
        name_len_diff=(
            pl.col("name_core").str.len_chars().cast(pl.Int32)
            - pl.col("name_core_s1").str.len_chars().cast(pl.Int32)
        ).abs(),
        legal_match=((pl.col("name_legal") == pl.col("name_legal_s1")) & (pl.col("name_legal") != "")).fill_null(False),
        house_eq=((pl.col("addr_house_no") == pl.col("addr_house_no_s1")) & (pl.col("addr_house_no") != "")).fill_null(False),
        house_diff=(
            (pl.col("addr_house_no") != pl.col("addr_house_no_s1"))
            & (pl.col("addr_house_no") != "") & (pl.col("addr_house_no_s1") != "")
        ).fill_null(False),
        house_missing=((pl.col("addr_house_no") == "") | (pl.col("addr_house_no_s1") == "")).fill_null(True),
        city_eq=((pl.col("addr_city") == pl.col("addr_city_s1")) & (pl.col("addr_city") != "")).fill_null(False),
        state_eq=((pl.col("addr_state") == pl.col("addr_state_s1")) & (pl.col("addr_state") != "")).fill_null(False),
        addr_missing_either=(pl.col("addr_missing") | pl.col("addr_missing_s1")).fill_null(True),
        name_script_nonlatin=pl.col("name_script").is_not_null(),
    ).drop("_inter_n", "_union_n", "_inter_a", "_union_a")

    cands = _fuzzy_features(cands)

    out_cols = KEY_COLS + FEATURE_COLS + (["is_true"] if split == "train" else [])
    cands = cands.with_columns(
        pl.col(c).cast(pl.Float32) for c in FEATURE_COLS if cands.schema[c] != pl.Float32
    ).select(out_cols)

    od = out_dir(split, smoke)
    od.mkdir(parents=True, exist_ok=True)
    out_path = od / "features.parquet"
    cands.write_parquet(out_path)
    print(f"[features] {split} smoke={smoke}: {cands.height:,} rows, {len(FEATURE_COLS)} features "
          f"-> {out_path.relative_to(DATASET_DIR.parent)} ({time.perf_counter() - t0:.0f}s)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--smoke", type=int, default=None, help="train only: sample this many S1")
    args = ap.parse_args()
    assert not (args.smoke and args.split == "test"), "--smoke is train-only"
    build_features(args.split, args.smoke)
