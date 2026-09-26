"""Sanity check for dataset/cache/<split>/norm_source{1,2,3}.parquet.

Per source: row count vs. expected and vs. the raw parquet cache, entity_id uniqueness,
null counts per country (any null in a KEY column is a failure; nulls elsewhere are
listed), and per-country empty rates of the columns blocking relies on.
Splits are processed one at a time.

    python check_norm.py            # both splits
    python check_norm.py --split test
"""

from __future__ import annotations

import argparse

import polars as pl

from data import cache_path, source_path
from evaluate import norm_path

EXPECTED_ROWS = {"train": (2_206_821, 5_034_616, 5_285_603), "test": (1_732_544, 4_887_273, 5_082_316)}
KEY_COLS = [
    "entity_id", "country", "name_core", "name_compact", "name_tokens", "name_fold",
    "addr_nolm", "addr_tokens", "addr_house_no", "addr_city", "addr_missing",
]
EMPTY_RATES = {
    "name_compact": pl.col("name_compact") == "",
    "name_core": pl.col("name_core") == "",
    "name_tok0": pl.col("name_tokens").list.len() == 0,
    "addr_missing": pl.col("addr_missing"),
    "addr_nolm": pl.col("addr_nolm") == "",
    "house_no": pl.col("addr_house_no") == "",
    "city": pl.col("addr_city") == "",
    "gibberish": pl.col("name_is_gibberish"),
}


def check(split: str) -> int:
    fails = 0
    for src in (1, 2, 3):
        path = norm_path(split, src)
        lf = pl.scan_parquet(path)
        cols = lf.collect_schema().names()
        missing = [c for c in KEY_COLS if c not in cols]
        if missing:
            print(f"FAIL {split} s{src}: missing columns {missing}")
            fails += 1
            continue
        n, n_ids = lf.select(pl.len(), pl.col("entity_id").n_unique()).collect().row(0)
        raw = cache_path(source_path(split, src))
        n_raw = pl.scan_parquet(raw).select(pl.len()).collect().item() if raw.exists() else None
        ok = n == EXPECTED_ROWS[split][src - 1] and n_ids == n and (n_raw is None or n_raw == n)
        fails += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {split} s{src}: rows={n:,} unique_ids={n_ids:,} raw_cache={n_raw}")

        nulls = (
            lf.group_by("country")
            .agg(pl.len().alias("_n"), *[pl.col(c).null_count() for c in cols if c != "country"])
            .collect()
        )
        for row in nulls.iter_rows(named=True):
            for c, v in row.items():
                if c in ("country", "_n") or not v:
                    continue
                bad = c in KEY_COLS
                fails += bad
                print(f"{'FAIL' if bad else 'note'} {split} s{src} [{row['country']}]: {c} nulls={v:,} ({v / row['_n']:.4f})")
        if nulls["country"].null_count():
            print(f"FAIL {split} s{src}: null country rows")
            fails += 1

        rates = (
            lf.group_by("country")
            .agg(pl.len().alias("n"), *[e.fill_null(False).mean().alias(k) for k, e in EMPTY_RATES.items()],
                 (pl.col("addr_missing").fill_null(False) != (pl.col("addr_nolm") == "")).sum().alias("miss_vs_nolm"))
            .sort("country")
            .collect()
        )
        for r in rates.iter_rows(named=True):
            body = " ".join(f"{k}={r[k]:.3f}" for k in EMPTY_RATES)
            print(f"     {split} s{src} [{r['country']}] n={r['n']:,} empty: {body} miss_vs_nolm={r['miss_vs_nolm']:,}")
    return fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"])
    a = ap.parse_args()
    total = sum(check(s) for s in ([a.split] if a.split else ["train", "test"]))
    print("SANITY OK" if total == 0 else f"SANITY FAIL ({total} issues)")
