"""Pairwise features for the ranker's wide (SAVE_CAP~80) candidate pool.

Reuses features.py's exact feature computation (compute_features()) unchanged -- the wide
pool has the identical (src, qi, s1, score, rank, name_s, addr_s, hits, best) schema as
candidates.parquet, just up to rank 79 instead of the final pruned <=2. Only the *source*
of the candidate frame differs: ranker_data.py's dataset/cache/blocking_v2_ranker/train/parts/
(via blocking.load_scored) instead of a single candidates.parquet.

    python ranker_features.py
"""

from __future__ import annotations

import time

import polars as pl

import cv_harness as ev
from blocking import load_scored
from data import DATASET_DIR
from features import KEY_COLS, compute_features

OD = DATASET_DIR / "cache" / "blocking_v2_ranker" / "train"


def build() -> None:
    t0 = time.perf_counter()
    cands = load_scored(OD)
    truth = ev.load_truth_idx()
    cands = cands.join(
        truth.with_columns(pl.lit(True).alias("is_true")), on=KEY_COLS, how="left"
    ).with_columns(pl.col("is_true").fill_null(False))

    cands = compute_features(cands, "train")

    out_path = OD / "features.parquet"
    cands.write_parquet(out_path)
    print(f"[ranker_features] {cands.height:,} rows -> {out_path.relative_to(DATASET_DIR.parent)} "
          f"({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    build()
