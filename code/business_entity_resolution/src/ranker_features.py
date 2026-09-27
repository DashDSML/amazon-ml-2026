"""Pairwise features for the ranker's wide (SAVE_CAP~80) candidate pool, chunked.

Reuses features.py's exact feature computation (compute_features()) unchanged -- the wide
pool has the identical (src, qi, s1, score, rank, name_s, addr_s, hits, best) schema as
candidates.parquet, just up to rank 79 instead of the final pruned <=2. Only the *source*
differs: ranker_data.py's dataset/cache/blocking_v2_ranker/train/parts/*.parquet (~780M rows
across 414 files) instead of one small candidates.parquet.

A first attempt loaded the whole wide pool at once (mirroring build_features()) and was
OOM-killed at ~61GB RSS on a 62GB machine. This processes a few part-files at a time
(CLAUDE.md's own memory-budget rule, already used everywhere else in this pipeline -- see
blocking.py's Q_CHUNK/S1_CHUNK), writing one output parquet per chunk under
features_parts/ rather than a single monolithic features.parquet.

    python ranker_features.py                # full run, ~100+ chunks
    python ranker_features.py --files-per-chunk 2 --limit-chunks 2   # quick check
"""

from __future__ import annotations

import argparse
import shutil
import time

import polars as pl

import cv_harness as ev
from data import DATASET_DIR
from features import KEY_COLS, compute_features

OD = DATASET_DIR / "cache" / "blocking_v2_ranker" / "train"
COLS = ["src", "qi", "s1", "score", "name_s", "addr_s", "rank"]


def build(files_per_chunk: int = 4, limit_chunks: int | None = None) -> None:
    t0 = time.perf_counter()
    files = sorted((OD / "parts").glob("*.parquet"))
    truth = ev.load_truth_idx().with_columns(pl.lit(True).alias("is_true"))
    out_dir = OD / "features_parts"
    if out_dir.exists() and limit_chunks is None:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = 0
    total_rows = 0
    for i in range(0, len(files), files_per_chunk):
        if limit_chunks is not None and n_chunks >= limit_chunks:
            break
        batch = files[i : i + files_per_chunk]
        cands = pl.concat([pl.read_parquet(f).select(COLS) for f in batch])
        cands = cands.join(truth, on=KEY_COLS, how="left").with_columns(pl.col("is_true").fill_null(False))
        out = compute_features(cands, "train")
        out.write_parquet(out_dir / f"chunk_{i:05d}.parquet")
        total_rows += out.height
        n_chunks += 1
        if n_chunks % 10 == 0 or limit_chunks is not None:
            el = time.perf_counter() - t0
            done_files = min(i + files_per_chunk, len(files))
            eta = el / done_files * (len(files) - done_files)
            print(f"  chunk {n_chunks} ({done_files}/{len(files)} files): {out.height:,} rows "
                  f"({el / 60:.1f} min elapsed, ~{eta / 60:.0f} min left)", flush=True)

    print(f"[ranker_features] {n_chunks} chunks, {total_rows:,} rows total -> "
          f"{out_dir.relative_to(DATASET_DIR.parent)} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files-per-chunk", type=int, default=4)
    ap.add_argument("--limit-chunks", type=int, default=None, help="quick check: process only this many chunks")
    args = ap.parse_args()
    build(args.files_per_chunk, args.limit_chunks)
