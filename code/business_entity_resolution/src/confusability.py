"""S1 "confusability" (NonStandard.md T8): for each S1 entity, similarity to its nearest
OTHER S1 entity. Every S1-S1 pair is a guaranteed non-match (S1 is deduplicated), so this is
a free, label-free signal: a high-confusability S1 (a common name pattern, chain-like) needs
stronger evidence before accepting a candidate; a low-confusability S1 (a distinctive name)
can afford to accept a weaker-looking true match with less risk.

Motivation (aimed at a teammate's two biggest matcher loss buckets, per their own
current_bottleneck.md): "in-list miss" (35-39% of loss -- true match in the candidate list,
scored too low) and "false positives"/"singleton given a match" (29-35% combined -- lookalike
candidates accepted). A per-S1 confusability feature could let the decision layer be more
lenient exactly where it's safe to be (low confusability) and stricter exactly where
lookalikes are a real risk (high confusability), rather than one global calibration for both.

Method: reuses blocking.py's own key-based retrieval + rapidfuzz scoring, unchanged, with S1
queried against its own index instead of S2/S3 against it -- the identical machinery, just
S1-vs-S1. The single best-scoring OTHER S1 (self-matches excluded) is this S1's nearest
neighbour; that score is the confusability feature.

    python confusability.py --split train --version v1 --smoke 20000
    python confusability.py --split train --version v1
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl

import blocking as bk
import cv_harness as ev
from data import DATASET_DIR


def build(split: str, version: str, smoke: int | None = None) -> None:
    t0 = time.perf_counter()
    countries = sorted(
        set(pl.scan_parquet(ev.norm_path(split, 1)).select(pl.col("country").fill_null("").unique()).collect()["country"].to_list())
    )
    cc_expr = pl.col("country").cast(pl.Enum(countries)).to_physical().cast(pl.Int16)

    s1 = bk.load_norm(split, 1)
    n_s1 = s1.height
    if smoke:
        rng = np.random.default_rng(bk.SEED)
        keep = np.sort(rng.choice(n_s1, size=min(smoke, n_s1), replace=False)).astype(np.uint32)
        query_ids = keep
    else:
        query_ids = np.arange(n_s1, dtype=np.uint32)

    s1_cc = s1.select(cc_expr).to_series().to_numpy()
    s1_pos = np.arange(n_s1, dtype=np.int64)
    keys, s1i = bk.build_index(s1, version=version)
    print(f"[confusability] index S1={n_s1:,} query={len(query_ids):,} ({time.perf_counter()-t0:.0f}s)")

    best_score = np.zeros(n_s1, np.float32)
    best_other = np.full(n_s1, -1, np.int64)
    for off in range(0, len(query_ids), bk.Q_CHUNK):
        batch_ids = query_ids[off : off + bk.Q_CHUNK]
        chunk = s1.filter(pl.Series(np.isin(s1["i"].to_numpy(), batch_ids)))
        q_cc = chunk.select(cc_expr).to_series().to_numpy()
        cand, _ = bk.retrieve(chunk, keys, s1i, s1_cc, q_cc, version=version)
        if cand.height == 0:
            continue
        cand = cand.filter(pl.col("qi") != pl.col("s1"))  # exclude self-match
        if cand.height == 0:
            continue
        scored = bk.score(cand, chunk, s1, s1_pos, version=version)
        top = (
            scored.sort(["qi", "score"], descending=[False, True])
            .group_by("qi", maintain_order=True).first()
        )
        qi_arr = top["qi"].to_numpy().astype(np.int64)
        sc_arr = top["score"].to_numpy()
        oth_arr = top["s1"].to_numpy().astype(np.int64)
        best_score[qi_arr] = sc_arr
        best_other[qi_arr] = oth_arr
        if (off // bk.Q_CHUNK + 1) % 20 == 0:
            print(f"  [progress] {off + len(batch_ids):,}/{len(query_ids):,} S1 queried "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    out = pl.DataFrame({
        "s1": np.arange(n_s1, dtype=np.uint32),
        "confusability": best_score,
        "nearest_other_s1": best_other,
    })
    od = bk.out_dir(split, smoke, version)
    od.mkdir(parents=True, exist_ok=True)
    out_path = od / "confusability.parquet"
    out.write_parquet(out_path)
    print(f"[confusability] mean={best_score[query_ids].mean():.4f} "
          f"p90={np.percentile(best_score[query_ids], 90):.4f} "
          f"max={best_score[query_ids].max():.4f} "
          f"-> {out_path.relative_to(DATASET_DIR.parent)} ({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--version", choices=["v1", "v2"], default="v1")
    ap.add_argument("--smoke", type=int, default=None)
    args = ap.parse_args()
    build(args.split, args.version, args.smoke)
