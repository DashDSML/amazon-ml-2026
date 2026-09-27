"""Turn ranker_model.py's finding into a real candidate set (NonStandard.md T21): apply the
trained ranker to the FULL wide pool (not just the 20k-S1 eval sample), re-rank by its
probability, prune with its own grid winner (floor=0.4, alpha=0.9, m=3 -- oracle 0.9977 vs
frozen v2's 0.9893, mean 4.28/S1 vs 5.40), then recompute the final matcher's features
(features.py's current FEATURE_COLS, including T8 confusability) on the much smaller pruned
set. Writes to dataset/cache/blocking_v2r/{split}/ -- a new tree; the frozen v2 output
feeding lb-3 is never touched.

The ranker (dataset/cache/blocking_v2_ranker/model/ranker.txt) was trained on the 33-column
FEATURE_COLS that existed before T8 confusability was added (frozen here as
RANKER_FEATURE_COLS, since the model file doesn't self-describe its own features) -- it is
applied to the wide pool's *already-computed* features_parts (same 33 columns), unrelated to
today's separate features.py change. Confusability is looked up from blocking_v2's own
confusability.parquet (an S1-only property, independent of which candidate-pruning version is
in use) via compute_features(..., version="v2").

    python ranker_apply.py --split train
    python ranker_apply.py --split test
"""

from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import polars as pl

from blocking import keep_mask, out_dir
from data import DATASET_DIR
from features import compute_features

RANKER_FEATURE_COLS = [
    "score", "name_s", "addr_s", "rank", "best", "gap_best", "gap_top2", "n_cand_record",
    "rank_s1", "n_cand_s1", "jaro_winkler", "token_sort", "partial_ratio", "name_jaccard",
    "name_len_diff", "legal_match", "alias_fwd", "alias_bwd", "house_eq", "house_diff",
    "house_missing", "city_eq", "state_eq", "addr_num_jaccard", "addr_missing_either",
    "name_is_gibberish", "name_translit", "addr_translit", "name_script_nonlatin",
    "is_mutual_top1", "reciprocal_rank_sum", "trans_max_sim", "trans_agree_count",
]
FLOOR, ALPHA, M = 0.4, 0.9, 3  # ranker_model.py's grid winner
FILES_PER_CHUNK = 8
RANKER_MODEL_PATH = DATASET_DIR / "cache" / "blocking_v2_ranker" / "model" / "ranker.txt"


def build(split: str) -> None:
    t0 = time.perf_counter()
    wide_dir = DATASET_DIR / "cache" / "blocking_v2_ranker" / split
    files = sorted((wide_dir / "features_parts").glob("*.parquet"))
    assert files, f"no features_parts under {wide_dir} -- run ranker_data.py + ranker_features.py first"
    booster = lgb.Booster(model_file=str(RANKER_MODEL_PATH))

    od = out_dir(split, None, "v2r")
    od.mkdir(parents=True, exist_ok=True)

    parts = []
    total_in = 0
    for i in range(0, len(files), FILES_PER_CHUNK):
        batch = files[i : i + FILES_PER_CHUNK]
        df = pl.concat([pl.read_parquet(f) for f in batch])
        total_in += df.height
        prob = booster.predict(df.select(RANKER_FEATURE_COLS).to_numpy())
        keep_cols = [c for c in df.columns if c not in ("rank", "score", "best")]
        df = (
            df.select(keep_cols)
            .with_columns(score=pl.Series("score", prob))
            .sort(["src", "qi", "score"], descending=[False, False, True])
            .with_columns(
                rank=pl.int_range(pl.len()).over(["src", "qi"]).cast(pl.UInt8),
                best=pl.col("score").max().over(["src", "qi"]),
            )
        )
        keep = keep_mask(df, FLOOR, ALPHA, M)
        pruned = df.filter(pl.Series(keep))
        out_cols = ["src", "qi", "s1", "score", "name_s", "addr_s", "rank"] + (
            ["is_true"] if "is_true" in pruned.columns else []
        )
        parts.append(pruned.select(out_cols))
        n_chunk = i // FILES_PER_CHUNK + 1
        if n_chunk % 20 == 0:
            print(f"  chunk {n_chunk} ({min(i + FILES_PER_CHUNK, len(files))}/{len(files)} files) "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    cands = pl.concat(parts, how="vertical_relaxed")
    del parts
    cands.write_parquet(od / "candidates.parquet")
    n_s1 = cands["s1"].n_unique()
    print(f"[ranker_apply] {split}: {total_in:,} wide rows -> {cands.height:,} pruned rows "
          f"({cands.height / max(n_s1, 1):.2f}/S1) -> "
          f"{(od / 'candidates.parquet').relative_to(DATASET_DIR.parent)} ({time.perf_counter() - t0:.0f}s)")

    feats = compute_features(cands, split, "v2")  # version="v2" only for the confusability lookup
    feats.write_parquet(od / "features.parquet")
    print(f"[ranker_apply] {split}: features -> "
          f"{(od / 'features.parquet').relative_to(DATASET_DIR.parent)} ({time.perf_counter() - t0:.0f}s total)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    args = ap.parse_args()
    build(args.split)
