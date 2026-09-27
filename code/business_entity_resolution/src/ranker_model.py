"""Train the ranker (supervised meta-blocking -- disclosed per NonStandard.md T21: this
model's own pruning decision *becomes* the final candidate_pairs.tsv content, the same way
the current hand-weighted score does; the separate, already-shipped LightGBM matcher still
runs afterward on whatever list this produces, unchanged) on the wide SAVE_CAP~80 pool.

The wide pool is extremely imbalanced (~781M rows, true matches only ~1%), so training uses
a class-balanced sample: every positive plus a random sample of negatives at NEG_RATIO:1,
GroupKFold-by-S1 (folds 0-3 train, fold 4 holdout, matching model.py exactly). Evaluation
re-ranks a SAMPLE of fold-4 holdout S1 entities (fastloop-style, ~20k S1, for a fast,
representative read rather than the full ~150M-row holdout) by the ranker's probability
instead of the current hand-weighted score, sweeps the same floor/alpha/m grid used
everywhere else in this project, and reports oracle F0.5 + mean candidates/S1 side by side
with the frozen v2 blocking's real numbers (oracle 0.9893, mean 5.40/S1, from STATUS.md).
Adopted only if it is a genuine improvement per BLOCKING_V2 #6's own criterion (oracle rises
at a similar or smaller size, or size falls at a similar oracle) -- not adopted otherwise,
consistent with every other decision-layer/training experiment this session.

    python ranker_model.py
"""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import polars as pl

import cv_harness as ev
from blocking import GRID, keep_mask
from data import DATASET_DIR
from features import FEATURE_COLS
from model import EARLY_STOP, HOLDOUT_FOLD, MAX_ROUNDS, PARAMS

OD = DATASET_DIR / "cache" / "blocking_v2_ranker" / "train"
NEG_RATIO = 5
SEED = 42
EVAL_N_S1 = 20_000
FILES_PER_CHUNK = 8  # matches ranker_features.py's proven-safe chunked-file pattern


def load_sampled() -> pl.DataFrame:
    """Every positive row + a random NEG_RATIO:1 sample of negatives, with fold assigned.

    Two passes over the part-files, both column-projected and chunked (never one
    lf.collect() over the whole 781M-row pool -- that pattern hit 50GB+ RSS and was
    guard-killed once already): pass 1 reads only the ``is_true`` column to get the
    exact global positive/negative counts (cheap, projection pushdown keeps this to
    one boolean column across 781M rows); pass 2 re-scans FILES_PER_CHUNK files at a
    time and collects only the rows that survive (is_true OR the hash-sampled
    negative keep-rate), so peak memory is bounded by one chunk, not the whole pool.
    """
    files = sorted((OD / "features_parts").glob("*.parquet"))

    n_pos = 0
    n_rows = 0
    for i in range(0, len(files), FILES_PER_CHUNK):
        batch = files[i : i + FILES_PER_CHUNK]
        agg = (
            pl.scan_parquet(batch)
            .select(pl.col("is_true").sum().alias("pos"), pl.len().alias("n"))
            .collect()
        )
        n_pos += int(agg["pos"][0])
        n_rows += int(agg["n"][0])
    n_neg_total = n_rows - n_pos
    n_neg_target = n_pos * NEG_RATIO
    frac = min(1.0, n_neg_target / max(n_neg_total, 1))
    print(f"[ranker_model] positives={n_pos:,}  negatives_total={n_neg_total:,}  sampling frac={frac:.5f}")

    row_hash = pl.concat_str([pl.col("src").cast(pl.String), pl.lit("|"), pl.col("qi").cast(pl.String),
                              pl.lit("|"), pl.col("s1").cast(pl.String)]).hash(seed=SEED)
    keep_expr = pl.col("is_true") | ((row_hash % 1_000_000) < int(frac * 1_000_000))

    parts = []
    for i in range(0, len(files), FILES_PER_CHUNK):
        batch = files[i : i + FILES_PER_CHUNK]
        parts.append(pl.scan_parquet(batch).filter(keep_expr).collect())
    df = pl.concat(parts, how="vertical_relaxed")
    del parts

    folds = ev.fold_array()
    fold_col = pl.DataFrame({"s1": np.arange(len(folds), dtype=np.uint32), "fold": folds.astype(np.int8)})
    df = df.join(fold_col, on="s1", how="left")
    assert df["fold"].null_count() == 0
    return df


def train(df: pl.DataFrame):
    tr = df.filter(pl.col("fold") != HOLDOUT_FOLD)
    ho = df.filter(pl.col("fold") == HOLDOUT_FOLD)
    Xtr, ytr = tr.select(FEATURE_COLS).to_numpy(), tr["is_true"].to_numpy().astype(np.int32)
    Xho, yho = ho.select(FEATURE_COLS).to_numpy(), ho["is_true"].to_numpy().astype(np.int32)
    print(f"[ranker_model] train rows={len(ytr):,} pos_rate={ytr.mean():.4f}  "
          f"holdout rows={len(yho):,} pos_rate={yho.mean():.4f}")

    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=FEATURE_COLS)
    dho = lgb.Dataset(Xho, label=yho, feature_name=FEATURE_COLS, reference=dtr)
    booster = lgb.train(
        PARAMS, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dho], valid_names=["holdout"],
        callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(100)],
    )
    print(f"[ranker_model] best_iteration={booster.best_iteration}")
    imp = sorted(zip(FEATURE_COLS, booster.feature_importance(importance_type="gain")), key=lambda x: -x[1])
    print("[ranker_model] top 10 feature importances (gain):")
    for name, val in imp[:10]:
        print(f"  {name:<22} {val:,.0f}")
    return booster


def evaluate_on_wide_holdout(booster: lgb.Booster) -> None:
    """Re-rank a sample of fold-4 S1 entities' FULL wide candidate set (not the training
    sample) by ranker probability, sweep floor/alpha/m, compare oracle/size to frozen v2."""
    folds = ev.fold_array()
    ho_s1 = np.flatnonzero(folds == HOLDOUT_FOLD).astype(np.uint32)
    rng = np.random.default_rng(SEED)
    eval_s1 = rng.choice(ho_s1, size=min(EVAL_N_S1, len(ho_s1)), replace=False)

    files = sorted((OD / "features_parts").glob("*.parquet"))
    eval_s1_set = pl.Series(eval_s1)
    parts = []
    for i in range(0, len(files), FILES_PER_CHUNK):
        batch = files[i : i + FILES_PER_CHUNK]
        parts.append(pl.scan_parquet(batch).filter(pl.col("s1").is_in(eval_s1_set)).collect())
    df = pl.concat(parts, how="vertical_relaxed")
    del parts
    print(f"[ranker_model] wide holdout sample: s1={len(eval_s1):,}  rows={df.height:,}")

    prob = booster.predict(df.select(FEATURE_COLS).to_numpy())
    df = (
        df.drop("rank").rename({"score": "old_score"})
        .with_columns(score=pl.Series("score", prob))
        .sort(["src", "qi", "score"], descending=[False, False, True])
        .with_columns(
            rank=pl.int_range(pl.len()).over(["src", "qi"]).cast(pl.UInt8),
            best=pl.col("score").max().over(["src", "qi"]),
        )
    )

    is_true = df["is_true"].to_numpy()
    s1_arr = df["s1"].to_numpy().astype(np.int64)
    n_s1 = int(folds.shape[0])
    ntrue = np.bincount(s1_arr[is_true], minlength=n_s1).astype(np.float64)
    mask = np.zeros(n_s1, bool)
    mask[eval_s1] = True

    print("\n[ranker_model] floor/alpha/m grid on ranker-reranked wide holdout "
          "(compare to frozen v2: oracle=0.9893 mean=5.40 at floor=0.5 alpha=0.95 m=3):")
    print(f"  {'floor':>6} {'alpha':>6} {'m':>3}   {'oracleF05':>9}   {'cands/S1 mean':>13}")
    for floor, alpha, m in GRID:
        keep = keep_mask(df, floor, alpha, m)
        s1k = s1_arr[keep]
        tp = np.bincount(s1k, weights=is_true[keep].astype(np.float64), minlength=n_s1)
        npred = np.bincount(s1k, minlength=n_s1).astype(np.float64)
        f = ev.per_s1_f05(tp, tp, ntrue)  # oracle: prediction = candidates themselves
        cs = ev.candidate_stats(npred, mask)
        print(f"  {floor:>6} {alpha:>6} {m:>3}   {f[mask].mean():>9.4f}   {cs['mean']:>13.2f}")


def main() -> None:
    t0 = time.perf_counter()
    df = load_sampled()
    booster = train(df)
    del df
    evaluate_on_wide_holdout(booster)
    md = DATASET_DIR / "cache" / "blocking_v2_ranker" / "model"
    md.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(md / "ranker.txt"))
    print(f"\n[ranker_model] saved -> {md / 'ranker.txt'} ({time.perf_counter() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
