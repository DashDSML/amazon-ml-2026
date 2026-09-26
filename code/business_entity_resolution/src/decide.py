"""Entity-level decision layer for the LightGBM matcher (CLAUDE.md D5).

One-owner (each S2/S3 record belongs to at most one S1, T30): for every record,
take its highest-probability candidate and accept the match iff that probability
is >= t. t is tuned by sweeping REAL macro F0.5 (src/evaluate.py, the official
metric) on the fold-4 holdout, scored against the model.py CV booster
(``lgbm_cv.txt``, trained on folds 0-3 only, so fold 4 is a genuine holdout).
Nothing here touches test (G1).

Reports overall / per country / singleton vs non-singleton at the chosen t,
alongside the blocking oracle ceiling for context (see ORACLE_F05).

    python decide.py --smoke 20000   # dev sample
    python decide.py                 # full train (run yourself; slow)
"""

from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import polars as pl

import cv_harness as ev
import evaluate as official
from data import load_ground_truth_pairs
from features import FEATURE_COLS
from model import MODEL_DIR, add_fold, load_features

ORACLE_F05 = 0.9778  # blocking v1 full-train grid (floor=0.5 alpha=0.9 m=3), experiments/log.csv
HOLDOUT_FOLD = 4
T_GRID = [round(0.05 * i, 3) for i in range(1, 20)]  # 0.05 .. 0.95


def id_maps(split: str) -> tuple[pl.DataFrame, dict[int, pl.DataFrame]]:
    s1 = pl.scan_parquet(ev.norm_path(split, 1)).select("entity_id", "country").with_row_index("s1").collect()
    q = {
        src: pl.scan_parquet(ev.norm_path(split, src)).select("entity_id").with_row_index("qi").collect()
        for src in (2, 3)
    }
    return s1, q


def top1_per_record(df: pl.DataFrame, prob: np.ndarray) -> pl.DataFrame:
    """(src, qi, s1, prob) restricted to each record's single highest-probability row."""
    d = (
        df.select("src", "qi", "s1")
        .with_columns(prob=pl.Series(prob))
        .sort(["src", "qi", "prob"], descending=[False, False, True])
        .with_columns(_r=pl.int_range(pl.len()).over(["src", "qi"]))
        .filter(pl.col("_r") == 0)
        .drop("_r")
    )
    return d


def predictions_at(top1: pl.DataFrame, t: float, s1_ids: pl.DataFrame, q_ids: dict[int, pl.DataFrame]) -> pl.DataFrame:
    accepted = top1.filter(pl.col("prob") >= t)
    parts = []
    for src in (2, 3):
        part = accepted.filter(pl.col("src") == src).join(q_ids[src], on="qi", how="left")
        parts.append(part.select("s1", pl.col("entity_id").alias("match_id")))
    long = pl.concat(parts)
    return long.join(
        s1_ids.select("s1", pl.col("entity_id").alias("source1_entity_id")), on="s1", how="left"
    ).select("source1_entity_id", "match_id")


def tune_threshold(top1, s1_ids, q_ids, truth, universe) -> tuple[float, float]:
    best_t, best_f05 = T_GRID[0], -1.0
    for t in T_GRID:
        pred = predictions_at(top1, t, s1_ids, q_ids)
        f05 = official.macro_f05(truth, pred, universe)
        print(f"  t={t:.2f}  f05={f05:.4f}")
        if f05 > best_f05:
            best_t, best_f05 = t, f05
    return best_t, best_f05


def main(smoke: int | None = None) -> None:
    df = add_fold(load_features(smoke))
    ho = df.filter(pl.col("fold") == HOLDOUT_FOLD)
    booster = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_cv.txt"))
    prob = booster.predict(ho.select(FEATURE_COLS).to_numpy())

    s1_ids, q_ids = id_maps("train")
    truth = load_ground_truth_pairs()
    universe = ho.select("s1").unique().join(s1_ids, on="s1")["entity_id"]
    top1 = top1_per_record(ho, prob)

    print(f"=== sweep: accept record's top-probability candidate iff prob>=t (fold {HOLDOUT_FOLD} holdout) ===")
    best_t, best_f05 = tune_threshold(top1, s1_ids, q_ids, truth, universe)
    print(f"-> best t={best_t:.2f}  f05={best_f05:.4f}  (blocking oracle ceiling={ORACLE_F05:.4f})")

    pred = predictions_at(top1, best_t, s1_ids, q_ids)
    scores = official.per_entity_scores(truth, pred, universe)
    s1_country = s1_ids.select(pl.col("entity_id").alias("source1_entity_id"), "country")
    report = official.score_report(scores, s1_country)
    print(f"\n--- report at t={best_t:.2f} ---")
    print(f"overall f05={report['overall']:.4f}  blocking oracle ceiling={ORACLE_F05:.4f}")
    print("by country:")
    print(report["by_country"])
    print("singleton vs non-singleton:")
    print(report["by_bucket"])

    (MODEL_DIR / "decision.json").write_text(json.dumps(
        {"t": best_t, "f05": best_f05, "holdout_fold": HOLDOUT_FOLD, "ceiling": ORACLE_F05}, indent=2
    ))
    print(f"\nsaved decision threshold -> {MODEL_DIR / 'decision.json'}")

    ev.log_experiment(f"decide.py t={best_t:.2f}", {"f05": report["overall"], "ceiling": ORACLE_F05})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=None, help="must match the sample used for model.py --smoke")
    args = ap.parse_args()
    main(args.smoke)
