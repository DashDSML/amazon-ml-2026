"""LightGBM matcher retrained with metric-aware sample weights (NonStandard.md T3).

Plain log-loss treats every pair mistake as equal; the metric does not. Derived directly
from the F0.5 formula (matches every value in challengecontext.md's worked table exactly):
  negative pair, S1 has t true matches: cost of one avoidable extra wrong ID once the rest
    are right = 1/(1.25t + 1) for t>=1, or 1.0 for a true singleton (t=0) -- any match at
    all on a singleton costs the full point.
  positive pair, S1 has t true matches: cost of missing just this one once the rest are
    right = 0.25/(1.25t - 1) for t>=1 (t=1 reduces to the "miss the only match" cost, 1.0).
Weights are normalised to mean 1 (so the loss scale is comparable to the unweighted model)
and passed to LightGBM's sample_weight. Everything else (features, folds, params, decision
layer) is identical to model.py/decide.py, so any change in CV F0.5 is attributable to the
weighting alone.

Adopted only if it beats the plain model on the same fold-4 holdout, evaluated with the
same decide.py record-centric rule -- compare *after* the decision layer (T3's own risk
note), not on raw loss.

    python model_weighted.py --version v1
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
from decide import T_GRID, id_maps, predictions_at, top1_per_record
from features import FEATURE_COLS
from model import EARLY_STOP, HOLDOUT_FOLD, MAX_ROUNDS, PARAMS, add_fold, load_features, model_dir


def sample_weights(df: pl.DataFrame) -> np.ndarray:
    # true cardinality per S1 (real ground truth, not just how many true matches happen to
    # be present in the candidate set -- those can differ when blocking misses one)
    truth = ev.load_truth_idx()
    t = truth.group_by("s1").agg(pl.len().alias("_t"))
    d = df.join(t, on="s1", how="left").with_columns(pl.col("_t").fill_null(0))
    tt = d["_t"].to_numpy().astype(np.float64)
    is_true = d["is_true"].to_numpy()
    w = np.where(
        is_true,
        np.where(tt >= 1, 0.25 / np.maximum(1.25 * tt - 1, 1e-6), 1.0),
        np.where(tt >= 1, 1.0 / (1.25 * tt + 1), 1.0),
    )
    return (w / w.mean()).astype(np.float64)


def to_xyw(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = df.select(FEATURE_COLS).to_numpy()
    y = df["is_true"].to_numpy().astype(np.int32)
    w = sample_weights(df)
    return X, y, w


def main(version: str = "v1") -> None:
    df = add_fold(load_features(None, version))
    tr = df.filter(pl.col("fold") != HOLDOUT_FOLD)
    ho = df.filter(pl.col("fold") == HOLDOUT_FOLD)
    Xtr, ytr, wtr = to_xyw(tr)
    Xho, yho, _ = to_xyw(ho)
    print(f"[model_weighted] train rows={len(ytr):,}  weight range=[{wtr.min():.3f},{wtr.max():.3f}] mean={wtr.mean():.3f}")

    dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr, feature_name=FEATURE_COLS)
    dho = lgb.Dataset(Xho, label=yho, feature_name=FEATURE_COLS, reference=dtr)
    booster = lgb.train(
        PARAMS, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dho], valid_names=["holdout"],
        callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(0)],
    )
    print(f"[model_weighted] best_iteration={booster.best_iteration}")

    prob = booster.predict(Xho)
    top1 = top1_per_record(ho, prob)
    s1_ids, q_ids = id_maps("train")
    truth = load_ground_truth_pairs()
    universe = ho.select("s1").unique().join(s1_ids, on="s1")["entity_id"]

    best_t, best_f = None, -1.0
    for t in T_GRID:
        pred = predictions_at(top1, t, s1_ids, q_ids)
        f = official.macro_f05(truth, pred, universe)
        if f > best_f:
            best_f, best_t = f, t
    print(f"[model_weighted] best t={best_t:.2f}  f05={best_f:.4f}  (compare to decide.py's unweighted result)")

    ev.log_experiment(
        f"model_weighted.py (T3) version={version} best_iter={booster.best_iteration} t={best_t:.2f}",
        {"f05": best_f, "best_iteration": booster.best_iteration},
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v1", "v2"], default="v1")
    args = ap.parse_args()
    main(args.version)
