"""Hyperparameter sweep for the LightGBM matcher -- never actually tuned this session: every
prior "improve the matcher" attempt (T1/T2/T4/T5/T6 entity-centric decision, T3 sample
weighting) kept the exact same fixed PARAMS (num_leaves=127, lr=0.05, feature_fraction=0.8)
and only changed post-hoc decision logic on top of the SAME trained model. This searches the
model's own hyperparameters instead.

Searches on a random S1 sample (fast per-config training, same GroupKFold fold assignment
as the real thing) using the real macro F0.5 metric via decide.py's own evaluation helpers
(not logloss, which doesn't necessarily track the one-owner-threshold F0.5 optimum). Reports
a ranked sweep table only -- promoting a winning config to a full retrain is a separate,
explicit step (model.py itself, with PARAMS edited), never automatic, so the already-
validated v2r model is never at risk of being silently overwritten by a config that only
looked good on the sample.

    python model_tune.py --version v2r --sample-s1 400000
"""

from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import decide as dec
import evaluate as official
from data import load_ground_truth_pairs
from features import FEATURE_COLS
from model import EARLY_STOP, MAX_ROUNDS, SEED, add_fold, load_features, HOLDOUT_FOLD

GRID = [
    dict(),  # baseline: num_leaves=127 lr=0.05 feature_fraction=0.8 (already known full-scale: f05=0.9661)
    dict(num_leaves=255),
    dict(num_leaves=63),
    dict(num_leaves=31),
    dict(learning_rate=0.03),
    dict(learning_rate=0.08),
    dict(learning_rate=0.1),
    dict(feature_fraction=1.0),
    dict(feature_fraction=0.6),
    dict(min_data_in_leaf=100),
    dict(min_data_in_leaf=10),
    dict(lambda_l2=1.0),
    dict(lambda_l1=0.1, lambda_l2=0.1),
    dict(num_leaves=255, learning_rate=0.03, min_data_in_leaf=50),
    dict(bagging_fraction=0.8, bagging_freq=1),
]
DEFAULTS = dict(num_leaves=127, learning_rate=0.05, feature_fraction=0.8)
BASE_PARAMS = dict(objective="binary", seed=SEED, deterministic=True, verbose=-1)


def sample_by_s1(df: pl.DataFrame, n_s1: int, seed: int = SEED) -> pl.DataFrame:
    uniq = df["s1"].unique().to_numpy()
    rng = np.random.default_rng(seed)
    pick = rng.choice(uniq, size=min(n_s1, len(uniq)), replace=False)
    return df.filter(pl.col("s1").is_in(pl.Series(pick)))


def eval_config(df: pl.DataFrame, overrides: dict, s1_ids, q_ids, truth) -> dict:
    tr = df.filter(pl.col("fold") != HOLDOUT_FOLD)
    ho = df.filter(pl.col("fold") == HOLDOUT_FOLD)
    Xtr, ytr = tr.select(FEATURE_COLS).to_numpy(), tr["is_true"].to_numpy().astype(np.int32)
    Xho, yho = ho.select(FEATURE_COLS).to_numpy(), ho["is_true"].to_numpy().astype(np.int32)

    params = {**BASE_PARAMS, **DEFAULTS, **overrides}
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=FEATURE_COLS)
    dho = lgb.Dataset(Xho, label=yho, feature_name=FEATURE_COLS, reference=dtr)
    t0 = time.perf_counter()
    booster = lgb.train(
        params, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dho], valid_names=["holdout"],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )
    prob = booster.predict(ho.select(FEATURE_COLS).to_numpy())
    universe = ho.select("s1").unique().join(s1_ids, on="s1")["entity_id"]
    top1 = dec.top1_per_record(ho, prob)
    best_t, best_f05 = -1.0, -1.0
    for t in dec.T_GRID:
        pred = dec.predictions_at(top1, t, s1_ids, q_ids)
        f05 = official.macro_f05(truth, pred, universe)
        if f05 > best_f05:
            best_t, best_f05 = t, f05
    return {
        "f05": best_f05, "t": best_t, "best_iteration": booster.best_iteration,
        "time": time.perf_counter() - t0,
    }


def main(version: str, sample_s1: int) -> None:
    t0 = time.perf_counter()
    df = add_fold(load_features(None, version))
    df = sample_by_s1(df, sample_s1)
    print(f"[model_tune] sample: {df['s1'].n_unique():,} S1, {df.height:,} rows "
          f"({time.perf_counter()-t0:.0f}s to load+sample)")

    s1_ids, q_ids = dec.id_maps("train")
    truth = load_ground_truth_pairs()

    results = []
    for i, overrides in enumerate(GRID):
        res = eval_config(df, overrides, s1_ids, q_ids, truth)
        res["overrides"] = overrides or "baseline"
        results.append(res)
        print(f"[{i+1}/{len(GRID)}] {res['overrides']} -> f05={res['f05']:.4f} t={res['t']:.2f} "
              f"best_iter={res['best_iteration']} ({res['time']:.0f}s)", flush=True)

    results.sort(key=lambda r: -r["f05"])
    print("\n[model_tune] ranked results (best first):")
    for r in results:
        print(f"  f05={r['f05']:.4f}  best_iter={r['best_iteration']:<5}  {r['overrides']}")
    print(f"\n[model_tune] total {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v1", "v2", "v2r"], default="v2r")
    ap.add_argument("--sample-s1", type=int, default=400_000)
    args = ap.parse_args()
    main(args.version, args.sample_s1)
