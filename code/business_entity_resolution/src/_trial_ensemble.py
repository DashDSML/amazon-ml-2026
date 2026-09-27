"""Same-family GBM ensemble (different seeds, same tuned PARAMS) as a genuine 'second
opinion' -- averaging probabilities across models trained on different random subsamples/
splits (feature_fraction/bagging randomness differs by seed) can reduce variance even when
architecture is identical. Uses the CV-holdout pattern only (no full-data retrain needed to
test whether ensembling helps in principle) -- fast, isolated, writes nothing to the live
v2r model.
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import polars as pl

import decide as dec
import evaluate as official
from data import load_ground_truth_pairs
from features import FEATURE_COLS
from model import EARLY_STOP, MAX_ROUNDS, PARAMS, add_fold, load_features, HOLDOUT_FOLD, model_dir

SEEDS = [123, 456]


def best_f05(ho, prob, s1_ids, q_ids, truth):
    universe = ho.select("s1").unique().join(s1_ids, on="s1")["entity_id"]
    top1 = dec.top1_per_record(ho, prob)
    best_t, best = -1.0, -1.0
    for t in dec.T_GRID:
        pred = dec.predictions_at(top1, t, s1_ids, q_ids)
        f = official.macro_f05(truth, pred, universe)
        if f > best:
            best_t, best = t, f
    return best_t, best


def main() -> None:
    t0 = time.perf_counter()
    df = add_fold(load_features(None, "v2r"))
    tr = df.filter(pl.col("fold") != HOLDOUT_FOLD)
    ho = df.filter(pl.col("fold") == HOLDOUT_FOLD)
    Xtr, ytr = tr.select(FEATURE_COLS).to_numpy(), tr["is_true"].to_numpy().astype(np.int32)
    Xho = ho.select(FEATURE_COLS).to_numpy()
    print(f"[ensemble] loaded {len(ytr):,} train / {len(Xho):,} holdout rows ({time.perf_counter()-t0:.0f}s)")

    s1_ids, q_ids = dec.id_maps("train")
    truth = load_ground_truth_pairs()

    live_booster = lgb.Booster(model_file=str(model_dir("v2r") / "lgbm_cv.txt"))
    probs = {"live(seed42)": live_booster.predict(Xho)}
    t_live, f_live = best_f05(ho, probs["live(seed42)"], s1_ids, q_ids, truth)
    print(f"[ensemble] live alone: f05={f_live:.4f} t={t_live:.2f}")

    for seed in SEEDS:
        t1 = time.perf_counter()
        params = {**PARAMS, "seed": seed}
        dtr = lgb.Dataset(Xtr, label=ytr, feature_name=FEATURE_COLS)
        dho_ds = lgb.Dataset(Xho, label=ho["is_true"].to_numpy().astype(np.int32), feature_name=FEATURE_COLS, reference=dtr)
        booster = lgb.train(
            params, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dho_ds], valid_names=["holdout"],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
        )
        prob = booster.predict(Xho)
        probs[f"seed{seed}"] = prob
        t_s, f_s = best_f05(ho, prob, s1_ids, q_ids, truth)
        print(f"[ensemble] seed{seed} alone: f05={f_s:.4f} t={t_s:.2f} best_iter={booster.best_iteration} "
              f"({time.perf_counter()-t1:.0f}s)")

    avg_all = np.mean(list(probs.values()), axis=0)
    t_a, f_a = best_f05(ho, avg_all, s1_ids, q_ids, truth)
    print(f"[ensemble] average of all {len(probs)}: f05={f_a:.4f} t={t_a:.2f}")

    for seed in SEEDS:
        avg2 = (probs["live(seed42)"] + probs[f"seed{seed}"]) / 2
        t_2, f_2 = best_f05(ho, avg2, s1_ids, q_ids, truth)
        print(f"[ensemble] live + seed{seed} avg: f05={f_2:.4f} t={t_2:.2f}")

    print(f"[ensemble] total {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
