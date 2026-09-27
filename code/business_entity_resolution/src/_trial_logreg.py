"""Fast logistic regression on v2r's train features, evaluated the same way as the LightGBM
matcher (real macro F0.5 via decide.py's helpers, not accuracy/logloss) -- both alone and
blended with the current tuned LightGBM's probabilities, to see whether a linear model
captures anything complementary. Standalone, isolated: writes only to
dataset/cache/blocking_v2r_trial2/ -- never touches the live v2r model.
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import decide as dec
import evaluate as official
from data import DATASET_DIR, load_ground_truth_pairs
from features import FEATURE_COLS
from model import HOLDOUT_FOLD, SEED, add_fold, load_features, model_dir


def best_f05(ho: pl.DataFrame, prob: np.ndarray, s1_ids, q_ids, truth) -> tuple[float, float]:
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
    Xtr = tr.select(FEATURE_COLS).to_numpy()
    ytr = tr["is_true"].to_numpy().astype(np.int32)
    Xho = ho.select(FEATURE_COLS).to_numpy()
    print(f"[logreg] loaded {len(ytr):,} train / {len(Xho):,} holdout rows ({time.perf_counter()-t0:.0f}s)")

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(np.nan_to_num(Xtr))
    Xho_s = scaler.transform(np.nan_to_num(Xho))

    t1 = time.perf_counter()
    lr = LogisticRegression(max_iter=200, n_jobs=-1, random_state=SEED)
    lr.fit(Xtr_s, ytr)
    lr_prob = lr.predict_proba(Xho_s)[:, 1]
    print(f"[logreg] fit + predict done ({time.perf_counter()-t1:.0f}s)")

    s1_ids, q_ids = dec.id_maps("train")
    truth = load_ground_truth_pairs()

    t_lr, f_lr = best_f05(ho, lr_prob, s1_ids, q_ids, truth)
    print(f"[logreg] alone: f05={f_lr:.4f} t={t_lr:.2f}")

    booster = lgb.Booster(model_file=str(model_dir("v2r") / "lgbm_cv.txt"))
    gbm_prob = booster.predict(Xho)
    t_gbm, f_gbm = best_f05(ho, gbm_prob, s1_ids, q_ids, truth)
    print(f"[logreg] current live gbm (recompute check): f05={f_gbm:.4f} t={t_gbm:.2f}")

    for w in (0.1, 0.2, 0.3, 0.5):
        blend = (1 - w) * gbm_prob + w * lr_prob
        t_b, f_b = best_f05(ho, blend, s1_ids, q_ids, truth)
        print(f"[logreg] blend w_lr={w:.1f}: f05={f_b:.4f} t={t_b:.2f}")

    print(f"[logreg] total {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
