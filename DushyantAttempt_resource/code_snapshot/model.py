"""LightGBM pairwise match classifier (CLAUDE.md D4), MIT licensed, trained on
features.py's output.

Uses the S1-level GroupKFold from cv_harness.fold_array(): folds 0-3 train, fold 4
is a true holdout used only for early stopping, so no fold-4 S1's candidate rows
ever enter training of the CV model. That CV model (``lgbm_cv.txt``) is what
decide.py tunes the decision threshold against, since it never saw fold 4.

The CV model's best_iteration is then reused, unchanged, to retrain on every fold
(``lgbm_full.txt``) -- the model make_submission.py applies to test. Nothing here
is fit on test (G1): features, folds and hyperparameters all come from train.

Fixed hyperparameters and a fixed seed (see PARAMS/SEED); no tuning beyond the
early-stopping round count.

    python model.py --smoke 20000     # dev sample
    python model.py                   # full train (run yourself; slow, ~12.8M rows)
    python model.py --version v2      # v2 features, saved to dataset/cache/blocking_v2/model/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

import cv_harness as ev
from blocking import out_dir
from data import DATASET_DIR
from features import FEATURE_COLS, build_features

SEED = 42
PARAMS = dict(
    objective="binary",
    num_leaves=127,
    learning_rate=0.05,
    feature_fraction=0.8,
    seed=SEED,
    deterministic=True,
    verbose=-1,
)
MAX_ROUNDS = 2000
EARLY_STOP = 50
HOLDOUT_FOLD = 4


def model_dir(version: str = "v1") -> Path:
    return DATASET_DIR / "cache" / f"blocking_{version}" / "model"


def load_features(smoke: int | None = None, version: str = "v1") -> pl.DataFrame:
    path = out_dir("train", smoke, version) / "features.parquet"
    if not path.exists():
        build_features("train", smoke, version)
        path = out_dir("train", smoke, version) / "features.parquet"
    return pl.read_parquet(path)


def add_fold(df: pl.DataFrame) -> pl.DataFrame:
    folds = ev.fold_array()  # per train S1 row index
    fold_col = pl.DataFrame({"s1": np.arange(len(folds), dtype=np.uint32), "fold": folds.astype(np.int8)})
    return df.join(fold_col, on="s1", how="left")


def to_xy(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df.select(FEATURE_COLS).to_numpy()
    y = df["is_true"].to_numpy().astype(np.int32)
    return X, y


def train(smoke: int | None = None, note: str = "", version: str = "v1") -> None:
    t0 = time.perf_counter()
    df = add_fold(load_features(smoke, version))
    assert df["fold"].null_count() == 0, "every candidate's s1 must have a fold"

    tr = df.filter(pl.col("fold") != HOLDOUT_FOLD)
    ho = df.filter(pl.col("fold") == HOLDOUT_FOLD)
    Xtr, ytr = to_xy(tr)
    Xho, yho = to_xy(ho)
    print(f"[model] train rows={len(ytr):,} pos_rate={ytr.mean():.4f}  "
          f"holdout(fold {HOLDOUT_FOLD}) rows={len(yho):,} pos_rate={yho.mean():.4f}")

    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=FEATURE_COLS)
    dho = lgb.Dataset(Xho, label=yho, feature_name=FEATURE_COLS, reference=dtr)
    cv_booster = lgb.train(
        PARAMS, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dho], valid_names=["holdout"],
        callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(100)],
    )
    best_iter = cv_booster.best_iteration
    print(f"[model] best_iteration={best_iter}")

    Xall, yall = to_xy(df)
    dall = lgb.Dataset(Xall, label=yall, feature_name=FEATURE_COLS)
    full_booster = lgb.train(PARAMS, dall, num_boost_round=best_iter)

    imp = sorted(
        zip(FEATURE_COLS, full_booster.feature_importance(importance_type="gain")), key=lambda x: -x[1]
    )
    print("[model] top 15 feature importances (gain, full-train model):")
    for name, val in imp[:15]:
        print(f"  {name:<22} {val:,.0f}")

    md = model_dir(version)
    md.mkdir(parents=True, exist_ok=True)
    cv_booster.save_model(str(md / "lgbm_cv.txt"))
    full_booster.save_model(str(md / "lgbm_full.txt"))
    (md / "meta.json").write_text(json.dumps(
        {"feature_cols": FEATURE_COLS, "best_iteration": best_iter, "seed": SEED,
         "params": PARAMS, "holdout_fold": HOLDOUT_FOLD, "version": version},
        indent=2,
    ))
    print(f"[model] saved cv/full boosters + meta.json -> "
          f"{md.relative_to(DATASET_DIR.parent)} ({time.perf_counter() - t0:.0f}s)")

    ev.log_experiment(
        note or f"lightgbm matcher version={version} num_leaves={PARAMS['num_leaves']} "
                f"lr={PARAMS['learning_rate']} best_iter={best_iter}",
        {"best_iteration": best_iter},
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=None, help="train only: sample this many S1 (via features.py)")
    ap.add_argument("--version", choices=["v1", "v2"], default="v1")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    train(args.smoke, args.note, args.version)
