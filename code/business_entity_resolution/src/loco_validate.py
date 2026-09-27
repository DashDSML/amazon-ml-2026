"""Leave-one-country-out validation (NonStandard.md T24): the France proxy.

Train has only US and India; France is fully unseen (G1: no France row informs anything,
ever). The only way to rehearse "how will the country-agnostic pipeline behave on a country
it has never seen" without touching test is to hold out one of the two countries we DO have
labels for entirely -- train on the other, evaluate on the held-out one exactly as if it were
France. If that number is close to the country's own normal (same-country) CV score, the
pipeline's country-blindness is working as intended. If it is much worse, that is a real,
actionable warning about France risk on the private leaderboard.

Retrains a fresh LightGBM per held-out country (same params/features as model.py; country is
never a feature, only used here to decide which rows are held out) and tunes+evaluates
decide.py's (already-proven-best) record-centric top1-per-record rule on that held-out
country alone, against real macro F0.5.

    python loco_validate.py --version v1
    python loco_validate.py --version v1 --smoke 20000
"""

from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import polars as pl

import cv_harness as ev
import evaluate as official
from data import load_ground_truth_pairs
from decide import T_GRID, id_maps, predictions_at, top1_per_record
from features import FEATURE_COLS, build_features
from model import EARLY_STOP, MAX_ROUNDS, PARAMS
from blocking import out_dir


def load_features_with_country(split: str, smoke: int | None, version: str) -> pl.DataFrame:
    path = out_dir(split, smoke, version) / "features.parquet"
    if not path.exists():
        build_features(split, smoke, version)
        path = out_dir(split, smoke, version) / "features.parquet"
    feat = pl.read_parquet(path)
    s1_country = (
        pl.scan_parquet(ev.norm_path(split, 1))
        .select(pl.col("country").fill_null(""))
        .with_row_index("s1")
        .collect()
    )
    return feat.join(s1_country, on="s1", how="left")


def run_holdout(df: pl.DataFrame, held_out: str, s1_ids, q_ids, truth) -> dict:
    tr = df.filter(pl.col("country") != held_out)
    ho = df.filter(pl.col("country") == held_out)
    Xtr, ytr = tr.select(FEATURE_COLS).to_numpy(), tr["is_true"].to_numpy().astype(np.int32)
    Xho, yho = ho.select(FEATURE_COLS).to_numpy(), ho["is_true"].to_numpy().astype(np.int32)
    print(f"  [{held_out}] train rows={len(ytr):,} (other country)  holdout rows={len(yho):,} ({held_out})")

    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=FEATURE_COLS)
    dho = lgb.Dataset(Xho, label=yho, feature_name=FEATURE_COLS, reference=dtr)
    booster = lgb.train(
        PARAMS, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dho], valid_names=["holdout"],
        callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(0)],
    )
    prob = booster.predict(Xho)
    top1 = top1_per_record(ho, prob)
    universe = ho.select("s1").unique().join(s1_ids, on="s1")["entity_id"]

    best_t, best_f = None, -1.0
    for t in T_GRID:
        pred = predictions_at(top1, t, s1_ids, q_ids)
        f = official.macro_f05(truth, pred, universe)
        if f > best_f:
            best_f, best_t = f, t
    return {"held_out": held_out, "best_iteration": booster.best_iteration, "t": best_t, "f05": best_f,
            "n_holdout_s1": int(universe.len())}


def main(version: str = "v1", smoke: int | None = None) -> None:
    df = load_features_with_country("train", smoke, version)
    s1_ids, q_ids = id_maps("train")
    truth = load_ground_truth_pairs()
    countries = sorted(c for c in df["country"].unique().to_list() if c)
    print(f"[loco] version={version} countries in train={countries}")

    results = []
    for held_out in countries:
        res = run_holdout(df, held_out, s1_ids, q_ids, truth)
        print(f"  -> held-out {held_out}: best t={res['t']:.2f}  f05={res['f05']:.4f} "
              f"(n_s1={res['n_holdout_s1']:,}, best_iter={res['best_iteration']})")
        results.append(res)

    print("\n[loco] summary (compare to each country's normal same-country CV number):")
    for res in results:
        print(f"  {res['held_out']:<8} unseen-country f05={res['f05']:.4f}")

    if not smoke:
        ev.log_experiment(
            f"loco_validate.py version={version} " + " ".join(
                f"{r['held_out']}={r['f05']:.4f}" for r in results
            ),
            {r["held_out"] + "_f05": r["f05"] for r in results},
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v1", "v2"], default="v1")
    ap.add_argument("--smoke", type=int, default=None, help="train only: sample this many S1 (via features.py)")
    args = ap.parse_args()
    main(args.version, args.smoke)
