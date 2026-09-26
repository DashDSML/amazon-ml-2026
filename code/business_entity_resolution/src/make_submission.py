"""Decide the entity-level matches and write the submission (CLAUDE.md D6).

Candidates come from blocking (dataset/cache/blocking_{version}/{split}/candidates.parquet,
columns src, qi, s1, score, name_s, addr_s, rank -- already the final candidate set, one
row per (S2/S3 record, its rank-th best S1) up to m=3). --version {v1,v2} (default v1)
selects which blocking's candidates/features/model to use; v1 is unchanged unless
requested (lb-2 depends on it staying that way).

Two decision methods, chosen with --method:

  rule (default): keep a pair if it is the record's rank-0 (blocking-score) destination
    with score >= t1, optionally also rank-1 with a stricter t2. t1/t2 are swept on TRAIN
    only against real macro F0.5 (not the blocking oracle).

  model: use the LightGBM matcher (model.py) -- each record's highest-probability
    candidate, accepted iff probability >= the threshold decide.py tuned on the fold-4
    holdout (dataset/cache/blocking_{version}/model/decision.json). Requires test features
    already built (python features.py --split test --version {version}) and the model
    already trained (python model.py --version {version}).

Either way: candidate_pairs.tsv is always the full test candidate set (unfiltered), the
decision is written to matching_results.tsv, matches are checked with
validate_submission.py, and the run is logged.

    python make_submission.py                              # v1 rule (tunes on train, ~1-2 min)
    python make_submission.py --method model                # v1 model (needs features/model built first)
    python make_submission.py --version v2 --method model   # v2 model
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

import cv_harness as ev
import decide
from blocking import make_ctx, out_dir
from data import DATASET_DIR, REPO_ROOT
from model import model_dir

T_GRID = [round(0.5 + 0.025 * i, 3) for i in range(19)]  # 0.500 .. 0.950
OUT_DIR = REPO_ROOT / "output"


def load_candidates(split: str, version: str = "v1") -> pl.DataFrame:
    return pl.read_parquet(out_dir(split, None, version) / "candidates.parquet")


# ---------------------------------------------------------------- method: rule
def eval_mask(s1: np.ndarray, is_true: np.ndarray, ctx: ev.EvalCtx, mask: np.ndarray) -> dict:
    return ev.score_pairs(s1[mask], is_true[mask], ctx, oracle=False)


def sweep_rank0(rank: np.ndarray, score: np.ndarray, s1: np.ndarray, is_true: np.ndarray, ctx: ev.EvalCtx):
    print("=== sweep: keep pair iff rank==0 and score>=t1 ===")
    best_t, best_f05 = None, -1.0
    for t in T_GRID:
        mask = (rank == 0) & (score >= t)
        res = eval_mask(s1, is_true, ctx, mask)
        print(f"  t1={t:.3f}  f05={res['f05']:.4f}  cands/S1={res['mean']:.2f}")
        if res["f05"] > best_f05:
            best_t, best_f05 = t, res["f05"]
    print(f"-> best t1={best_t:.3f}  f05={best_f05:.4f}")
    return best_t, best_f05


def sweep_rank1(rank: np.ndarray, score: np.ndarray, s1: np.ndarray, is_true: np.ndarray, ctx: ev.EvalCtx, t1: float):
    print(f"\n=== sweep: also allow rank==1 with score>=t2 (t1 fixed at {t1:.3f}) ===")
    base = (rank == 0) & (score >= t1)
    best_t2, best_f05 = None, -1.0
    for t2 in [t for t in T_GRID if t >= t1]:
        mask = base | ((rank == 1) & (score >= t2))
        res = eval_mask(s1, is_true, ctx, mask)
        print(f"  t2={t2:.3f}  f05={res['f05']:.4f}  cands/S1={res['mean']:.2f}")
        if res["f05"] > best_f05:
            best_t2, best_f05 = t2, res["f05"]
    print(f"-> best t2={best_t2:.3f}  f05={best_f05:.4f}")
    return best_t2, best_f05


def report(name: str, res: dict, ctx: ev.EvalCtx) -> None:
    print(f"\n--- breakdown: {name} ---")
    print(f"  overall f05={res['f05']:.4f}  pair_recall={res['pair_recall']:.4f}  cands/S1 mean={res['mean']:.2f}")
    for line in ev.breakdown_lines(res, ctx):
        print(line)


def apply_rule(cands_test: pl.DataFrame, version: str = "v1") -> pl.DataFrame:
    """Tune the rank/score rule on TRAIN, then apply the chosen rule to TEST candidates."""
    truth = ev.load_truth_idx()
    cands_train = load_candidates("train", version)
    df = cands_train.join(
        truth.with_columns(pl.lit(True).alias("is_true")), on=["src", "qi", "s1"], how="left"
    ).with_columns(pl.col("is_true").fill_null(False))
    ctx = make_ctx("train", out_dir("train", None, version), truth)
    s1 = df["s1"].to_numpy().astype(np.int64)
    rank = df["rank"].to_numpy()
    score = df["score"].to_numpy()
    is_true = df["is_true"].to_numpy()

    t1_best, f05_a = sweep_rank0(rank, score, s1, is_true, ctx)
    mask_a = (rank == 0) & (score >= t1_best)
    res_a = eval_mask(s1, is_true, ctx, mask_a)
    report(f"rank==0, t1={t1_best:.3f}", res_a, ctx)

    t2_best, f05_b = sweep_rank1(rank, score, s1, is_true, ctx, t1_best)
    mask_b = mask_a | ((rank == 1) & (score >= t2_best))
    res_b = eval_mask(s1, is_true, ctx, mask_b)
    report(f"rank<=1, t1={t1_best:.3f} t2={t2_best:.3f}", res_b, ctx)

    if f05_b > f05_a:
        rule, res = ("rank01", t1_best, t2_best), res_b
        print(f"\nchosen: rank<=1 (t1={t1_best:.3f}, t2={t2_best:.3f}) beats rank==0 ({f05_b:.4f} > {f05_a:.4f})")
    else:
        rule, res = ("rank0", t1_best, None), res_a
        print(f"\nchosen: rank==0 (t1={t1_best:.3f}); rank<=1 did not improve ({f05_b:.4f} <= {f05_a:.4f})")

    ev.log_experiment(
        f"make_submission version={version} rule={rule[0]} t1={rule[1]:.3f} t2={rule[2]}",
        {"f05": res["f05"], "pair_recall": res["pair_recall"], "cands_mean": res["mean"]},
    )

    rank_t, score_t = cands_test["rank"].to_numpy(), cands_test["score"].to_numpy()
    if rule[0] == "rank0":
        mask_t = (rank_t == 0) & (score_t >= rule[1])
    else:
        mask_t = ((rank_t == 0) & (score_t >= rule[1])) | ((rank_t == 1) & (score_t >= rule[2]))
    return cands_test.filter(pl.Series(mask_t))


# ---------------------------------------------------------------- method: model
def apply_model(cands_test: pl.DataFrame, version: str = "v1") -> pl.DataFrame:
    """LightGBM matcher (model.py) + decide.py's tuned one-owner threshold, applied to TEST."""
    feat_path = out_dir("test", None, version) / "features.parquet"
    assert feat_path.exists(), f"missing {feat_path}; run: python features.py --split test --version {version}"
    md = model_dir(version)
    meta = json.loads((md / "meta.json").read_text())
    decision_path = md / "decision.json"
    assert decision_path.exists(), f"missing {decision_path}; run: python decide.py --version {version}"
    decision = json.loads(decision_path.read_text())

    feat = pl.read_parquet(feat_path)
    booster = lgb.Booster(model_file=str(md / "lgbm_full.txt"))
    prob = booster.predict(feat.select(meta["feature_cols"]).to_numpy())
    top1 = decide.top1_per_record(feat, prob)
    accepted = top1.filter(pl.col("prob") >= decision["t"]).select("src", "qi", "s1")
    print(
        f"[model] t={decision['t']:.2f} (tuned on fold {decision['holdout_fold']}, "
        f"holdout f05={decision['f05']:.4f}, blocking oracle ceiling={decision['ceiling']:.4f}) "
        f"-> {accepted.height:,} matched pairs"
    )
    # semi join: keeps cands_test's own columns/order, guarantees matches subset candidates
    return cands_test.join(accepted, on=["src", "qi", "s1"], how="semi")


# ---------------------------------------------------------------- shared: id mapping + output
def build_id_lists(cands: pl.DataFrame, split: str, id_col: str) -> pl.DataFrame:
    """One row per test S1 (entity_id, comma-joined ids), including S1 with no rows in cands."""
    s1_ids = pl.scan_parquet(ev.norm_path(split, 1)).select("entity_id").with_row_index("s1").collect()
    parts = []
    for src in (2, 3):
        q_ids = pl.scan_parquet(ev.norm_path(split, src)).select("entity_id").with_row_index("qi").collect()
        part = cands.filter(pl.col("src") == src).join(q_ids, on="qi", how="left")
        assert part["entity_id"].null_count() == 0, f"unmapped qi in source{src}"
        parts.append(part.select("s1", "entity_id"))
    long = pl.concat(parts)
    grouped = (
        long.group_by("s1")
        .agg(pl.col("entity_id").unique().sort().alias(id_col))
        .with_columns(pl.col(id_col).list.join(",").alias(id_col))
    )
    return (
        s1_ids.join(grouped, on="s1", how="left")
        .with_columns(pl.col(id_col).fill_null(""))
        .select(pl.col("entity_id").alias("source1_entity_id"), id_col)
        .sort("source1_entity_id")
    )


def write_tsv(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path, separator="\t", quote_style="never", include_bom=False)
    print(f"wrote {df.height:,} rows -> {path.relative_to(REPO_ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["rule", "model"], default="rule")
    ap.add_argument("--version", choices=["v1", "v2"], default="v1")
    args = ap.parse_args()

    cands_test = load_candidates("test", args.version)
    if args.method == "model":
        pred_test = apply_model(cands_test, args.version)
    else:
        pred_test = apply_rule(cands_test, args.version)
    print(f"\ntest: version={args.version} {cands_test.height:,} candidate pairs -> "
          f"{pred_test.height:,} matched pairs")

    matching = build_id_lists(pred_test, "test", "matched_entity_ids")
    candidates_out = build_id_lists(cands_test, "test", "candidate_entity_ids")
    write_tsv(matching, OUT_DIR / "matching_results.tsv")
    write_tsv(candidates_out, OUT_DIR / "candidate_pairs.tsv")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "utils" / "validate_submission.py"),
            "--matching", str(OUT_DIR / "matching_results.tsv"),
            "--candidate", str(OUT_DIR / "candidate_pairs.tsv"),
            "--test-dir", str(DATASET_DIR / "test"),
        ],
        capture_output=True,
        text=True,
    )
    print("\n" + result.stdout)
    if result.stderr:
        print(result.stderr)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
