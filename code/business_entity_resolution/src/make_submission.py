"""Tune the entity-level decision rule on TRAIN, apply it to TEST, write the submission.

Candidates come from blocking v1 (dataset/cache/blocking_v1/{split}/candidates.parquet,
columns src, qi, s1, score, name_s, addr_s, rank -- already the final candidate set, one
row per (S2/S3 record, its rank-th best S1) up to m=3). The decision rule keeps a pair if
it is the record's rank-0 (best) destination with score >= t1, and optionally also keeps
rank-1 with a stricter score >= t2 when that improves CV.

t1 is chosen by sweeping real macro F0.5 (not the blocking oracle) on TRAIN only, over the
full candidate set; nothing is fit on TEST. The chosen rule is applied to the TEST
candidates, mapped back to entity IDs, and written as output/matching_results.tsv (the
decision) and output/candidate_pairs.tsv (the full test candidate set), then checked with
utils/validate_submission.py.

    python make_submission.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

import cv_harness as ev
from blocking import make_ctx, out_dir
from data import DATASET_DIR, REPO_ROOT

T_GRID = [round(0.5 + 0.025 * i, 3) for i in range(19)]  # 0.500 .. 0.950
OUT_DIR = REPO_ROOT / "output"


def eval_mask(s1: np.ndarray, is_true: np.ndarray, ctx: ev.EvalCtx, mask: np.ndarray) -> dict:
    return ev.score_pairs(s1[mask], is_true[mask], ctx, oracle=False)


def sweep_rank0(rank: np.ndarray, score: np.ndarray, s1: np.ndarray, is_true: np.ndarray, ctx: ev.EvalCtx):
    print("=== sweep: keep pair iff rank==0 and score>=t1 ===")
    best_t, best_f05, rows = None, -1.0, []
    for t in T_GRID:
        mask = (rank == 0) & (score >= t)
        res = eval_mask(s1, is_true, ctx, mask)
        rows.append((t, res["f05"], res["mean"]))
        print(f"  t1={t:.3f}  f05={res['f05']:.4f}  cands/S1={res['mean']:.2f}")
        if res["f05"] > best_f05:
            best_t, best_f05 = t, res["f05"]
    print(f"-> best t1={best_t:.3f}  f05={best_f05:.4f}")
    return best_t, best_f05


def sweep_rank1(rank: np.ndarray, score: np.ndarray, s1: np.ndarray, is_true: np.ndarray, ctx: ev.EvalCtx, t1: float):
    print(f"\n=== sweep: also allow rank==1 with score>=t2 (t1 fixed at {t1:.3f}) ===")
    base = (rank == 0) & (score >= t1)
    best_t2, best_f05, rows = None, -1.0, []
    for t2 in [t for t in T_GRID if t >= t1]:
        mask = base | ((rank == 1) & (score >= t2))
        res = eval_mask(s1, is_true, ctx, mask)
        rows.append((t2, res["f05"], res["mean"]))
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


def load_candidates(split: str) -> pl.DataFrame:
    return pl.read_parquet(out_dir(split, None) / "candidates.parquet")


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
    truth = ev.load_truth_idx()
    cands_train = load_candidates("train")
    df = cands_train.join(
        truth.with_columns(pl.lit(True).alias("is_true")), on=["src", "qi", "s1"], how="left"
    ).with_columns(pl.col("is_true").fill_null(False))
    ctx = make_ctx("train", out_dir("train", None), truth)
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
        f"make_submission rule={rule[0]} t1={rule[1]:.3f} t2={rule[2]}",
        {"f05": res["f05"], "pair_recall": res["pair_recall"], "cands_mean": res["mean"]},
    )

    # ---- apply chosen rule to test ----
    cands_test = load_candidates("test")
    rank_t, score_t = cands_test["rank"].to_numpy(), cands_test["score"].to_numpy()
    if rule[0] == "rank0":
        mask_t = (rank_t == 0) & (score_t >= rule[1])
    else:
        mask_t = ((rank_t == 0) & (score_t >= rule[1])) | ((rank_t == 1) & (score_t >= rule[2]))
    pred_test = cands_test.filter(pl.Series(mask_t))
    print(f"\ntest: {cands_test.height:,} candidate pairs -> {pred_test.height:,} matched pairs")

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
