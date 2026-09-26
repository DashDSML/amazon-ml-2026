"""Validation harness (CLAUDE.md "Evaluation").

- Macro F0.5 over S1 (empty truth + empty prediction = 1.0, empty vs non-empty = 0.0),
  reported overall, per country, singleton vs non-singleton and per fold.
- Blocking oracle: prediction = candidates ∩ truth, i.e. the best any scorer can do on
  that candidate set; always reported with mean/median/p95 candidates per S1 (G2, T22).
- GroupKFold by S1: each S2/S3 record belongs to at most one S1 (one-owner), so folding
  S1 keeps every match group intact. The S2/S3 pool is never split: every fold is scored
  against the full distractor pool (T23).
- Experiment log: experiments/log.csv.

Index conventions (shared with blocking.py): S1 records are identified by their row index
``s1`` in norm_source1.parquet; S2/S3 records by (``src`` in {2,3}, row index ``qi``).

    python cv_harness.py --prepare      # build truth index + folds (train only)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from data import DATASET_DIR, REPO_ROOT, load_ground_truth_pairs

BETA2 = 0.25  # F0.5
N_FOLDS = 5
SEED = 42
EXP_DIR = REPO_ROOT / "experiments"
LOG_PATH = EXP_DIR / "log.csv"
FOLDS_PATH = EXP_DIR / "folds_train_s1.parquet"
TRUTH_PATH = DATASET_DIR / "cache" / "train" / "truth_idx.parquet"


def norm_path(split: str, source: int) -> Path:
    return DATASET_DIR / "cache" / split / f"norm_source{source}.parquet"


def _fresh(out: Path, deps: list[Path]) -> bool:
    return out.exists() and all(out.stat().st_mtime >= d.stat().st_mtime for d in deps)


# ---------------------------------------------------------------- truth + folds
def load_truth_idx(rebuild: bool = False) -> pl.DataFrame:
    """Train positives as integer indexes: (s1 u32, src u8, qi u32). Cached as parquet."""
    deps = [norm_path("train", s) for s in (1, 2, 3)]
    if not rebuild and _fresh(TRUTH_PATH, deps):
        return pl.read_parquet(TRUTH_PATH)

    s1 = pl.scan_parquet(deps[0]).select("entity_id").with_row_index("s1").collect()
    gt = load_ground_truth_pairs()
    n_gt_s1 = gt["source1_entity_id"].n_unique()
    if n_gt_s1 != s1.height:
        print(f"WARN truth: ground truth has {n_gt_s1} S1 ids, norm_source1 has {s1.height}")
    gt = gt.join(s1.rename({"entity_id": "source1_entity_id"}), on="source1_entity_id", how="left")
    assert gt["s1"].null_count() == 0, "ground-truth S1 id missing from norm_source1"
    pos = gt.filter(pl.col("match_id").is_not_null()).select("s1", "match_id")
    del gt

    parts = []
    for src in (2, 3):
        ids = pl.scan_parquet(deps[src - 1]).select("entity_id").with_row_index("qi").collect()
        parts.append(
            pos.join(ids.rename({"entity_id": "match_id"}), on="match_id", how="inner")
            .select("s1", pl.lit(src, dtype=pl.UInt8).alias("src"), "qi", "match_id")
        )
        del ids
    t = pl.concat(parts)
    assert t.height == pos.height, (
        f"truth: {pos.height} positive pairs but {t.height} mapped to S2/S3 rows "
        "(unmapped ids or ids present in both S2 and S3)"
    )
    assert t["match_id"].n_unique() == t.height, "one-owner violated or id collision"
    t = t.select(pl.col("s1").cast(pl.UInt32), "src", pl.col("qi").cast(pl.UInt32))
    TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    t.write_parquet(TRUTH_PATH)
    print(f"truth: {t.height} positive pairs, {s1.height} S1 ({n_gt_s1} in GT)")
    return t


def make_folds(force: bool = False) -> pl.DataFrame:
    """Seeded random fold per train S1 entity_id (GroupKFold by S1)."""
    dep = norm_path("train", 1)
    if not force and _fresh(FOLDS_PATH, [dep]):
        return pl.read_parquet(FOLDS_PATH)
    ids = pl.scan_parquet(dep).select("entity_id").collect().sort("entity_id")
    perm = np.random.default_rng(SEED).permutation(ids.height)
    fold = np.empty(ids.height, dtype=np.int8)
    fold[perm] = np.arange(ids.height) % N_FOLDS
    out = ids.with_columns(pl.Series("fold", fold))
    FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(FOLDS_PATH)
    print(f"folds: {N_FOLDS} folds over {ids.height} S1 -> {FOLDS_PATH.name}")
    return out


def fold_array() -> np.ndarray:
    """Fold per train S1 row index."""
    folds = make_folds()
    s1 = pl.scan_parquet(norm_path("train", 1)).select("entity_id").with_row_index("s1").collect()
    f = s1.join(folds, on="entity_id", how="left").sort("s1")["fold"]
    assert f.null_count() == 0
    return f.to_numpy()


# ---------------------------------------------------------------- metrics
@dataclass
class EvalCtx:
    """Per-S1 arrays (length = rows in norm_source1) describing what is evaluated."""

    mask: np.ndarray  # bool, S1 rows included in the evaluation
    cc: np.ndarray  # country code per S1
    countries: list[str]
    ntrue: np.ndarray | None = None  # float, true matches per S1 (None for test)
    fold: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.mask)


def per_s1_f05(tp: np.ndarray, npred: np.ndarray, ntrue: np.ndarray) -> np.ndarray:
    fp = npred - tp
    fn = ntrue - tp
    denom = (1 + BETA2) * tp + BETA2 * fn + fp
    return np.where(denom > 0, (1 + BETA2) * tp / np.maximum(denom, 1e-12), 1.0)


def candidate_stats(npred: np.ndarray, mask: np.ndarray) -> dict:
    x = npred[mask]
    if x.size == 0:
        return dict(mean=0.0, median=0.0, p95=0.0, max=0, zero=0.0)
    return dict(
        mean=float(x.mean()),
        median=float(np.median(x)),
        p95=float(np.percentile(x, 95)),
        max=int(x.max()),
        zero=float((x == 0).mean()),
    )


def score_pairs(s1: np.ndarray, is_true: np.ndarray | None, ctx: EvalCtx, oracle: bool) -> dict:
    """Score a set of (S1, candidate) pairs.

    oracle=True: pairs are a candidate set, prediction = candidates ∩ truth.
    oracle=False: pairs are the final prediction.
    Returns overall metrics plus per-S1 arrays under "_f" and "_npred".
    """
    npred = np.bincount(s1, minlength=ctx.n).astype(np.float64)
    res = {"n_pairs": int(ctx.mask[s1].sum()), **candidate_stats(npred, ctx.mask), "_npred": npred}
    if ctx.ntrue is None or is_true is None:
        return res
    tp = np.bincount(s1, weights=is_true.astype(np.float64), minlength=ctx.n)
    f = per_s1_f05(tp, tp if oracle else npred, ctx.ntrue)
    m = ctx.mask
    res.update(
        f05=float(f[m].mean()),
        pair_recall=float(tp[m].sum() / max(ctx.ntrue[m].sum(), 1.0)),
        _f=f,
    )
    return res


def breakdown_lines(res: dict, ctx: EvalCtx) -> list[str]:
    """Per country / singleton / fold lines for one scored setting."""
    lines = []
    f, npred, m = res.get("_f"), res["_npred"], ctx.mask
    for code, name in enumerate(ctx.countries):
        mc = m & (ctx.cc == code)
        if not mc.any():
            continue
        cs = candidate_stats(npred, mc)
        s = f"  {name:<12} n_s1={int(mc.sum()):>8}  cands mean={cs['mean']:.2f} med={cs['median']:.0f} p95={cs['p95']:.0f} zero={cs['zero']:.3f}"
        if f is not None:
            s += f"  oracleF05={f[mc].mean():.4f}"
        lines.append(s)
    if f is not None and ctx.ntrue is not None:
        single = ctx.ntrue == 0
        for label, mm in (("singleton", m & single), ("non-singleton", m & ~single)):
            if mm.any():
                cs = candidate_stats(npred, mm)
                lines.append(
                    f"  {label:<12} n_s1={int(mm.sum()):>8}  cands mean={cs['mean']:.2f}  oracleF05={f[mm].mean():.4f}"
                )
        if ctx.fold is not None:
            per = [f[m & (ctx.fold == k)].mean() for k in range(N_FOLDS) if (m & (ctx.fold == k)).any()]
            lines.append(f"  folds        oracleF05 min={min(per):.4f} max={max(per):.4f} (k={len(per)})")
    return lines


# ---------------------------------------------------------------- experiment log
def git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "nogit"


def log_experiment(change: str, scores: dict, lb_score: str = "") -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    new = not LOG_PATH.exists()
    cv = ";".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in scores.items())
    with open(LOG_PATH, "a", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["date", "git_hash", "change", "cv_scores", "lb_score"])
        w.writerow([dt.date.today().isoformat(), git_hash(), change, cv, lb_score])
    print(f"logged to {LOG_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true", help="build train truth index and folds")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.prepare:
        t = load_truth_idx(rebuild=args.force)
        f = make_folds(force=args.force)
        print(f"truth pairs={t.height}  folds sizes={f['fold'].value_counts().sort('fold')['count'].to_list()}")
