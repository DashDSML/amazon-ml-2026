"""Wide candidate pool for the ranker (train only): reruns blocking's retrieval+scoring
stage with SAVE_CAP raised from 5 to ~PRE_CAP_V2, so every retrieved-and-scored candidate
is written, not just each record's top 5.

Why a rerun is needed, not a reread: score() already scores every retrieved candidate
(up to PRE_CAP_V2 per record) before keep_top() cuts to SAVE_CAP -- but only the kept rows
are ever written to parts/. Nothing else changes: same keys (P/N composite keys included),
same rarity weighting (fix A3), same key-evidence score boost (fix S1/S2), same version="v2"
code path in blocking.py, completely unmodified. SAVE_CAP is overridden here, in this driver,
not in blocking.py, so v1/v2's own frozen behaviour and outputs are untouched.

Output: dataset/cache/blocking_v2_ranker/train/{parts/, eval_s1.npy} -- a new, separate tree,
never dataset/cache/blocking_v2/train/ (the frozen v2 output feeding lb-3 stays exactly as is).

This only reruns retrieval+scoring (the expensive rapidfuzz stage, ~2h on the real full
train set -- the same cost already paid for the frozen v2 run, since compute is identical,
only the write cutoff differs). It does not run the floor/alpha/m grid or write a
candidates.parquet: load the wide pool directly via blocking.load_scored(od) for feature
engineering and ranker training.

    python ranker_data.py             # full train (run on EC2; ~2h, matches the original run)
    python ranker_data.py --smoke 500  # tiny check before committing to the full run
"""

from __future__ import annotations

import argparse
import shutil

import blocking as bk
from data import DATASET_DIR

WIDE_SAVE_CAP = bk.PRE_CAP_V2  # keep every retrieved+scored candidate (80), not just top 5


def build(smoke: int | None = None) -> None:
    bk.SAVE_CAP = WIDE_SAVE_CAP  # overrides the module global keep_top() reads at call time
    od = DATASET_DIR / "cache" / "blocking_v2_ranker" / "train"
    if smoke:
        od = od / "smoke"
    if od.exists():
        shutil.rmtree(od)
    od.mkdir(parents=True)
    bk.run_retrieval("train", smoke, od, version="v2")
    print(f"[ranker_data] wide pool (SAVE_CAP={WIDE_SAVE_CAP}) written to "
          f"{od.relative_to(DATASET_DIR.parent)}/parts/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=None, help="sample this many S1 for a quick check")
    args = ap.parse_args()
    build(args.smoke)
