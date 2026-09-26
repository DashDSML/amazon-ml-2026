"""Blocking v1: S2/S3-side retrieval of S1 candidates within country, adaptive top-m.

Stage 1, retrieve + score (per S2/S3 chunk, memory-bounded):
  1. Every record gets blocking keys, each prefixed with its country so a key can only
     hit S1 records of the same country (country is a hard key; France needs no special
     case). Kinds:  C exact name_compact | T name token | G 4-gram of name_compact
     (typo-robust) | H name token + house no | Y name token + city | A house no + addr token.
  2. S1 keys are hashed into one sorted uint64 array (inverted index via searchsorted).
  3. Per query record, per kind, keep the `quota` most selective keys whose posting list
     is non-empty and <= `cap`, then stop adding keys once PAIR_BUDGET postings are used.
     Selectivity is the posting length in the index being queried: an index property at
     query time, nothing is fitted (no IDF/vocabulary is learned from test).
  4. Candidates are ranked by number of shared keys; top PRE_CAP per record are scored
     with rapidfuzz (name token_set/ratio, address token_set, house-number agreement).
  5. Top SAVE_CAP per record are saved with score, rank and the record's best score.

Stage 2, adaptive top-m (one-owner cap, T30/T31): a query record keeps candidate r if
      r < m  and  score >= floor  and  score >= alpha * best_score_of_that_record.
  train: scores a grid of (floor, alpha, m) with the blocking oracle + candidate sizes,
         then writes candidates for the chosen setting and logs it (full runs only).
  test:  applies the chosen setting and prints candidate-size stats.

Outputs: dataset/cache/blocking_v1/<split>[/smoke]/{parts/, candidates.parquet}
candidates.parquet columns: src (2|3), qi (row in norm_source{src}), s1 (row in
norm_source1), score, name_s, addr_s, rank.

  python blocking.py --split train --smoke 20000   # quick check on a sample
  python blocking.py --split train                 # full run + grid
  python blocking.py --split train --prune-only --floor 0.5 --alpha 0.9 --m 3
  python blocking.py --split test  --floor 0.5 --alpha 0.9 --m 3
"""

from __future__ import annotations

import argparse
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

import cv_harness as ev
from data import DATASET_DIR

warnings.filterwarnings("ignore", category=DeprecationWarning)  # polars 1.x explode/is_in notices

SEED = 42
# kind code -> (letter, quota per record, max posting length)
KINDS = {0: ("C", 1, 2000), 1: ("T", 2, 500), 2: ("G", 3, 500), 3: ("H", 2, 200), 4: ("Y", 1, 200), 5: ("A", 2, 200)}
QUOTA = np.array([KINDS[k][1] for k in sorted(KINDS)], dtype=np.int64)
CAP = np.array([KINDS[k][2] for k in sorted(KINDS)], dtype=np.int64)
PAIR_BUDGET = 2000  # max postings expanded per query record
PRE_CAP = 30  # candidates per record sent to the scorer
SAVE_CAP = 5  # candidates per record kept for stage 2 (max usable m)
Q_CHUNK = 25_000
S1_CHUNK = 200_000
SCORE_BATCH = 400_000
W_NAME, W_ADDR, HOUSE_ADJ = 0.65, 0.35, 0.05

GRID = [  # (floor, alpha, m)
    (0.0, 0.0, 5),
    (0.0, 0.0, 1),
    (0.5, 0.0, 1),
    (0.5, 0.9, 2),
    (0.5, 0.9, 3),
    (0.5, 0.9, 5),
    (0.4, 0.9, 3),
    (0.6, 0.9, 3),
    (0.5, 0.8, 3),
    (0.5, 0.95, 3),
    (0.6, 0.95, 2),
]

COLS = [
    "entity_id", "country", "name_core", "name_compact", "name_tokens",
    "addr_nolm", "addr_tokens", "addr_house_no", "addr_city", "addr_missing",
]
STR_COLS = ["entity_id", "country", "name_core", "name_compact", "addr_nolm", "addr_house_no", "addr_city"]
LIST_COLS = ["name_tokens", "addr_tokens"]


def out_dir(split: str, smoke: int | None) -> Path:
    d = DATASET_DIR / "cache" / "blocking_v1" / split
    return d / "smoke" if smoke else d


def load_norm(split: str, source: int) -> pl.DataFrame:
    df = pl.scan_parquet(ev.norm_path(split, source)).select(COLS).with_row_index("i").collect()
    return df.with_columns(
        *[pl.col(c).fill_null("") for c in STR_COLS],
        *[pl.col(c).fill_null(pl.lit([], dtype=pl.List(pl.String))) for c in LIST_COLS],
        has_addr=~pl.col("addr_missing").fill_null(True) & (pl.col("addr_nolm") != ""),
    ).drop("addr_missing")


# ---------------------------------------------------------------- keys
def _k(letter: str, *parts: pl.Expr) -> pl.Expr:
    return pl.concat_str([pl.col("country"), pl.lit(f"\x1f{letter}\x1f"), *parts])


def key_frame(df: pl.DataFrame) -> pl.DataFrame:
    """(i u32, kind u8, k u64) for every blocking key of every record in df."""
    base = df.select("i", "country", "name_compact", "name_tokens", "addr_house_no", "addr_city", "addr_tokens")
    kind = lambda c: pl.lit(c, dtype=pl.UInt8).alias("kind")  # noqa: E731
    parts = [base.filter(pl.col("name_compact") != "").select("i", kind(0), _k("C", pl.col("name_compact")).alias("s"))]

    tok = (
        base.select("i", "country", "addr_house_no", "addr_city", pl.col("name_tokens").alias("t"))
        .explode("t")
        .filter(pl.col("t").str.len_chars() >= 2)
    )
    parts.append(tok.select("i", kind(1), _k("T", pl.col("t")).alias("s")))
    parts.append(
        tok.filter(pl.col("addr_house_no") != "")
        .select("i", kind(3), _k("H", pl.col("t"), pl.lit("|"), pl.col("addr_house_no")).alias("s"))
    )
    parts.append(
        tok.filter(pl.col("addr_city") != "")
        .select("i", kind(4), _k("Y", pl.col("t"), pl.lit("|"), pl.col("addr_city")).alias("s"))
    )

    n_grams = pl.max_horizontal(pl.col("name_compact").str.len_chars().cast(pl.Int32) - 3, pl.lit(1, dtype=pl.Int32))
    grams = (
        base.filter(pl.col("name_compact") != "")
        .select("i", "country", "name_compact", pl.int_ranges(0, n_grams).alias("off"))
        .explode("off")
    )
    parts.append(grams.select("i", kind(2), _k("G", pl.col("name_compact").str.slice(pl.col("off"), 4)).alias("s")))

    addr = (
        base.filter(pl.col("addr_house_no") != "")
        .select("i", "country", "addr_house_no", pl.col("addr_tokens").alias("a"))
        .explode("a")
        .filter((pl.col("a").str.len_chars() >= 3) & ~pl.col("a").str.contains(r"^\d+$"))
    )
    parts.append(addr.select("i", kind(5), _k("A", pl.col("addr_house_no"), pl.lit("|"), pl.col("a")).alias("s")))

    return (
        pl.concat(parts)
        .select(pl.col("i").cast(pl.UInt32), "kind", pl.col("s").hash(seed=SEED).alias("k"))
        .unique()
    )


def build_index(s1: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ks, ss = [], []
    for off in range(0, s1.height, S1_CHUNK):
        kf = key_frame(s1.slice(off, S1_CHUNK))
        ks.append(kf["k"].to_numpy())
        ss.append(kf["i"].to_numpy())
    keys, s1i = np.concatenate(ks), np.concatenate(ss)
    order = np.argsort(keys, kind="stable")
    return keys[order], s1i[order]


# ---------------------------------------------------------------- retrieval
def retrieve(chunk: pl.DataFrame, keys: np.ndarray, s1i: np.ndarray, s1_cc: np.ndarray, q_cc: np.ndarray):
    """Candidate (qi, s1, hits) for one query chunk, top PRE_CAP by hits per qi."""
    kf = key_frame(chunk)
    qi, kind, k = kf["i"].to_numpy(), kf["kind"].to_numpy(), kf["k"].to_numpy()
    lo = np.searchsorted(keys, k, "left")
    post = np.searchsorted(keys, k, "right") - lo
    ok = (post >= 1) & (post <= CAP[kind])
    sel = (
        pl.DataFrame({"qi": qi[ok], "kind": kind[ok], "post": post[ok], "lo": lo[ok], "quota": QUOTA[kind[ok]]})
        .sort(["qi", "kind", "post", "lo"])
        .filter(pl.int_range(pl.len()).over(["qi", "kind"]) < pl.col("quota"))
        .sort(["qi", "post", "lo"])
        .filter((pl.col("post").cum_sum().over("qi") - pl.col("post")) < PAIR_BUDGET)
    )
    qs, ps, ls = sel["qi"].to_numpy(), sel["post"].to_numpy(), sel["lo"].to_numpy()
    total = int(ps.sum())
    if total == 0:
        empty = pl.DataFrame(schema={"qi": pl.UInt32, "s1": pl.UInt32, "hits": pl.UInt16})
        return empty, np.empty(0, np.uint64)
    rep_q = np.repeat(qs, ps)
    pos = np.repeat(ls - (np.cumsum(ps) - ps), ps) + np.arange(total)
    s1 = s1i[pos]
    same = s1_cc[s1] == q_cc[np.searchsorted(chunk["i"].to_numpy(), rep_q)]  # hash-collision guard
    code = (rep_q[same].astype(np.uint64) << np.uint64(32)) | s1[same].astype(np.uint64)
    del rep_q, pos, s1, same
    uq, hits = np.unique(code, return_counts=True)
    cand = (
        pl.DataFrame({
            "qi": (uq >> np.uint64(32)).astype(np.uint32),
            "s1": (uq & np.uint64(0xFFFFFFFF)).astype(np.uint32),
            "hits": np.minimum(hits, 65535).astype(np.uint16),
        })
        .sort(["qi", "hits", "s1"], descending=[False, True, False])
        .filter(pl.int_range(pl.len()).over("qi") < PRE_CAP)
    )
    return cand, uq


# ---------------------------------------------------------------- scoring
def _sim(a: pl.Series, b: pl.Series, scorer) -> np.ndarray:
    return cpdist(a.to_list(), b.to_list(), scorer=scorer, workers=-1, dtype=np.float32) / 100.0


def score(cand: pl.DataFrame, chunk: pl.DataFrame, s1: pl.DataFrame, s1_pos: np.ndarray) -> pl.DataFrame:
    qloc = np.searchsorted(chunk["i"].to_numpy(), cand["qi"].to_numpy())
    sloc = s1_pos[cand["s1"].to_numpy()]
    n = cand.height
    name_s = np.empty(n, np.float32)
    addr_s = np.full(n, np.nan, np.float32)
    house = np.zeros(n, np.float32)
    for a in range(0, n, SCORE_BATCH):
        qb, sb = qloc[a : a + SCORE_BATCH], sloc[a : a + SCORE_BATCH]
        q, s = chunk[qb], s1[sb]
        name_s[a : a + len(qb)] = np.maximum(
            _sim(q["name_core"], s["name_core"], fuzz.token_set_ratio),
            _sim(q["name_compact"], s["name_compact"], fuzz.ratio),
        )
        both = (q["has_addr"] & s["has_addr"]).to_numpy()
        if both.any():
            idx = np.flatnonzero(both)
            addr_s[a + idx] = _sim(q["addr_nolm"].gather(idx), s["addr_nolm"].gather(idx), fuzz.token_set_ratio)
        hq, hs = q["addr_house_no"], s["addr_house_no"]
        has_h = ((hq != "") & (hs != "")).to_numpy()
        eq = (hq == hs).to_numpy()
        house[a : a + len(qb)] = np.where(has_h, np.where(eq, HOUSE_ADJ, -HOUSE_ADJ), 0.0)
    total = np.where(np.isnan(addr_s), name_s, W_NAME * name_s + W_ADDR * np.nan_to_num(addr_s)) + house
    return cand.with_columns(
        pl.Series("name_s", name_s),
        pl.Series("addr_s", addr_s),
        pl.Series("score", np.clip(total, 0, 1).astype(np.float32)),
    )


def keep_top(scored: pl.DataFrame, src: int) -> pl.DataFrame:
    return (
        scored.sort(["qi", "score", "s1"], descending=[False, True, False])
        .with_columns(pl.int_range(pl.len()).over("qi").cast(pl.UInt8).alias("rank"))
        .filter(pl.col("rank") < SAVE_CAP)
        .with_columns(pl.col("score").max().over("qi").alias("best"), pl.lit(src, dtype=pl.UInt8).alias("src"))
        .select("src", "qi", "s1", "hits", "name_s", "addr_s", "score", "rank", "best")
    )


def _codes(df: pl.DataFrame) -> np.ndarray:
    return (df["qi"].to_numpy().astype(np.uint64) << np.uint64(32)) | df["s1"].to_numpy().astype(np.uint64)


# ---------------------------------------------------------------- stage 1
def run_retrieval(split: str, smoke: int | None, od: Path) -> None:
    t0 = time.perf_counter()
    countries = sorted(
        set().union(*[
            pl.scan_parquet(ev.norm_path(split, s)).select(pl.col("country").fill_null("").unique()).collect()["country"].to_list()
            for s in (1, 2, 3)
        ])
    )
    cc_expr = pl.col("country").cast(pl.Enum(countries)).to_physical().cast(pl.Int16)

    s1 = load_norm(split, 1)
    n_s1_rows = s1.height
    truth = ev.load_truth_idx() if split == "train" else None
    q_rows: dict[int, np.ndarray | None] = {2: None, 3: None}
    if smoke:
        rng = np.random.default_rng(SEED)
        eval_s1 = np.sort(rng.choice(n_s1_rows, size=min(smoke, n_s1_rows), replace=False)).astype(np.uint32)
        s1 = s1.filter(pl.Series(np.isin(s1["i"].to_numpy(), eval_s1)))
        truth = truth.filter(pl.Series(np.isin(truth["s1"].to_numpy(), eval_s1)))
        for src in (2, 3):
            n_src = pl.scan_parquet(ev.norm_path(split, src)).select(pl.len()).collect().item()
            matched = truth.filter(pl.col("src") == src)["qi"].to_numpy()
            extra = rng.choice(n_src, size=min(n_src, int(0.35 * len(matched)) + 1), replace=False)
            q_rows[src] = np.unique(np.concatenate([matched, extra])).astype(np.uint32)
    else:
        eval_s1 = np.arange(n_s1_rows, dtype=np.uint32)

    if od.exists():
        shutil.rmtree(od)
    (od / "parts").mkdir(parents=True)
    np.save(od / "eval_s1.npy", eval_s1)

    s1_cc = np.full(n_s1_rows, -1, np.int16)
    s1_cc[s1["i"].to_numpy()] = s1.select(cc_expr).to_series().to_numpy()
    s1_pos = np.full(n_s1_rows, -1, np.int64)
    s1_pos[s1["i"].to_numpy()] = np.arange(s1.height)
    keys, s1i = build_index(s1)
    print(f"[index] S1={s1.height:,} keys={len(keys):,} ({keys.nbytes / 2**20:.0f} MiB) {time.perf_counter() - t0:.0f}s")

    tot = dict(q=0, q_nocand=0, ret=0, pre=0, saved=0, t_ret=0, t_pre=0, t_saved=0, t_all=0)
    for src in (2, 3):
        q = load_norm(split, src)
        if q_rows[src] is not None:
            q = q.filter(pl.Series(np.isin(q["i"].to_numpy(), q_rows[src])))
        tcodes = None
        if truth is not None:
            tcodes = np.sort(_codes(truth.filter(pl.col("src") == src)))
            tot["t_all"] += len(tcodes)
        t_src = time.perf_counter()
        for n_chunk, off in enumerate(range(0, q.height, Q_CHUNK)):
            chunk = q.slice(off, Q_CHUNK)
            q_cc = chunk.select(cc_expr).to_series().to_numpy()
            cand, uq = retrieve(chunk, keys, s1i, s1_cc, q_cc)
            top = keep_top(score(cand, chunk, s1, s1_pos), src) if cand.height else None
            tot["q"] += chunk.height
            tot["q_nocand"] += chunk.height - cand["qi"].n_unique()
            tot["ret"] += len(uq)
            tot["pre"] += cand.height
            if top is not None:
                tot["saved"] += top.height
                top.write_parquet(od / "parts" / f"s{src}_{n_chunk:05d}.parquet")
            if (n_chunk + 1) % 20 == 0:
                el = time.perf_counter() - t_src
                done = off + chunk.height
                eta = el / done * (q.height - done)
                print(f"  [progress] source{src}: {done:,}/{q.height:,} records, "
                      f"{el / 60:.1f} min elapsed, ~{eta / 60:.0f} min left for this source", flush=True)
            if tcodes is not None:
                tot["t_ret"] += int(np.isin(uq, tcodes).sum())
                tot["t_pre"] += int(np.isin(_codes(cand), tcodes).sum()) if cand.height else 0
                tot["t_saved"] += int(np.isin(_codes(top), tcodes).sum()) if top is not None else 0
        print(f"[stage1] source{src}: {q.height:,} records done {time.perf_counter() - t0:.0f}s")
        del q

    nq = max(tot["q"], 1)
    print(
        f"[stage1] queries={tot['q']:,} no_candidate={tot['q_nocand'] / nq:.3f} "
        f"pairs/query retrieved={tot['ret'] / nq:.1f} scored={tot['pre'] / nq:.1f} saved={tot['saved'] / nq:.2f}"
    )
    if truth is not None:
        ta = max(tot["t_all"], 1)
        print(
            f"[stage1] true-pair recall retrieved={tot['t_ret'] / ta:.4f} "
            f"after_hits_cap={tot['t_pre'] / ta:.4f} after_score_cap={tot['t_saved'] / ta:.4f}"
        )
    print(f"[stage1] total {time.perf_counter() - t0:.0f}s")


# ---------------------------------------------------------------- stage 2
def load_scored(od: Path) -> pl.DataFrame:
    return pl.scan_parquet(od / "parts" / "*.parquet").select("src", "qi", "s1", "score", "rank", "best", "name_s", "addr_s").collect()


def make_ctx(split: str, od: Path, truth: pl.DataFrame | None) -> ev.EvalCtx:
    s1c = pl.scan_parquet(ev.norm_path(split, 1)).select(pl.col("country").fill_null("")).collect()["country"]
    countries = sorted(s1c.unique().to_list())
    cc = s1c.cast(pl.Enum(countries)).to_physical().to_numpy().astype(np.int16)
    mask = np.zeros(len(cc), bool)
    mask[np.load(od / "eval_s1.npy")] = True
    ntrue = fold = None
    if truth is not None:
        ntrue = np.bincount(truth["s1"].to_numpy(), minlength=len(cc)).astype(np.float64)
        fold = ev.fold_array()
    return ev.EvalCtx(mask=mask, cc=cc, countries=countries, ntrue=ntrue, fold=fold)


def keep_mask(df: pl.DataFrame, floor: float, alpha: float, m: int) -> np.ndarray:
    r, sc, b = df["rank"].to_numpy(), df["score"].to_numpy(), df["best"].to_numpy()
    return (r < m) & (sc >= floor) & (sc >= alpha * b)


def _fmt(res: dict) -> str:
    s = f"cands/S1 mean={res['mean']:.2f} med={res['median']:.0f} p95={res['p95']:.0f} zero={res['zero']:.3f}"
    if "f05" in res:
        s = f"oracleF05={res['f05']:.4f} pair_recall={res['pair_recall']:.4f} " + s
    return s


def run_prune(split: str, smoke: int | None, od: Path, floor: float, alpha: float, m: int, note: str) -> None:
    assert m <= SAVE_CAP, f"m must be <= SAVE_CAP={SAVE_CAP}"
    df = load_scored(od)
    truth = None
    if split == "train":
        truth = ev.load_truth_idx()
        truth = truth.filter(pl.Series(np.isin(truth["s1"].to_numpy(), np.load(od / "eval_s1.npy"))))
        df = df.join(truth.with_columns(pl.lit(True).alias("is_true")), on=["src", "qi", "s1"], how="left").with_columns(
            pl.col("is_true").fill_null(False)
        )
    ctx = make_ctx(split, od, truth)
    s1 = df["s1"].to_numpy().astype(np.int64)
    is_true = df["is_true"].to_numpy() if truth is not None else None

    if split == "train":
        print(f"[grid] eval S1={int(ctx.mask.sum()):,} (true matches/S1 = {ctx.ntrue[ctx.mask].mean():.2f})")
        for fl, al, mm in GRID:
            k = keep_mask(df, fl, al, mm)
            res = ev.score_pairs(s1[k], is_true[k], ctx, oracle=True)
            print(f"[grid] floor={fl:.2f} alpha={al:.2f} m={mm}  {_fmt(res)}")

    k = keep_mask(df, floor, alpha, m)
    res = ev.score_pairs(s1[k], is_true[k] if is_true is not None else None, ctx, oracle=True)
    print(f"[chosen] floor={floor} alpha={alpha} m={m}  {_fmt(res)}")
    for line in ev.breakdown_lines(res, ctx):
        print("[chosen]" + line)

    out = df.filter(pl.Series(k)).select("src", "qi", "s1", "score", "name_s", "addr_s", "rank")
    out.write_parquet(od / "candidates.parquet")
    print(f"[chosen] wrote {out.height:,} pairs -> {(od / 'candidates.parquet').relative_to(DATASET_DIR.parent)}")

    if split == "train" and not smoke:
        ev.log_experiment(
            note or f"blocking_v1 floor={floor} alpha={alpha} m={m}",
            {"oracle_f05": res["f05"], "pair_recall": res["pair_recall"], "cands_mean": res["mean"],
             "cands_median": res["median"], "cands_p95": res["p95"]},
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--smoke", type=int, default=None, help="train only: sample this many S1")
    ap.add_argument("--prune-only", action="store_true", help="skip stage 1, reuse saved parts")
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    assert not (a.smoke and a.split == "test"), "--smoke is train-only"
    od = out_dir(a.split, a.smoke)
    if not a.prune_only:
        run_retrieval(a.split, a.smoke, od)
    run_prune(a.split, a.smoke, od, a.floor, a.alpha, a.m, a.note)
