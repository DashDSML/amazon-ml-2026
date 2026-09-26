"""Blocking fast loop + miss diagnostics (BLOCKING_V2.md sections 3 and 6).

Indexes ALL train S1 (so block sizes and competition are realistic) but queries only:
  * every S2/S3 record that truly matches one of N sampled S1 (inclusion prob 1), plus
  * a uniform RATE sample of all other S2/S3 records (inclusion prob RATE).
Candidate counts are re-weighted by 1/inclusion-prob, so mean/median/p95 candidates
per S1 are unbiased estimates of the full run. Recall and oracle are exact for the
sampled S1. Uses blocking.py's own key/retrieve/score/keep_top, so any v2 change to
blocking.py is measured here unchanged. ~5 min per run instead of ~1.7 h.

--version v1 (default) or v2 (BLOCKING_V2.md fixes) is passed straight through to
blocking.retrieve(); it only changes the ranking/cap inside retrieve(), so the rest of
this loop (scoring, pruning, bucketing) is identical between versions.

Every true pair of a sampled S1 that is missing from the final candidate set gets
exactly one bucket:
  A1 block>cap      shared a key with its S1, but every shared block exceeded the cap
  A2 quota/budget   shared a within-cap key, but it lost to rarer keys (quota/budget)
  A3 hits-cap       retrieved, dropped by the top-PRE_CAP-by-shared-keys cap
  S1 score-top5     scored, dropped by the top-SAVE_CAP-by-score cap
  S2 pruned         saved, dropped by the floor/alpha/m rule
  B gibberish | C script residue | D typo (token edit dist 1-2) |
  E reordered/partial (token_set >= 90) | F address-only (name weak, address strong) | G other
(B-G apply only to pairs that shared no key at all.)

    python fastloop.py --note "v1 baseline"
    python fastloop.py --floor 0.5 --alpha 0.9 --m 2 --note "v1 m=2"
    python fastloop.py --version v2 --note "v2 fix A3"
    python fastloop.py --version v2 --grid --note "v2 with size/oracle trade-off table"
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

import blocking as bk
import cv_harness as ev

N_S1 = int(__import__("os").environ.get("FAST_N", 20_000))
RATE = 0.03
SEED = 7  # fixed: every experiment uses the same sample
N_EXAMPLES = 20
STAGE_BUCKETS = ["A1 block>cap", "A2 quota/budget", "A3 hits-cap", "S1 score-top5", "S2 pruned"]
CONTENT_BUCKETS = ["B gibberish", "C script", "D typo", "E reorder/partial", "F addr-only", "G other"]
BUCKETS = STAGE_BUCKETS + CONTENT_BUCKETS


def peak_mem_gb() -> float:
    try:
        import psutil

        mi = psutil.Process().memory_info()
        return getattr(mi, "peak_wset", mi.rss) / 2**30
    except Exception:
        return float("nan")


def _bool_col(split: str, src: int, col: str) -> np.ndarray:
    return pl.scan_parquet(ev.norm_path(split, src)).select(pl.col(col).fill_null(False)).collect()[col].to_numpy()


def content_bucket(q: dict, s: dict, gq: bool, gs: bool) -> str:
    if gq or gs:
        return "B gibberish"
    if any(not x.isascii() for x in (q["name_compact"], s["name_compact"])):
        return "C script"
    qt = [t for t in q["name_tokens"] if len(t) >= 4]
    st = [t for t in s["name_tokens"] if len(t) >= 4]
    if any(1 <= Levenshtein.distance(a, b, score_cutoff=2) <= 2 for a in qt for b in st):
        return "D typo"
    name = fuzz.token_set_ratio(q["name_core"], s["name_core"])
    if name >= 90:
        return "E reorder/partial"
    house = q["addr_house_no"] != "" and q["addr_house_no"] == s["addr_house_no"]
    if name < 60 and (fuzz.token_set_ratio(q["addr_nolm"], s["addr_nolm"]) >= 70 or house):
        return "F addr-only"
    return "G other"


def diagnose_unretrieved(miss: np.ndarray, q: pl.DataFrame, s1: pl.DataFrame, s1_pos: np.ndarray,
                         keys: np.ndarray, gib_q: np.ndarray, gib_s1: np.ndarray,
                         version: str = "v1") -> tuple[list[str], list[str]]:
    """Bucket for each never-retrieved true pair (codes qi<<32|s1). Returns (buckets, G examples)."""
    if len(miss) == 0:
        return [], []
    qi = (miss >> np.uint64(32)).astype(np.uint32)
    si = (miss & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    qsub = q[np.searchsorted(q["i"].to_numpy(), qi)]
    ssub = s1[s1_pos[si]]
    kq = bk.key_frame(qsub, version=version).rename({"i": "qi"})
    ks = bk.key_frame(ssub, version=version).rename({"i": "s1"})
    shared = (
        pl.DataFrame({"qi": qi, "s1": si})
        .join(kq, on="qi")
        .join(ks, on=["s1", "kind", "k"])
    )
    lo = np.searchsorted(keys, shared["k"].to_numpy(), "left")
    post = np.searchsorted(keys, shared["k"].to_numpy(), "right") - lo
    within = post <= bk.CAP[shared["kind"].to_numpy()]
    agg = (
        shared.with_columns(pl.Series("within", within))
        .group_by("qi", "s1")
        .agg(pl.col("within").any())
    )
    lookup = {(a, b): w for a, b, w in agg.iter_rows()}

    buckets, examples = [], []
    qrows, srows = qsub.to_dicts(), ssub.to_dicts()
    for n, (a, b) in enumerate(zip(qi.tolist(), si.tolist())):
        if (a, b) in lookup:
            buckets.append("A2 quota/budget" if lookup[(a, b)] else "A1 block>cap")
            continue
        bkt = content_bucket(qrows[n], srows[n], bool(gib_q[a]), bool(gib_s1[b]))
        buckets.append(bkt)
        if bkt == "G other" and len(examples) < N_EXAMPLES:
            qq, ss = qrows[n], srows[n]
            examples.append(
                f"  G  [{qq['country']}] {qq['name_core'][:32]!r:<35} | {ss['name_core'][:32]!r:<35} "
                f"| {qq['addr_nolm'][:30]!r} | {ss['addr_nolm'][:30]!r}"
            )
    return buckets, examples


def main(floor: float, alpha: float, m: int, note: str, version: str = "v1", grid: bool = False) -> None:
    t0 = time.perf_counter()
    split = "train"
    countries = sorted(set().union(*[
        pl.scan_parquet(ev.norm_path(split, s)).select(pl.col("country").fill_null("").unique()).collect()["country"].to_list()
        for s in (1, 2, 3)
    ]))
    cc_expr = pl.col("country").cast(pl.Enum(countries)).to_physical().cast(pl.Int16)

    s1 = bk.load_norm(split, 1)
    n_s1 = s1.height
    truth = ev.load_truth_idx()
    rng = np.random.default_rng(SEED)
    eval_s1 = np.sort(rng.choice(n_s1, size=N_S1, replace=False)).astype(np.uint32)
    tr = truth.filter(pl.Series(np.isin(truth["s1"].to_numpy(), eval_s1)))

    s1_cc = s1.select(cc_expr).to_series().to_numpy()
    s1_pos = np.arange(n_s1, dtype=np.int64)
    keys, s1i = bk.build_index(s1, version=version)
    gib_s1 = _bool_col(split, 1, "name_is_gibberish")
    print(f"[fast] index S1={n_s1:,} keys={len(keys):,} {time.perf_counter() - t0:.0f}s", flush=True)

    tops, bucket_rows, examples = [], [], []
    stage = {"ret": [], "pre": [], "saved": []}
    own_sets: dict[int, np.ndarray] = {}
    n_queries = 0
    for src in (2, 3):
        q = bk.load_norm(split, src)
        own = np.unique(tr.filter(pl.col("src") == src)["qi"].to_numpy())
        own_sets[src] = own
        sel = np.isin(q["i"].to_numpy(), own) | (rng.random(q.height) < RATE)
        q = q.filter(pl.Series(sel))
        n_queries += q.height
        gib_q = _bool_col(split, src, "name_is_gibberish")
        tcodes = np.sort(bk._codes(tr.filter(pl.col("src") == src)))
        ret_true = []
        for off in range(0, q.height, bk.Q_CHUNK):
            chunk = q.slice(off, bk.Q_CHUNK)
            q_cc = chunk.select(cc_expr).to_series().to_numpy()
            cand, uq = bk.retrieve(chunk, keys, s1i, s1_cc, q_cc, version=version)
            ret_true.append(uq[np.isin(uq, tcodes)])
            if cand.height:
                pc = bk._codes(cand)
                stage["pre"].append((src, pc[np.isin(pc, tcodes)]))
                top = bk.keep_top(bk.score(cand, chunk, s1, s1_pos, version=version), src)
                sc = bk._codes(top)
                stage["saved"].append((src, sc[np.isin(sc, tcodes)]))
                tops.append(top)
        ret = np.concatenate(ret_true) if ret_true else np.empty(0, np.uint64)
        stage["ret"].append((src, ret))
        miss = np.setdiff1d(tcodes, ret)
        b, ex = diagnose_unretrieved(miss, q, s1, s1_pos, keys, gib_q, gib_s1, version=version)
        bucket_rows.append(pl.DataFrame({
            "src": np.full(len(miss), src, np.uint8),
            "code": miss, "bucket": pl.Series(b, dtype=pl.String),
        }))
        examples += ex
        print(f"[fast] source{src}: {q.height:,} queries {time.perf_counter() - t0:.0f}s", flush=True)
        del q

    df = pl.concat(tops)
    keep = bk.keep_mask(df, floor, alpha, m)
    kept = df.filter(pl.Series(keep))

    # stage buckets for pairs that were retrieved but lost later
    def codes_of(key: str, src: int) -> np.ndarray:
        arrs = [c for s, c in stage[key] if s == src]
        return np.concatenate(arrs) if arrs else np.empty(0, np.uint64)

    for src in (2, 3):
        ret, pre, saved = codes_of("ret", src), codes_of("pre", src), codes_of("saved", src)
        kc = bk._codes(kept.filter(pl.col("src") == src))
        for lost, name in (
            (np.setdiff1d(ret, pre), "A3 hits-cap"),
            (np.setdiff1d(pre, saved), "S1 score-top5"),
            (np.setdiff1d(saved, kc), "S2 pruned"),
        ):
            bucket_rows.append(pl.DataFrame({
                "src": np.full(len(lost), src, np.uint8), "code": lost,
                "bucket": pl.Series([name] * len(lost), dtype=pl.String),
            }))
    buckets = pl.concat(bucket_rows).with_columns(
        (pl.col("code") & 0xFFFFFFFF).cast(pl.UInt32).alias("s1")
    )

    # metrics: exact recall/oracle, re-weighted candidate counts
    ntrue = np.bincount(tr["s1"].to_numpy(), minlength=n_s1).astype(np.float64)
    s1k = kept["s1"].to_numpy().astype(np.int64)
    qk, srck = kept["qi"].to_numpy(), kept["src"].to_numpy()
    tset = {src: np.sort(bk._codes(tr.filter(pl.col("src") == src))) for src in (2, 3)}
    is_true = np.zeros(kept.height, bool)
    w = np.full(kept.height, 1.0 / RATE)
    kc_all = bk._codes(kept)
    for src in (2, 3):
        msk = srck == src
        is_true[msk] = np.isin(kc_all[msk], tset[src])
        w[msk & np.isin(qk, own_sets[src])] = 1.0
    tp = np.bincount(s1k, weights=is_true.astype(np.float64), minlength=n_s1)
    npred = np.bincount(s1k, weights=w, minlength=n_s1)
    f = ev.per_s1_f05(tp, tp, ntrue)
    mask = np.zeros(n_s1, bool)
    mask[eval_s1] = True

    cols = [("Overall", mask)] + [
        (c, mask & (s1_cc == i)) for i, c in enumerate(countries) if (mask & (s1_cc == i)).any()
    ]
    bc = buckets.with_columns(pl.Series("cc", s1_cc[buckets["s1"].to_numpy()])).group_by("bucket", "cc").len()

    def cell_bucket(name: str, i: int | None, mm: np.ndarray) -> str:
        sub = bc.filter(pl.col("bucket") == name)
        if i is not None:
            sub = sub.filter(pl.col("cc") == i)
        n = int(sub["len"].sum()) if sub.height else 0
        return f"{n:,} ({100 * n / max(ntrue[mm].sum(), 1):.2f}%)"

    rows: list[tuple[str, list[str]]] = []
    stats = [ev.candidate_stats(npred, mm) for _, mm in cols]
    rows.append(("Oracle macro F0.5", [f"{f[mm].mean():.4f}" for _, mm in cols]))
    rows.append(("Pair recall", [f"{tp[mm].sum() / max(ntrue[mm].sum(), 1):.4f}" for _, mm in cols]))
    rows.append(("Cands/S1 mean/med/p95", [f"{s['mean']:.2f}/{s['median']:.0f}/{s['p95']:.0f}" for s in stats]))
    rows.append(("% S1 empty list", [f"{100 * s['zero']:.2f}%" for s in stats]))
    for bname in BUCKETS:
        rows.append((f"miss {bname}", [
            cell_bucket(bname, None if c == "Overall" else countries.index(c), mm) for c, mm in cols
        ]))
    width = max(len(r[0]) for r in rows)
    print(f"[fast] version={version}  floor={floor} alpha={alpha} m={m}  eval S1={N_S1:,}  queries={n_queries:,}")
    print("  " + " " * width + "".join(f"{c:>22}" for c, _ in cols))
    for name, vals in rows:
        print(f"  {name:<{width}}" + "".join(f"{v:>22}" for v in vals))
    print(f"  runtime {time.perf_counter() - t0:.0f}s   peak memory {peak_mem_gb():.1f} GB")
    if examples:
        print(f"[fast] G examples (q name | S1 name | q addr | S1 addr):")
        print("\n".join(examples))

    if grid:
        print(f"\n[fast] floor/alpha/m grid (candidate size counts toward the final ranking, G2):")
        print(f"  {'floor':>6} {'alpha':>6} {'m':>3}   {'oracleF05':>9}   {'cands/S1 mean':>13}")
        for fl, al, mm in bk.GRID:
            keep_g = bk.keep_mask(df, fl, al, mm)
            kept_g = df.filter(pl.Series(keep_g))
            s1g = kept_g["s1"].to_numpy().astype(np.int64)
            qg, srcg = kept_g["qi"].to_numpy(), kept_g["src"].to_numpy()
            is_true_g = np.zeros(len(s1g), bool)
            wg = np.full(len(s1g), 1.0 / RATE)
            kc_g = bk._codes(kept_g)
            for src in (2, 3):
                m_ = srcg == src
                is_true_g[m_] = np.isin(kc_g[m_], tset[src])
                wg[m_ & np.isin(qg, own_sets[src])] = 1.0
            tp_g = np.bincount(s1g, weights=is_true_g.astype(np.float64), minlength=n_s1)
            npred_g = np.bincount(s1g, weights=wg, minlength=n_s1)
            f_g = ev.per_s1_f05(tp_g, tp_g, ntrue)
            cs = ev.candidate_stats(npred_g, mask)
            print(f"  {fl:>6} {al:>6} {mm:>3}   {f_g[mask].mean():>9.4f}   {cs['mean']:>13.2f}")

    ev.log_experiment(
        f"fastloop {version} {note or 'unnamed'} (floor={floor} alpha={alpha} m={m})",
        {"oracle_f05": float(f[mask].mean()), "pair_recall": float(tp[mask].sum() / ntrue[mask].sum()),
         "cands_mean": stats[0]["mean"], "cands_p95": stats[0]["p95"]},
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--version", choices=["v1", "v2"], default="v1")
    ap.add_argument("--grid", action="store_true", help="also sweep bk.GRID: oracle vs mean cands/S1")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    main(a.floor, a.alpha, a.m, a.note, a.version, a.grid)
