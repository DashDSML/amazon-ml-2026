"""Would a char n-gram TF-IDF retrieval channel raise the blocking oracle ceiling?

NonStandard.md T20 (char-n-gram TF-IDF as one of several fused retrieval channels); the
teammate's improvement_ideas.md item A2 proposes a related "pair-difference n-gram model" as
a *matcher* feature -- this tests the same underlying signal (character n-gram overlap) as a
*retrieval* channel instead, to directly answer "does this change the ceiling."

Method:
  1. Fit a char n-gram TF-IDF vectorizer on TRAIN S1 names only (G1: vocabulary/IDF is a
     fitted artefact -- train-only, exactly like blocking v2's name-token DF table).
  2. Sample a fastloop-scale set of S1 (default 20k), rerun the EXISTING key-based
     retrieval (blocking.py, --version v1 or v2) to find which true pairs are currently
     MISSED (retrieved==False).
  3. For each missed pair, compute char-ngram cosine similarity between the query record
     and every S1 of the same country (sparse matrix multiply -- TF-IDF vectors are
     L2-normalised, so dot product == cosine similarity), and find the true S1's RANK.
  4. Report what fraction of misses would be recovered at several candidate-budget levels K
     -- the direct "effect on ceiling" this channel would have if added and capped at K.

    python test_ngram_retrieval.py --version v1 --n-s1 20000
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

import blocking as bk
import cv_harness as ev

SEED = 42
K_GRID = [5, 10, 20, 30, 50, 100]


def fit_vectorizer(split: str) -> TfidfVectorizer:
    """Char n-gram TF-IDF, vocabulary + IDF fit on TRAIN S1 names only."""
    names = pl.scan_parquet(ev.norm_path("train", 1)).select(
        pl.col("name_compact").fill_null("")
    ).collect()["name_compact"].to_list()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, dtype=np.float32)
    vec.fit(names)
    print(f"[ngram] vectorizer fit on {len(names):,} TRAIN S1 names, vocab size {len(vec.vocabulary_):,}")
    return vec


def main(version: str, n_s1: int) -> None:
    t0 = time.perf_counter()
    split = "train"
    vec = fit_vectorizer(split)

    countries = sorted(set().union(*[
        pl.scan_parquet(ev.norm_path(split, s)).select(pl.col("country").fill_null("").unique()).collect()["country"].to_list()
        for s in (1, 2, 3)
    ]))
    cc_expr = pl.col("country").cast(pl.Enum(countries)).to_physical().cast(pl.Int16)

    s1 = bk.load_norm(split, 1)
    n_s1_total = s1.height
    truth = ev.load_truth_idx()
    rng = np.random.default_rng(SEED)
    eval_s1 = np.sort(rng.choice(n_s1_total, size=min(n_s1, n_s1_total), replace=False)).astype(np.uint32)
    tr = truth.filter(pl.Series(np.isin(truth["s1"].to_numpy(), eval_s1)))

    s1_cc = s1.select(cc_expr).to_series().to_numpy()
    s1_pos = np.arange(n_s1_total, dtype=np.int64)
    keys, s1i = bk.build_index(s1, version=version)
    print(f"[ngram] index built, S1={n_s1_total:,} eval_s1={len(eval_s1):,} ({time.perf_counter()-t0:.0f}s)")

    # find current misses (true pairs never retrieved by the existing key-based channels)
    miss_rows = []  # (src, qi, s1, country_code)
    for src in (2, 3):
        q = bk.load_norm(split, src)
        own = np.unique(tr.filter(pl.col("src") == src)["qi"].to_numpy())
        q_own = q.filter(pl.Series(np.isin(q["i"].to_numpy(), own)))
        tcodes = np.sort(bk._codes(tr.filter(pl.col("src") == src)))
        ret_true = []
        for off in range(0, q_own.height, bk.Q_CHUNK):
            chunk = q_own.slice(off, bk.Q_CHUNK)
            q_cc = chunk.select(cc_expr).to_series().to_numpy()
            _, uq = bk.retrieve(chunk, keys, s1i, s1_cc, q_cc, version=version)
            ret_true.append(uq[np.isin(uq, tcodes)])
        ret = np.concatenate(ret_true) if ret_true else np.empty(0, np.uint64)
        miss = np.setdiff1d(tcodes, ret)
        qi_miss = (miss >> np.uint64(32)).astype(np.uint32)
        s1_miss = (miss & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        for qi_, s1_ in zip(qi_miss.tolist(), s1_miss.tolist()):
            miss_rows.append((src, qi_, s1_))
        print(f"[ngram] source{src}: {q_own.height:,} true-pair queries, {len(miss)} currently missed")
        del q, q_own

    print(f"[ngram] total currently-missed true pairs in this sample: {len(miss_rows)}")
    if not miss_rows:
        print("[ngram] nothing missed in this sample -- try a larger --n-s1")
        return

    miss_df = pl.DataFrame(miss_rows, schema=["src", "qi", "s1"], orient="row")

    # for each country present among misses, rank the true S1 by char-ngram cosine similarity
    recovered_at_k = {k: 0 for k in K_GRID}
    ranks = []
    for src in (2, 3):
        sub = miss_df.filter(pl.col("src") == src)
        if sub.height == 0:
            continue
        q = bk.load_norm(split, src)
        q_cc_all = q.select(cc_expr).to_series().to_numpy()
        q_rows = q.filter(pl.Series(np.isin(q["i"].to_numpy(), sub["qi"].to_numpy())))
        for code, cname in enumerate(countries):
            s1_mask = s1_cc == code
            if not s1_mask.any():
                continue
            s1_idx = np.flatnonzero(s1_mask)
            s1_names = s1.filter(pl.Series(s1_mask))["name_compact"].fill_null("").to_list()
            s1_mat = vec.transform(s1_names)

            sub_c = sub.join(q_rows.select("i", "name_compact"), left_on="qi", right_on="i", how="left")
            sub_c = sub_c.filter(pl.Series(q_cc_all[sub_c["qi"].to_numpy()] == code))
            if sub_c.height == 0:
                continue
            q_mat = vec.transform(sub_c["name_compact"].fill_null("").to_list())
            sim = (q_mat @ s1_mat.T).toarray()  # (n_missed_in_country, n_s1_in_country)

            true_s1_local = np.searchsorted(s1_idx, sub_c["s1"].to_numpy())
            for row in range(sim.shape[0]):
                true_local = true_s1_local[row]
                if true_local >= len(s1_idx) or s1_idx[true_local] != sub_c["s1"].to_numpy()[row]:
                    continue  # true S1 not in this country's index (shouldn't happen; safety)
                score_true = sim[row, true_local]
                rank = int((sim[row] > score_true).sum())  # 0 = best
                ranks.append(rank)
                for k in K_GRID:
                    if rank < k:
                        recovered_at_k[k] += 1
        del q, q_rows

    n_miss = len(ranks)
    print(f"\n[ngram] ranked {n_miss} of {len(miss_rows)} misses (rest had no valid country index)")
    print("[ngram] fraction of CURRENT misses that a char-ngram channel would recover, by budget K:")
    for k in K_GRID:
        frac = recovered_at_k[k] / max(n_miss, 1)
        print(f"  K={k:<4} recovered={recovered_at_k[k]:>4}/{n_miss}  ({frac:.1%})")
    if ranks:
        r = np.array(ranks)
        print(f"[ngram] miss rank stats: median={np.median(r):.0f}  p90={np.percentile(r,90):.0f}  "
              f"mean={r.mean():.1f}  (0=best)")
    print(f"[ngram] total runtime {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v1", "v2"], default="v1")
    ap.add_argument("--n-s1", type=int, default=20_000)
    args = ap.parse_args()
    main(args.version, args.n_s1)
