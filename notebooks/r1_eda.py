"""R1 exploratory analysis of the label structure (challengecontext.md section 7).

Run:  python notebooks/r1_eda.py
Train and test are processed one after the other; train is freed before test is
touched. From test only row counts, country counts and missingness are read
(format/sanity checks allowed by NonStandard.md G1).
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import polars as pl
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution" / "src"))
from data import (  # noqa: E402
    load_ground_truth_pairs,
    scan_ground_truth,
    scan_source,
)

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(200)
pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
pl.Config.set_float_precision(4)

PROC = psutil.Process()
S1 = "source1_entity_id"


def rss() -> str:
    return f"{PROC.memory_info().rss / 2**30:.2f} GiB"


_T = [time.perf_counter()]


def h(title: str) -> None:
    now = time.perf_counter()
    print(f"\n## {title}  (RSS {rss()}, previous section {now - _T[0]:.1f}s)\n", flush=True)
    _T[0] = now


def slim_source(split: str, source: int) -> pl.DataFrame:
    """ID, country and missingness flags only: keeps memory small."""
    df = (
        scan_source(split, source)
        .select(
            "entity_id",
            pl.col("country").cast(pl.Categorical),
            (pl.col("business_name").str.strip_chars() == "").alias("name_missing"),
            (pl.col("business_address").str.strip_chars() == "").alias("addr_missing"),
        )
        .collect()
    )
    # Row count vs raw line count is verified once when the parquet cache is built.
    prefix_ok = df["entity_id"].str.starts_with(f"S{source}-").all()
    assert prefix_ok, f"{split} S{source}: entity_id prefix mismatch"
    assert df["entity_id"].n_unique() == df.height, f"{split} S{source}: duplicate entity_id"
    return df


def source_overview(split: str, sources: dict[int, pl.DataFrame]) -> None:
    h(f"{split}: rows, countries, missingness per source")
    rows = []
    for s, df in sources.items():
        rows.append(
            df.group_by("country")
            .agg(
                pl.len().alias("rows"),
                pl.col("name_missing").mean().alias("name_missing"),
                pl.col("addr_missing").mean().alias("addr_missing"),
            )
            .with_columns(pl.lit(f"S{s}").alias("source"), pl.col("country").cast(pl.String))
        )
    tbl = pl.concat(rows).select("source", "country", "rows", "name_missing", "addr_missing")
    print(tbl.sort("source", "country"))
    print()
    print(tbl.group_by("source").agg(pl.col("rows").sum()).sort("source"))


def train_eda() -> dict[str, int]:
    sources = {s: slim_source("train", s) for s in (1, 2, 3)}
    source_overview("train", sources)
    s1 = sources[1].select(pl.col("entity_id").alias(S1), pl.col("country").cast(pl.String))
    others = pl.concat([sources[2], sources[3]]).select(
        pl.col("entity_id").alias("match_id"),
        pl.col("country").cast(pl.String).alias("match_country"),
    )

    # ---- ground-truth integrity -------------------------------------------
    h("Ground truth integrity")
    gt_raw = scan_ground_truth().collect()
    n_dup_s1_rows = gt_raw.height - gt_raw[S1].n_unique()
    gt = load_ground_truth_pairs()
    raw_ids = gt_raw["matched_entity_ids"].str.split(",").list.eval(pl.element().filter(pl.element() != ""))
    n_dup_within_list = int((raw_ids.list.len() - raw_ids.list.n_unique()).sum())
    pos = gt.filter(pl.col("match_id").is_not_null())
    gt_s1 = gt.select(S1).unique()
    print(f"GT rows: {gt_raw.height:,}   unique S1 in GT: {gt_s1.height:,}   duplicate S1 rows: {n_dup_s1_rows}")
    print(f"S1 in source1 but not in GT: {s1.join(gt_s1, on=S1, how='anti').height:,}")
    print(f"S1 in GT but not in source1: {gt_s1.join(s1, on=S1, how='anti').height:,}")
    print(f"Duplicate IDs inside one matched list: {n_dup_within_list}")
    bad_prefix = pos.filter(~pl.col("match_id").str.contains(r"^S[23]-")).height
    print(f"Matched IDs without S2-/S3- prefix: {bad_prefix}")
    missing_ids = pos.join(others, on="match_id", how="anti").height
    print(f"Matched IDs absent from train S2/S3 files: {missing_ids:,}")
    print(f"Positive pairs: {pos.height:,}")

    # ---- cardinality --------------------------------------------------------
    h("Match cardinality per S1 entity")
    card = (
        s1.join(pos.group_by(S1).agg(
            pl.len().alias("n_match"),
            pl.col("match_id").str.starts_with("S2-").sum().alias("n_s2"),
            pl.col("match_id").str.starts_with("S3-").sum().alias("n_s3"),
        ), on=S1, how="left")
        .with_columns(pl.col("n_match", "n_s2", "n_s3").fill_null(0))
        .with_columns(
            bucket=pl.when(pl.col("n_match") >= 6).then(pl.lit("6+")).otherwise(pl.col("n_match").cast(pl.String))
        )
    )
    dist = card.group_by("bucket").len("n_s1").with_columns((pl.col("n_s1") / card.height).alias("share"))
    print(dist.sort(pl.col("bucket").str.replace(r"\+", "").cast(pl.Int32)))
    singleton_rate = float((card["n_match"] == 0).mean())
    print(f"\nSingleton rate (= macro F0.5 of all-empty prediction): {singleton_rate:.4f}")
    nz = card.filter(pl.col("n_match") > 0)["n_match"]
    print(f"Mean matches per S1: {card['n_match'].mean():.3f} (all S1), {nz.mean():.3f} (non-singletons)")
    print(f"Median / p95 / max matches among non-singletons: {nz.median():.0f} / {nz.quantile(0.95):.0f} / {nz.max()}")

    h("S2 vs S3 contribution")
    print(f"Matches from S2: {int(card['n_s2'].sum()):,}   from S3: {int(card['n_s3'].sum()):,}")
    combo = card.filter(pl.col("n_match") > 0).with_columns(
        combo=pl.when((pl.col("n_s2") > 0) & (pl.col("n_s3") > 0)).then(pl.lit("S2+S3"))
        .when(pl.col("n_s2") > 0).then(pl.lit("S2 only")).otherwise(pl.lit("S3 only"))
    )
    print(combo.group_by("combo").len("n_s1").with_columns((pl.col("n_s1") / combo.height).alias("share")).sort("combo"))
    per_src = pl.concat([
        combo.group_by("n_s2").len("n_s1").rename({"n_s2": "k"}).with_columns(pl.lit("S2").alias("src")),
        combo.group_by("n_s3").len("n_s1").rename({"n_s3": "k"}).with_columns(pl.lit("S3").alias("src")),
    ]).pivot(on="src", index="k", values="n_s1").fill_null(0).sort("k")
    print("\nMatches per non-singleton S1, by source (k = count from that source):")
    print(per_src)

    # ---- one-owner property -------------------------------------------------
    h("One-owner check: does any S2/S3 ID appear under more than one S1?")
    owners = pos.group_by("match_id").agg(pl.col(S1).n_unique().alias("n_owner"))
    multi = owners.filter(pl.col("n_owner") > 1)
    print(f"Distinct matched S2/S3 IDs: {owners.height:,}")
    print(f"IDs claimed by >1 S1: {multi.height:,}   max owners: {owners['n_owner'].max()}")
    if multi.height:
        print(multi.group_by("n_owner").len().sort("n_owner"))
        print(pos.join(multi.head(5), on="match_id").sort("match_id"))

    # ---- unmatched S2/S3 (distractors) --------------------------------------
    h("S2/S3 records that match no S1 (distractors)")
    matched = owners.select("match_id")
    un = others.with_columns(
        source=pl.col("match_id").str.slice(0, 2),
        matched=pl.col("match_id").is_in(matched["match_id"].implode()),
    )
    print(
        un.group_by("source", "match_country")
        .agg(pl.len().alias("rows"), (~pl.col("matched")).mean().alias("unmatched_share"))
        .sort("source", "match_country")
    )
    print(f"\nOverall unmatched share: {float((~un['matched']).mean()):.4f}")

    # ---- country agreement --------------------------------------------------
    h("Country agreement between S1 and its matches")
    cc = pos.join(s1, on=S1).join(others, on="match_id")
    print(cc.group_by("country", "match_country").len("pairs").sort("country", "match_country"))
    print(f"\nPairs with differing country: {cc.filter(pl.col('country') != pl.col('match_country')).height:,} of {cc.height:,}")

    # ---- per-country breakdown ----------------------------------------------
    h("Per-country breakdown of S1 cardinality")
    print(
        card.group_by("country").agg(
            pl.len().alias("n_s1"),
            (pl.col("n_match") == 0).mean().alias("singleton_rate"),
            (pl.col("n_match") == 1).mean().alias("share_1"),
            (pl.col("n_match") == 2).mean().alias("share_2"),
            (pl.col("n_match") >= 3).mean().alias("share_3plus"),
            pl.col("n_match").mean().alias("mean_matches"),
            pl.col("n_match").filter(pl.col("n_match") > 0).mean().alias("mean_nonsingleton"),
            (pl.col("n_s2").sum() / pl.col("n_match").sum()).alias("s2_share_of_matches"),
        ).sort("country")
    )

    counts = {f"train_S{s}": df.height for s, df in sources.items()}
    del sources, s1, others, gt_raw, gt, pos, card, owners, un, cc, combo
    gc.collect()
    return counts


def test_eda(train_counts: dict[str, int]) -> None:
    sources = {s: slim_source("test", s) for s in (1, 2, 3)}
    source_overview("test", sources)
    h("Train / test size ratio")
    for s, df in sources.items():
        tr = train_counts[f"train_S{s}"]
        print(f"S{s}: train {tr:,}  test {df.height:,}  test/train {df.height / tr:.3f}")
    del sources
    gc.collect()


def main() -> None:
    print(f"# R1 EDA  (polars {pl.__version__})")
    counts = train_eda()
    gc.collect()
    h("Train freed")
    test_eda(counts)
    h("Done")


if __name__ == "__main__":
    main()
