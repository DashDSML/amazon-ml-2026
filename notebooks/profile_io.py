"""Time each I/O step of the R1 EDA per file: raw line count, TSV parse,
parquet cache build and parquet parse.

Run:  python notebooks/profile_io.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution" / "src"))
from data import (  # noqa: E402
    GT_COLUMNS,
    SOURCE_COLUMNS,
    _scan_tsv,
    build_cache,
    count_data_lines,
    ground_truth_path,
    source_path,
)


def slim(lf: pl.LazyFrame, gt: bool) -> pl.DataFrame:
    if gt:
        return lf.collect()
    return lf.select(
        "entity_id",
        pl.col("country").cast(pl.Categorical),
        (pl.col("business_name").str.strip_chars() == "").alias("name_missing"),
        (pl.col("business_address").str.strip_chars() == "").alias("addr_missing"),
    ).collect()


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def main() -> None:
    files = [(source_path(sp, s), SOURCE_COLUMNS, False) for sp in ("train", "test") for s in (1, 2, 3)]
    files.append((ground_truth_path(), GT_COLUMNS, True))
    rows = []
    for path, cols, gt in files:
        _, t_lines = timed(lambda: count_data_lines(path))
        _, t_tsv = timed(lambda: slim(_scan_tsv(path, cols), gt))
        cached, t_build = timed(lambda: build_cache(path, cols, force=True))
        _, t_pq = timed(lambda: slim(pl.scan_parquet(cached), gt))
        rows.append(dict(
            file=path.name,
            tsv_mb=path.stat().st_size / 2**20,
            parquet_mb=cached.stat().st_size / 2**20,
            line_count_s=t_lines,
            tsv_parse_s=t_tsv,
            cache_build_s=t_build,
            parquet_parse_s=t_pq,
        ))
        print(rows[-1], flush=True)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_float_precision(1)
    df = pl.DataFrame(rows)
    print(df)
    print(df.select(pl.exclude("file").sum()))


if __name__ == "__main__":
    main()
