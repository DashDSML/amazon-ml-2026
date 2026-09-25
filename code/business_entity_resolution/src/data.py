"""TSV loading for the business entity resolution pipeline.

Every file is read with tab separators, all columns as strings, empty cells kept
as "" (never null), and quoting disabled so stray quote characters inside names
or addresses cannot swallow tabs/newlines. The header is asserted on load.

Uses polars (Arrow-backed strings) so full files never become pandas object
columns. Prefer the lazy ``scan_*`` functions plus ``.select(...)`` when only a
few columns are needed; never hold train and test in memory at the same time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "dataset"

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

Split = Literal["train", "test"]
Source = Literal[1, 2, 3]

_CSV_KWARGS = dict(
    separator="\t",
    has_header=True,
    infer_schema=False,  # every column is pl.String
    quote_char=None,
    missing_utf8_is_empty_string=True,
    encoding="utf8",
)


def source_path(split: Split, source: Source, dataset_dir: Path = DATASET_DIR) -> Path:
    return Path(dataset_dir) / split / f"{split}_source{source}.tsv"


def ground_truth_path(dataset_dir: Path = DATASET_DIR) -> Path:
    return Path(dataset_dir) / "train" / "train_ground_truth.tsv"


def _check_columns(names: list[str], expected: list[str], path: Path) -> None:
    assert len(names) == len(expected), (
        f"{path}: expected {len(expected)} columns, got {len(names)}: {names}"
    )
    assert names == expected, f"{path}: expected columns {expected}, got {names}"


def _scan(path: Path, expected: list[str]) -> pl.LazyFrame:
    lf = pl.scan_csv(path, **_CSV_KWARGS)
    _check_columns(lf.collect_schema().names(), expected, path)
    # Any NULL here would mean a short/ragged row slipped through.
    return lf.with_columns(pl.all().fill_null(""))


def scan_source(split: Split, source: Source, dataset_dir: Path = DATASET_DIR) -> pl.LazyFrame:
    """Lazy scan of one source file. Select columns before collecting."""
    return _scan(source_path(split, source, dataset_dir), SOURCE_COLUMNS)


def load_source(
    split: Split,
    source: Source,
    columns: list[str] | None = None,
    dataset_dir: Path = DATASET_DIR,
) -> pl.DataFrame:
    lf = scan_source(split, source, dataset_dir)
    if columns is not None:
        lf = lf.select(columns)
    return lf.collect()


def scan_ground_truth(dataset_dir: Path = DATASET_DIR) -> pl.LazyFrame:
    return _scan(ground_truth_path(dataset_dir), GT_COLUMNS)


def load_ground_truth_pairs(dataset_dir: Path = DATASET_DIR) -> pl.DataFrame:
    """Ground truth in long form: one row per (source1_entity_id, match_id).

    Singletons are kept as a single row with a null ``match_id`` so the set of
    S1 IDs is preserved; filter with ``match_id.is_not_null()`` for positives.
    """
    return (
        scan_ground_truth(dataset_dir)
        .select(
            "source1_entity_id",
            pl.col("matched_entity_ids").str.split(",").alias("match_id"),
        )
        .explode("match_id")
        .with_columns(
            pl.col("match_id").str.strip_chars().replace("", None)
        )
        .unique(maintain_order=True)
        .collect()
    )


def count_data_lines(path: Path, chunk_size: int = 1 << 24) -> int:
    """Number of non-header lines, counted from raw bytes (sanity check vs. parser)."""
    n = 0
    last = b"\n"
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            n += chunk.count(b"\n")
            last = chunk[-1:]
    if last != b"\n":
        n += 1  # final line without trailing newline
    return n - 1
