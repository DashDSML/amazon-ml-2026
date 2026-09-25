# R1 EDA  (polars 1.44.2)

## train: rows, countries, missingness per source  (RSS 1.41 GiB, previous section 3.7s)

| source | country | rows    | name_missing | addr_missing |
| ---    | ---     | ---     | ---          | ---          |
| str    | str     | u32     | f64          | f64          |
|--------|---------|---------|--------------|--------------|
| S1     | India   | 883188  | 0.0000       | 0.0000       |
| S1     | US      | 1323633 | 0.0000       | 0.0000       |
| S2     | India   | 2017799 | 0.0000       | 0.0287       |
| S2     | US      | 3016817 | 0.0000       | 0.0368       |
| S3     | India   | 2115547 | 0.0000       | 0.0307       |
| S3     | US      | 3170056 | 0.0000       | 0.0350       |

| source | rows    |
| ---    | ---     |
| str    | u32     |
|--------|---------|
| S1     | 2206821 |
| S2     | 5034616 |
| S3     | 5285603 |

## Ground truth integrity  (RSS 0.68 GiB, previous section 0.4s)

GT rows: 2,206,821   unique S1 in GT: 2,206,821   duplicate S1 rows: 0
S1 in source1 but not in GT: 0
S1 in GT but not in source1: 0
Duplicate IDs inside one matched list: 0
Matched IDs without S2-/S3- prefix: 0
Matched IDs absent from train S2/S3 files: 0
Positive pairs: 7,638,365

## Match cardinality per S1 entity  (RSS 2.45 GiB, previous section 4.8s)

| bucket | n_s1   | share  |
| ---    | ---    | ---    |
| str    | u32    | f64    |
|--------|--------|--------|
| 0      | 123247 | 0.0558 |
| 1      | 119157 | 0.0540 |
| 2      | 375212 | 0.1700 |
| 3      | 530841 | 0.2405 |
| 4      | 484115 | 0.2194 |
| 5      | 321957 | 0.1459 |
| 6+     | 252292 | 0.1143 |

Singleton rate (= macro F0.5 of all-empty prediction): 0.0558
Mean matches per S1: 3.461 (all S1), 3.666 (non-singletons)
Median / p95 / max matches among non-singletons: 4 / 6 / 11

## S2 vs S3 contribution  (RSS 2.40 GiB, previous section 0.9s)

Matches from S2: 3,693,619   from S3: 3,944,746
| combo   | n_s1    | share  |
| ---     | ---     | ---    |
| str     | u32     | f64    |
|---------|---------|--------|
| S2 only | 143029  | 0.0686 |
| S2+S3   | 1776047 | 0.8524 |
| S3 only | 164498  | 0.0789 |

Matches per non-singleton S1, by source (k = count from that source):
| k   | S2     | S3     |
| --- | ---    | ---    |
| u32 | u32    | u32    |
|-----|--------|--------|
| 0   | 164498 | 143029 |
| 1   | 789108 | 716417 |
| 2   | 652779 | 668375 |
| 3   | 333957 | 372443 |
| 4   | 119078 | 145116 |
| 5   | 24154  | 35378  |
| 6   | 0      | 2816   |

## One-owner check: does any S2/S3 ID appear under more than one S1?  (RSS 2.41 GiB, previous section 0.1s)

Distinct matched S2/S3 IDs: 7,638,365
IDs claimed by >1 S1: 0   max owners: 1

## S2/S3 records that match no S1 (distractors)  (RSS 3.08 GiB, previous section 0.6s)

| source | match_country | rows    | unmatched_share |
| ---    | ---           | ---     | ---             |
| str    | str           | u32     | f64             |
|--------|---------------|---------|-----------------|
| S2     | India         | 2017799 | 0.2663          |
| S2     | US            | 3016817 | 0.2664          |
| S3     | India         | 2115547 | 0.2535          |
| S3     | US            | 3170056 | 0.2538          |

Overall unmatched share: 0.2599

## Country agreement between S1 and its matches  (RSS 2.20 GiB, previous section 9.4s)

| country | match_country | pairs   |
| ---     | ---           | ---     |
| str     | str           | u32     |
|---------|---------------|---------|
| India   | India         | 3059843 |
| US      | US            | 4578522 |

Pairs with differing country: 0 of 7,638,365

## Per-country breakdown of S1 cardinality  (RSS 3.48 GiB, previous section 1.8s)

| country | n_s1    | singleton_rate | share_1 | share_2 | share_3plus | mean_matches | mean_nonsingleton | s2_share_of_matches |
| ---     | ---     | ---            | ---     | ---     | ---         | ---          | ---               | ---                 |
| str     | u32     | f64            | f64     | f64     | f64         | f64          | f64               | f64                 |
|---------|---------|----------------|---------|---------|-------------|--------------|-------------------|---------------------|
| India   | 883188  | 0.0559         | 0.0537  | 0.1698  | 0.7206      | 3.4645       | 3.6696            | 0.4839              |
| US      | 1323633 | 0.0558         | 0.0542  | 0.1702  | 0.7198      | 3.4591       | 3.6636            | 0.4834              |

## Train freed  (RSS 3.53 GiB, previous section 0.1s)


## test: rows, countries, missingness per source  (RSS 1.27 GiB, previous section 3.9s)

| source | country | rows    | name_missing | addr_missing |
| ---    | ---     | ---     | ---          | ---          |
| str    | str     | u32     | f64          | f64          |
|--------|---------|---------|--------------|--------------|
| S1     | France  | 259452  | 0.0000       | 0.0000       |
| S1     | India   | 809986  | 0.0000       | 0.0000       |
| S1     | US      | 663106  | 0.0000       | 0.0000       |
| S2     | France  | 703378  | 0.0000       | 0.0306       |
| S2     | India   | 2312565 | 0.0000       | 0.0228       |
| S2     | US      | 1871330 | 0.0000       | 0.0294       |
| S3     | France  | 731615  | 0.0000       | 0.0294       |
| S3     | India   | 2405000 | 0.0000       | 0.0246       |
| S3     | US      | 1945701 | 0.0000       | 0.0284       |

| source | rows    |
| ---    | ---     |
| str    | u32     |
|--------|---------|
| S1     | 1732544 |
| S2     | 4887273 |
| S3     | 5082316 |

## Train / test size ratio  (RSS 0.79 GiB, previous section 0.2s)

S1: train 2,206,821  test 1,732,544  test/train 0.785
S2: train 5,034,616  test 4,887,273  test/train 0.971
S3: train 5,285,603  test 5,082,316  test/train 0.962

## Done  (RSS 0.79 GiB, previous section 0.0s)


## Profiling (2026-09-25)

I/O per step, summed over all 7 files (`notebooks/profile_io.py`, warm OS file cache):

| step | seconds |
|---|---|
| raw line count (2.4 GB) | 3.9 |
| TSV parse, slim columns | 4.7 |
| parquet cache build (one-off, 973 MB zstd) | 11.6 |
| parquet parse, slim columns | 2.8 |

A full EDA run on the parquet cache takes 27 s wall-clock. The slowest section is country agreement (about 10 s: a join of 7.6M pairs against 10.3M S2/S3 IDs). Peak RSS is about 3.5 GiB. RSS stays high after train is freed because the allocator keeps the memory rather than returning it.

The first run took more than 10 minutes, so its time was not in the script's compute. That command also ran `pip install psutil`, and the TSVs were read cold for the first time (possibly with antivirus scanning). I have not confirmed the cause with a cold re-run.
