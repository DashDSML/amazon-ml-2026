# Business Entity Resolution — Pipeline README

Reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv` end-to-end from
the raw TSVs in `dataset/`, using only what is in this `code/` folder.

## Environment

- Python 3.12, Windows or Linux (all paths use `pathlib`; the pipeline has been run on both
  Windows and an Ubuntu 24.04 EC2 instance without changes).
- `py -3.12 -m venv .venv` at the repo root, then activate it
  (`.venv\Scripts\Activate.ps1` on Windows, `source .venv/bin/activate` on Linux) and
  `pip install -r code/business_entity_resolution/requirements.txt`.
- Run every command below from the repo root, or from `code/business_entity_resolution/src/`
  (both work — `python` adds the script's own directory to `sys.path`).

## Pipeline shape

```
raw TSVs (dataset/{train,test}/*.tsv)
  -> normalize.py       multi-view name/address normalisation (cached to dataset/cache/{split}/norm_source*.parquet)
  -> blocking.py         S2/S3-side retrieval + adaptive top-m pruning -> candidates.parquet
  -> features.py         29 pairwise features per candidate -> features.parquet
  -> model.py             LightGBM binary classifier (GroupKFold folds 0-3 train, fold 4 early-stop)
  -> decide.py             one-owner "record's top-probability candidate, prob>=t" decision layer
  -> make_submission.py   writes output/matching_results.tsv + candidate_pairs.tsv, runs the validator
```

`--version {v1,v2}` runs through every stage from `blocking.py` onward, selecting
`dataset/cache/blocking_{version}/` for both input and output at each step. `v1` is the
original blocking design; `v2` (BLOCKING_V2.md) adds rarity-weighted retrieval ranking,
composite blocking keys, and key-evidence-boosted scoring — see `experiments/log.csv` and
`STATUS.md` for the measured oracle F0.5 / candidates-per-S1 of each. Default is `v1`.

## Exact commands (full data — each of these is a genuine multi-hour job on a laptop;
budget accordingly, or run on a larger machine)

```bash
# 1. Build the normalised parquet cache (both splits, all three sources each)
python code/business_entity_resolution/src/normalize.py --split train
python code/business_entity_resolution/src/normalize.py --split test

# 2. Blocking (choose --version v1 or v2; the operating point below is v2's frozen choice)
python code/business_entity_resolution/src/blocking.py --split train --version v2 --floor 0.5 --alpha 0.95 --m 3
python code/business_entity_resolution/src/blocking.py --split test  --version v2 --floor 0.5 --alpha 0.95 --m 3

# 3. Pairwise features (train and test, same --version)
python code/business_entity_resolution/src/features.py --split train --version v2
python code/business_entity_resolution/src/features.py --split test --version v2

# 4. Train the matcher
python code/business_entity_resolution/src/model.py --version v2

# 5. Tune the one-owner decision threshold on the fold-4 holdout
python code/business_entity_resolution/src/decide.py --version v2

# 6. Write the submission and validate it
python code/business_entity_resolution/src/make_submission.py --version v2 --method model
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

`make_submission.py` also supports `--method rule` (a lighter rank/score threshold rule
tuned directly on train candidates, no LightGBM training required) as a faster fallback path
that does not depend on steps 3-5.

## What each script is (`code/business_entity_resolution/src/`)

**Pipeline (needed to reproduce `output/`):**
- `data.py` — TSV loading, parquet caching, column-count assertions.
- `normalize.py` — multi-view name/address normalisation; mines nothing from test.
- `cv_harness.py` — GroupKFold-by-S1 folds, the official-metric evaluation harness, experiment logging.
- `blocking.py` — retrieval + adaptive top-m pruning; `--version v2`'s fixes are documented in its own docstring and in `BLOCKING_V2.md`.
- `features.py` — pairwise feature engineering (country excluded by design — see `NonStandard.md` G1).
- `model.py` — LightGBM training (CV booster + full-retrain booster).
- `decide.py` — the decision layer that ships in `output/`.
- `make_submission.py` — orchestrates the above into the two output TSVs and runs the validator.
- `evaluate.py` — the official macro F0.5 implementation (self-testing).

**Development / diagnostic only (not required to reproduce `output/`):**
- `fastloop.py` — a ~5-minute sampled proxy for a full blocking run, used to iterate on blocking changes without paying the full runtime.
- `mine_resources.py`, `check_norm.py` — build/check the train-only normalisation resource tables under `resources/`.
- `decide_entity.py`, `model_weighted.py` — alternative decision-layer / training approaches (entity-centric gate, metric-aware sample weights) that were tried and validated *not* to beat the shipped approach; kept for the record, not part of the reproduction path.
- `loco_validate.py` — leave-one-country-out robustness check (a France proxy); a validation tool, produces no pipeline artefact.
- `ranker_data.py` — generates a wider candidate pool for an in-progress supervised meta-blocking ranker (see `STATUS.md` for its status as of the last commit).

## Reproducibility

Fixed seeds throughout (`SEED = 42` in `blocking.py`/`model.py`/`features.py`); pinned
versions in `requirements.txt`. `experiments/log.csv` has one row per real (non-smoke)
experiment: date, git commit, change, CV scores, and leaderboard score where submitted.
