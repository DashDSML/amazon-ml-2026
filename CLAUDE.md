# Project: Amazon ML Challenge 2026 – Business Entity Resolution

Read these before doing anything:
- `challengecontext.md`: problem, data, metric, output format, deliverables
- `NonStandard.md`: advanced techniques. Use its symptom → technique index when tuning.

## Hard rules
- Read every TSV with `sep="\t"`, `dtype=str`, `keep_default_na=False`. Assert the column count after loading.
- Never make network calls to look up businesses, addresses or geocodes. Use only the provided data. Package installs from PyPI are fine.
- Any pretrained model in the final pipeline must be MIT or Apache 2.0 licensed and ≤8B parameters. Check the license of the exact checkpoint before adding it.
- `country` is an open set. France appears only in test. Never hard-code {US, India}.
- IDs in `matching_results.tsv` must be a subset of `candidate_pairs.tsv`. `candidate_pairs.tsv` is the final candidate set the model scores.
- Test data is inference-only. Never fit anything on test records: no training, no TF-IDF/vocabulary/IDF fitting, no dictionaries or rules derived from test rows, no threshold tuning. France (test-only) is treated as fully unseen. See NonStandard.md G1.
- Candidate-set size per S1 counts toward the final ranking (smaller ranks higher). Every blocking change must report oracle macro F0.5 *and* mean/median/p95 candidates per S1. See NonStandard.md G2 and T22.
- Blocking must scale: no all-pairs comparisons; use inverted indexes, chunked sparse top-K, or ANN indexes.
- Run `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` before calling any output ready.

## Layout
- `dataset/`, `utils/`: provided files (do not modify)
- `code/business_entity_resolution/src/`: all pipeline code. Suggested modules: `data.py`, `normalize.py`, `blocking.py`, `features.py`, `model.py`, `decide.py`, `evaluate.py`, `run_pipeline.py`
- `code/business_entity_resolution/README.md` and `requirements.txt` (pinned versions)
- `notebooks/`: exploration only. Pipeline logic lives in `src/`.
- `output/`: `matching_results.tsv`, `candidate_pairs.tsv`
- `experiments/log.csv`: one row per experiment

## Evaluation
- Judge every change by macro F0.5 on GroupKFold-by-S1 validation with the full S2/S3 distractor pool (NonStandard T23).
- Always report: overall score, per country, singleton vs non-singleton, the blocking oracle score, and mean candidates per S1 (T22).
- Log each experiment to `experiments/log.csv`: date, git hash, change, CV scores, leaderboard score if submitted.

## Workflow
- One change at a time. Commit before and after each experiment.
- Tag every leaderboard submission `lb-<n>` (max 5 per day).
- Fixed random seeds. Single entry point: `python code/business_entity_resolution/src/run_pipeline.py --stage {all,blocking,train,predict}`.

## Data scale
- The dataset is about 2.5 GB (each Source 2/3 file is about 500 MB) and the dev laptop has 16 GB RAM.
- Load with polars or pyarrow. Do not use pandas object (Python string) columns for full files. Convert only small slices to pandas.
- Never load train and test at the same time. Finish with one split and free it before loading the other.
- Every step must stay within a fixed memory budget: process in chunks (S1 batches, streaming or lazy scans, chunked sparse top-K), select only the columns you need, and don't materialise large Python lists or dicts of strings.

## Environment
- Windows, PowerShell. Use `python` (not `python3`), `pathlib` for all paths, and keep the code OS-independent so it also runs on Linux/EC2.
- Python 3.12 in `.venv`, created with `py -3.12 -m venv .venv`. Activate with `.venv\Scripts\Activate.ps1` (Windows) or `source .venv/bin/activate` (Linux).
