# HANDOFF: Context and Plan for the Remaining Challenge

Read this together with the attached CLAUDE.md and STATUS.md. It fills in what the previous chat did not have (deadline, ranking rules, deliverables) and sets the order of the remaining work.

## 1. Hard facts from the challenge rules

| Item | Detail |
|---|---|
| Deadline | **27 Sep 2026, 23:59 IST** |
| Leaderboard submissions | Max **5 per day**; unused ones expire at midnight IST |
| Leaderboard file | `matching_results.tsv` (tab-separated, header `source1_entity_id	matched_entity_ids`, one row per test S1, comma-separated S2/S3 IDs, empty for no match) |
| Metric | Macro F0.5 per S1 entity, singletons included (empty prediction on a singleton = 1.0, any prediction on it = 0.0) |
| Final ranking | Private leaderboard score **plus** a review of `candidate_pairs.tsv` and the code that produces it. **A smaller candidate set per S1 ranks higher.** Blocking must scale (no all-pairs). |
| `candidate_pairs.tsv` | Same format (`source1_entity_id	candidate_entity_ids`). Must be the **exact** set the matcher runs inference on. Every matched ID must appear in it. |
| Validator | `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` must print PASS before any upload |
| Model constraint | Any pretrained model: MIT or Apache 2.0 license, at most 8B parameters |

### Final zip (all teams)

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/business_entity_resolution/
│   ├── src/                 # full pipeline
│   ├── README.md            # exact end-to-end run steps
│   └── requirements.txt     # pinned versions
└── Documentation_template.md  # filled in: methodology, blocking strategy,
                               # model architecture + features, other details
```

The pipeline must regenerate both output files from the raw data using only what is in `code/`.

## 2. Test-data policy (G1)

Test data is inference-only. Nothing is fitted on test records: no training, no TF-IDF/IDF/vocabulary fitting, no dictionaries or rules derived from test rows, no threshold tuning, no leaderboard-driven tuning. France appears only in test and is treated as fully unseen. It gets the global model and conservative, relative (rank/margin-based) cutoffs.

## 3. Current state (as of 26 Sep evening)

- Normalisation is done for train and test (`dataset/cache/{train,test}/norm_source{1,2,3}.parquet`).
- Blocking v1 is built. Oracle macro F0.5: **0.968 India, 0.984 US**. Known losses: 3.0% dropped by the top-30 cap, 1.8% missed at key lookup.
- One full blocking run takes about 1.7 hours per split. The test blocking run is in progress.
- **No leaderboard submission made yet.**

## 4. Revised order of work

The matcher comes **before** blocking v2. The oracle ceiling is already high, so blocking v2 can add at most about 2 points. A trained matcher with a proper decision layer is where most of the remaining score is.

| When (IST) | Work |
|---|---|
| 26 Sep, before midnight | Test blocking finishes → `make_submission.py` with a simple rule tuned on train candidates → validator → submit **lb-1** (day-2 slots expire at midnight). Start `features.py`. |
| 27 Sep, morning | `features.py` + `model.py` (LightGBM) + `decide.py` → score on CV → submit **lb-2**. |
| 27 Sep, early afternoon | One or two improvements chosen from the NonStandard.md symptom index → **lb-3 / lb-4**. Blocking v2 only if time remains. |
| **27 Sep, 18:00: freeze experiments** | No new modelling after this point. |
| 18:00–23:00 | `run_pipeline.py --stage {all,blocking,train,predict}`, pinned `requirements.txt`, README, filled `Documentation_template.md`, fresh-clone rerun on the laptop, validator, final submission, build the zip. |
| 23:00–23:59 | Buffer only. |

### Matcher details

- **Training data:** sample about **300k S1 entities** and all their candidates (roughly 2M pairs), grouped by S1 with the existing GroupKFold folds. Do not train on all ~13M train pairs on a 16 GB laptop.
- **Memory:** compute features in chunks, and score the full test candidate set in chunks.
- **Decision layer:**
  - Each S2/S3 record goes to at most one S1 (its best-scoring S1 above a threshold).
  - Tune the thresholds for macro F0.5 on CV.
  - Report overall, per-country, singleton vs non-singleton, oracle, and mean candidates per S1.
- **Candidate size:** keep the *final* candidate list per S1 small (target about 6 or fewer on average), and report it next to every score.

## 5. Working protocol (token economy)

- Write complete files in one reply. The user saves them, runs them in PowerShell, and pastes back only the result lines or errors.
- Nothing long-running is executed by the assistant. Every long job gets an exact command for the user to run.
- Output stays short: summaries only, never full tables.
- Commit after every working step, and tag each leaderboard submission `lb-<n>`.
- Laptop: Windows, 16 GB RAM, Python 3.12 venv. Run scripts with `.venv\Scripts\python.exe` from the project root.
- Usage is limited (weekly allowance about 28% used as of midday 26 Sep), so prefer fewer, larger replies over many small ones.

## 6. Immediate next task

Write `make_submission.py` as a complete file. It reads the test candidates from blocking, applies a simple rule tuned on train candidates (e.g. best S1 per S2/S3 record above a similarity threshold), writes both TSVs to `output/`, and prints the validator command to run. Include the command that tunes the rule's threshold on train candidates for macro F0.5.
