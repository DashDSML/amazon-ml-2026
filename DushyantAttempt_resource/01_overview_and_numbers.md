# Our approach: overview and headline numbers

Written for a teammate whose own pipeline (see `PrajanyaAttempt_resource/`) is already ahead
on the leaderboard (0.978). This folder is what we found that might still be useful to you --
one real compliance flag on your side (`00_peer_review_of_Prajanya_approach.md`), one idea we
validated that you haven't tried (blocking v2's rarity-weighted retrieval ranking), several
negative results that converge with your own D42/D48 findings, and code you're welcome to run
as-is on your own compute if useful (`05_code_and_how_to_run.md`).

## Pipeline shape
```
raw TSVs -> normalize.py -> blocking.py (--version v1|v2) -> features.py -> model.py
-> decide.py -> make_submission.py
```
`--version v2` switches every stage to a separate `dataset/cache/blocking_v2/` tree; `v1`
(default) is byte-identical to the original pipeline. See `code/business_entity_resolution/
README.md` for exact commands.

## Numbers, most to least mature

| Stage | Metric | v1 (submitted, `lb-2`) | v2 (built, not submitted) |
|---|---|---|---|
| Blocking oracle F0.5 (train, real full-scale) | | 0.9778 | **0.9893** |
| Mean candidates/S1 (train) | | 5.82 | **5.40** (smaller *and* better) |
| Mean candidates/S1 (test) | | 7.85 | 7.08 |
| Matcher CV F0.5 (fold-4 holdout) | | 0.9541 | **0.9628** |
| Public leaderboard | | **0.943** | not submitted |

v2's blocking is a strict improvement over v1 (higher oracle, smaller candidate list --
see `02_blocking_v2_fixes.md`). The matcher (LightGBM + a one-owner decision rule) improved
alongside it since it's trained on v2's better candidates, but four separate attempts to
improve the *decision layer or feature set itself* (independent of the blocking swap) all
failed to move F0.5 beyond noise -- see `03_matcher_experiments_negative_results.md`. This
converges with your own D42 finding (decision-rule sophistication moves F0.5 by ≤0.0002).

## What's still running / not finished
- A supervised meta-blocking "ranker" (train a LightGBM on the wide, unpruned candidate pool
  to make the pruning decision instead of a hand-weighted score) -- data built (781M rows),
  training in progress at the time of writing. See `05_code_and_how_to_run.md` for how to
  pick this up; disclosed explicitly as changing what `candidate_pairs.tsv` content means,
  per the same T21-style concern that applies to your stage-2 model if it ever became the
  pruning stage rather than a post-blocking re-scorer.
- T19 (noise-operator augmentation on training records, to build in the invariances France's
  orthography needs without touching French rows) -- identified as the most promising
  *untried* lever for the France gap, not attempted due to time.

## Files in this folder
1. `00_peer_review_of_Prajanya_approach.md` -- read this first, especially §1.
2. `01_overview_and_numbers.md` -- this file.
3. `02_blocking_v2_fixes.md` -- the three fixes that raised oracle 0.9778→0.9893, with the
   G1 reasoning for each (relevant given your own IDF-table question).
4. `03_matcher_experiments_negative_results.md` -- four validated "didn't help" results.
5. `04_loco_unseen_country_findings.md` -- our France-risk quantification (cross-validates
   your D47).
6. `05_code_and_how_to_run.md` -- how to run any of this yourself, including the ranker.
