# Code snapshot and how to run it

`code_snapshot/` is a self-contained copy of every script referenced in this folder (pinned
`requirements.txt` included). It's not a diff or a patch -- these are the actual files, so you
can drop them into your own environment and run them directly, adjusting only the data paths
if your directory layout differs from ours (`dataset/train/`, `dataset/test/`,
`dataset/cache/...` -- see `data.py`'s `REPO_ROOT`/`DATASET_DIR` for the exact expected
layout, computed relative to wherever you place these files under `code/<pkg>/src/`).

**Not included** (needed to actually run normalize.py from raw data): the `resources/`
directory (mined abbreviation/legal-form/gibberish tables, built by `mine_resources.py` from
training pairs only). If you want to run the normalization step fresh rather than reuse ours,
run `mine_resources.py` first against your own copy of the training data -- it's train-only
by construction (see its own docstring), so this is safe and reproducible from scratch.

## What's genuinely worth running on better compute

**The ranker** (`ranker_data.py` -> `ranker_features.py` -> `ranker_model.py`) is the
heaviest thing we built and the one most likely to benefit from more RAM/cores than we had:
1. `ranker_data.py` reruns blocking's retrieval+scoring with the per-record save-cap raised
   from 5 to ~80 (everything retrieved and scored, not just the top 5 kept) -- same cost as a
   normal full blocking run since the expensive part (scoring) is identical either way, only
   the write cutoff differs. Produces a ~780M-row wide candidate pool (train only -- needs
   ground-truth labels).
2. `ranker_features.py` computes the same 29 features as the shipped matcher over that wide
   pool, chunked by part-file (an unchunked version OOM-killed at ~61GB RSS on a 62GB
   machine -- chunking is not optional at this scale, and your compute budget will determine
   how large a chunk is safe).
3. `ranker_model.py` trains on a class-balanced sample of the wide pool (positive rate is
   only ~1%) and evaluates by re-ranking a sample of held-out S1's full wide candidate set by
   the ranker's probability instead of the hand-weighted blocking score, sweeping the usual
   floor/alpha/m grid, and comparing oracle F0.5 / mean candidates-per-S1 to the frozen
   blocking-v2 baseline (0.9893 / 5.40). **This still needs a real evaluation run to know if
   it helps at all -- we ran out of wall-clock time on our own EC2 instance before getting a
   validated result.** If you have compute to spare, this is the most valuable thing to
   actually finish.

Disclosure note for whichever of us ships this: if the ranker's decision *becomes* the
content of `candidate_pairs.tsv` (replacing the current hand-weighted score as the pruning
criterion), that's the same category of thing as your own stage-2 model would be if it were
moved earlier in the pipeline -- worth stating explicitly in the methodology doc either way
(see our peer review, §1, for why we're being careful about this kind of thing right now).

## What's cheap and safe to just re-run for sanity

- `loco_validate.py --version v1` (or point it at your own features/model if you adapt the
  `--version` plumbing) -- reruns our leave-one-country-out check. Real full-scale run takes
  ~15-20 min per held-out country (two retrains) on an 8-core/16GB-class machine.
- `decide_entity.py` / `model_weighted.py` -- our two negative decision-layer results. Useful
  mainly as a reference implementation if you want to re-verify the "decision layer isn't the
  bottleneck" finding on your own features rather than take our word for it.

## Everything else
`blocking.py`, `features.py`, `model.py`, `decide.py`, `make_submission.py` are the shipped
pipeline (`--version {v1,v2}`, see `01_overview_and_numbers.md`). `cv_harness.py` has the
official metric implementation and the GroupKFold-by-S1 fold assignment; `evaluate.py` is a
second, self-testing implementation of the same macro-F0.5 formula, used for final checks.
`normalize.py` + `mine_resources.py` + `check_norm.py` are the normalization layer and its
train-only resource-mining step, only needed if you want to reproduce from raw TSVs rather
than reuse a cache.
