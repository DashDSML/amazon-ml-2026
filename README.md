# Amazon ML Challenge 2026: Business Entity Resolution (Dushyant's attempt)

Given business records from three sources with noisy names and addresses, find every Source 2 / Source 3 record that refers to the same real-world business as each Source 1 entity. Scoring is macro F0.5 per S1 entity, and the size of the candidate set counts toward the ranking. Test data adds France, a country never seen in training.

**Final result:** leaderboard F0.5 **0.954** (submission `lb-3`), from CV F0.5 0.9669.

| Submission | Pipeline | CV F0.5 | Leaderboard F0.5 |
|---|---|---|---|
| `lb-2` | blocking v1 + LightGBM matcher | 0.9541 | 0.943 |
| `lb-3` | blocking v2 + supervised ranker pruning + tuned matcher (v2r) | 0.9669 | 0.954 |

## Pipeline

```
raw TSVs -> normalize -> blocking -> features -> LightGBM matcher -> decide -> submission
```

All code is in `code/business_entity_resolution/src/` (see its `README.md` for commands). Entry point: `run_pipeline.py --stage {all,blocking,train,predict}`.

1. **Normalization.** Legal forms, street types and abbreviations are mined from training pairs only. A French lexicon is added from general knowledge, because France is test-only and no rule may be derived from test rows.
2. **Blocking v1.** An inverted index over hashed name and address tokens, with country as a hard key. Oracle F0.5 is 0.9778 at 5.82 candidates per S1.
3. **Blocking v2.** Three fixes found by bucketing every v1 miss by cause:
   - ranking by rarity-weighted key evidence instead of raw block size;
   - folding key evidence into the stage-1 score;
   - composite keys, one from the two rarest name tokens and one from an address token plus city.

   Oracle rises to 0.9893 at 5.40 candidates per S1, so the list got smaller and better.
4. **Supervised meta-blocking ranker (the biggest win).** A LightGBM model trained on a wide pool of about 80 candidates per record (about 781M rows) replaces the hand-weighted pruning score. Oracle rises to 0.9977 and mean candidates per S1 fall to about 4.0. Retraining the matcher on these candidates lifted CV from 0.9628 to 0.9661.
5. **Matcher.** LightGBM over 34 features, including *confusability*: the similarity of an S1 record to its nearest other S1, which is a free non-match signal because S1 is deduplicated. A 15-config hyperparameter sweep, scored with the real F0.5 metric, took CV from 0.9661 to 0.9669.
6. **Decision.** A one-owner rule (each S2/S3 record matches at most one S1) on top of a probability threshold.

## What didn't help

Ten techniques (four decision-layer rules, sample weighting, a logistic-regression blend, multi-seed ensembles and others) converged to within about 0.008 of each other. This suggests the current feature set is near its ceiling. Details are in `SESSION_CHRONOLOGY.md` and `DushyantAttempt_resource/03_matcher_experiments_negative_results.md`.

## Unseen-country risk

A leave-one-country-out test, with India held out as a stand-in for France, dropped F0.5 from 0.9541 to 0.9042. Tuning the threshold on the held-out country recovered only about 0.009 of that. The cause is a feature and model generalization limit, not threshold calibration. This is why the leaderboard score sits about 0.011–0.013 below CV. See `DushyantAttempt_resource/04_loco_unseen_country_findings.md`.

## Untried ideas

- Noise-operator augmentation (accents, apostrophes, hyphens, casing) to build in invariances for French.
- A character n-gram TF-IDF retrieval channel. It recovered about 44% of the pairs the current blocking misses, but was never wired into the pipeline.

## Repository guide

| Path | Contents |
|---|---|
| `code/business_entity_resolution/` | Pipeline source, pinned `requirements.txt`, mined `resources/` |
| `SESSION_CHRONOLOGY.md` | Full decision log: options considered, what was chosen and rejected, and results |
| `DushyantAttempt_resource/` | Write-ups of blocking v2, negative results and the unseen-country test, plus a code snapshot |
| `BLOCKING_V2.md`, `HANDOFF.md`, `STATUS.md` | Planning and status notes |
| `challengecontext.md`, `NonStandard.md` | Problem statement and technique catalog |
| `experiments/log.csv` | One row per experiment |

The dataset (about 2.5 GB), `output/` and all `.tsv` files are git-ignored. Place the provided data in `dataset/` to reproduce.
