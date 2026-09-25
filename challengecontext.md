# Challenge Context: Amazon ML Challenge 2026, Business Entity Resolution

## 1. Timeline and Submission Budget

| Item | Detail |
|---|---|
| Window | 25 Sep 2026, 00:00 IST to 27 Sep 2026, 23:59 IST (~72 hours) |
| Leaderboard submissions | Max 5 per day, 3 days, so **15 total**. Unused ones do not roll over in any useful sense, but each one is precious. |
| Leaderboards | Public (subset of test, live) and Private (remaining test, revealed after). The leaderboard part of the ranking uses the Private score. |
| Final ranking (organisers' update) | Private leaderboard score **plus** a review of `candidate_pairs.tsv` and the code that produces it. **A smaller candidate set per Source 1 entity ranks higher.** Blocking must scale (billions of records in production; no all-pairs comparison). |
| Implication | Do not overfit to the public leaderboard. Trust a well-built local validation split more than small public LB swings. |

Keep a version history of every submission (code commit + config + output + LB score). Shortlisting is based on submitted solutions and final source code may be requested.

---

## 2. The Problem in One Paragraph

Three independent data sources describe businesses with noisy, partial fields and **no shared identifiers**. Source 1 (S1) is a **deduplicated reference** list. For every S1 entity, find **all** records in Source 2 (S2) and Source 3 (S3) that refer to the same real-world business. An S1 entity can match **zero, one, or many** S2/S3 records. This is a classic Entity Resolution (ER) task: candidate generation (blocking) followed by pairwise or group-level match classification.

---

## 3. Data

All files are **tab-separated**. Always read with `sep="\t"` (addresses and ID lists contain commas). Reading without it silently yields a single column holding the whole line, so assert the expected column count after loading. Also consider `dtype=str`, `keep_default_na=False` so empty fields and IDs are not mangled.

All paths below (`dataset/...`, `utils/...`, `output/...`) are relative to the provided `student_resource/` directory.

### Source files (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`)

| Column | Notes |
|---|---|
| `entity_id` | Prefix gives the source: `S1-`, `S2-`, `S3-`. No separate source column. |
| `business_name` | Abbreviations, legal suffixes, DBA/trade names, typos, transliterations, `&` vs `and`, word-order swaps |
| `business_address` | Partial addresses, abbreviations (Rd/Road, St/Street), missing PIN/ZIP/state, landmark references ("Near SBI ATM"), municipal numbering formats, reordered components, transliteration variants |
| `country` | Train: **US, India**. Test: US, India **and France (unseen in training)**. Treat as an open set of strings. |

### Ground truth (`train_ground_truth.tsv`)

| Column | Notes |
|---|---|
| `source1_entity_id` | S1 ID |
| `matched_entity_ids` | Comma-separated S2/S3 IDs; empty for singletons |

### Files

```
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv   <- every entity here must appear in the output
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

No test labels exist. **You must build your own validation split from train.**

---

## 4. Evaluation Metric (read this carefully, it drives most design decisions)

**Macro-averaged F0.5 per S1 entity.**

```
F_0.5 = (1.25 * P * R) / (0.25 * P + R)
```

Computed for each S1 entity, then averaged across all S1 entities (singletons included).

### Scoring cases per S1 entity

| Truth | Prediction | Score |
|---|---|---|
| No matches (singleton) | Empty list | **1.0** |
| No matches (singleton) | Any match | **0.0** |
| Has matches | Empty list | **0.0** (recall 0) |
| Has matches | Non-empty list | F0.5 of that list |

### Worked sensitivities

Entity with *k* true matches, predicting only correct ones (P = 1):

| k true | predicted correct | F0.5 |
|---|---|---|
| 1 | 1 | 1.000 |
| 2 | 1 | 0.833 |
| 3 | 1 | 0.714 |
| 3 | 2 | 0.909 |
| 4 | 2 | 0.833 |

Entity with 1 true match, prediction contains it plus wrong IDs:

| extra false IDs | P | F0.5 |
|---|---|---|
| 0 | 1.00 | 1.000 |
| 1 | 0.50 | 0.556 |
| 2 | 0.33 | 0.385 |

### What this means

1. **The "match anything vs. predict empty" decision per S1 entity is the single biggest lever.** Every entity is worth equal weight, so a wrongly-matched singleton costs a full point, and a wrongly-emptied non-singleton also costs a full point.
2. **Precision beats recall within an entity.** Getting one sure match is worth a lot; adding a doubtful one is risky. One false positive on a single-match entity cuts its score nearly in half.
3. Because it is macro-averaged, entities with many matches do not count more than entities with one. Optimise per-entity decisions, not global pair-level F1.
4. Thresholds should be tuned by **directly maximising macro F0.5 on validation**, not by pair-level AUC or accuracy. Consider an entity-level decision layer (e.g., "include top-1 if score > t1, include further candidates only if score > t2 where t2 > t1").

---

## 5. Required Outputs

Both are tab-separated, placed in `output/`.

### `matching_results.tsv` (scored)

```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003
```

### `candidate_pairs.tsv` (not on the leaderboard, but counts toward the final ranking)

Organisers' update: this file and the code that produces it are reviewed when deciding final rankings, alongside the `matching_results.tsv` score. The approach that generates a smaller candidate set per Source 1 entity ranks higher. It must still be the exact set the matching model runs inference over, so do not shrink it by relabelling part of the matcher as blocking.

```
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
```

This must be the **final candidate set that the matching model actually scores**, i.e., the last filtering stage before inference, not an earlier looser pass.

### Format rules (both files)

- Exactly one row per test S1 entity; no duplicate `source1_entity_id` rows
- Empty list allowed (singleton / no candidates)
- Only `S2-`/`S3-` IDs that exist in the test set; no S1 IDs
- No duplicate IDs within a list; commas, no quotes, no spaces
- Every matched ID must also be in that entity's candidate list
- Validate locally before every upload (run from the `student_resource/` directory; the script is stdlib-only):

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

  It prints `PASS` (exit 0) or a numbered list of issues (exit 1). It checks format only and does **not** compute a score. A matched ID missing from the candidate list triggers a warning, which signals a pipeline bug.
- On the Portal, a correctly formatted upload shows a `SCORED` status with its F0.5. Files that fail validation are not evaluated at all.

### Final submission package

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md          # exact end-to-end reproduction steps
│       └── requirements.txt   # pinned versions
└── Documentation_template.md  # filled-in methodology write-up
```

The pipeline must regenerate both output files from the raw train/test data using only what is in `code/`. Set random seeds and pin versions so results are reproducible.

The methodology doc must cover: methodology, candidate generation / blocking strategy, model architecture and feature engineering, and any other relevant details. No page limit; depth over brevity. Top 100 teams will be asked for these details again.

### Model constraint

The final model must be at most **8B parameters** under an **MIT or Apache 2.0** license. This affects which pretrained encoders / LLMs are usable (check the license of any embedding model or cross-encoder before building on it).

---

## 6. Structural Properties Worth Exploiting (verify on train first)

- **S1 is deduplicated**, so each S2/S3 record should belong to **at most one** S1 entity. This makes it a constrained assignment problem: if two S1 entities both claim the same S2 record, keep only the stronger claim. Verify this against the ground truth before relying on it.
- **S2 and S3 records that match the same S1 entity match each other.** Within-source and cross-source duplicates in S2/S3 can be clustered, and cluster evidence can support (or veto) matches to S1 transitively.
- **Country is a strong but not perfect blocking key.** Check in train whether matched pairs ever have differing country labels before using it as a hard filter.
- **Many-match entities** (S1 matched to several S2/S3 records) may look like chains/branches or like duplicate listings. Understand which it is, because chains with the same name at different addresses are prime false-positive traps.

---

## 7. Research Burdens (things that need to be figured out)

### R1. Exploratory analysis of the label structure
- Sizes of each source and the train/test ratio
- Distribution of match cardinality per S1 entity (0, 1, 2, 3+) and the **singleton rate**; this sets the baseline score of "predict everything empty"
- S2 vs S3 contribution to matches; does each S2/S3 record map to at most one S1?
- Fraction of S2/S3 records that match nothing (noise / distractors)
- Per-country breakdown of all of the above
- What kind of noise dominates: name vs address, which abbreviations, how much is missing

### R2. Blocking / candidate generation design
- Blocking sets the **recall ceiling**; anything not in candidates can never be matched
- Candidate-set size per S1 is now a **ranking criterion in its own right** (organisers' update). Blocking has two objectives: end-to-end macro F0.5 and small mean candidates per S1. Aim for per-entity adaptive list sizes (often 0–3), not a fixed top-K.
- Must scale: inverted indexes, sparse top-K products, ANN indexes; never all-pairs
- Options to evaluate and combine (fuse multiple blockers, then prune; see NonStandard.md T20, T30, T31):
  - Normalised name token blocking (rare tokens, first significant token)
  - Character n-gram TF-IDF with approximate nearest neighbours (top-K)
  - Postal code / PIN / ZIP blocking when present
  - City / locality token blocking
  - Phonetic keys (Soundex, Metaphone, Double Metaphone) for transliteration noise
  - Dense embeddings of name + address with ANN (FAISS / hnswlib) for semantic variants
  - Sorted neighbourhood on normalised names
- Measure per blocker: oracle macro F0.5, pair completeness (recall), mean/median/p95 candidates per S1, share of empty lists, reduction ratio, runtime, per country (NonStandard.md T22)
- Must generalise to **France**, where rare-token and postcode heuristics tuned on US/India may behave differently

### R3. Unseen-country generalisation (France)
- No French training data, but French test entities count equally in the macro average
- France exists only in the test set, so **nothing is fitted on it**: no training, no vocabulary/IDF fitting, no threshold tuning, no rules written from looking at test France records, no leaderboard probing (NonStandard.md G1). French handling comes from country-agnostic features, general knowledge of French conventions, and robustness checks on training data.
- French specifics to anticipate:
  - Legal forms: SARL, SAS, SASU, SA, EURL, SNC, SCI, "Société", "Ets" (établissements), "Cie"
  - Address words: rue, avenue/av, boulevard/bd, place/pl, chemin/ch, allée, impasse/imp, quai, route/rte, "bis/ter" house numbers, CEDEX
  - 5-digit postal codes (like US ZIP but different meaning), city after postcode
  - Accents and diacritics (é, è, ç, à), apostrophes and elision (l', d'), hyphenated names and cities (Saint-/St-)
  - "et" as the ampersand word (not "and")
- Approaches: country-agnostic features (character n-grams, normalised edit distances) as the backbone; unicode normalisation (NFKD + accent stripping); multilingual encoders; hand-built normalisation dictionaries per country; avoid features that only exist for US/India (e.g., a PIN-code-specific parser as a required feature)
- Risk: thresholds calibrated on US/India may be miscalibrated on France. Use "leave-one-country-out" validation (train on US, validate on India and vice versa) as the proxy for transfer to an unseen country, and prefer relative/rank-based cutoffs over absolute ones.

### R4. Match model selection
- Classic: gradient boosted trees (LightGBM / XGBoost / CatBoost) on engineered similarity features; fast, strong, easy to calibrate
- Neural: bi-encoder for retrieval + cross-encoder for re-ranking (fine-tuned on train pairs); must respect the 8B / MIT-Apache constraint
- Small LLM as a re-ranker or tie-breaker on ambiguous pairs only (cost and latency heavy)
- Likely best: GBDT on hand-crafted features + embedding similarity features, with an entity-level decision layer on top
- Research whether hard negative mining (same-name-different-branch, same-address-different-business) improves precision

### R5. Decision / thresholding strategy for macro F0.5
- Pairwise probabilities are not the final answer; the per-entity list must be chosen to maximise expected F0.5
- Candidate strategies:
  - Global threshold on match probability
  - Top-1 gating (decide "any match?" first), then a stricter threshold for additional matches
  - Relative thresholds (keep candidates within a margin of the best score)
  - Expected-F0.5 optimisation per entity using calibrated probabilities
  - Separate "is singleton?" classifier using entity-level features (best score, score gap, number of candidates)
  - Per-country thresholds, with a sensible fallback for France
- Resolve conflicts where one S2/S3 record is assigned to multiple S1 entities

### R6. Validation protocol
- Split train **by S1 entity** (GroupKFold) so all an entity's candidates stay together; all S2/S3 records in the fold should remain as candidates/distractors
- Stratify by country and by match cardinality (singleton vs. non-singleton)
- Score with the exact macro F0.5 implementation (write it once, reuse everywhere)
- Also track blocking recall, candidates per entity, pair-level precision/recall, and score by country and by cardinality bucket
- Use cross-validation to choose thresholds, then refit on full train

---

## 8. Development Burdens (things that need to be built)

### D1. Data loading and normalisation layer
- Robust TSV reader (`sep="\t"`, string dtype, empty strings preserved)
- Unicode normalisation, lowercasing, accent stripping (keep an un-stripped copy too), punctuation and whitespace cleanup
- Name normaliser: expand/contract abbreviations (corp/corporation, pvt/private, ltd/limited, co/company, inc/incorporated, llc, `&`/and/et), strip or separately extract legal suffixes, remove DBA/trade-name markers but keep both parts, handle stopwords
- Address normaliser: street-type abbreviations per country (St/Street, Rd/Road, Ave/Avenue, Marg, Nagar, rue, bd ...), extract postal code (US 5/9-digit ZIP, India 6-digit PIN, France 5-digit), extract house/unit numbers, state/city tokens, detect landmark phrases ("near", "opp", "behind", "beside", "next to")
- Country-aware but **country-open** code paths: unknown countries fall back to generic normalisation, never crash or get dropped

### D2. Blocking module
- Multiple retrievers fused by rank or normalised score, then pruned per entity (reciprocal / cardinality node pruning, adaptive cutoffs, empty lists for confident singletons)
- Scalable implementation: chunked sparse top-K, ANN indexes, block purging for oversized keys; logs comparisons and runtime
- Evaluation harness reporting oracle macro F0.5 against mean candidates per S1 on validation
- Writes `candidate_pairs.tsv` from the **final** filtered candidate set

### D3. Feature engineering (per S1-candidate pair)
- Name similarities: Jaro-Winkler, Levenshtein ratio, token set/sort ratio (rapidfuzz), Jaccard on tokens and char n-grams, TF-IDF cosine (word and char), prefix match, acronym match (e.g., "IBM" vs "International Business Machines"), legal-suffix agreement, core-name similarity after suffix removal
- Address similarities: same set of metrics on full and normalised address, postal code exact/partial match, house number match/conflict, street name similarity, city match, landmark overlap, token overlap weighted by IDF
- Rarity features: IDF of shared tokens (sharing "Sharma" in India or "Main St" in the US means little; sharing a rare token means a lot)
- Embedding similarities: cosine of sentence embeddings for name, address, and combined
- Missingness indicators: address missing, postal code missing on either side, name length
- Context features: rank of this candidate among the S1 entity's candidates, gap to best score, number of candidates, how many S1 entities this candidate is close to (ambiguity)
- Source indicator (S2 vs S3), country (as a categorical that must tolerate unseen values, or excluded)

### D4. Matching model
- Train on labelled pairs from the candidate set (positives from ground truth, negatives from the remaining candidates, plus mined hard negatives)
- Watch class imbalance; use the model's calibrated probabilities downstream
- Optionally a cross-encoder / fine-tuned transformer as a second stage on top-N candidates

### D5. Post-processing and decision layer
- Per-entity selection logic tuned for macro F0.5
- One-S2/S3-record-to-one-S1 conflict resolution
- Optional transitive clustering of S2/S3 records to add or veto matches
- Guarantee: matches ⊆ candidates

### D6. Output writer and validation
- Emit every test S1 ID exactly once, deduplicate lists, strip spaces, write empty cells for no matches
- Run `utils/validate_submission.py` as the final pipeline step

### D7. Experiment tracking and reproducibility
- Log every run: config, blocking stats, CV macro F0.5 (overall, per country, singleton vs non-singleton), LB score if submitted
- Fixed seeds, pinned `requirements.txt`, a single entry point (e.g., `python src/run_pipeline.py --stage all`)
- README with the exact commands from raw data to both output files

### D8. Documentation
- Fill in `Documentation_template.md`: methodology, blocking strategy with recall/reduction numbers, model architecture, features, thresholding, experiments and what did not work

---

## 9. Suggested Pipeline Shape

```
raw TSVs
  -> normalise (name, address, postal code, country-open)
  -> blocking (union of token / n-gram TF-IDF ANN / postal / phonetic / embedding blockers)
  -> candidate_pairs.tsv  (final set fed to the model)
  -> pairwise features
  -> match model (GBDT, optionally + cross-encoder re-rank)
  -> entity-level decision layer tuned for macro F0.5
  -> conflict resolution (+ optional S2/S3 clustering)
  -> matching_results.tsv
  -> validate_submission.py
```

---

## 10. Key Risks

| Risk | Mitigation |
|---|---|
| Blocking misses true matches (recall ceiling) | Fused diverse retrievers; measure oracle score per country |
| Candidate sets too large (hurts final ranking) | Adaptive per-entity cutoffs, reciprocal pruning, one-owner cap; report size alongside score |
| Accidentally fitting on test data (e.g. TF-IDF fit on test, rules from France rows) | Fit every artefact on training data only; keep test strictly inference-only |
| False positives on singletons (each costs a full point) | Dedicated "any match?" gating with a conservative threshold |
| Chains / branches (same name, different location) | Address-conflict features, house number and postcode conflict flags, hard negatives |
| France behaves differently from training countries | Country-agnostic features, French normalisation rules, leave-one-country-out validation, avoid hard-coding the country set |
| Overfitting to the public leaderboard | Trust CV; use LB only as a sanity check; final rank is private LB |
| Wasted submissions from format errors | Always run the validator locally |
| Irreproducible final package | Single entry point, seeds, pinned deps, tested fresh-clone run before zipping |

---

## 11. Early-Hours Checklist

1. Load all files correctly and run EDA from R1 (especially singleton rate and cardinality)
2. Implement the macro F0.5 scorer and a GroupKFold split by S1 entity
3. Submit a simple baseline early (e.g., TF-IDF top-1 with a high threshold) to confirm format and get an LB anchor
4. Build and measure blocking recall before investing heavily in the matcher
5. Iterate on features and model, tuning the decision layer on CV
6. Spend remaining submissions deliberately, one controlled change at a time
