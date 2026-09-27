# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

S2/S3-side blocking with rarity-weighted retrieval and composite keys (oracle macro F0.5
0.9895, ~5.4-7.1 candidates/S1) feeds a LightGBM pairwise classifier with a one-owner
decision layer (CV macro F0.5 0.9628). The main technical contributions are (a) meta-blocking
that caps candidates from the S2/S3 side rather than a fixed top-K per S1, ranked by a
rarity-weighted key evidence score rather than raw key overlap, and (b) a disciplined
train-only validation protocol (GroupKFold-by-S1, a leave-one-country-out check as a proxy
for the fully-unseen France country) that surfaced several plausible improvements that did
*not* actually help and were deliberately left out.

---

## 2. Methodology

### 2.1 Problem Analysis

Key EDA findings (full detail in `experiments/r1_eda_output.md`):
- Singleton rate is 5.6% — predicting all-empty scores only 0.056, so missing matches costs
  far more than the generic "precision-heavy metric" advice suggests.
- Mean 3.46 true matches per S1 (3.67 among non-singletons) — this is the list size a
  perfect blocker would output; our shipped candidate lists (5.4 train / 7.1 test mean) are
  roughly 1.5-2x that, which is the cost of blocking robustly instead of at oracle-only cost.
- Each S2/S3 record matches at most one S1 (one-owner holds exactly, 0 violations in 7.64M
  positive pairs) — this is exploited directly in blocking (S2/S3-side top-m cap) and in the
  decision layer (each record independently owns at most one final match).
- Country never differs within a true match — a hard blocking key, but an *open set*: France
  appears only in test, and every normalisation/blocking/decision artefact that "learns"
  anything is fit on training rows only, never on test, so an unseen country degrades
  gracefully rather than crashing or being silently mishandled.
- US and India are statistically near-identical (singleton rate, cardinality, S2/S3 split),
  consistent with one synthetic noise generator across countries.

### 2.2 Solution Strategy

**Approach Type:** Blocking (meta-blocking / rarity-weighted retrieval) + LightGBM classifier
+ a one-owner decision layer, with a disciplined train-only validation protocol throughout.

**Core innovation:** blocking ranks and prunes S2/S3-side candidates by a *rarity-weighted
sum of shared blocking keys* (fixed, hand-reasoned per-kind multipliers, not fit on any row)
rather than raw key-overlap count, and folds that same weighted evidence into the
retrieval-stage similarity score as a small saturating boost — recovering true pairs that
share strong, rare keys but score weakly on plain name/address string similarity (e.g.
scrambled names, addresses missing a house number). Two new composite key kinds (a record's
two rarest name tokens as an unordered pair; an address token + city with no house-number
requirement) target, respectively, oversized single-token blocks and gibberish/scrambled
names retrievable only by address.

---

## 3. Candidate Generation (Blocking)

Full detail in `BLOCKING_V2.md` and each blocking-fix commit's message.

- **Blocking keys used** (all prefixed by country, a hard key): exact compacted name; single
  name tokens; 4-grams of the compacted name (typo-robust); name token + house number; name
  token + city; house number + address token; **(v2)** the two rarest name tokens as an
  unordered pair (rarity from one global document-frequency table fit on train S1+S2+S3 only,
  never recomputed per-batch — an earlier per-batch version was found to be both inconsistent
  across differently-sized index/query batches and a G1 violation on a test run, and was
  fixed before being used); **(v2)** an address token + city requiring no house number, so
  records with unusable names are still retrievable by address alone.
- **Retrieval ranking (v2):** candidates ranked by `Σ per-kind-multiplier / log2(block_size+1)`
  over shared keys (a fixed, hand-reasoned formula — G1: nothing here is fit on any row's
  content, only the block size of the index actually being queried, which G1 explicitly
  permits at inference time), instead of raw shared-key count; this recovered candidates
  previously dropped by a fixed top-K cap without raising the cap indiscriminately.
- **Stage-1 score:** rapidfuzz name/address similarity, further boosted by a small saturating
  function of the same rarity-weighted key evidence, so a true pair that wins on keys but
  scores weakly on string similarity survives the top-5-by-score cut.
- **Pruning (one-owner cap, T30):** each S2/S3 record keeps at most `m=3` S1 candidates,
  subject to a score floor and a relative margin from its own best score — tuned on training
  folds by oracle F0.5 vs. mean candidates/S1, never on test.
- **Candidate pairs generated (v2, real full-scale):** train 11,916,319 pairs (mean 5.40/S1,
  oracle macro F0.5 0.9893); test 12,267,845 pairs (mean 7.08/S1; by country: US 6.32, India
  7.36, France 8.14 — France was not used to choose any parameter, this is simply its
  resulting candidate-set size).
- **How true matches were not lost:** oracle macro F0.5 (candidates ∩ truth, i.e. the ceiling
  if the matcher were perfect) is reported for every blocking change alongside mean
  candidates/S1, so a size reduction is only kept when the ceiling holds or rises at a
  similar or smaller size (never traded blindly for a smaller file). v1 -> v2 raised the
  ceiling 0.9778 -> 0.9893 while *shrinking* the mean train candidate list 5.82 -> 5.40.

---

## 4. Matching Model

**Features used** (29 total, `features.py`; country deliberately excluded — France is unseen
in training, see G1 in `NonStandard.md`):
- Blocking-derived: retrieval score, name/address sub-scores, this candidate's rank and gap
  to its record's best and second-best score, candidates competing for the same record/S1.
- Name: Jaro-Winkler, token-sort ratio, partial ratio, token Jaccard, length difference, legal-form
  agreement, alias match (DBA/trading-as text) both directions.
- Address: house-number equal/different/missing, city/state agreement, address-number Jaccard,
  missing-address flag.
- Record flags: gibberish-name score, transliteration flags (name/address), non-Latin script.

**Model type:** LightGBM binary classifier (MIT licensed, well under the 8B-parameter limit).
`num_leaves=127`, `learning_rate=0.05`, `feature_fraction=0.8`, up to 2000 rounds with early
stopping (50 rounds) on a true holdout fold (GroupKFold-by-S1, fold 4 of 5), then retrained
on all folds at the tuned iteration count for the model that ships.

**Threshold selection method:** each S2/S3 record takes its single highest-probability
candidate and accepts it iff probability >= t (one-owner by construction — no post-hoc
conflict resolution needed); t is swept on the fold-4 holdout against the *real* macro F0.5
(`evaluate.py`, not a proxy), never on test/leaderboard feedback.

**Alternatives tried and deliberately not adopted** (validated on real, full-scale data, not
smoke samples — kept in the codebase and in `experiments/log.csv` for the record):
- An entity-centric decision layer (per-S1 "any match?" gate + stricter rule for additional
  matches + isotonic probability calibration) landed within noise of the shipped
  record-centric rule and was individually worse on both singleton and non-singleton buckets.
- Metric-aware sample weights (from the exact F0.5 cost table in this document's evaluation
  section) gave +0.0003 F0.5 — not a real improvement despite shifting the optimal threshold
  materially.
Both are documented rather than silently discarded, since a negative result validated on real
data is still useful evidence about where the remaining score gap does *not* come from.

---

## 5. Results & Error Analysis

- **F0.5 Score (macro), fold-4 holdout, v2:** overall 0.9628 (blocking oracle ceiling 0.9895);
  India 0.9546, US 0.9683; non-singleton 0.9635, singleton 0.9512.
- **v1 comparison** (submitted `lb-2`): CV 0.9541, public leaderboard 0.943 (oracle 0.9778).
- **Singleton scoring below non-singleton** is the opposite of the usual pattern (an empty
  prediction on a true singleton should be an easy win) — diagnosed as the symptom NonStandard.md's
  T2 describes (a crowd of mediocre candidates makes an entity look like it has a match when
  it does not), but the entity-centric gate built to address it did not outperform the
  simpler rule in testing (see above) — left as a known, quantified weakness rather than
  patched with an unvalidated fix.
- **Unseen-country risk (leave-one-country-out, a France proxy):** retraining with India held
  out entirely (never seen in training) drops F0.5 from 0.9541 to 0.9042 using a threshold
  never tuned on India; holding out US only drops it to 0.9453. This gap barely closes even
  with a threshold optimally tuned on the held-out country itself (0.9129 for India), meaning
  it is a genuine feature/model generalisation limit rather than a threshold-calibration
  problem — the recommended next step (not completed) is train-record noise-operator
  augmentation (accent/apostrophe/hyphen/case perturbations) to build in the invariances
  France's orthography needs without ever using a French row.
- **Common false negatives:** true matches lost to the blocking-stage score-cut before the
  matcher ever sees them (~1.2 points of pair recall in the real v2 train run, between
  retrieval recall 0.9912 and post-score-cut recall 0.9732) — an in-progress supervised
  meta-blocking ranker targets exactly this gap; see `STATUS.md` for its status.

---

## 6. Conclusion

Meta-blocking with rarity-weighted retrieval and two new composite key kinds raised the
oracle ceiling from 0.9778 to 0.9895 while shrinking the mean candidate list, and the
downstream LightGBM matcher plus a simple one-owner decision rule reaches 0.9628 CV macro
F0.5 against that ceiling. The most useful lesson from this session was arguably negative:
several theoretically-motivated decision-layer improvements (entity-centric gating,
probability calibration, metric-aware training weights) were implemented, validated on real
full-scale data, and found not to help — which is as informative as a positive result, and
kept the shipped pipeline simple. The leave-one-country-out check gives a concrete, if
sobering, estimate of the unseen-country (France) risk that a purely US/India-trained,
country-blind pipeline carries into the private leaderboard.

---

## Appendix

### A. Code Artefacts

Complete, runnable code ships under `code/business_entity_resolution/` (`src/`, `README.md`,
`requirements.txt`). Entry point: `README.md`'s numbered command sequence
(`normalize.py` -> `blocking.py --version {v1,v2}` -> `features.py` -> `model.py` ->
`decide.py` -> `make_submission.py`) reproduces `output/matching_results.tsv` and
`output/candidate_pairs.tsv` from the raw `dataset/` TSVs. `evaluate.py` is the official,
self-testing macro F0.5 implementation used for every reported score in this document.

### B. Additional Results

See `experiments/log.csv` for the full experiment history (one row per real, non-smoke run:
date, git commit, change, CV scores, leaderboard score where submitted) and `STATUS.md` for
the most recent state, including work still in progress at submission time.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and
technical depth.
