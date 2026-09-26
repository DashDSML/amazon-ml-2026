# BLOCKING_V2.md: Brief for Blocking v2

This file distils the blocking-related parts of NonStandard.md, and the strict test-data rule agreed for this project, into one short brief. Use it with CLAUDE.md, STATUS.md and HANDOFF.md.

## 1. Where we are

| Number | Value |
|---|---|
| Blocking v1 oracle macro F0.5 (train CV) | 0.978 overall: 0.968 India, 0.984 US |
| Simple-rule end-to-end F0.5 on v1 candidates | ~0.83 |
| Known recall losses | 3.0% of true pairs dropped by the top-30 cap on shared keys; 1.8% never found by any key |
| Full blocking run | ~1.7 h per split |

**What each number means:**
- The *oracle* is the score if the matcher were perfect within the candidate lists. It is the ceiling.
- The gap between 0.83 and 0.978 belongs to the **matcher** (features + LightGBM + decision layer), not to blocking.
- Blocking v2 raises the ceiling. The matcher decides how close the final score gets to it.

**Targets for v2:** oracle ≥ 0.99, with mean candidates per S1 ≤ ~5 (the ideal is ~3.5, the true mean number of matches per S1). Candidate-set size counts toward the final ranking, so a higher oracle bought with much bigger lists is not a win.

## 2. Strict test-data rule for blocking (G1, clarified)

**Fitted artefacts: training data only.** This covers:
- IDF / token-frequency tables
- key rarity weights
- caps, quotas, margins, score floors, α
- TF-IDF vocabularies
- any learned weight

France has no training data, so unseen tokens get a fixed default weight chosen on train, e.g. "unseen = rare", validated with leave-one-country-out.

**Allowed at test time (pure inference, identical code on validation):**
- building the index over test S1 records
- looking up keys, computing block sizes and ranks for each query
- applying the train-fitted weights and cutoffs

**Not allowed:**
- choosing or adjusting any parameter by looking at test outputs, including test candidate counts or test block-size distributions
- writing keys, aliases or lexicons from test rows
- any leaderboard-driven tuning

Test candidate sizes are **reported, never tuned on**.

## 3. Step 1: diagnose before fixing (fast loop)

Index all train S1 records, but run queries for only ~20k sampled S1 entities: their true S2/S3 records plus a proportional sample of distractors. Each experiment should take ~5 minutes instead of 1.7 hours.

For every true pair missed by blocking, assign **one** reason bucket and report counts per country:

| Bucket | How to detect |
|---|---|
| A. Capped out | Shared a key with the true S1, but the block exceeded the top-30 cap |
| B. Gibberish name | Name trigram log-odds flags gibberish (resources/gibberish_*); only the address can match |
| C. Script residue | Native-script characters or a failed transliteration remain after normalisation |
| D. Typo in a key token | The key tokens differ by edit distance 1–2 |
| E. Reordered / partial name | Same tokens but a different first or main token, or a DBA / extra words |
| F. Address-only divergence | Name matches weakly, and the address dropped the city or reordered components |
| G. Other | Show 20 examples |

Fix the biggest buckets first. Stop when a fix's gain drops below ~0.1 oracle points.

## 4. Fix menu (bucket → fix → NonStandard reference)

| Bucket | Fix | Ref |
|---|---|---|
| A | **Do not truncate large blocks arbitrarily.** Score candidates by the *sum of rarity weights of shared keys* (a meta-blocking edge weight). Split oversized blocks with composite keys: name-token ∧ city, name-token ∧ house number, street-token ∧ house number. Keep block purging only for truly generic keys (the threshold is chosen on train). | T12, T30, T32 |
| B | **Address-only keys:** house number ∧ street token, street token ∧ city. Records flagged gibberish are retrieved by address keys only. | T16, T13 |
| C | Extend the train-mined Indic lexicon, and add transliteration-folding keys (collapse vowels, v/w, sh/s, ph/f, double letters). | T17, T18 |
| D | Deletion-neighbourhood keys (every variant of a key token with one character deleted), or phonetic keys (Double Metaphone), on the rarest name tokens only. | T17, T20 |
| E | **Token-set keys:** the two rarest name tokens as an unordered pair, not the first token. Strip legal suffixes, honorifics and DBA markers before key generation. | T11, T15 |
| F | A second retriever: char-trigram TF-IDF (fitted on train) over name+address, queried **only** for S2/S3 records whose best key-based score is weak. That's cheap because it touches few records. | T20 |

## 5. Keeping candidate lists small

- Blocking stays **S2/S3-side**. Each S2/S3 record goes to its best S1, and to a 2nd or 3rd S1 **only** if that one's fused score is within a margin δ of the best (an adaptive m). δ is tuned on train.
- A per-S1 relative cutoff drops candidates below α × the S1's best score. A score floor lets confident no-match records go nowhere, which is how singletons get empty lists.
- The one-owner rule holds exactly in train, so the S2/S3-side cap bounds the total size: mean candidates per S1 ≤ m × (|S2|+|S3|) / |S1| (about 4.7 at m=1 in train).
- Prefer **relative** cutoffs (margin, α, rank) over absolute floors. Absolute similarity levels may shift for France.

## 6. Report for every blocking experiment (train CV, fast loop)

One table, nothing more:

| Metric | Overall | India | US |
|---|---|---|---|
| Oracle macro F0.5 | | | |
| Pair recall | | | |
| Mean / median / p95 candidates per S1 | | | |
| % S1 with empty list | | | |
| Missed pairs by bucket (A–G) | | | |
| Runtime, peak memory | | | |

Keep a v2 change only if the oracle rises at a similar or smaller mean candidate size, **or** the size falls at a similar oracle. After the fast loop settles, run once on full train and once on full test. On test, report sizes only.

## 7. Scale constraints

- No all-pairs comparisons. Use inverted indexes, chunked sparse top-K, ANN.
- Process by country, and by chunks of S1 queries. Keep peak memory under ~10 GB on the 16 GB laptop.
- Pairwise features are computed only for the final candidate pairs.
- Log comparisons and runtime per stage. State the complexity in the methodology doc.

## 8. When to do this

**After** the matcher gives its first CV score and `lb-2` is submitted (see HANDOFF.md). If the matcher leaves a large gap to the oracle, improve the matcher first. If it reaches close to the oracle, blocking v2 is the next lever. Hard stop for all experiments: **27 Sep, 18:00 IST**.
