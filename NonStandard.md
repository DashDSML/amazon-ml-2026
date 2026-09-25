# NonStandard.md: Non-Obvious Techniques for Raising Macro F0.5

Companion to `challengecontext.md`. This file assumes the basic pipeline already exists (normalise → block → pairwise features → GBDT → threshold → write TSVs). Everything here is what comes *after* that. It is distilled from winning and gold-medal solutions to closely related problems and from the entity-resolution research literature (sources at the bottom).

**Two ground rules (G1, G2 below) override anything else in this file.** They were added after the organisers' update that makes candidate-set size a ranking criterion, and to keep France strictly unseen.

## How to use this file

Every technique has the same five parts:

- **Use when**: the symptom in validation or on the leaderboard that says this trick is warranted
- **Why the obvious approach fails**: the trap this avoids
- **Technique**: what to do
- **Sketch**: minimal implementation guidance
- **Risk**: how it can backfire, and what to check

Rule for any agent editing the pipeline: **change one thing at a time, and judge it by macro F0.5 on the validation protocol in T23, reported together with the mean candidate-set size per S1 entity (G2)**. Pair-level AUC or F1 are not reliable proxies for this metric.

---

## Ground Rules (read before applying any technique)

### G1. No test data is used to fit anything; France is fully unseen

France appears only in the test set. Treat it as a genuinely unseen country: nothing is learned, tuned or designed from test records of any country.

**Not allowed on test data:**
- training or fine-tuning any model or embedding
- fitting vocabularies, TF-IDF vectorisers, IDF/frequency tables, normalisation or abbreviation dictionaries, calibrators, thresholds or cutoffs
- generating synthetic training pairs from test records
- reading France (or any test) records to write rules, regexes or lexicons
- tuning parameters on leaderboard feedback (T25)

**Allowed (pure inference; the identical code also runs on validation):**
- running the train-fitted pipeline on test records
- per-query context computed from the records being matched: candidate ranks, score gaps, similarity between candidates, counts of near-duplicates in the input pool (T7, T9, T10, chain frequency in T12). These are inference-time computations, like retrieval itself.
- format and sanity checks: row counts, every S1 ID present, no crash on an unseen country label

Consequences: everything France-related comes from (a) country-agnostic features, (b) general knowledge written down *without* looking at test records (e.g. a French legal-form list), and (c) robustness checks on training data (T24).

### G2. Candidate-set size is a ranking criterion, and blocking must scale

Organisers' update (September 2026): `candidate_pairs.tsv` and the code that produces it are reviewed for the final ranking alongside the `matching_results.tsv` score. **The approach that generates a smaller candidate set per Source 1 entity ranks higher.** Blocking must also scale: Amazon resolves entities across billions of records, so all-pairs comparison is ruled out.

Consequences:
- Blocking now has two objectives: end-to-end macro F0.5 and mean candidates per S1. The usual advice "maximise the recall ceiling" no longer holds on its own.
- The exact weighting between the two is not published. Always report both and choose operating points where extra candidates buy little score (T22).
- `candidate_pairs.tsv` must be the real input of the matching model. Do not shrink it by relabelling part of the matcher as "blocking" (T21).
- Post-processing may never add an ID that is not in that entity's candidate list (T9).
- Relevant techniques: T20, T21, T22, T30, T31, T32.

---

## 0. Symptom → Technique Index

| Symptom observed on validation | Look at |
|---|---|
| Score on singletons (true-empty entities) is well below 1.0 | T2, T1, T6, T8, T12 |
| Precision on single-match entities is low (extra wrong IDs) | T1, T4, T7, T6 |
| Many false merges between branches of the same chain / same-name businesses | T8, T10, T12, T13, T16 |
| False merges where names and addresses look near-identical but numbers differ | T13 |
| Recall ceiling (oracle score) is low: true matches missing from candidates | T20, T22, T31, T17, T18 |
| Candidate lists are large (now also a ranking criterion, G2) | T30, T31, T20, T21, T22 |
| The same generic S2/S3 record sits in many S1 candidate lists | T30, T7, T6 |
| Blocking is slow, memory-hungry, or compares too many pairs | T32, T30 |
| One S2/S3 ID appears in several S1 rows of the output | T6 |
| Abbreviation/DBA/branch-suffix pairs are being missed | T11, T15, T18 |
| Indian records with landmarks or transliteration are missed | T16, T17, T18 |
| Worried about France (unseen country) | G1, T24, T12, T19, T28, T14, T31 |
| CV looks great but leaderboard is clearly worse | T23, T24, T25 |
| Good pairwise model but the final score barely moves | T1, T3, T5 |
| Two models each good alone but blending hurts | T29 |

---

## 1. Metric-Level Tricks (usually the largest gains after a baseline)

### T1. Per-entity expected-F0.5 decision instead of one global threshold

**Use when:** you have calibrated pair probabilities and are currently picking matches with a single `p > t` rule.

**Why the obvious approach fails:** The metric is F0.5 computed *per S1 entity* and then averaged. A global threshold ignores how much probability mass an entity has in total, how many candidates it has, and how much one extra ID hurts that specific entity. The costs are very asymmetric:

| Mistake | Cost to that entity's score |
|---|---|
| Any match on a true singleton | 1.000 |
| Miss the only match (t=1) | 1.000 |
| One extra wrong ID when t=1 | 0.444 |
| One extra wrong ID when t=2 | 0.286 |
| One extra wrong ID when t=3 | 0.211 |
| Miss one of two matches (t=2) | 0.167 |
| Miss one of three matches (t=3) | 0.091 |

So adding a doubtful third ID is rarely worth it, but getting the first ID right is worth a whole point.

**Technique:** For each S1 entity, sort its candidates by calibrated probability. Consider predicting the top-k for every k = 0..K, estimate the *expected* F0.5 of each choice, and output the best k. This is the Jansche (2007) maximum-expected-utility framework, which Bai, Binette & Reiter apply to record linkage (outer loop over number of matches, inner choice of which ones).

**Sketch (Monte Carlo, treats candidate truths as independent Bernoullis):**

```python
import numpy as np

def best_k(p, n=4000, rng=np.random.default_rng(0)):
    """p: calibrated match probabilities for ONE S1 entity's candidates."""
    order = np.argsort(-p); p = p[order]
    Z = rng.random((n, len(p))) < p          # sampled ground truth
    T = Z.sum(1)                              # true matches among candidates
    best_score, best = (T == 0).mean(), 0     # k=0 scores 1 only if truth is empty
    for k in range(1, len(p) + 1):
        tp = Z[:, :k].sum(1)
        P = tp / k
        R = np.where(T > 0, tp / np.maximum(T, 1), 0.0)
        f = np.where(tp > 0, 1.25 * P * R / (0.25 * P + R), 0.0)
        if f.mean() > best_score:
            best_score, best = f.mean(), k
    return order[:best]                       # indices of candidates to output
```

Refinements:
- Blocking misses: true matches outside the candidate list still count in the denominator. Add a sampled "missed matches" count per draw, using the per-entity miss rate you measured on validation (T22). With the tighter blocking G2 calls for, misses are more common, so this refinement matters more.
- Use the singleton gate (T2) for the k=0 vs k≥1 decision, and use this routine only to choose k given "at least one match".

**Risk:** Needs calibrated probabilities (T5). Candidate probabilities are correlated, not independent. Tune a temperature on the probabilities (T5) by maximising validation macro F0.5 to absorb that miscalibration.

---

### T2. Separate entity-level "does this S1 entity have any match?" gate

**Use when:** the singleton bucket scores poorly, or the singleton rate in training is substantial.

**Why the obvious approach fails:** "max pair probability > t" is a weak singleton detector. The pairwise model is trained to separate pairs, not to answer whether a match exists anywhere. An entity with a crowd of mediocre candidates (a common name) looks like it has a match when it does not.

**Technique:** Train a second, small classifier with **one row per S1 entity** and label `has_any_match`. Features:
- max, 2nd max and sum of pair probabilities
- gap between best and second-best
- number of candidates
- best name similarity and best address similarity
- postcode-agree flag of the best candidate
- S1 "confusability" (T8), neighbourhood density (T10), country, missing-address flag

Output empty whenever the gate says no; otherwise hand over to T1/T4.

**Risk:** The gate's threshold directly trades singleton points against non-singleton points. Tune it on validation macro F0.5, not on gate accuracy.

---

### T3. Metric-aware sample weights for the pairwise model

**Use when:** the pairwise classifier is trained with uniform weights.

**Why the obvious approach fails:** Uniform log-loss treats all pair mistakes as equal. The metric does not: a false positive on a singleton entity costs 1.0, while a false negative on a 4-match entity costs about 0.06. The 7th-place Foursquare team weighted each training pair by the metric loss its misclassification would cause, because false positives hurt more than true negatives help.

**Technique:** Weight each training pair by the score its misclassification would cost, from the table in T1:
- Negative pair of an S1 entity with t true matches: weight = cost of one extra wrong ID at that t (1.0 for singletons, 0.444 for t=1, ...)
- Positive pair: weight = cost of missing it at that t (1.0 for t=1, 0.167 for t=2, ...)

Normalise the weights to mean 1.

**Risk:** Heavily weighted singleton negatives can make the model timid. Compare to the unweighted model *after* the T1/T2 decision layer, not before.

---

### T4. Different rules for the first match and for additional matches

**Use when:** you are not using T1 yet, or you want a simpler and more robust alternative to it.

**Why the obvious approach fails:** The first ID and the second ID carry very different risk (see the T1 table). One threshold for both is either too strict for the first (losing whole points) or too loose for the rest (adding 0.2–0.44 penalties). Shopee winners made rank-specific rules too. The 1st place applied a separate threshold ("min2") to the second-nearest neighbour. The 4th place added rank-2 matches only when mutual or when the rank-2/rank-3 gap was large. There, a known minimum group size pushed rules toward *adding* matches. Here the metric pushes the other way.

**Technique:**
- First match: accept the top candidate if the gate (T2) passes.
- Additional candidates: require all of
  - an absolute threshold `t2` (tuned; typically stricter than for the first match)
  - a relative criterion such as `p >= alpha * p_best`
  - optionally a consistency check with already-accepted matches (T9)

**Risk:** Three knobs → overfitting risk. Tune them with cross-validation across folds, not on one split.

---

### T5. Calibrate, then tune a temperature for the decision layer

**Use when:** using T1, T2 or any probability-based rule.

**Why the obvious approach fails:** GBDT probabilities, especially with weighting (T3) or class imbalance handling, are not calibrated. Expected-F computations on uncalibrated scores choose the wrong k.

**Technique:**
1. Fit isotonic regression on out-of-fold predictions.
2. Apply a single scalar temperature/shift `p' = sigmoid((logit(p) + b) / τ)`.
3. Grid-search `(b, τ)` directly on validation macro F0.5 after the full decision layer.

Do this per country only if you have enough validation data per country. France gets the global values or a conservative variant (T24).

---

## 2. Structural and Graph Tricks

### T6. One-owner constraint: each S2/S3 record belongs to at most one S1 entity

**Use when:** always check it first. Verify on training ground truth that no S2/S3 ID appears under two S1 entities. S1 is deduplicated, so this should hold.

**Why the obvious approach fails:** Independent per-S1 decisions can assign the same S2 record to two S1 entities (chains, similar names). At least one of those is a false positive, and it may land on a singleton, which costs a full point.

**Technique:** After scoring, for every S2/S3 record keep only its best-scoring S1 claim. Drop it entirely if the best and second-best claims are within a margin (ambiguous → precision). This is *many-to-one*, not one-to-one, so no Hungarian algorithm is needed: an argmax per S2/S3 record is exact. (Research on clean-clean ER covers the harder one-to-one case with bipartite matching algorithms; that doesn't apply here unless the data also shows each S1 matches at most one record per source.)

Also add "margin between this S1 and the candidate's next-best S1" as a **pairwise feature**, since it is highly predictive.

**Sketch:** `pairs.sort_values('p', ascending=False).drop_duplicates('cand_id')`, then apply the margin rule.

**Risk:** If ground truth violates the constraint (check!), apply it softly (feature only).

---

### T7. Mutual (reverse) retrieval consistency

**Use when:** precision problems where one S2 record is "near" many S1 entities.

**Why the obvious approach fails:** Retrieval from the S1 side alone finds, for each S1, its nearest S2 records. A generic S2 record ("Sharma Traders") is near everything. Checking from the other side exposes that. Shopee's 2nd place deleted any A→B link not reciprocated by B→A. The 4th place added rank-2 matches when the ranking was mutual.

**Technique:** Also retrieve top-k S1 for every S2/S3 record. Features:
- rank of this S1 in the S2 record's list
- `is_mutual_top1`
- reciprocal rank sum

Optionally hard-require mutual top-N for additional matches (T4). The same idea also works at blocking time as a pruning rule that shrinks the candidate file (T30).

---

### T8. S1 "confusability" and free hard negatives from S1–S1 pairs

**Use when:** false merges between similar but distinct businesses (branches, franchises, common names).

**Why the obvious approach fails:** The model never sees how similar two *different* businesses can look unless you show it. Random negatives are too easy.

**Technique:** S1 is deduplicated, so **every S1–S1 pair is a guaranteed non-match**. Exploit that twice:
1. **Confusability feature:** for each S1 entity, compute the similarity to its nearest *other* S1 entity (name, address, combined). A highly confusable S1 needs stronger address evidence before accepting a match.
2. **Hard negatives:** candidates that are true matches of a confusable neighbour S1 are excellent hard negatives. Training on them teaches the model the fine distinctions (house number, locality) that separate chain branches.

---

### T9. Transitive evidence through S2↔S3 links, with guards

**Use when:** entities have multiple matches and recall on the 2nd/3rd match is weak.

**Why the obvious approach fails:** Scoring each (S1, candidate) pair in isolation ignores that the S2 and S3 records of one business should also look like *each other*. But naive transitive closure (union-find over all accepted edges) chains distinct businesses together, which is catastrophic for precision.

**Technique:** Build a small graph per S1 entity over its candidates, with S2–S3 and S2–S2 similarity edges.
- Features: max similarity of this candidate to any *other high-scoring* candidate of the same S1, and how many such candidates it agrees with.
- Post-processing (Foursquare 7th place pattern): link at a threshold with union-find, remove bridge edges with high betweenness centrality, and keep only nodes within graph distance 2. Shopee's 2nd place also recursively removed the highest-betweenness edges.

A GNN over this graph was the headline gain for one Foursquare gold solution. It is too heavy for 72 hours, but it shows how much signal is in the graph.

**Candidate-set rule (G2):** all of this operates *inside* each S1 entity's candidate list. Post-processing may never add an ID that is not in the candidate list. If you want to reach records through S2↔S3 links ("candidates of candidates"), do it as a blocking step, before `candidate_pairs.tsv` is written, and accept that it enlarges the candidate set.

**Risk:** Always cap path length and prune bridges. Validate on single-match entities specifically, which is where chaining does damage.

---

### T10. Neighbourhood density / hubness features

**Use when:** common names or dense localities produce false merges.

**Technique:** For each record, compute:
- mean and std of its top-5/10/20 similarities to records of the *other* sources
- the number of neighbours above a similarity level

Dense neighbourhoods (chains, generic names, busy streets) mean a high raw similarity is less informative. Shopee's 2nd place used exactly these top-K similarity statistics (and PageRank) as features for its pair model; the 3rd place added density around both points.

---

## 3. Feature Tricks

### T11. Asymmetric containment scores (not just symmetric similarity)

**Use when:** DBA/trade names, branch suffixes, or truncated names are being missed.

**Why the obvious approach fails:** Jaccard, cosine and ratio are symmetric and punish "Starbucks" vs "Starbucks Coffee MG Road Branch" for the extra tokens. The Foursquare 7th place computed token- and character-level *precision and recall* separately (how much of A is in B, and vice versa) because pairs like "X convenience store" vs "X convenience store Y branch" were common.

**Technique:** For word tokens and char n-grams, compute `|A∩B|/|A|` and `|A∩B|/|B|`, plus their min and max. `max` catches containment; `min` still penalises genuinely different names. Apply to names and addresses separately.

---

### T12. Rarity-weighted agreement (term-frequency adjustment) and chain frequency

**Use when:** false merges on common tokens (Sharma, Enterprises, Traders, Main St, MG Road), or missed matches on rare distinctive tokens.

**Why the obvious approach fails:** An exact match on "John" is weak evidence; an exact match on a rare name is strong. Splink (the UK Ministry of Justice's record-linkage library) builds this directly into its match weights.

**Technique:**
- Fit token frequency / IDF tables on **training records only** (all three training sources) and save them as pipeline artefacts (G1).
- Tokens never seen in training (most French words) need a deliberate default weight. Compare "unseen = rare" (IDF of a token seen once) with "unseen = median IDF" using leave-one-country-out validation (T24). "Unseen = rare" makes a shared French proper name strong evidence, which is right, but also makes shared common French words (rue, avenue, boulangerie) look strong, which is wrong. So strip the generic-word and street-type lists from T15 *before* computing rarity features.
- Character n-gram IDF transfers to unseen languages better than word IDF, because n-grams overlap across languages. Apply accent folding before vectorising.
- Features:
  - IDF-weighted overlap
  - rarest shared token's IDF
  - rarest *unshared* token's IDF (a rare token present on one side only is strong negative evidence)
  - **chain frequency**: how many S1 entities in the *current input set* share this core name (T15). This is computed at inference over the records being matched, with the same code on validation and test; nothing is fitted (allowed under G1).

---

### T13. Numeric-token agreement and conflict, plus postcode hierarchy

**Use when:** near-identical strings are wrongly merged, e.g. "Sector 56" vs "Sector 55", "Shop 12" vs "Shop 21", "6th Cross" vs "64th Cross".

**Why the obvious approach fails:** String similarity rates these as nearly identical. In business ER, a conflicting number is one of the strongest *negative* signals and a matching rare number is one of the strongest positive ones.

**Technique:** Extract all digit sequences from names and addresses. Features:
- exact number-set equality, any-conflict flag, count of shared numbers
- conflict *given both present*, kept separate from missing-on-one-side (missing ≠ conflict)
- postcode equal / postcode prefix equal:
  - India: 6-digit PIN; first 3 digits ≈ sorting district
  - US: 5-digit ZIP (+4); first 3 digits ≈ sectional centre
  - France: 5-digit code; first 2 digits ≈ département

Parse postcodes country-aware. A US house number can be 5 digits too, so use position (US ZIP near the end; French postcode directly before the city).

---

### T14. Multi-view normalisation (keep the raw view too)

**Use when:** always. Especially relevant for France, and whenever aggressive cleaning seems to help one error type while creating another.

**Why the obvious approach fails:** One aggressive normalisation (strip suffixes, expand abbreviations, drop punctuation) destroys information that sometimes matters. The Foursquare 7th place built three separate views of each name and address: one for character-level comparison, one for word-level comparison, one for neural-network input.

**Technique:** Keep at least:
1. **Raw-ish:** Unicode NFKC, lowercase
2. **Char view:** accents folded, punctuation removed, whitespace collapsed
3. **Token view:** abbreviations expanded, legal suffixes and stopwords removed, numbers normalised (T15)

Compute the key similarities on each view and let the GBDT pick.

---

### T15. Core-name extraction, acronym detection and suffix handling

**Use when:** legal suffix noise, generic words, or acronyms cause misses.

**Technique:**
- **Core name** = name minus legal forms (Pvt Ltd, Private Limited, LLP, Inc, LLC, Corp, Co, SARL, SAS, SA, EURL, Ets), generic business words (Traders, Enterprises, Store, Boulangerie, Restaurant) and articles (The, Le/La/Les/L').
- Compare core names as the main feature, but keep **suffix-class agreement** as a separate feature rather than discarding it.
- **Acronym feature:** does the initials string of one name equal a token of the other ("IBM" vs "International Business Machines", "SBI" vs "State Bank of India")?

---

### T16. Segment-level address comparison with landmark handling

**Use when:** Indian addresses with landmarks, or addresses with reordered or missing components.

**Why the obvious approach fails:** A whole-string address similarity mixes informative parts (number, street, locality, postcode) with noise (landmarks, reordering). Commercial Indian address matchers score each component separately (house number, street, landmark, city, PIN) for this reason.

**Technique:** Rule-based parse into components:
- number, unit/floor, street, locality/sector, city, state, postcode
- **landmark phrase**: text after near / opp / opposite / behind / beside / next to / adjacent

Features:
- per-component similarity and presence flags
- address similarity *with landmarks removed*
- landmark overlap as its own feature

A shared landmark is weak positive evidence ("Near SBI ATM" appears in thousands of addresses).

---

### T17. Transliteration-folding keys for Indian names (heuristic, validate)

**Use when:** Indian transliteration variants are missed (Agarwal/Aggarwal, Srinivas/Shrinivas, Krishna/Krushna, Laxmi/Lakshmi).

**Technique:** Build a folding function and use the folded string as an extra blocking key and similarity view:
- collapse repeated letters
- map common alternations: aa→a, ee→i, oo→u, w→v, ph→f, sh→s, ksh→x/ks, th→t, dh→d, bh→b
- drop terminal 'a'

Combine with Double Metaphone keys.

**Risk:** Over-folding merges distinct names. Measure its effect on precision separately, and apply only to India-labelled records unless validation says otherwise.

---

## 4. Exploiting How the Data Was Generated

### T18. Reverse-engineer the noise generator; mine the abbreviation dictionary from training pairs

**Use when:** early. This usually unlocks several other tricks.

**Why it matters:** Competition ER datasets are often synthetic or semi-synthetic. The SIGMOD 2022 organisers generated their datasets by shuffling, deleting and re-casing words. Their retrospective notes that the best solutions were optimised for the provided data and relied heavily on preprocessing, regex feature extraction and dictionaries of aliases. This problem statement lists its noise patterns explicitly, which strongly suggests a generator.

**Technique:**
1. For all positive training pairs, align tokens (exact → fuzzy) and collect the unaligned leftovers on each side.
2. Count `(tokenA, tokenB)` substitution pairs. The frequent ones *are* the generator's abbreviation/synonym table. Use it as your normalisation dictionary instead of guessing.
3. Measure per-source behaviour:
   - which source drops PIN codes
   - which uses landmarks
   - which reorders components
   - typo types and rates (swap, drop, insert, replace)
   - whether S1 is the "clean canonical" side

   Source-specific patterns become features (source × feature interactions) and blocking choices.
4. Check the typo profile. If typos are mostly single edits, an edit-distance-1 token blocking key is cheap and very high-recall.


**Data rule (G1):** mine substitutions and noise statistics from **training pairs only**. Do not run this analysis on test records, including France.

---

### T19. Noise-operator augmentation from training records (robustness and France)

**Use when:** after T18, especially for a transformer cross-encoder or metric-learning embeddings, or to reduce France risk.

**Technique:**
- Apply the discovered noise operations to **training records** to create extra positive pairs, and pair records with confusable neighbours for hard negatives. This is analogous to Ditto's data augmentation, which adds harder training examples via span deletion, token shuffling and attribute changes.
- Because no French data may be used (G1), add **script-level invariance augmentations** to training records: random accent insertion/removal on vowels, apostrophe elision (l'/d' ↔ le/de), hyphen ↔ space, case changes, and abbreviation of long street-type words. These teach invariances that French text needs without any French examples.
- A French abbreviation and legal-form lexicon (rue/r, avenue/av, boulevard/bd, place/pl, saint/st, société/sté, établissements/ets; SARL, SAS, SASU, SA, EURL, SNC, SCI) may be written from general knowledge of French conventions and used in normalisation (T14/T15). Do not derive or refine it by inspecting test-set France records.

**Risk:** Over-augmentation shifts the training distribution. Check that US/India validation scores do not fall.

---

## 5. Candidate-Generation Tricks

Candidate-set size counts toward the final ranking and blocking must scale (G2). Every technique here should be judged on both end-to-end macro F0.5 and mean candidates per S1 (T22).

### T20. Union of heterogeneous retrievers with small budgets, fused then pruned

**Use when:** the recall ceiling is below target, or one retriever dominates the candidate list.

**Why the obvious approach fails:** One retriever with one big K wastes the budget on near-duplicates of the same signal. A plain union of many retrievers fixes recall but inflates list size, which now counts against you.

**Technique:** Run several different retrievers, each with its own small K. The Foursquare 7th place used five with separate small budgets:
- spatial nearest neighbours
- a learned combination of distance and embedding similarity
- word-level overlap
- char-level overlap
- name-embedding similarity

Here, the equivalents are:
- char-n-gram TF-IDF on name
- same on address
- name+address combined
- postcode/locality key
- phonetic/transliteration key (T17)
- dense embedding kNN

Instead of a plain union, **fuse** them: reciprocal rank fusion, or a weighted sum of normalised similarities with a handful of weights tuned on training folds by oracle score. Then prune with T30/T31 so the final list per S1 stays small. The SIGMOD 2022 blocking winner (Mannheim WBSG) re-ranked embedding neighbours with a symbolic similarity score.

The Foursquare 7th place learned its merge weights with a ranking model. Whether a learned ranker still counts as blocking here is discussed in T21.

---

### T21. Multi-stage shrinking: keep every stage recognisably "blocking"

**Use when:** candidate lists are large and you want a cheap stage to cut them before the expensive model.

**Context:** The Foursquare 1st place (re:waiwai) used a limited-feature LightGBM to cut candidates to the top 40 per ID before a richer second-stage model. In this challenge, `candidate_pairs.tsv` must be the exact set the matching model runs inference over, and reviewers now compare candidate sizes (G2). A learned classifier that scores (S1, candidate) pairs from pairwise features is arguably *part of the matching model*. If so, the candidate file must be its **input**, not its output. Reporting its output as "candidates" could be judged as misreporting.

**Technique (conservative default):**
- Blocking stages use only index lookups and similarity-based retrieval and pruning: key blocking, TF-IDF / embedding kNN, rank fusion (T20), meta-blocking pruning (T30), adaptive cutoffs (T31). Their few parameters (K, fusion weights, cutoffs) are tuned on training folds.
- The first learned pairwise classifier marks the boundary: `candidate_pairs.tsv` is its input.
- If you want a learned pruning stage (the literature calls this supervised meta-blocking), ask through the official query form how it will be treated, and describe it explicitly in the methodology document either way.

---

### T22. Measure blocking with the oracle F0.5 *against* candidate-set size

**Use when:** evaluating any blocking change.

**Why the obvious approach fails:** Pair-level recall over-weights entities with many matches, and recall alone ignores list size, which is now a ranking criterion (G2).

**Technique:** For every blocking configuration, report on validation:
- **oracle macro F0.5**: the score if the matcher output exactly the true matches present in each candidate list (the Foursquare 7th place tracked the equivalent "max IoU" at each stage)
- **mean, median and p95 candidates per S1**, share of S1 entities with an empty list, total pairs, reduction ratio versus the full cross product
- pair recall and per-entity miss rate (T1 needs it), by country and by cardinality bucket
- runtime and peak memory

Plot oracle F0.5 against mean candidates per S1 and pick an operating point near the knee. A useful anchor: the true mean number of matches per S1 (from EDA) is the average size a perfect blocker would output. Report your mean size as a multiple of it.

Because F0.5 is precision-heavy, a tighter candidate set costs less than intuition suggests. Losing the 2nd or 3rd match of an entity costs 0.09–0.17 on that entity, and the dropped candidates were also possible false positives. So also measure the **end-to-end** score (after matcher and decision layer) at two or three candidate budgets, not only the oracle.

---

### T30. Meta-blocking: reciprocal and cardinality node pruning, plus the one-owner cap

**Use when:** after retrieval the lists are still larger than needed, or a few generic S2/S3 records appear in many S1 lists.

**Why the obvious approach fails:** A global top-K per S1 keeps K candidates for every entity, including easy ones and singletons, and lets the same generic S2 record ("Sharma Traders") into hundreds of lists.

**Technique:** Treat candidates as a bipartite graph S1 ↔ S2/S3, with edge weight = fused retrieval score (T20). The meta-blocking literature (Papadakis et al.) prunes this kind of graph:
- **Cardinality node pruning (CNP):** keep the top-k edges per node.
- **Weighted node pruning (WNP):** keep edges above a node-local threshold (e.g. the mean, or a fraction of the max, weight in that node's neighbourhood).
- **Reciprocal variants:** keep an edge only if it passes the criterion at *both* ends. Papadakis et al. report that reciprocal pruning discards over 66% more comparisons at a limited cost in recall.

The S2/S3-side criterion is especially strong here because of the one-owner property (T6): each S2/S3 record belongs to at most one S1. So keep each S2/S3 record only in the lists of its top-m S1 entities (m = 1–3, tuned by oracle score). Total candidates are then bounded by m × |S2 ∪ S3|, however many S1 entities look alike.

**Risk:** For chains and franchises, an S2 record's true S1 may not be its top-m. Check on validation which m preserves the oracle score, and look at the confusable-S1 bucket (T8) separately.

---

### T31. Adaptive per-entity candidate counts, including empty lists

**Use when:** you still use a fixed top-K per S1.

**Why the obvious approach fails:** Most S1 entities have 0–2 true matches. A fixed K pads every list with K − t non-matches. For a singleton, the ideal list is empty.

**Technique:** Cut each S1's list at:
- a **score floor**: below it, true matches almost never occur in training folds
- a **relative gap**: drop candidates below α × best score, or after the largest drop in score
- a hard cap K_max

An S1 whose best fused score is below the floor gets an **empty candidate list**, and therefore an empty prediction. That shrinks the candidate file *and* banks the singleton point, provided the floor sits where validation shows almost no true matches below it. Tune α, the floor and K_max on training folds by end-to-end score at a given mean size.

**Risk:** Absolute floors tuned on US/India may be too strict for France, where similarity scores could be systematically lower. Relative and rank-based criteria transfer better under distribution shift. Check with leave-one-country-out validation (T24) and lean on relative criteria.

---

### T32. Blocking engineered for scale

**Use when:** always. The organisers state blocking must scale to billions of records, and they review the code that produces `candidate_pairs.tsv` (G2).

**Technique:**
- Never compare all pairs, even if the dataset fits in memory. Retrieve with inverted indexes (key → record IDs), sparse top-K matrix products computed in chunks (e.g. `sparse_dot_topn`), and approximate nearest-neighbour indexes (FAISS, hnswlib).
- Guard against giant blocks. Skip or sub-block any key shared by more than B records, such as the token "enterprises" or a very common PIN. The meta-blocking literature calls this block purging.
- Process S1 in chunks so memory stays bounded.
- Compute pairwise features only on the final candidate pairs.
- Log comparisons made and runtime per stage. State the complexity (e.g. roughly O(N · K) retrieval) in the methodology document.

---

## 6. Validation Tricks

### T23. Validate against the full distractor pool

**Use when:** always. This is the single most common cause of CV/LB gaps in matching competitions.

**Why the obvious approach fails:** If you split S1 into folds *and* restrict the validation candidate pool to the S2/S3 records matched to validation S1 entities, validation is far easier than test. Thresholds come out too loose and you over-merge on the leaderboard.

**Technique:** Split by S1 entity (GroupKFold). For validation S1 entities, retrieve candidates from **all** S2/S3 training records, including those that belong to training-fold S1 entities. Those act as realistic distractors. The Shopee 3rd place searched for candidates across the whole training set rather than only within the validation fold, and reported that this narrowed the CV/LB gap and made the two correlate.

---

### T24. Leave-one-country-out as the France proxy (no France data used)

**Use when:** deciding anything that might not transfer to France.

**Why it matters:** Matchers fine-tuned on one dataset transfer poorly. Peeters et al. measured large F1 drops when fine-tuned PLM matchers (Ditto, RoBERTa) were applied to other datasets, while LLM-based matchers were more robust to unseen entities. France is also only in the test set, so under G1 nothing about it can be learned from data.

**Technique:**
- Train on US → validate on India, and vice versa. Features and models whose score collapses in this setting are country-specific; keep them only behind the country-aware parts of the pipeline.
- France gets **no parameters of its own**. It runs through the global model. For thresholds and cutoffs, use the more conservative of the two cross-country optima (precision-heavy metric: under uncertainty, err toward fewer matches and smaller candidate lists).
- Prefer relative and rank-based criteria (T4's α, T31's gap rule) over absolute score thresholds, because they transfer better under distribution shift.
- Rely on the country-agnostic backbone for France: char n-grams, numbers, containment, and rarity with deliberate unseen-token handling (T12).
- Smoke-test that every test S1 ID, France included, appears in both output files. That is a format check, not training.

---

### T25. Use the leaderboard for diagnostics, not for fitting

**Use when:** deciding what to submit and how to read the public score.

**Why the obvious approach fails:** The leaderboard is feedback from the test set. Tuning parameters to it, especially France-specific ones via probe submissions, fits the test set indirectly, overfits the public subset (the final ranking uses the private part), and breaks G1. Submissions are also scarce: 15 in total.

**Technique:** Submit only pipelines chosen on validation. Use the public score as a sanity check: a sharp drop relative to cross-validation signals a bug or a train/test mismatch, which you then investigate *on validation* (T23), not by probing. Do not make France-only probe submissions.

---

## 7. Model Tricks

### T26. Ditto-style cross-encoder input

**Use when:** adding a transformer re-ranker on top of the GBDT.

**Technique:**
- Serialise pairs with explicit field tags: `[NAME] ... [ADDR] ... [PIN] ... [SEP] [NAME] ... [ADDR] ...`.
- "Inject domain knowledge" by marking extracted numbers, postcodes and core names with special tokens, so the model attends to them. This is Ditto's domain-knowledge injection.
- Put the parsed components first and the landmark text last.
- Feed the re-ranker's score into the GBDT as a feature rather than replacing it.

---

### T27. Metric-learning embeddings trained on entity clusters

**Use when:** the pretrained embedding retriever misses matches that a fine-tuned one would catch.

**Technique:** Treat each S1 entity plus its matches as one class and fine-tune a multilingual encoder with ArcFace or contrastive loss. One Foursquare gold solution's first stage was exactly this: ArcFace metric learning predicting the POI, which put matching records close together in embedding space. The Shopee 1st place gradually increased the ArcFace margin during training and found the margin value mattered for embedding quality.

**Risk:** Singletons have no positives: include them as single-member classes or as negatives only.

---

### T28. Small LLM adjudicator only for the borderline band

**Use when:** after the main pipeline, for the pairs whose probability sits in an uncertain middle band. Particularly useful for France.

**Technique:**
- Run a license-compliant model of at most 8B parameters on the borderline pairs only (hundreds to low thousands).
- Ask a strict binary question, and use the answer (or its token probability) as a feature or tie-breaker.
- A cascade (cheap rules → model → LLM for the hardest cases) is how practitioners like Overture's Drew Breunig structure place conflation.
- Evidence that LLMs generalise better to unseen entities than fine-tuned PLMs makes this most valuable for France.

**Risk:** Latency, and non-determinism unless temperature is 0. Log prompts and outputs for reproducibility.

---

### T29. Threshold-normalised ensembling

**Use when:** blending models (GBDT variants, cross-encoder, embedding scores) makes things worse.

**Why the obvious approach fails:** Averaging raw scores from models with different operating points shifts the effective threshold unpredictably.

**Technique:** Shopee's 2nd place:
1. Find each model's own best threshold on validation.
2. Subtract it from that model's score.
3. Sum the shifted scores; a pair is a match if the sum > 0.

Alternatively, stack with a logistic regression on out-of-fold scores, then re-run T5 calibration.

---

## 8. Things That Failed or Backfired in Related Competitions

- **Generic sentence-encoder features** (Universal Sentence Encoder) added to the GBDT made the Foursquare 7th-place leaderboard score *worse*. Task-trained embeddings helped instead. → Do not assume an off-the-shelf embedding adds value; test it.
- **Unpruned transitive closure** chains distinct entities together. → Always use T9's guards.
- **Fine-tuned matchers overfit their training domain** and lose a lot of F1 on other datasets. → Relevant to France; see T24.

---

## 9. Sources

- Foursquare Location Matching (Kaggle 2022), winners overview: https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/
- Foursquare 7th place write-up (asymmetric containment, metric-aware weights, graph post-processing, multi-retriever budgets): https://future-architect.github.io/articles/20220720a/
- Foursquare 4th place code: https://github.com/TheoViel/kaggle_foursquare
- Shopee Price Match Guarantee (Kaggle 2021), summary of 1st–4th place solutions: https://masatakashiwagi.com/blog/kaggle-shopee-solution/
- SIGMOD Programming Contests 2020–2022 retrospective: https://iris.unimore.it/retrieve/handle/11380/1310126/574065/09_Reports_DeAngelis.pdf
- SIGMOD 2022 blocking winner (WBSG, Mannheim): https://www.uni-mannheim.de/en/news/wbsg-wins-sigmod-programming-contest-2022/
- Ditto (deep entity matching with pre-trained LMs): https://arxiv.org/abs/2004.00584
- Entity matching using LLMs (Peeters et al.): https://arxiv.org/pdf/2310.11244
- Fine-tuning LLMs for entity matching (Steiner, Peeters, Bizer): https://arxiv.org/abs/2409.08185
- Optimal F-score clustering for bipartite record linkage (Bai, Binette, Reiter): https://arxiv.org/pdf/2311.13923
- Bipartite graph matching for clean-clean ER (Papadakis et al.): https://arxiv.org/abs/2112.14030
- Splink term-frequency adjustments: https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html
- Overture places conflation walkthrough (Drew Breunig): https://www.dbreunig.com/2024/09/27/conflating-overture-points-of-interests-with-duckdb-ollama-and-more.html
- Segment-level Indian address matching (HyperVerge): https://hyperverge.co/blog/address-match-onboarding-ai-india/
- Contextual address matching patent examples (Indian addresses, numeric conflicts): https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8595219
- Meta-blocking (Papadakis et al., TKDE 2014): https://helios2.mi.parisdescartes.fr/~themisp/publications/tkde13-metablocking.pdf
- Enhanced meta-blocking, reciprocal pruning results: https://www.sciencedirect.com/science/article/abs/pii/S2214579616300168
- End-to-end ER for big data survey (node/edge pruning taxonomy): https://arxiv.org/pdf/1905.06397
- Supervised meta-blocking (GSM): https://www.sciencedirect.com/science/article/pii/S0306437923001436
