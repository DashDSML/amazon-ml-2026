# Four validated negative results on the matcher/decision layer

All four were implemented properly and validated on real, full-scale data (not smoke
samples) -- not abandoned early, genuinely tried and measured. Listed because a negative
result validated on real data is still evidence, and because two of these converge directly
with findings already in your own log (D42, D48).

Baseline throughout: LightGBM binary classifier (num_leaves=127, lr=0.05, feature_fraction=0.8,
GroupKFold-by-S1, fold 4 holdout) + a record-centric decision rule (each S2/S3 record takes
its single highest-probability candidate S1, accepted iff probability >= a threshold t tuned
on the fold-4 holdout). Baseline F0.5 = 0.9541 (v1, real full-scale, fold-4 holdout).

## 1. Entity-centric decision layer (gate + additional-match rule + isotonic calibration)
**Idea:** per-S1-entity "does this entity have any match at all" gate, then a stricter
absolute+relative rule for additional matches beyond the top pick -- this is the
NonStandard.md T1/T2/T4 family (per-entity expected-utility decisions, a separate singleton
gate), the same family your own D42 "learned singleton gate" and expected-F0.5 rule belong
to.

**Result:** 0.9543 vs baseline 0.9541 -- within noise, and individually *worse* on both the
singleton and non-singleton buckets. Isotonic calibration of the raw probabilities came back
as an identity mapping (mean unchanged) -- the raw LightGBM probabilities were already
well-calibrated, so miscalibration wasn't hiding a real gain either.

**Converges with your D42:** "the exact expected-F rule differs by ≤ 0.0002; a learned
singleton gate gave +0.0001." Two independently-built pipelines landing on the same
conclusion is fairly strong evidence that decision-layer sophistication is not where the
remaining gap lives, for this kind of pairwise-classifier + per-entity-decision architecture.

## 2. Metric-aware sample weights (train pairs weighted by their F0.5 misclassification cost)
**Idea:** weight each training pair by the score its misclassification would cost, using the
exact formula derived from the F0.5 definition (`neg_weight(t) = 1/(1.25t+1)` for an S1 with
t true matches, or 1.0 for a true singleton; `pos_weight(t) = 0.25/(1.25t-1)`) -- verified the
formula reproduces every value in `challengecontext.md`'s worked table before running.

**Result:** 0.9544 vs 0.9541 -- +0.0003, not a real improvement, despite the optimal
threshold shifting materially (0.65 -> 0.40). The model already learns to separate these
cases from the raw features; reweighting only shifts the prior, which the decision threshold
already absorbs.

## 3. T7 (mutual/reverse retrieval consistency) + T9 (transitive same-S1 co-candidate evidence)
**Idea:** two feature groups from the NonStandard.md literature-mapped list.
- T7: `is_mutual_top1` (this record's own top pick agrees with being this S1's own top pick)
  and a reciprocal-rank-sum feature. Free -- derived from already-existing rank columns.
- T9: for each candidate, max name similarity to, and count of "agreeing" (>=0.7 similarity),
  *other* candidates already in the same S1's own list (never adds a new ID -- G2-safe).
  Capped the self-join at 200 candidates/S1 to bound the O(n^2) worst case.

**Individually, both features show strong signal**: `is_mutual_top1=1` correlates with a 92%
true-match rate vs. a ~49% baseline, and both groups land in the top-15 LightGBM feature
importances (ahead of several existing features).

**Result: F0.5 0.9542 vs 0.9541 -- still within noise.** `gap_top2` (an existing feature: the
gap between a record's best and second-best blocking score) dominates feature importance by
a wide margin in both the old and new models (85M gain vs. the next feature's 19M) -- the new
features are real but largely *redundant* with what the model already extracts from existing
features, not adding net-new separability the single-stage decision layer can act on.

**Converges with your D48 in spirit, if not mechanism:** your stage-2 model (a genuine
second-stage LightGBM trained on out-of-fold stage-1 probabilities, not just concatenated
features into one model) gave a real, measured gain (+0.0017 to +0.0022 held-out) using
list-context features conceptually similar to our T9 (rank/gap/agreement with the top
candidate, best-name-similarity-with-confident-candidates). **The likely lesson: stacking
(a properly-staged second model on OOF outputs) can extract value from list-context
information that simply adding similar features to a single-stage model cannot.** If we
revisit T9, the right move is probably a stage-2-style stack, not more feature concatenation.

## Bottom line
None of these four should be read as "don't bother with decision-layer or list-context work"
in general -- your own stage 2 proves list-context information *can* help, properly staged.
They should be read narrowly: on a single-stage LightGBM + record-centric threshold, these
four specific interventions don't move the needle, and further effort in that exact direction
is probably better spent elsewhere (the France/unseen-country gap, or a genuine second-stage
model).
