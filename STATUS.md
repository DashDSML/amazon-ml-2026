# Status (2026-09-27, commit d0eabce)

**Final submission is v2r (tuned), DONE and ready (not submitted -- that's your call):** T21 ranker-pruned candidates + a tuned LightGBM matcher (`min_data_in_leaf=100`, found via a 15-config hyperparameter sweep, `model_tune.py`). `output/matching_results.tsv` + `candidate_pairs.tsv` on EC2, re-validated PASS. CV F0.5 **0.9628 (v2) -> 0.9661 (v2r) -> 0.9669 (v2r tuned)**, blocking oracle ceiling 0.9925. Local backup: `output_v2r/`. Fallback still available: v2 (`lb-3`, CV 0.9628) in `output_v2_lb3/`.

**Hyperparameter tuning (new this push):** never actually tuned before -- every prior "improve the matcher" attempt kept the same fixed params and only changed post-hoc decision logic. Swept 15 configs on a 400k-S1 sample (real F0.5, not logloss); `min_data_in_leaf=100` won and, critically, converged in ~180s at full scale (not the ~1300 rounds/hours the small sample implied -- the sample's regularization pressure didn't transfer to 8.7M rows). Promoted, confirmed, adopted.

**Six further things tried in this same push, all landed within noise (not adopted), on top of the four already-negative results from earlier in the session:**
- `decide_entity.py` (entity-centric decision) re-tested against the tuned matcher: 0.9673 vs 0.9669 -- same ~noise-level pattern as its three prior tests on different matchers.
- Logistic regression alone (0.9558) and GBM+logreg blends (monotonically worse as blend weight rises) -- clean negative, no complementary signal.
- `lambda_l2=1.0` combo -- killed at a 20-min time cutoff without finishing (unlike `min_data_in_leaf`, its round count did not collapse at full scale); abandoned for time, not conclusively ruled out, but not worth revisiting given the pattern below.
- 3-seed GBM ensemble: 0.9669-0.9674 across seeds/averages -- within noise; picking the best seed after the fact would be cherry-picking, not a real generalizable gain.

**Ten independent techniques across this whole session now converge to within ~0.008 of each other** (0.966-0.974 band): this is strong evidence 0.9669 is close to a real ceiling for the current feature set, not a sign the right technique hasn't been found yet.

**Not attempted:** T19 noise-operator augmentation for the France/unseen-country gap -- still the one lever that targets a *different* part of the problem (generalization, not matcher fit) and was never built out, due to time.

**Everything above is committed to git and mirrored on EC2.** Nothing has been submitted to the leaderboard.
