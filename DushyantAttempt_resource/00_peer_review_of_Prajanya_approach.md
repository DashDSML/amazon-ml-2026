# Peer review of PrajanyaAttempt_resource (studied in depth, 2026-09-27)

Read all four files (`current_bottleneck.md`, `improvement_ideas.md`, `strategy_current.md`,
`project_state.md`, D01-D65). This is a stronger pipeline than ours on every measured axis
(leaderboard 0.978 vs our unsubmitted ~0.96 estimate for blocking-v2+matcher-v2). What follows
is a direct, honest assessment: one likely-important compliance issue, several things done
right that we should copy, and a couple of things worth double-checking.

## 1. Flag: the IDF table is fit on TRAIN + TEST combined — likely a rules violation

`strategy_current.md` §2 row 9 says the IDF/Soft-TFIDF features (`idf_feats.py`) use a
"**fixed train + test corpus**", and D52 confirms the built table (`runs/c1_idf/idf.parquet`)
has **24.2M records** — exactly train (2.2M+5.03M+5.29M ≈ 12.5M) + test (1.73M+4.89M+5.08M ≈
11.7M). This table is live in v4/v4r, the model behind the best submission (0.978, D53/D62).

Under the rules as we've read them (`challengecontext.md`'s "no ground truth for test,
build your own validation" plus our own `NonStandard.md` G1, which explicitly lists "fitting
vocabularies, TF-IDF vectorisers, IDF/frequency tables... on test data" as **not allowed**),
this is a real problem: term document-frequency counts computed over the test corpus mean
France's own vocabulary is informing a fitted artefact the model then uses on France. This
is different in kind from things that are explicitly allowed at inference time (e.g. block
sizes, per-query context) — an IDF table is a *fitted statistic over test row content*, not a
per-query computation.

**This is worth resolving before the deadline, not after.** Two ways to fix it without losing
much: (a) refit the IDF table on train-only (12.5M records) and re-measure — if `cand_rarity`/
`soft_tfidf_a` still rank near the top of feature importance, the gain likely survives, since
document frequency for common tokens (legal forms, generic words) is unlikely to shift much
between train-only and train+test counts; (b) if the organizers' code review would treat this
generously (it's a borderline case, not a lookup service or external data), at minimum
disclose it explicitly in the methodology document rather than let a reviewer discover it.
Given "final rankings... a review of candidate_pairs.tsv and the code that produces it" is
explicit in the rules, I'd treat this as the single highest-priority thing to check before
packaging.

**Related, lower-severity:** D58 records "the user" saying transductive self-training
("using test data for training... nothing in the problem statement forbids it") is allowed,
superseding an earlier no-self-training decision. D60 shows this was tried (pseudo-labelling
test pairs) and **dropped for hurting France-proxy accuracy**, not for a rules reason, and it
does not appear to be in the shipped v4r model. So this one is probably moot in practice (not
shipped), but it reflects the same underlying rules-interpretation gap as the IDF table, and
is worth aligning on explicitly rather than leaving implicit.

## 2. Things done right that we should copy

**Test-like density training (`--hide-permille`, D49 on).** This is the best idea in either
pipeline. They hide a random fraction of *training* S1 (never touching test) to inflate
distractor density in the training pool, closely reproducing the actual test shift (test has
~23% more pool records per S1 than train, mostly lookalikes — D46/D48). This is fully
train-only (G1-clean) and it visibly worked: held-out→leaderboard gap shrank from −0.013 (v1)
to −0.005 (v4r+shrink). We have never simulated this shift at all; our CV is entirely on
train's own density. **This is the single most actionable thing to adopt from their work.**

**Adversarial validation of features before shipping (D48).** Submission 2 (sibling/count
features) scored 0.886 on the leaderboard despite 0.973 held-out — the held-out set couldn't
detect that count features (`c_n`, `c_rank`, sibling counts) shift meaning under the denser
test pool. They caught this by comparing train vs. test feature-value distributions directly,
not just by CV score. **We should audit our own feature set for the same risk**: `n_cand_record`
and `n_cand_s1` in our `features.py` are exactly this kind of count feature, and we've never
checked whether their distribution shifts between train and test densities. Given test's
candidate lists are already measurably larger than train's in our own pipeline (7.08 vs 5.40
mean cands/S1 for v2), this is a real, unchecked risk for us too.

**Numeric go/no-go gates set before running expensive jobs (D25).** They wrote down exact
thresholds (pair recall ≥ 0.99, oracle F0.5 ≥ 0.995, etc.) *before* the 10+ hour full blocking
run, so the run's success/failure was judged against a pre-committed bar, not vibes after the
fact. We've been more ad hoc (see our own `BLOCKING_V2.md` targets, which are looser).

**Paired bootstrap confidence intervals before choosing a submission (D64).** 1000-resample
bootstrap on 220K held-out S1 gave a confidence interval on "is v4r really better than v4,"
not just a point estimate. We reported point-estimate deltas all session (e.g. our T7/T9
features: 0.9542 vs 0.9541) without ever asking whether that's inside the noise band. Given
several of our own "no improvement" verdicts rest on deltas of 0.0001-0.0003, a bootstrap CI
would make those verdicts more rigorous rather than relying on eyeballing.

## 3. A convergent finding worth noting to them directly

Their D42/current_bottleneck.md independently found that decision-layer sophistication
(their expected-F0.5 rule vs a fixed threshold) differs by **≤ 0.0002**, and a learned
singleton gate added only **+0.0001**. This is the *same conclusion* our own session reached
after four separate attempts (entity-centric gate + calibration, metric-aware sample weights,
and T7/T9 feature engineering — see `03_matcher_experiments_negative_results.md`): none of it
moved our F0.5 by more than 0.0003. Two independent pipelines, built independently, landing on
the same "the decision layer isn't the bottleneck" conclusion is fairly strong convergent
evidence — worth stating to them plainly since it validates their own D42 finding and should
save them from re-litigating it.

Similarly, our leave-one-country-out numbers (`04_loco_unseen_country_findings.md`) are the
same order of magnitude as their D47 (unseen-country loss ≈ 3-5 points of F0.5 either way),
which cross-validates both teams' France-risk estimates independently.

## 4. Smaller things worth a second look (not necessarily wrong, just unverified)

- **Learned partition/alias artefacts (D12) and the Indic dictionary (D10)** — these appear to
  be train-only (D10 explicitly measures the dictionary on a held-out slice), unlike the IDF
  table. Worth a final confirmation that no test rows entered these specifically, since the
  same 24.2M-style combined-corpus pattern that happened with IDF could in principle have
  happened elsewhere without being called out as clearly.
- **India's candidate lists sit at the p90=60 cap on both train and test** (D39/D40) — their
  own notes flag this as the main remaining blocking-side lever. Given our blocking v2 (fix
  A3: rank by summed rarity-weighted key evidence instead of raw channel scores) is a
  genuinely different mechanism from their channel-floor cap, it might be worth a quick,
  cheap comparison on their India partition specifically if there's time — though given their
  oracle (0.997-0.998) is already higher than ours (0.9893-0.9895), I wouldn't expect a large
  gain and wouldn't prioritize it over the IDF-table fix above.
