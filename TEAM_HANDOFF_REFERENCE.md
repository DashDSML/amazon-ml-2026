# Team Handoff Reference — Dushyant's Pipeline (amazon-ml-2026)

**Purpose of this document:** this project (business entity resolution, Amazon ML Challenge 2026) was developed independently on this machine while a teammate (Prajanya) developed a separate, ultimately higher-scoring pipeline on their own account (best known internal submission: **0.982**; public leaderboard top score: **0.992**). The final documentation package is being assembled on the teammate's account using their pipeline. This document is a complete technical recap of everything done on *this* system, written as reference material for whichever agent/person assembles that final documentation — so it can cite, reuse, or explicitly disclose anything relevant from this side of the work without re-deriving it. It assumes no prior context.

A parallel, more narrative document (`SESSION_CHRONOLOGY.md`, same repo) covers the same ground in situation/options/decision/result prose, aimed at a human reader. This document is organized for fast lookup instead.

---

## 1. The problem, in one paragraph

Business entity resolution: a reference registry (`Source1`, ~2.2M deduplicated records) must be matched against two other sources (`Source2`, `Source3`, together several million records) that each contain zero, one, or more records referring to the same real-world business as a given Source1 entity. Countries: US and India in training; **France appears only in test, fully unseen**. Metric: macro F0.5 per Source1 entity (singletons — entities with zero true matches — count too: an empty prediction on a true singleton scores 1.0, any prediction on it scores 0.0). Deliverables: `matching_results.tsv` (final matched IDs per Source1) and `candidate_pairs.tsv` (the exact candidate set the matcher ran inference on — **this file's size counts directly toward the final ranking**, smaller is better, per the organizers' stated rules). Full rules in `challengecontext.md`.

Hard constraints that shaped every decision below (`CLAUDE.md`'s G1/G2, confirmed against `challengecontext.md`):
- **G1 — nothing may be fit on test data.** No vocabularies, IDF/frequency tables, thresholds, or dictionaries derived from test rows. France must be handled as fully unseen, using only things fit on train (or general-knowledge lexicons written without inspecting test rows — explicitly permitted by `NonStandard.md`'s T19).
- **G2 — candidate-set size counts toward the ranking.** A bigger candidate list is a cost, not a free lunch, even if it raises recall.
- Any pretrained model: MIT/Apache-2.0 license, ≤8B parameters. (This pipeline used none — LightGBM, MIT-licensed, trained from scratch, was the only model.)

---

## 2. Pipeline version lineage (the core evolution)

Everything below is **blocker (candidate generation) × matcher (classification/ranking) × decision layer**, evolved mostly independently of each other. "Matcher" is the same LightGBM architecture throughout (gradient-boosted trees, binary objective) — what changed between versions is *what it was trained on* (candidates) and, only at the very end, *its own hyperparameters*.

| Pipeline | Blocker | Matcher | Decision layer | CV F0.5 | Oracle ceiling | Real LB score | Status |
|---|---|---|---|---|---|---|---|
| **P0** | v1 (key-based inverted index) | — (rank/score rule only) | threshold rule tuned on train | f05≈0.84 (rule, train) | 0.978 | not submitted standalone | superseded same day |
| **P1** | v1 | LightGBM v1 | global threshold (t=0.65) | **0.9541 → 0.9550** (T7/T9/T8 features added later) | 0.978–0.980 | **0.943 (lb-2, real)** | submitted |
| **P2** | v2 (see §3) | LightGBM retrained on v2 candidates | global threshold (t=0.65) | **0.9628** | 0.9893 | not submitted standalone | superseded by v2r before submission |
| **P3** | v2r — T21 ranker-pruned (see §4) | LightGBM retrained on v2r candidates | global threshold (t=0.70) | **0.9661** | 0.9925 | not submitted standalone | superseded by tuned version |
| **P4 (final)** | v2r | LightGBM, **tuned hyperparameters** (`min_data_in_leaf=100`, found by a 15-config sweep) | global threshold (t=0.70) | **0.9669** | 0.9925 | **0.954 (lb-3, real)** | **this pipeline's final submission** |

Blocking v1 is frozen and byte-identical throughout — every later blocking version (`v2`, `v2r`) was built as a separate, parallel tree (`dataset/cache/blocking_v2/`, `blocking_v2r/`) so v1 always remained a known-good fallback. This was an explicit, repeated constraint throughout development, not an accident.

**Two real leaderboard data points exist for this pipeline**, and they're the only ground truth connecting CV score to actual test performance:
- lb-2: CV 0.9541 → LB 0.943 (gap −0.0111)
- lb-3: CV 0.9669 → LB 0.954 (gap −0.0129)

Both gaps are close to each other, which is itself informative: it suggests the CV→LB gap is a fairly stable property of *this feature set's* generalization to the test distribution (principally the unseen-France problem, see §6), not something that moved much as the candidate set and matcher improved underneath it.

---

## 3. Blocking v2: what changed and why (candidate generation)

v1 was a plain key-based inverted index (hashed name/address tokens, city, house-number, etc.) with rarity-agnostic ranking. v2 added, in this order, each validated independently via a fast ~20k-S1 diagnostic loop before committing (`BLOCKING_V2.md` has the full method — bucket every missed true pair by cause, fix the biggest bucket first, stop once a fix's marginal gain drops below ~0.1 oracle points):

1. **Rarity-weighted retrieval ranking** (fix A3): rank candidates by the sum of rarity weights of shared keys, not raw block size. Oracle 0.9778 → 0.9849.
2. **Key-evidence folded into the stage-1 score** (fix S1/S2). Oracle → 0.9863.
3. **Composite keys**: a rarest-token-pair key ("P") and an address-token+city key that doesn't need a house number ("N"), on top of the existing single-token keys. Oracle → 0.9898.

**A real bug caught and fixed here (G1-relevant):** the P-key's token document-frequency table was first computed per-batch — inconsistent across batches of different composition, and a latent G1 violation since batches could include test-side query rows. Fixed to a single global table fit once on train S1 (+train S2/S3), cached (`dataset/cache/blocking_v2/name_token_df.parquet`), loaded unchanged for train and test scoring. Unseen tokens (expected for France) default to document-frequency 0 (treated as rarest), ties broken alphabetically.

**Rejected:** shrinking `PRE_CAP_V2` (the pre-ranking pool size before pruning) from 80 to 50 — oracle dropped (0.9893→0.9883) for no real candidate-size benefit (5.37→5.34 mean). Kept at 80.

---

## 4. T21 — the supervised meta-blocking ranker (single biggest win)

**Idea (NonStandard.md T21):** rather than hand-tune the blocking score's pruning rule, train a *second* supervised model on the same features the final matcher uses, over a much wider candidate pool (SAVE_CAP≈80 instead of the final ≈5), and let **that model's own pruning decision become the actual candidate_pairs.tsv content** — i.e., it replaces the hand-weighted blocking score, not just an internal implementation detail. This needs disclosing explicitly in documentation since it changes what counts as "blocking" vs "matching."

**Result:** at the ranker's own grid-search optimum (floor=0.4, alpha=0.9, m=3), it beat the frozen v2 hand-weighted pruning on **both** axes simultaneously:
- oracle F0.5: 0.9893 → **0.9925** (full-scale confirmed)
- mean candidates/S1: 5.40 → **4.04** (full-scale confirmed)

This is an unambiguous win by the project's own adoption rule (raise oracle without growing the list, or shrink the list without losing oracle) — most blocking changes in this project traded one for the other; this moved both favorably at once.

**Build artifacts:** `ranker_data.py` (wide-pool retrieval), `ranker_features.py` (chunked feature computation over the wide pool), `ranker_model.py` (trains the ranker on a class-balanced sample), `ranker_apply.py` (applies the trained ranker to the full wide pool for both train and test, prunes, and recomputes final features — this is what actually produces the v2r candidate/feature trees that `model.py`/`decide.py`/`make_submission.py` train and run inference on).

**Two real operational incidents here, worth knowing if reproducing this elsewhere:**
- `ranker_features.py`'s first version loaded the entire ~781M-row wide pool at once and was **OOM-killed at ~61.3GB RSS** on a 62GB machine (confirmed via `dmesg`). Fixed by processing a handful of part-files at a time. General lesson: any step touching the *wide* (pre-pruning) candidate pool needs explicit chunking; the final pruned pool (≈5/S1) is small enough that this isn't an issue, but the wide pool (≈80/S1) at full data scale is not.
- Applying the ranker to build test-side candidates hit a **disk-full crash** (58GB EC2 disk at 100%) partway through, because the wide pool's raw retrieval output and its derived features for *both* train and test together exceeded available disk. Fixed by deleting already-consumed train-side intermediates (26.6GB) once they'd served their purpose. General lesson: budget disk, not just memory, when working with the wide pool at this data scale — it's roughly 4-5x the size of the final pruned artifacts.

---

## 5. Feature engineering — what's in the final feature set, and its evolution

`features.py`'s `FEATURE_COLS` grew from an initial ~29 columns to the final 34, in two additions, both validated before being kept:

- **T7 (mutual/reverse retrieval consistency)** — `is_mutual_top1`, `reciprocal_rank_sum`: free, derived purely from each candidate's rank within its own record vs. its rank within its own S1 (no new computation needed, both ranks already existed).
- **T9 (transitive S2↔S3 evidence)** — `trans_max_sim`, `trans_agree_count`: for each candidate, similarity to and count of "agreeing" *other* candidates already in the same S1's own list (capped at 200 candidates per S1 to bound the self-join cost for a few very high-degree/generic-name S1 groups). G2-safe — never introduces an ID outside the already-retrieved candidate set.
- **T8 (confusability)** — `confusability`: each S1's similarity to its *nearest other* S1, reusing the exact same key-based retrieval+scoring machinery already built for blocking, just run S1-vs-S1 instead of S2/S3-vs-S1. Free and label-free, since every S1-S1 pair is a guaranteed non-match by construction (S1 is deduplicated). **Motivation was explicitly aimed at the teammate's two biggest loss buckets** (see §6) — a high-confusability S1 (a common/chain-like name pattern) needs stronger evidence before accepting a match; a low-confusability S1 (a distinctive name) can safely accept a weaker-looking true match. Validated qualitatively first (top results were exactly same-name-different-address lookalikes) before trusting it at scale. On the final tuned v2r matcher, this ranked as the **4th-most-important feature by gain** — a strong, real signal, not a marginal addition.

The remaining ~31 features (unchanged since early in the project): blocking score/rank/gap features, rapidfuzz string-similarity scores (Jaro-Winkler, token-sort, partial-ratio on normalized name fields), name/address token-Jaccard, legal-form match, alias forward/backward similarity, house-number/city/state agreement flags, address-number Jaccard, and record-quality flags (gibberish-name, transliteration-residue, non-Latin script) carried only on the S2/S3 side (S1 is a clean reference registry with no missing names/addresses, confirmed in R1 EDA).

---

## 6. Decision-layer and matcher-hyperparameter experiments — the full negative-result inventory

This project treated negative results as load-bearing evidence, not failures to hide. **Ten independent techniques across the whole project landed within ~0.008 F0.5 of each other (roughly 0.966–0.974)** regardless of which matcher or candidate set sat underneath them — strong evidence this is close to a real ceiling for *this specific feature set*, not a sign the right trick hadn't been found yet.

| # | Technique | What it does | Result | Verdict |
|---|---|---|---|---|
| 1 | `decide_entity.py` (T1/T2/T4/T5/T6): per-entity expected-F0.5 decision, singleton gate, first-vs-additional-match asymmetric thresholds, isotonic calibration | Replace the one global threshold with a cost-aware per-entity rule, motivated directly by F0.5's own asymmetric cost table (missing an entity's only match costs 1.0; a doubtful 3rd ID costs only ~0.21) | Tested 4 times across different matchers (v1 baseline, v1 post-T7/T9/T8, v2r baseline, v2r tuned): 0.9543 vs 0.9541, then 0.9664 vs 0.9661, then 0.9673 vs 0.9669 — always +0.0002–0.0004 | **Rejected every time.** Calibration specifically came back as the identity function (probabilities already well-calibrated). |
| 2 | `model_weighted.py` (T3): LightGBM sample weights set to the exact F0.5 marginal-cost formula | Weight training rows by their real cost to the metric instead of treating all rows equally | 0.9544 vs 0.9541 (+0.0003); optimal threshold shifted materially (0.65→0.40) despite the score not moving | **Rejected.** |
| 3 | `PRE_CAP_V2=50` (smaller pre-ranking pool) | — | oracle 0.9883 vs the kept value (80)'s 0.9893, no size benefit | **Rejected**, kept 80. |
| 4 | 15-config LightGBM hyperparameter sweep (`model_tune.py`) — first time the matcher's *own* hyperparameters were ever searched; every prior attempt above only changed post-hoc decision logic on a fixed model | Swept tree depth, learning rate, regularization, leaf-size constraints on a fast 400k-S1 sample, scored by real F0.5 | `min_data_in_leaf=100` won; promoted to full data, converged in 184s (not the ~1,300 rounds/hours the small sample's round-count implied — the constraint is far weaker relative to 8.7M rows) | **Adopted.** CV 0.9661→0.9669, the only non-noise gain in this table besides T7/T9/T8 above. |
| 5 | Logistic regression alone, as a simple linear second model on the same features | — | 0.9558, well below the GBM's 0.9669 | **Rejected** — expected, linear model can't capture the nonlinear interactions the GBM already does. |
| 6 | GBM + logistic-regression probability blend, weights 0.1–0.5 | — | 0.9669 → 0.9652, **monotonically worse** with more logreg weight | **Rejected** — logreg has zero complementary signal here. |
| 7 | `lambda_l2=1.0` regularization combo (on top of the already-tuned `min_data_in_leaf=100`) | — | killed at a 20-minute time cutoff under deadline pressure, no confirmed result | **Inconclusive, not adopted** — unlike #4, this config's round-count did not collapse at full scale within the time available to check safely. |
| 8 | 3-seed GBM ensemble (independently-seeded models averaged) | — | 0.9669–0.9674 across seeds and averages | **Rejected** — within noise; picking the best seed after the fact is cherry-picking, not a generalizable technique. |
| 9 | PCA+logreg, KNN-as-last-resort-for-unmatched-tuples, transformer cross-encoder (all proposed, feasibility-checked, not attempted) | — | — | **Declined on feasibility grounds**, not tested: PCA adds nothing with only 34 features; KNN needed more design/validation time than remained before a hard deadline; a cross-encoder needs GPU-scale throughput the available CPU-only EC2 instance could not provide in minutes. |
| 10 | Char n-gram TF-IDF as an additional **retrieval** channel (`test_ngram_retrieval.py`, distinct from the decision-layer items above — this targets blocking, not matching) | Fit on train-only S1 names, tested whether it would raise the blocking ceiling | Would recover **~44% of currently-missed true pairs** at a K=20 candidate budget, with a sensible bimodal failure pattern (typo-recoverable vs genuinely-different content) | **Validated as a real finding, never built into production** — time ran out before this was prioritized over the ranker work (§4), which had a larger expected payoff. This is a genuinely open, promising lever if anyone picks this pipeline back up. |

---

## 7. The one real, unresolved generalization finding: unseen-country risk (LOCO test)

`loco_validate.py` (T24) held out an entire country from training (as a France proxy, since France has zero training data) and measured the *real* degradation, with the decision threshold also not allowed to see the held-out country ("blind" case):

- Holding out **India**: F0.5 dropped from 0.9541 to **0.9042** (blind) / 0.9129 (best case, threshold optimally tuned on the held-out country anyway).
- Holding out **US**: dropped only to 0.9453 — a much smaller, 0.88-point cost.

This proved the unseen-country gap is a genuine feature/model generalization limit, not a threshold-calibration problem (since even an unfairly-optimal threshold barely helped) — and explains why every CV→LB estimate on this pipeline landed meaningfully below the oracle ceiling (see the two real gaps in §2). **The India asymmetry (much worse than US) is notable and unexplained** — worth investigating if continuing this line of work, since "any unseen country costs ~1 point" was not the finding; "this specific country's patterns are harder to generalize away from" was.

The ceiling-to-CV gap (matcher imperfection, ~2.5–2.6 points, stable across every version in §2) and this CV-to-LB gap (~1.1–1.3 points, generalization) are **separate, independently-measured** costs that don't shrink together — closing one doesn't touch the other. This matters for whoever reads the teammate's numbers alongside this pipeline's: if their approach achieves a smaller total gap, it is closing one or both of these *specific* things differently, not just "tuning harder."

---

## 8. Communication with the other team's pipeline (Prajanya's)

**Context:** partway through the project, learned a teammate (Prajanya) had built an independent pipeline that reached 0.977–0.978 on the leaderboard, well ahead of this pipeline's submitted scores at the time. Four of their planning documents (`current_bottleneck.md`, `improvement_ideas.md`, `strategy_current.md`, `project_state.md`) were available in `PrajanyaAttempt_resource/` and studied in full.

### 8.1 — The single most important thing to carry forward: a likely compliance issue

**Flagged in `DushyantAttempt_resource/00_peer_review_of_Prajanya_approach.md`:** their IDF/soft-TFIDF feature table (`idf_feats.py`) was fit on a **combined train+test corpus** (confirmed via their own logged record count: 24.2M ≈ train 12.5M + test 11.7M exactly), and this table is live in **v4/v4r — the model behind their best submission (0.978 at the time it was flagged, later improved further to 0.982)**.

Under the same G1 rule this pipeline followed strictly throughout (`NonStandard.md`'s explicit list of disallowed test-fitting includes "fitting vocabularies, TF-IDF vectorisers, IDF/frequency tables... on test data"), **this is a real rules problem, not a style nitpick**: document-frequency counts computed over the test corpus mean France's own vocabulary is informing a fitted statistic the model then uses *on France* — materially different from allowed inference-time operations like per-query block sizes.

**This needs to be resolved or explicitly disclosed before the final package goes out**, since the organizers' own rules state the final ranking includes "a review of candidate_pairs.tsv and the code that produces it." Two fixes were proposed at the time: (a) refit the IDF table on train-only and re-measure (likely survives largely intact, since document frequency for common tokens is unlikely to shift much train-only vs train+test) or (b) if the team judges it borderline enough to be defensible, disclose it explicitly in the methodology write-up rather than let a reviewer discover it unexplained. **Whoever is finalizing documentation on the teammate's pipeline needs to know this flag exists and needs to be addressed, one way or the other.**

A second, lower-severity item noted in the same review: an earlier decision to allow transductive self-training (pseudo-labelling test pairs) was later dropped for hurting France-proxy accuracy, not for a rules reason, and did not appear to be in the shipped model — probably moot, but worth a sanity check that it's genuinely absent from the final submitted pipeline.

### 8.2 — What was shared back, and why

Once the team's priority explicitly became *"help Prajanya's pipeline do its absolute best"* (a direct instruction partway through the project), a five-file package (`DushyantAttempt_resource/`) was built and shared, containing:
- `01_overview_and_numbers.md` — this pipeline's real numbers for comparison.
- `02_blocking_v2_fixes.md` — the fix-menu methodology (§3 above) in case any bucket-diagnosis approach was reusable.
- `03_matcher_experiments_negative_results.md` — the decision-layer negative results (§6 above), specifically so effort wouldn't be duplicated re-trying things already shown not to work here.
- `04_loco_unseen_country_findings.md` — the LOCO finding (§7 above), since France-generalization is a shared risk regardless of whose pipeline is used.
- `05_code_and_how_to_run.md` + a full `code_snapshot/` of this pipeline's source, in case any individual technique (e.g., T8 confusability, the ranker) was worth porting directly.

**Independent corroboration:** the teammate's own materials (their D47/D19-style findings, referenced in the peer review) independently found a similar unseen-country generalization cost on their own pipeline — the two pipelines arrived at compatible conclusions about this risk via different routes, which is reasonably strong evidence it's a property of the *data/problem*, not an artifact of either specific pipeline.

### 8.3 — Outcome

The team's final decision (communicated directly) was to submit from the teammate's account using their higher-scoring pipeline (internal best 0.982, vs. this pipeline's real 0.954), with this document serving as the reference material for anything from this side worth incorporating into the final documentation package — most importantly the compliance flag in §8.1, and secondarily any of the validated techniques in §4–§6 that aren't already present in their pipeline (the ranker/T21 approach and T8 confusability are the two most likely candidates to be genuinely novel relative to their work, based on what their own documents described).

---

## 9. Final artifact locations (this system)

All on the local machine (`C:\Users\Dushyant\amazon-ml-2026\`); the EC2 instance used for heavy compute (`3.109.206.248`) was stopped (not terminated) at the end of this work — disk/data preserved, compute billing stopped, but currently unreachable without restarting it via the AWS console.

- `output_v2r/matching_results.tsv`, `output_v2r/candidate_pairs.tsv` — the final submitted (lb-3) files, validated PASS twice (EC2 + local) via the organizers' own `utils/validate_submission.py`.
- `output_v2_lb3/` — the pre-hyperparameter-tuning v2r submission (CV 0.9628/0.9661), kept as a fallback, never itself submitted.
- `output_lb2/` — the original lb-2 (v1) submission files.
- `code/business_entity_resolution/src/` — full pipeline source (listed in full in §10 below).
- `code/business_entity_resolution/README.md`, `Documentation_template.md` — already filled in with this pipeline's own methodology write-up.
- `experiments/log.csv` — every experiment run this project, with CV scores and (where applicable) real LB scores, in chronological order.
- `STATUS.md` — latest-state summary (superseded in detail by this document and `SESSION_CHRONOLOGY.md`, but kept for continuity).
- `SESSION_CHRONOLOGY.md` — full narrative, hour-by-hour account of the same work, for human reading.
- `scripts/run_with_memcap.sh` — the memory-guard wrapper built after the OOM incident (§4); reusable for any future heavy EC2 job regardless of which pipeline.

---

## 10. Full source file inventory (`code/business_entity_resolution/src/`)

| File | Role |
|---|---|
| `data.py` | Core data loading (TSV with `sep="\t"`, `dtype=str`, `keep_default_na=False` per CLAUDE.md), ground-truth pair loading. |
| `normalize.py` | Canonical name/address normalization: legal forms, street types, transliteration folding, gibberish detection, French general-knowledge lexicon (G1-safe additions). |
| `mine_resources.py` | Mines equivalence/abbreviation tables from training pairs; merges in the hand-written French lexicon constants. |
| `check_norm.py` | Sanity-checks normalization output. |
| `cv_harness.py` | GroupKFold-by-S1 evaluation harness (folds 0–3 train, fold 4 true holdout) — used by every experiment in the project. |
| `evaluate.py` | Official macro-F0.5 scorer and per-entity/per-bucket breakdown reporting. |
| `blocking.py` | Core blocking engine: v1 (frozen) and v2 (composite keys, rarity weighting, P-key DF table) key-based retrieval, scoring, pruning. Parameterized by `version` throughout. |
| `fastloop.py` | Fast (~20k-S1) diagnostic loop for blocking experiments, and the floor/alpha/m grid sweep. |
| `confusability.py` | T8: S1-vs-S1 nearest-other-S1 similarity feature. |
| `test_ngram_retrieval.py` | T20: tests whether a char n-gram TF-IDF retrieval channel would raise the blocking ceiling (validated, not productionized — §6 item 10). |
| `ranker_data.py` | T21: builds the wide (SAVE_CAP≈80) candidate pool for ranker training/application, for either split. |
| `ranker_features.py` | T21: chunked feature computation over the wide pool (fixed an OOM crash — §4). |
| `ranker_model.py` | T21: trains the ranker on a class-balanced sample of the wide pool; evaluates its own floor/alpha/m grid against the frozen blocking baseline. |
| `ranker_apply.py` | T21: applies the trained ranker to the full wide pool, prunes, recomputes final features — produces the actual v2r candidate/feature trees. |
| `features.py` | `compute_features()`/`FEATURE_COLS` — the full 34-feature set (§5), shared by the standard and ranker feature-building paths. |
| `model.py` | LightGBM matcher training (`PARAMS`, GroupKFold early-stopping then full retrain), versioned by candidate-set (`v1`/`v2`/`v2r`). |
| `model_tune.py` | The 15-config hyperparameter sweep (§6 item 4) that produced the final tuned matcher. |
| `model_weighted.py` | T3: metric-aware sample weights (§6 item 2, not adopted). |
| `decide.py` | Global-threshold decision layer, tuned per version against real F0.5 on the holdout fold. |
| `decide_entity.py` | T1/T2/T4/T5/T6: entity-centric decision + isotonic calibration (§6 item 1, not adopted, tested 4 times). |
| `loco_validate.py` | T24: leave-one-country-out validation (§7). |
| `make_submission.py` | End-to-end: loads candidates, applies either the rule or model decision method, writes both TSVs, runs the validator, logs the experiment. |
| `_trial_ensemble.py`, `_trial_lambda_l2.py`, `_trial_logreg.py` | Throwaway scripts from the final hyperparameter-tuning push (§6 items 5–8), kept for the audit trail rather than deleted. |

---

## 11. Compliance/licensing summary

- Only model used: LightGBM (MIT license, far under the 8B-parameter limit — it's a gradient-boosted tree ensemble, not a parameter-counted neural net in the usual sense, but included for completeness). No pretrained embeddings, no transformer, no external API calls of any kind at any point in this pipeline.
- Every fitted artifact (normalization resource tables, blocking rarity weights, the P-key document-frequency table, TF-IDF vocabularies tested in the n-gram experiment, the matcher itself) was fit on train only, confirmed for each one specifically (not just asserted) — the P-key bug in §3 is the one place this was *caught failing* and fixed, which is itself useful evidence of active rather than assumed compliance.
- The T21 ranker (§4) is disclosed here explicitly as changing what counts as "blocking" vs "matching," per the reasoning given in its own section — this kind of disclosure is exactly the category of thing the compliance flag in §8.1 is asking the teammate's pipeline to also provide.
