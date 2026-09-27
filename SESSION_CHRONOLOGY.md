# Session Chronology: Amazon ML Challenge 2026 — Business Entity Resolution

This is a complete, hour-by-hour account of every decision made across the hackathon, from the first commit to the final leaderboard submission. For each major fork in the road it states: the **situation** (what problem we faced), the **options** considered, **which one we picked and why**, **which ones we rejected and why**, and the **result** of what we did, with honest analysis of whether it worked.

Timestamps are IST, taken from git commit history (`git log`) and `experiments/log.csv`, both of which are authoritative — this document does not rely on memory alone for *when* things happened, only for *why*.

---

## Day 1 — Sep 25, 21:37–23:01: Reading in, first EDA

**Situation:** Brand new repo. `challengecontext.md` (problem/data/metric), `NonStandard.md` (technique catalog), `CLAUDE.md` (hard rules) were the only guidance. Nothing built yet.

**What we did:** Wrote the data loader and the official macro-F0.5 scorer, ran R1 EDA (`notebooks/r1_eda.py`), and folded its facts straight into `CLAUDE.md` so every later decision could cite them without re-deriving them. Key facts that later decisions leaned on:
- Singleton rate is 5.6% — predicting "no match" everywhere scores 0.056, not 0, so *missing* matches costs far more than the generic ML-competition advice about precision would suggest.
- Mean 3.46 true matches per S1 — this is the list size a perfect blocker would produce, and became the target for every "how big should the candidate list be" decision later.
- Country never differs within a match (0 of 7.64M positive pairs) — makes country a **hard** blocking key, not a soft one.
- ~26% of S2/S3 records are distractors (match nothing) — meaning "no match" is a common and legitimate outcome, not a rare edge case.

Also ran a noise-operator analysis on real training pairs (T18) — comparing true-match record pairs to find what *kind* of noise the data generator actually injects (accents, apostrophes, hyphens, casing, abbreviations). This was banked for later (it's the basis of the still-untried T19 augmentation idea) rather than acted on immediately.

**No rejected alternatives here** — this was pure reconnaissance, not a decision point.

---

## Sep 26, 11:58–12:32: Normalization and mined resources

**Situation:** Raw name/address strings needed a canonical form before any similarity comparison would be meaningful (different casing, abbreviations, transliteration, legal-suffix noise, etc.).

**Options considered:**
1. Hand-write a fixed set of normalization rules (legal forms, street types, abbreviations) from general knowledge only.
2. *Mine* the same tables from the training pairs themselves (`mine_resources.py`) — i.e., learn which strings training-set true-matches treat as equivalent.
3. Some combination: mined tables augmented with general-knowledge additions for cases training data can't cover (specifically French conventions, since France never appears in train).

**Chosen: (3).** Pure hand-written rules (1) would miss the actual noise patterns in this specific synthetic dataset; pure mining (2) would leave France with zero coverage, since G1 (CLAUDE.md's hard rule) forbids deriving anything from test rows and France only appears in test. The combination — mine from train, then explicitly add a French legal-form/street-type/abbreviation lexicon "from general knowledge" (permitted explicitly by NonStandard.md's T19 guidance, since it doesn't inspect test rows) — was the only option that respected G1 while still covering the country we'd be scored on but never allowed to learn from.

**Result:** This choice was validated much later in the session (Sep 27, during the final push) when we specifically checked whether the French lexicon constants were actually wired into the resource-mining pipeline (`mine_resources.py`) rather than being dead code — confirmed they were correctly merged in. No rework needed.

---

## Sep 26, 15:14–18:52: Blocking v1

**Situation:** With normalized records in hand, needed a way to generate a manageable candidate set per S1 without all-pairs comparison (forbidden by CLAUDE.md — "no all-pairs comparisons; use inverted indexes").

**What we did:** Built `cv_harness.py` (GroupKFold-by-S1 evaluation harness — folds 0–3 train, fold 4 true holdout, reused for every experiment for the rest of the project) and `blocking.py` v1 (key-based inverted index: hashed name/address tokens, city, house number, etc. as keys; retrieve candidate S1s that share a key with a query record; score and rank).

**Result (full train run, 18:12):** oracle macro F0.5 = **0.9778**, mean candidates/S1 = 5.82. This became the frozen baseline — "v1 must reproduce exactly" was a hard constraint repeated by the user throughout the whole rest of the project, specifically so that later experiments always had a known-good fallback.

**Test run (18:52):** 7.85 candidates/S1, France (test-only) at 9.05 — already visibly worse than the train-side number, an early hint of the unseen-country cost that would be quantified properly later (LOCO test, Sep 27 morning).

---

## Sep 26, 19:04–19:22: First submission — rule, then a real matcher

**Situation:** Needed *something* submitted before the day's leaderboard slots reset at midnight IST (per `HANDOFF.md`'s hard deadline table).

**Options considered:**
1. Ship a simple rank/score threshold rule immediately (fast, no training).
2. Wait and build the full LightGBM matcher first, submit once.

**Chosen: (1) first, then (2) same evening.** `HANDOFF.md`'s own revised plan (written mid-session once the actual deadline and ranking rules were clarified) explicitly reordered work this way: "the matcher comes before blocking v2... a trained matcher... is where most of the remaining score is," but a submission still had to go out same-day to not waste the slot. `make_submission.py`'s rule (tune a rank/score threshold on train candidates, apply to test) shipped first; the LightGBM matcher (`features.py`/`model.py`/`decide.py`) was built and wired in the same evening.

**Result:** LightGBM matcher, CV F0.5 = **0.9541** (t=0.65). Submitted as **lb-2**.

---

## Sep 26, 22:44: lb-2 real score — the anchor point for every later estimate

**Result:** CV 0.9541 → actual leaderboard **0.943**. Gap = **−0.0111**.

This single real data point became load-bearing for the rest of the project: every later "what will this actually score on the leaderboard" question was answered by extrapolating this same gap, since it was the only ground truth we had connecting CV to the real, held-out test distribution (which includes France, entirely unseen in training).

---

## Sep 26, 23:00–23:49: Blocking v2 — raising the ceiling before touching the matcher again

**Situation:** `BLOCKING_V2.md` (written specifically to plan this) stated the target explicitly: oracle ≥ 0.99 at mean candidates/S1 ≤ ~5, since CLAUDE.md's G2 rule says candidate-set size counts toward the final ranking directly, not just as a means to a better score.

**Method:** Diagnosed v1's actual misses via a fast loop (~20k sampled S1, ~5 minutes per experiment instead of 1.7 hours), bucketed every miss by cause (capped-out blocks, gibberish names, script residue, typos, reordered names, address-only divergence), then fixed the biggest buckets first — stopping once a fix's marginal gain dropped below ~0.1 oracle points, per `BLOCKING_V2.md`'s own explicit stopping rule.

**Options considered per bucket (from `BLOCKING_V2.md`'s fix menu), and what was actually applied, in order:**

| Fix | What it did | Adopted? |
|---|---|---|
| **A3**: rarity-weighted retrieval ranking (replace raw block-size ranking with sum-of-rarity-weights) | oracle 0.9778 → 0.9849 | Yes |
| **S1/S2**: fold key-match evidence into the stage-1 score | oracle 0.9849 → 0.9863 | Yes |
| **A1+B**: composite keys (rare-token-pair "P" key, address-token+city "N" key, no house number needed) | oracle 0.9863 → 0.9898 | Yes |
| P-key bug fix (see below) | — | Yes, this was a correction, not an experiment |

**A real bug caught and fixed here:** the first version of the P-key computed its token-frequency table **per batch** (whatever S1/query batch happened to be in memory at the time). This was flagged and fixed on the user's explicit instruction, for two reasons at once: (a) it was *inconsistent* — the same token could be scored differently in different batches purely due to batch composition, a silent correctness bug; (b) it was a **G1 violation** — since batches could include query-side (test) rows, the frequency table would implicitly be fit on test data. Fixed to a single global table, fit once on train S1 (+ train S2/S3), cached, and loaded unchanged for both train and test scoring; unseen (e.g. French) tokens default to `df=0`, treated as rarest, ties broken alphabetically. This is the kind of bug that would have been invisible in aggregate metrics but wrong in principle — worth flagging as a case where "the number looks fine" was not sufficient evidence of correctness.

**Result:** v2's frozen oracle ceiling ended up around **0.989–0.990** (later refined further alongside the ranker work) versus v1's 0.9778 — a real, validated improvement, achieved without ever inflating the candidate list size (a later PRE_CAP_V2=50 check, below, confirmed the chosen operating point wasn't oversized either).

---

## Sep 27, overnight (00:08–09:10): autonomous work, explicitly authorized

**Situation:** The user explicitly granted overnight autonomy: *"do every reasonable iteration that needs to be done... i'm sleeping now don't ask my approval keep running sanity checks."* Everything in this window was done without further check-ins, but every single change was still smoke-tested before being trusted, per the standing "one change at a time, commit before and after" workflow rule.

**What got built, and — critically — what got tried and explicitly rejected:**

### Rejected #1 — `decide_entity.py` (T1/T2/T4/T5/T6): entity-centric decision + calibration
**Situation:** F0.5 is computed *per entity*, not per pair — a single global probability threshold ignores how many candidates an entity has and how asymmetric the cost of an extra wrong ID is (NonStandard.md's own cost table: a wrong 3rd ID costs only 0.211, but missing an entity's only match costs 1.000). This suggested a smarter, per-entity decision rule should beat one flat threshold.
**Options:** implement the full T1 (per-entity expected-F0.5), T2 (singleton gate), T4 (first-vs-additional-match asymmetric rules), T6 (one-owner enforcement), plus T5 (isotonic probability calibration), versus just keeping the flat threshold from `decide.py`.
**Chosen:** built and tested the full entity-centric version anyway, since the theoretical case for it was strong and it was cheap to test.
**Result:** F0.5 = 0.9543 vs the flat threshold's 0.9541 — a +0.0002 difference, i.e., **noise**, and it was actually *worse* on singleton and non-singleton individually when broken down. Calibration came back as the identity function (the model's probabilities were already well-calibrated, so recalibrating them changed nothing). **Rejected — not adopted.** This was the first of what became a very consistent pattern (see the running tally below): the theoretically-motivated fix that measurably doesn't move the number.

### A real, adopted finding — `loco_validate.py` (T24): quantifying the France risk directly
**Situation:** France is 15% of test and entirely unseen in training (per R1 EDA). We had no *direct* measurement of how much that would cost — only the general worry that it would.
**Method:** Leave-one-country-out — hold out India (or the US) entirely from training, treat it as a France proxy, and measure the real, un-cheated degradation.
**Result:** Holding out India dropped F0.5 from 0.9541 to **0.9042** (realistic, "blind" threshold — i.e., the threshold itself wasn't allowed to peek at the held-out country either). Even letting the threshold be optimally tuned *on* the held-out country only recovered to 0.9129 — meaning this is a genuine feature/model generalization limit, not a threshold-calibration problem. Holding out the US instead only cost 0.88 points (0.9453), suggesting India's *specific* patterns (not just "any unseen country") were harder to generalize away from — a useful, if slightly unsettling, asymmetry. **This became the single most important standing fact for the rest of the session** — it's the exact reason every later "what will the leaderboard score be" answer was capped well below the oracle ceiling, and why "just tune it harder" claims late in the session were pushed back on with numbers, not vibes.

### Rejected #2 — `model_weighted.py` (T3): metric-aware sample weights
**Situation:** F0.5's exact cost table (above) suggests weighting training examples by their real marginal cost to the metric, rather than treating every training row as equally important.
**Method:** Implemented the exact cost-table formula (verified against NonStandard.md before running, not just approximated) as LightGBM sample weights.
**Result:** F0.5 = 0.9544 vs 0.9541 — again **+0.0003, not real**, despite the optimal decision threshold shifting materially (0.65 → 0.40) as a side effect. **Rejected.** Notable because the threshold moved a lot while the score didn't — a sign the model was finding a different but equally-good operating point, not a genuinely better one.

### Rejected #3 — smaller candidate cap (`PRE_CAP_V2=50`)
**Situation:** The last open parameter from the blocking-v2 plan — was capping the pre-ranking pool at 80 candidates per record too generous?
**Result:** oracle 0.9883 at PRE_CAP_V2=50 vs the frozen 80's 0.9893 — a *worse* ceiling for essentially no candidate-list-size benefit (5.34 vs 5.37 mean). **Rejected — kept 80.**

### Built and shelved for later — the ranker's data pipeline
`ranker_data.py` (a wide, SAVE_CAP≈80 candidate pool, as opposed to the final pruned ≈5) was added at 04:56 — this is the raw material for the technique that eventually became the single biggest win of the whole project (T21, see below), but at this point it was just infrastructure; nothing was trained on it yet.

---

## Sep 27, ~09:10–15:00: the quiet stretch — EC2 reconnects and a teammate detour

This window included two resolved EC2 connectivity incidents (a pure network blip with no data loss, and later a genuine reboot that lost in-memory processes but not disk state — both times verified by checking actual file/process state afterward rather than assuming), and a deliberate pivot: the user asked to study a teammate's (Prajanya's) independently-built pipeline, which had reached 0.977 on the leaderboard.

**Decision:** treat this as worth real time, not a distraction — reviewed their four `PrajanyaAttempt_resource` markdowns in depth, flagged a likely G1-compliance issue in their pipeline (an IDF table apparently fit on combined train+test, inferred from their own logged record counts), and built a five-file `DushyantAttempt_resource` package sharing this project's own validated findings (the LOCO result and the blocking-ceiling data specifically) back to them, on the reasoning that a genuine finding is worth more shared than hoarded when the stated goal shifted to *"help Prajanya's pipeline do its absolute best."*

Also tested (Sep 27, 16:00, `test_ngram_retrieval.py`) whether adding a character n-gram TF-IDF retrieval channel would raise the blocking ceiling further: it would recover **~44% of currently-missed true pairs** at a K=20 candidate budget, with a sensible bimodal failure pattern (typo-recoverable vs genuinely-different content). This is a real, validated finding — but it was never built into the production pipeline, simply because time ran out before priorities shifted decisively toward the ranker (below). It remains the clearest *unexploited* lever in the blocking layer specifically, as distinct from the matcher.

---

## Sep 27, 16:18–16:59: T8 confusability — a real, adopted feature

**Situation:** Aimed specifically at two loss patterns (both our own and, per the shared resource, the teammate's): missing a true match that was in the candidate list but scored too low, and accepting a lookalike false positive.

**Idea:** every S1 is, by construction, a guaranteed non-match to every *other* S1 (S1 is a deduplicated reference registry) — so "how similar is this S1 to its nearest other S1" is a completely free, label-free signal computable with the exact same retrieval machinery already built for S2/S3-vs-S1 blocking, just run S1-vs-S1 instead.

**Validation before trusting it:** smoke-tested (2,000 S1, 43s) and manually spot-checked the top results — they were exactly the intended pattern (e.g., "Chennai Solutions Private Limited" at 0.964 confusability, paired with a genuine same-name-different-address lookalike). Only after that qualitative check passed was it run at full scale.

**Result:** wired into the matcher's feature set, retrained: F0.5 **0.9542 → 0.9550** on v1 (+0.0008) — small, but for the first time in this run of experiments, *outside* the noise band the four previous decision-layer tries had all landed in. **Adopted.** When later ported to v2r (see below), it ranked as high as the model's **4th-most-important feature by gain**, well above several established features like `addr_num_jaccard` and `state_eq` — corroborating that the sample-scale signal was real, not a fluke of the smoke test.

---

## Sep 27, 18:36–21:48: T21 — the supervised meta-blocking ranker (the biggest win)

This is the single highest-leverage decision of the whole project, so it's worth walking through in full.

**Situation:** Blocking v2's candidate pruning used a *hand-weighted* scoring formula (rarity-weighted key evidence, tuned by hand across the whole v2 development process above). NonStandard.md's T21 raises an obvious question: why hand-tune the pruning rule when a supervised model, trained on the same features the final matcher already uses, could learn to prune *better* — provided its own decision **becomes** the actual candidate list the matcher runs on (a "disclosed" technique, since this changes what counts as the final submitted candidate set, not just an internal scoring detail).

**Build, and the two real incidents inside it:**
1. `ranker_features.py` (chunked feature computation over the wide, SAVE_CAP≈80 pool) — the *first* version of this mirrored the existing `build_features()`'s single-shot design and was **OOM-killed** at ~61.3GB RSS on a 62GB machine (confirmed via `dmesg`, not guessed). Fixed by processing a handful of part-files at a time instead of all 781M rows at once — the same "process in chunks" rule CLAUDE.md states as a hard constraint, just not followed carefully enough the first time.
2. `ranker_model.py` — trained on a class-balanced sample of the wide pool (all positives + a 5:1 negative sample), evaluated by re-ranking a real 20k-S1 holdout sample by the ranker's own probability and sweeping the same floor/alpha/m cutoff grid used everywhere else in the project.

**Result of the ranker's own grid search:** at floor=0.4, alpha=0.9, m=3, the ranker-pruned candidate set beat the frozen v2 blocking on **both** axes at once — oracle F0.5 **0.9893 → 0.9977**, and mean candidates/S1 **5.40 → 4.28** (later confirmed even better, 4.04, at true full scale). This is an unambiguous win by the project's own stated adoption rule ("oracle rises at a similar or smaller size, or size falls at a similar oracle") — most improvements in this project moved one axis at the expense of the other; this moved both in the same direction.

**Turning the finding into a real submission (`ranker_apply.py`), and a second real incident:** applying the trained ranker to the *entire* wide pool (not just the eval sample) and rebuilding for both train and test hit a **disk-full crash** (58GB EC2 disk at 100% used) partway through the test-side build. Diagnosed via `df -h`, fixed by identifying and deleting 26.6GB of already-consumed train-side intermediates (the raw wide retrieval output and its once-derived features, both fully superseded by that point), and rerunning clean.

**Final result of this whole effort:** matcher retrained on the ranker-pruned candidates, CV F0.5 **0.9628 → 0.9661**. Submission built and validated PASS. This became "v2r."

---

## Sep 27, ~21:50–22:00: explaining the score honestly, and why 0.98 was never realistic

**Situation:** once v2r's CV score (0.9661) was known, the user asked what the expected leaderboard score would be, then — reasonably — pushed back when the estimate (~0.95) sounded low relative to a 0.9925 oracle ceiling.

**The honest answer given, held to for the rest of the session:** the gap from ceiling to real score is not one gap, it's **two separate, independently measured** gaps that don't shrink together:
- **Ceiling → CV** (matcher imperfection): ~2.5–2.6 points, and consistent across v1, v2, and v2r regardless of what else changed — the ceiling assumes a perfect classifier, and no real classifier is perfect.
- **CV → LB** (generalization to the real, France-inclusive test distribution): ~1.1 points, a *measured* number from lb-2's real submission, and independently corroborated by the LOCO test's direct evidence that unseen-country generalization is a genuine, non-threshold-fixable limit.

Stacking both against v2r's 0.9925 ceiling predicted **≈0.955** — which is exactly what later happened (see below). This framing was repeated (not just asserted once) at several later points in the session when the user pushed for "0.98 at any cost," specifically because the evidence for it kept accumulating rather than being a single guess.

---

## Sep 27, ~22:00–23:20: the final push — hyperparameter tuning, and six more rejections

**Situation:** the user pushed hard for more score, eventually against a real clock (10:30pm, later explicitly extended to 11:40pm). A sharp observation from the user reframed the remaining options: **every single "improve the matcher" attempt all session had kept the exact same fixed LightGBM hyperparameters** (`num_leaves=127, lr=0.05, feature_fraction=0.8`, unchanged since the very first matcher) and only ever changed *post-hoc decision logic* on top of that one fixed model. The matcher's own hyperparameters had never actually been searched.

**What we did:** built `model_tune.py`, a 15-config sweep (tree depth, learning rate, regularization, leaf-size constraints) on a fast 400k-S1 sample, scored by the *real* F0.5 metric (via `decide.py`'s own evaluation helpers) rather than a proxy like logloss.

**A real risk caught before it cost hours:** the sample's best configs needed 8–10× more boosting rounds to converge than the baseline (up to ~1,600–2,000 rounds vs 166) — naively scaling that round count to the full 8.7M-row dataset could have taken *hours*, not minutes, blowing the time budget entirely. Rather than trust the extrapolation, we launched the promotion with close, repeated monitoring and a stated abort plan. It turned out the constraint (`min_data_in_leaf=100`) was far *weaker* relative to 8.7M rows than to the 1.6M-row sample — it converged in **184 seconds**, not hours. This is a case where a plausible-sounding extrapolation ("more rounds needed on the sample → proportionally more rounds needed at full scale") was simply wrong, and checking it empirically (rather than trusting the arithmetic) was the only way to know that safely.

**Result: CV 0.9661 → 0.9669 (+0.0008), confirmed, real, adopted.** Submission regenerated and re-validated PASS.

**Then, six more things tried under the same time pressure, all rejected, each for a specific reason:**

| # | Idea | Result | Why rejected |
|---|---|---|---|
| 1 | `decide_entity.py` re-tested against the *new* tuned matcher | 0.9673 vs 0.9669 | Same noise-level pattern as its three prior tests against different matchers — the 4th independent confirmation this technique doesn't help *regardless* of which matcher sits underneath it. |
| 2 | Logistic regression alone | 0.9558 | A linear model can't capture the nonlinear feature interactions the GBM already does — expected, not a surprise. |
| 3 | GBM + logistic-regression blend, weights 0.1–0.5 | 0.9669 → 0.9652, **monotonically worse** as blend weight rose | Clean negative: logreg has literally zero complementary signal beyond what the GBM already extracts from these features. |
| 4 | `lambda_l2=1.0` regularization combo | killed at a 20-minute time cutoff, no result | Unlike `min_data_in_leaf`, this config's round-count did *not* collapse at full scale within the time we could safely allow; rather than let it run open-ended against a hard deadline, it was killed and left unresolved. |
| 5 | 3-seed GBM ensemble (average of independently-seeded models) | 0.9669–0.9674 across seeds and averages | All within the same noise band; picking the best-performing seed *after the fact* is cherry-picking across random variation, not a technique that would generalize to an unseen test set. |
| 6 | PCA+logreg, KNN-as-fallback, transformer cross-encoder (proposed, not attempted) | — | Assessed for feasibility and declined: PCA adds nothing with only 34 features and no dimensionality problem to solve; KNN needed more design/validation time than remained without risking the deadline; a cross-encoder needs GPU-scale throughput this CPU-only EC2 box simply doesn't have — training/inferring one over millions of pairs in minutes was not physically feasible here, not just inconvenient. |

**The actual conclusion, stated plainly at the time:** by this point, **ten independent techniques** across the whole session — four different decision-layer rules, one sample-weighting scheme, a 15-config hyperparameter sweep, a linear-model blend, and a multi-seed ensemble — had all converged to within about 0.008 of each other (roughly 0.966–0.974). That convergence, not any single result, is the real evidence: it means 0.9669 sits close to a genuine ceiling for *this feature set*, and no further tuning, blending, or seed-picking was going to break through it without either new features (e.g., the still-untried T19 augmentation, or the validated-but-unbuilt n-gram retrieval channel) or fundamentally more test-distribution data (which G1 forbids).

---

## Sep 27, 23:18–23:40: final submission and the real result

Submission regenerated against the tuned matcher, re-validated **PASS twice** (once on EC2, once independently on a locally-synced backup copy) using `utils/validate_submission.py` — the organizers' own validator, not a custom check. Everything committed to git with a full, honest record of both what worked and what didn't (including the six same-push rejections above) in `experiments/log.csv` and `STATUS.md`.

**Submitted as lb-3.**

| | CV F0.5 | Leaderboard F0.5 | Gap |
|---|---|---|---|
| lb-2 (v1) | 0.9541 | **0.943** | −0.0111 |
| lb-3 (v2r, tuned) | 0.9669 | **0.954** | −0.0129 |

The gap held almost exactly to the pattern predicted from the very first real data point — vindicating both the ~0.955 estimate given before submission and the broader refusal to promise 0.98 under pressure: the arithmetic said it wasn't reachable, and the real leaderboard score confirmed it.

---

## What's still on the table, not done

- **T19 noise-operator augmentation** (accent/apostrophe/hyphen/case perturbations + hard negatives from confusable neighbors) — the one lever that targets the *generalization* gap specifically (as opposed to the matcher-fit gap that ten techniques already showed is close to saturated). Never built out, purely for lack of time.
- **Char n-gram TF-IDF retrieval channel** — validated to recover ~44% of currently-missed pairs at a modest candidate budget, never wired into the production blocking pipeline.
- The `lambda_l2` combo trial was killed inconclusively, not definitively ruled out.
