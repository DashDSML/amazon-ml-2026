# Improvement ideas, mapped to the measured bottlenecks

Based on `current_bottleneck.md`: loss 0.0167 (h0) / 0.0181 (h210) / 0.022 (leaderboard 0.978).

**Tags:** (M) measured by us · (E) estimate · (Lit) literature.
Gains are held-out F0.5 unless stated.

## A. In-list misses (35–39% of loss): the biggest bucket

| Idea | Why it should work | Expected gain | Cost | Status |
|---|---|---|---|---|
| **Stage 2 on v4r** (list-level re-scoring with OOF probabilities) | True matches agree with the confident top record; measured +0.0017 on v4 (M) | +0.0015–0.002 | ~3 h | **Running** (`runs/c1_v4rs2` → submission 8). Test transfer unconfirmed (upload sub 5 to check) |
| **Pair-difference n-gram model** as a stacked feature (sparse logistic regression on the tokens/characters that differ between the two records) | Learns which edits are noise (typos, dropped legal forms) and which are decoy edits (inserted words, number changes) directly from labels (Lit: WDC corner-case benchmark) | +0.0005–0.0015 (E) | 2–3 h | Not started. French vocabulary is unseen → gain mostly US/India |
| **Recall-side decision tuning** on test-like data (bias 0.8–1.3, isotonic calibration of p, max list length) | FN pairs outnumber FP pairs ~7:1; the rule may be over-cautious for entities with many true records | +0.0001–0.0005 (E) | 30 min | Bias grid 1.0–2.0 showed ≤ 0.0002 differences (M); isotonic not tried |
| **Larger model** (lr 0.05, 3,500 rounds, 255 leaves) | v4r stopped at 1,762 rounds at lr 0.1; more capacity for rare patterns | +0.0002–0.0006 (E) | 2–3 h | Not started |

## B. False positives (19–22%) and singleton false merges (10–13%): same-address lookalikes

| Idea | Why | Expected gain | Cost | Status |
|---|---|---|---|---|
| **Candidate-vs-candidate address/number conflict features** (continuous, not counts): similarity of this candidate to the S1's *other* confident candidates at the *same* street with a *different* number | A lookalike business shares the street but not the number with the S1's true cluster. Stage 2 has part of this (`top_eq`, `grp_hn_maxp`) | +0.0003–0.001 (E) | 2 h | Partly covered by stage 2 |
| **Adversarial check of stage-2 features** (train-vs-test, per class) | Stage 2 raised test predictions 2× more than held-out on v4 (M); counts broke v2 | risk control | 30 min | Not run |
| Seed bagging | Smooths borderline decisions | +0.0002–0.0008 (Lit) | ~2 h | **Running** at low priority |

## C. Retention: true match not in the shortlist (14–15%)

| Idea | Why | Expected gain | Cost | Status |
|---|---|---|---|---|
| **Ranker retrained at test-like density with v4-style features** (IDF, decoy relations) | The ranker still uses count features that shift on test and has no rarity features; a better ranker keeps more true pairs at the same size | +0.0005–0.001 (E), and possibly a smaller candidate set | 4–5 h (new shortlist → full feature rebuild) | Not started; doesn't fit before the deadline |
| Keep 12 instead of 10 at the same threshold | 8.8K shortlist FN pairs on h0 | +0.0002 (E) | 1.5 h | Conflicts with the candidate-size criterion |

## D. Entities with matches predicted empty (11–13%)

| Idea | Why | Expected gain | Cost | Status |
|---|---|---|---|---|
| Learned list-level gate (hurdle rule) | Dependence between lookalike candidates | +0.0001–0.0002 (M) | done | **Dropped** (too small) |
| "At least one match if the top candidate agrees on name + number" rule | Empties cost 1.0 each | small; risky for singletons | 20 min | Not tried |

## E. Test shift and France

| Idea | Why | Evidence | Status |
|---|---|---|---|
| Self-training on test pseudo-labels | Learn test vocabulary and density | **−0.001 to −0.003** on the France proxy (M) | Dropped |
| Leave-one-country-out-tuned logit offset for France | Principled, label-free | Unmeasured | Not started; small |
| France diagnostics (conflict rate, empty rate, expected-F self-estimate) | Detect a France failure | Size histogram healthy (M) | Partly done |

## F. Ranking criterion (candidate-set size)
- Current: 7.21 candidates/S1 after the 0.005 shrink (−15%, −0.0002 F0.5).
- Next step: 0.01 (6.67/S1) costs −0.0005, at the user's limit. Not recommended.

## Realistic outlook
- Stacking stage 2 + bagging + decision tuning ≈ +0.002–0.003 held-out → leaderboard ≈ **0.980–0.981**.
- 0.988 would need a new source of information that removes most in-list misses **and** most false
  positives. None of the measured or planned levers is that large. The most promising unexplored one is the
  pair-difference n-gram model (A2).
