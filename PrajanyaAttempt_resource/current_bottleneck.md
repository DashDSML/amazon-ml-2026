# Current bottleneck (2026-09-27 ~09:30 UTC)

Model: **submission 7** = matcher v4r (v3a features + decoy + IDF/Soft-TFIDF, trained on 100% of non-held S1 at
test-like density) on the ranker shortlist shrunk at threshold 0.005 (7.21 candidates/S1), expected-F0.5 rule
(bias 1.3) + exclusivity.
- **Leaderboard 0.978** (best).
- Numbers below are measured today on the 220,140 held-out S1 (`runs/c1_lossdecomp.py`).

## 1. How much is left, and where

| Level | h0 (normal held-out) | h210 (test-like) | Test (leaderboard) |
|---|---|---|---|
| F0.5 | 0.9833 | 0.9819 | **0.978** |
| Loss (1 − F0.5) | 0.0167 | 0.0181 | **0.022** |
| Shortlist ceiling (oracle) | 0.9967 | 0.9966 | – |

**Loss breakdown**, as share of loss (F0.5 points in brackets):

| Error type | h0 | h210 | What it is |
|---|---|---|---|
| **In-list miss** (true match in the shortlist, not selected) | **0.391** (0.0065) | **0.348** (0.0063) | The decision stage is too cautious on weaker true records |
| **False positives only** | 0.194 (0.0032) | 0.220 (0.0040) | Lookalikes accepted; grows under test-like density |
| **Shortlist miss** (retention: true match not in the 7.2 candidates) | 0.147 (0.0025) | 0.142 (0.0026) | Blocking/ranker ceiling (oracle 0.9967) |
| **Has matches, predicted empty** | 0.128 (0.0021) | 0.113 (0.0020) | Whole entity missed (costs 1.0 each) |
| **Singleton given a match** | 0.096 (0.0016) | 0.128 (0.0023) | False merge on a no-match entity (costs 1.0 each); grows under test-like density |
| FP + misses | 0.044 | 0.049 | Mixed |

**Pair counts (h0):** TP 731,028; **FP 4,210**; in-list FN 20,746; shortlist FN 8,793.
- The model is **very precise**: about 7× more missed true pairs than false pairs.
- Singletons: 5.7% of S1, correctly left empty 97.2% (h0) / 95.9% (h210).
- US and India are equal (0.9832 / 0.9834 on h0).

## 2. Held-out → leaderboard gap
| Model | Held-out | Leaderboard | Gap |
|---|---|---|---|
| v1 | 0.968 | 0.955 | −0.013 |
| v3a | 0.9788 | 0.971 | −0.008 |
| **v4r + shrink** | **0.9833** | **0.978** | **−0.005** |

Remaining causes of the gap:
1. **Same-address lookalike shift.** The test set has ~23% more pool records per S1, mostly lookalikes at the
   same or nearby address. The FP and singleton shares grow from h0 to h210, and test is harsher still.
2. **France** (15% of test, unseen). The leave-one-country-out proxy loses ~0.025–0.03 for an unseen country.
   France's predicted list-size histogram matches US/India, but its accuracy can't be measured.

## 3. What the target implies
- 0.988 on the leaderboard means a test loss of 0.012, i.e. removing **~45% of the current test loss**.
- With the gap at −0.005, that needs held-out ≈ 0.993, i.e. removing ~60% of the held-out loss. For example:
  all in-list misses (0.0065) **plus** most false positives (0.0032).
- No single measured lever is near that. Stage 2 removed ~10% of the loss; v4's features ~20%.

## 4. What has been ruled out as a bottleneck
- **Blocking recall:** pair recall 0.993; oracle 0.998 at 50 candidates, 0.9967 at 7.2.
- **Decision formula:** the exact expected-F rule differs by ≤ 0.0002; a learned singleton gate gave +0.0001.
- **Exclusivity:** per-record argmax is optimal for this utility.
- **Re-weighting and self-training:** add no information (phase D ±0.0001; self-training −0.001 to −0.003 on the
  France proxy).
- **Model capacity/learner:** two independent lineages correlate 0.996; CatBoost/averaging adds nothing.

## 5. Where the code lives
Everything is in `code/business_entity_resolution/src/`:

| Stage | Files |
|---|---|
| Blocking | `common.py`, `learn_artifacts.py`, `block.py` |
| Pruning ranker | `prune_ranker.py` |
| Matcher | `matcher.py` |
| Features | `decoy.py`, `idf_feats.py` |
| Stage 2 | `c1_stage2.py` |
| Decision | `decision.py` |
| Shrink and submission | `c1_shrink.py`, `make_submission.py` |
| Validation | `testlike_eval.py`, `c1_confidence.py` |
| Experiments | `c1_hurdle.py`, `c1_selftrain.py`, `c1_decoyweight.py`, `c1_bag.py`, `exp_loco_matcher.py`, `adversarial_check.py` |
