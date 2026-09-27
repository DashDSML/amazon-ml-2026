# Current strategy (as of 2026-09-27 01:00 UTC)

Every component in use, the problem it solves, the evidence, and what was tried and dropped.
Leaderboard best: **0.971** (submission 3). Best validated model: **stage 2 on v4** (submission 5, held-out
0.9846, test-like 0.9835; leaderboard pending).

Related files:
- `code/business_entity_resolution/PIPELINE.md`: stage details and commands.
- `code/business_entity_resolution/LOGIC.md`: why v3a worked.
- `code/business_entity_resolution/project_state.md`: decision log D01–D56.
- `output/SUBMISSIONS.md`: every submission with md5 and score.
- `strategy.md`: the original hackathon plan, unchanged.

---

## 1. Constraints that shape everything

| Constraint | Consequence |
|---|---|
| F0.5 per S1, macro-averaged; precision counts double; singletons score 1 only if nothing is predicted | Predict several matches per S1, don't guess on uncertain ones, allow an explicit "no match" |
| 2.2M S1 × 10.3M pool records (train); 1.7M × 10M (test) | Blocking is required |
| `candidate_pairs.tsv` must be the exact set the model scores; smaller sets rank higher | A short learned shortlist (8.5/S1) feeds the matcher. No post-hoc pruning |
| France only in test; no external data; MIT/Apache models | Country-agnostic, learned from the provided data |
| **The test set differs from training**: ~23% more pool records per S1, mostly lookalikes at the same or nearby addresses | Features must keep their meaning under that shift; validation must imitate it |

## 2. The pipeline in use (submission 5)

```
blocking (7 TF-IDF channels) → pruning ranker (8.5 cand/S1) → stage 1: matcher v4 → stage 2: list-level
matcher → expected-F0.5 decision rule + exclusivity → matching_results.tsv
```

| # | Component (file) | Problem it solves | Evidence |
|---|---|---|---|
| 1 | Normalisation: learned Indic→Latin dictionary, `anyascii`, compact / skeleton / address views (`common.py`, `learn_artifacts.py`) | Indian scripts, typos, domains, accents | 1,347-token dictionary; −42% blocking misses (Kerala) |
| 2 | Learned region aliases + per-country partitions (`block.py`) | Scale; "TN" = "Tennessee"; France = one whole partition | Pair recall 0.9926 |
| 3 | 7 TF-IDF retrieval channels | Name-only or address-only search each miss different pairs | Oracle F0.5 0.998 at ~50 candidates |
| 4 | LightGBM pruning ranker (`prune_ranker.py`) | Candidate-size criterion; matcher cost | 8.5 candidates at an oracle of 0.997 |
| 5 | String features (`matcher.py`) | Blocking scores can't tell a match from a lookalike | 0.867 → 0.968 held-out |
| 6 | **Decoy features** (`decoy.py`): house-number relation, number difference, legal form, extra words, glyph repair, alias, anchor-vs-S1 number | Generator lookalikes (small number differences, inserted words) | Leaderboard 0.955 → **0.971** |
| 7 | **Test-like density training** (`--hide-permille 210`) | Lookalike records with no competing S1 | Test-like 0.963 → 0.977 |
| 8 | No sibling or count features | Counts in lists change meaning on test (v2: 0.886) | Adversarial check AUC 0.79 on count features |
| 9 | **IDF + Soft-TFIDF features** (`idf_feats.py`), fixed train + test corpus | 178K names shared across entities; generic vs rare words; typos in rare words | v4: held-out 0.9829 (+0.0041), test-like 0.9815 (+0.0044) |
| 10 | **50% of training S1** (vs 20%) | More examples of rare patterns | part of the v4 gain |
| 11 | **Stage 2: list-level matcher** (`c1_stage2.py`) | Pairwise scoring ignores that an S1's true matches agree with each other and that a lookalike business forms its own group | On v4: held-out **0.9846** (+0.0017), test-like **0.9835** (+0.0020) |
| 12 | Expected-F0.5 decision rule + exclusivity (`decision.py`) | Per-S1 choice of how many matches; explicit empty option; each pool record goes to one S1 | Beat every fixed threshold |
| 13 | Validation + upload gate (`testlike_eval.py`) | Plain held-out data can't see the test shift | Gate: better on held-out and test-like; test predictions per S1 ≈ 3.3–3.4; validator passes |

**Stage 2 in one paragraph.** Stage 1 (v4) scores each (S1, candidate) pair independently, giving m1.
Stage 2 adds, for each candidate:
- its rank and gap in m1;
- name and house-number agreement with the S1's top candidate;
- its best name similarity with other confident candidates;
- m1-weighted agreement;
- the best m1 inside its same-number group.

A second LightGBM, trained on **out-of-fold** m1 (honest stage-1 mistakes), re-scores the pair. True matches
get pulled up when they agree with the confident top record; a tight group at a different house number gets
pushed down. The "safe" variant avoids count features.

## 3. Scores

| Version | Held-out | Test-like | Test pred/S1 | Leaderboard |
|---|---|---|---|---|
| v1 | 0.9680 | 0.9630 | 3.36 | 0.955 |
| v2 | 0.9733 | 0.9645 | 3.92 | 0.886 |
| v3a | 0.9788 | 0.9771 | 3.34 | **0.971** |
| stage 2 on v3a (sub 4) | 0.9810 | 0.9795 | 3.37 | pending |
| v4 alone (sub 5a, backup) | 0.9829 | 0.9815 | 3.34 | not uploaded |
| **stage 2 on v4 (sub 5)** | **0.9846** | **0.9835** | 3.37 | pending (expected ~0.976–0.978) |

Held-out → leaderboard gap so far: v1 −0.013, v3a −0.008. The remaining gap is attributed to the
same-address lookalike shift and France (unseen country; the leave-one-country-out proxy loses ~0.025–0.03).

## 4. Tried and dropped

| Idea | Result / reason |
|---|---|
| v2 sibling count features | Leaderboard 0.886: counts inside lists shift on test |
| Within-country rank normalisation | France proxy worse (0.935 vs 0.943) |
| Dropping count features only | France proxy worse (0.916) |
| Random hiding as the only validation | Doesn't reproduce the same-address shift; ranked v2 above v1 |
| Indic vocabulary extension | Unseen test Indic words are decoy markers (inserted shop words) that the matcher already rejects (p≈0) |
| Stage 2 "full" (with counts) | +0.0003 only; count features are the v2 failure mode |
| **Phase D: decoy-weighted stage 1** (1.8× / 3.0× on 47% of negatives) | 0.9828/0.9816 and 0.9826/0.9814 vs 0.9829/0.9815: no gain. Re-weighting adds no information |
| MinHash/FAISS/phonetic blocking, SPLADE | Blocking loss is only ~0.003; costly rebuild |
| CatBoost switch | Features almost all numeric; country as a categorical is against the rules; models correlate 0.996 |
| Correlation clustering | Needs pool–pool probabilities; stage 2 covers the safe part; sibling logic backfired on test |
| Conformal / per-country thresholds | Needs labelled calibration data per domain; France has none |
| Post-matching candidate pruning | Rules require the file to be the exact set the model scores |
| Cross-encoder / embeddings | No GPU; low expected gain on rule-generated decoys |

## 5. Open options (deadline Sep 27 18:29 UTC; packaging from ~12:30)
1. **Final retrain on ~90% of S1** for v4 and stage 2 → submission 6. Reliable +0.0005–0.0015; ~4–5 h.
2. **Cross-entity joint assignment** (S1s competing for the same pool record decided jointly) → +0.0005–0.0015,
   uncertain; ~3–4 h.
3. **Packaging:**
   - README with commands;
   - requirements (lightgbm, rapidfuzz, indic-transliteration);
   - documentation from `PIPELINE.md` / `LOGIC.md` / this file;
   - zip with the best validated `output/`;
   - final validator run.
