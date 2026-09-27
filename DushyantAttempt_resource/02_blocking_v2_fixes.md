# Blocking v2: three fixes, oracle 0.9778 -> 0.9893 (real, full-scale train)

Your blocking already has a higher oracle than ours (0.997-0.998 vs our 0.9893) using
TF-IDF channels + learned partitions + reverse retrieval + channel-floor caps, so treat this
as "what we tried, in case any single piece is new to you" rather than "you should switch."
The one part that's architecturally different from anything in your pipeline is fix A3
(rank by summed rarity-weighted key evidence rather than a channel/count-based score) --
possibly worth a cheap comparison on your India partition specifically, since that's where
your own lists are hitting the p90=60 cap.

## Fix A3: rank retrieval candidates by rarity-weighted key evidence, not raw count

**Problem:** the original blocking ranked candidates by raw shared-key count, then cut to a
fixed cap. True pairs sharing a rare, distinctive key alongside common candidates were being
capped out before scoring ever saw them (bucket "A3 hits-cap": 3.10% of train misses).

**Fix:** rank by `Σ per-kind-multiplier / log2(block_size + 1)` over shared keys instead.
`block_size` is the posting-list size in the index being queried -- an inference-time
property allowed even on test under G1 (this is the same category of thing as your own
per-query block/candidate-count statistics, not a fitted artefact). The per-kind multipliers
(exact-name-match=3.0, name+house-number=3.0, name+city=2.0, house+addr-token=2.0, single
name token=1.0, 4-gram=1.0) are fixed constants reasoned from each key kind's specificity,
not fit from any row's content. Also raised the pre-scoring cap 30->80 to give the reranking
room to work.

**Result (fastloop, 20k S1 sample):** oracle 0.9778 -> 0.9849, pair recall 0.9372 -> 0.9526.

## Fix S1/S2: fold the same rarity-weighted evidence into the stage-1 score

**Problem:** true pairs that win on rare shared keys but score weakly on plain string
similarity (rapidfuzz name/address ratio) were still losing the top-5-by-score cut and the
floor/alpha/m prune.

**Fix:** add a small saturating boost, `W_KEY * (1 - exp(-weight/KEY_TAU))`, to the existing
rapidfuzz-based stage-1 score, where `weight` is fix A3's same per-edge evidence. `W_KEY=0.15`,
`KEY_TAU=3.0` are fixed constants (weight scale reasoned, not fit).

**Result:** oracle 0.9849 -> 0.9863.

## Fix A1+B: two composite key kinds

- **P key**: a record's two *rarest* name tokens (by document frequency), as an unordered
  pair. Splits oversized single-token/4-gram blocks a common name word creates. Rarity here
  had to be a **single global table fit on TRAIN S1+S2+S3 only**, cached once and reused
  unchanged for every run (train or test, index side or query side) -- an earlier per-batch
  version was inconsistent (an S1 record and its true match, indexed/queried in
  differently-sized batches, could disagree on which two tokens were "rarest" and silently
  never share the composite key) *and* would have been a G1 violation on a test run (a
  token-frequency table fit from test rows). This is the same category of concern as our
  flag on your IDF table (`00_peer_review...md` §1) -- the fix here was to fit the table
  once, from train only, and never recompute it from whatever split is currently being
  processed.
- **N key**: an address token + city, with **no house-number requirement** (unlike the
  existing house-number+token key) -- so a record with a scrambled or gibberish name is
  still retrievable by address alone.

**Result (fastloop, real full-scale train after all three fixes):** oracle **0.9893**
(India 0.9864, US 0.9912), pair recall 0.9657, mean candidates/S1 **5.40** (down from v1's
5.82 -- smaller *and* better, at the frozen operating point floor=0.5/alpha=0.95/m=3).

## Checked and rejected: `PRE_CAP_V2` 80 -> 50
Oracle 0.9893 -> 0.9883 for essentially zero candidate-list-size benefit (5.37 -> 5.34 mean).
Not worth it; kept at 80. (Analogous to your own D21 channel-floor-cap tuning, just on a
different underlying ranking signal.)
