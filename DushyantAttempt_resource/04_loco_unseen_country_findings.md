# Leave-one-country-out: quantifying the France risk (cross-validates your D47)

Train only has US and India; France is fully unseen. The only way to rehearse "how will a
country-blind pipeline behave on a country it's never seen" without touching test is to hold
out one of the two countries we *do* have labels for entirely, train on the other, and score
the held-out one as if it were France.

## Method
Retrain a fresh LightGBM holding out one country's rows entirely (country is never a
model feature, only used to decide which rows are held out), then tune+evaluate the
record-centric decision rule on the held-out country alone, against real macro F0.5.

Two variants, both real full-scale (12.8M-row v1 features, both retrains to convergence):
- **Ceiling**: threshold tuned *on the held-out country itself* -- optimistic, not realistic,
  since France has no labels to tune on.
- **Blind (realistic)**: F0.5 at a *fixed* threshold (0.65, the value a mixed-country fit
  would pick) never tuned on the held-out country -- this is what actually happens to France.

## Results
| Held out | Normal (in-domain) F0.5 | Ceiling (optimistic) | Blind (realistic) |
|---|---|---|---|
| India | 0.9541 | 0.9129 (-4.1 pts) | **0.9042 (-4.99 pts)** |
| US | 0.9541 | 0.9453 (-0.9 pts) | **0.9453 (-0.88 pts)** |

Two things worth noting:
1. **Highly asymmetric.** A model that's never seen India-style noise generalizes much worse
   than one that's never seen US-style noise. If French records resemble India's complexity
   (transliteration-adjacent orthographic variation) more than US's simpler generation
   pattern, France could plausibly land nearer the worse end of this range.
2. **The gap barely closes even with the optimal (ceiling) threshold** -- 0.9129 vs. 0.9042
   blind is a small difference relative to the 4-point drop from in-domain. This means it's a
   genuine feature/model generalization limit, not primarily a threshold-calibration problem
   -- consistent with our four negative decision-layer results (`03_...md`): tuning the
   decision layer harder is unlikely to close this gap either.

## Cross-validation with your D47
Your own leave-India-out matcher experiment found India F0.5 0.9432 vs 0.9746 in-domain
(-3.14 points), and your D65 independently estimates "France... likely ≥ 0.03" (3 points).
Our number (-4.99 for India-unseen, blind/realistic) is somewhat larger but the same order
of magnitude and same direction. Two independently-built pipelines landing on a 3-5 point
unseen-country F0.5 penalty is a reasonably strong, convergent signal that this is real and
not an artifact of either team's specific feature set.

## What we didn't get to
The recommended next step -- not attempted, flagged as the most promising *untried* lever for
this specific gap -- is NonStandard.md's T19: apply noise-operator augmentation (random
accent insertion/removal, apostrophe elision, hyphen<->space, case changes, abbreviation of
long street-type words) to **training** records only, to build in the invariances France's
orthography needs without ever using a French row. This is different from, and could combine
with, your own test-like density training (`--hide-permille`) -- that technique addresses the
*density/competition* shift; T19 would address the *orthographic* shift specific to an unseen
country's writing conventions.
