# Status (2026-09-27, commit ede486f, work continuing autonomously per explicit instruction)

**lb-3 candidate is DONE and ready** (not submitted -- that's your call): v2 blocking + LightGBM matcher, `output/matching_results.tsv` + `candidate_pairs.tsv` on the EC2 instance, validator PASS. t=0.65, CV F0.5=**0.9628** (up from v1's 0.9541), blocking oracle ceiling=0.9895. Public LB estimate (rough, from v1's -0.011 CV->LB gap): **~0.94-0.96**. A >0.987 target is not realistic tonight -- it would need ~99.7% of the oracle ceiling captured end-to-end on a set that includes an unseen country.

**EC2 mid-work incident (resolved):** SSH access was refused for a while (network-level, instance uptime never broke -- no data lost). During the blackout, two independent nohup watchers both fired and launched duplicate `ranker_data.py` processes racing on the same output files. Killed both, wiped the directory, restarted clean (single process, confirmed healthy since).

**Three things tried and NOT adopted, real full-scale validation, documented in git (not smoke samples):**
- `decide_entity.py` (T1/T2/T4/T5/T6 entity-centric decision + isotonic calibration): 0.9543 vs decide.py's 0.9541 -- within noise, worse on both singleton and non-singleton individually. Calibration came back as identity (probabilities already fine).
- `model_weighted.py` (T3, metric-aware sample weights, exact cost-table formula verified before running): 0.9544 vs 0.9541 -- +0.0003, not real, despite the optimal threshold shifting materially (0.65->0.40).
- `PRE_CAP_V2=50` (the last open question from the blocking plan): oracle 0.9883 vs the frozen 80's 0.9893, for essentially zero candidate-size benefit (5.34 vs 5.37 mean). Keeping 80.

**Real, actionable finding (not yet acted on):** `loco_validate.py` (T24, France proxy) -- holding out India entirely drops F0.5 from 0.9541 to **0.9042** (realistic, blind-threshold case); holding out US drops it to 0.9453. Barely closes even with a threshold optimally tuned on the held-out country (0.9129 for India) -- a genuine feature/model generalisation limit, not a threshold problem, so further decision-layer work (see above, all three tries failed) won't fix it. **Recommended next step, not attempted:** T19 noise-operator augmentation on training records (accent/apostrophe/hyphen/case perturbations) to build in invariances France's orthography needs without ever touching a French row.

**In progress:** `ranker_data.py` on EC2 (wide SAVE_CAP=80 candidate pool for train, single process confirmed healthy, ~1-1.5h more expected). Will build+validate the ranker once it lands, adopt only if it beats the current v2 blocking oracle (0.9893) on a real holdout check, and report/log regardless of outcome.

**Documentation gap closed:** `code/business_entity_resolution/README.md` (didn't exist) and `Documentation_template.md` (was blank) now reflect the real pipeline and real results, including the negative findings above, honestly reported.

**Everything above is committed to git and mirrored on the EC2 instance.** Nothing has been submitted to the leaderboard.
