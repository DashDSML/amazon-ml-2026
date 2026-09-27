# Status (2026-09-27, commit ab546e0, work continuing autonomously per explicit instruction)

**lb-3 candidate is DONE and ready** (not submitted -- that's your call): v2 blocking + LightGBM matcher, `output/matching_results.tsv` + `candidate_pairs.tsv` on the EC2 instance, validator PASS. t=0.65, CV F0.5=**0.9628** (up from v1's 0.9541), blocking oracle ceiling=0.9895. Public LB estimate (rough, from v1's -0.011 CV->LB gap): **~0.94-0.96**. A >0.987 target is not realistic tonight -- it would need ~99.7% of the oracle ceiling captured end-to-end on a set that includes an unseen country; see reasoning in conversation.

**EC2 mid-work incident (resolved):** SSH access was refused for a while (network-level, instance uptime never broke -- no data lost). During the blackout, two independent nohup watchers both fired and launched duplicate `ranker_data.py` processes racing on the same output files. Killed both, wiped the directory, restarted clean (single process, confirmed).

**Two things tried and NOT adopted (real, validated, documented in git):**
- `decide_entity.py` (T1/T2/T4/T5/T6 entity-centric decision + isotonic calibration): does not beat decide.py's simple record-centric rule (0.9543 vs 0.9541, worse on both singleton and non-singleton individually). Calibration came back as identity -- probabilities were already fine. Not a bug, just doesn't help here.
- `loco_validate.py` (T24, France proxy): real full-scale result -- holding out India entirely drops F0.5 from 0.9541 to **0.9042** (realistic blind-threshold case); holding out US only drops to 0.9453. This is a genuine, structural generalization gap (barely closes even with the country's own optimal threshold), not a threshold-tuning problem -- more decision-layer work won't fix it. T19 (noise-augmentation on training records) is the documented next step if there's time; not attempted yet given the ranker was judged higher-value for remaining time.

**In progress:** `ranker_data.py` on EC2 (wide SAVE_CAP=80 candidate pool for train, ~2-2.5h, single process confirmed healthy). Will build+validate the ranker once it lands, adopt only if it beats the current v2 blocking oracle (0.9893) on a real holdout check, and report/log regardless.

**Known gaps, not addressed tonight:** README.md and Documentation_template.md are stale (pre-date blocking v2/matcher/ranker); need a pass before the final zip.
