# Status (2026-09-27, commit 864ed0d)

**v2r is the new best candidate, DONE and ready (not submitted -- that's your call):** T21 supervised meta-blocking (ranker) replaces v2's hand-weighted candidate pruning. `output/matching_results.tsv` + `candidate_pairs.tsv` on EC2, validator PASS, 5,790,638 matched pairs. CV F0.5 **0.9628 -> 0.9661** (t=0.70), blocking oracle ceiling 0.9893 -> 0.9925. Local backup: `output_v2r/`.

**Pipeline:** `ranker_model.py` (trained on wide SAVE_CAP=80 pool, class-balanced sample) found floor=0.4/alpha=0.9/m=3 beats frozen v2 on both oracle (+0.0084) and size (5.40->4.28 mean/S1). `ranker_apply.py` (new) applies it to the full wide pool, prunes, recomputes features (incl. T8 confusability, new this session: v1 F0.5 0.9542->0.9550, adopted) -> a new `blocking_v2r` tree. Built for both train and test (`ranker_data.py`/`ranker_features.py` generalised to `--split test`).

**Incidents, both resolved:** (1) `ranker_model.py`'s first run OOM'd at 50.9GB doing single-shot `.collect()` over 781M rows -- rewrote to chunk over part-files, added `scripts/run_with_memcap.sh` (RSS-polling guard, kills cleanly with a logged reason instead of a silent kernel OOM-kill) and used it on every EC2 launch since. (2) `ranker_features.py --split test` crashed disk-full (58GB EC2 disk at 100%) at chunk 41/100 -- freed 26.6GB of already-consumed train intermediates, reran clean.

**v2 (`lb-3`, CV 0.9628) remains available as a fallback**, unaffected -- its files were backed up locally (`output_v2_lb3/`) before v2r overwrote EC2's `output/`.

**Not yet acted on:** T19 noise-operator augmentation for the France/unseen-country gap (still the best untried lever for that specific weakness); confirming v2r's LOCO/unseen-country behavior specifically has not been re-checked.

**Everything above is committed to git and mirrored on EC2.** Nothing has been submitted to the leaderboard.
