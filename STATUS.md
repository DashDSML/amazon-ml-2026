# Status (2026-09-26, commit 62d6e57)

Done:
- Blocking v1 (S2/S3-side, one-owner top-m) built and tuned. Train: oracle F0.5 0.9778, pair recall 0.9384, 5.82 cands/S1 (floor 0.5, alpha 0.9, m 3). Test: 7.85 cands/S1, 13.6M pairs.
- `make_submission.py`: tunes an entity-level decision rule on train candidates only (rank==0 & score>=t1, sweep t1 0.5-0.95; also tried rank<=1 with a stricter t2, which did not improve). Best: rank==0, t1=0.950, real macro F0.5=0.8388 (US 0.8745, India 0.7852, singleton 0.8583, non-singleton 0.8376, fold range 0.8382-0.8394).
- Applied to test, wrote `output/matching_results.tsv` + `output/candidate_pairs.tsv` (1,732,544 S1 rows each), passed `validate_submission.py` (PASS).

Note (not acted on, flagged only): the t1 sweep kept improving up to the 0.95 boundary — a higher t1 may score even better; out of scope for this task's fixed grid.

Next: decide whether to extend the t1 sweep past 0.95, then consider LB submission (`lb-1`).
