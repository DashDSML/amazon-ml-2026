# Status (2026-09-26, commit 410f375)

Done:
- Fixed broken `merge_letter_runs` import in `mine_resources.py` (renamed to `merge_letters_expr`).
- Fixed name-prefix mining rule (s1_start_rate check) instead of patching "kim"; verified fix on real S1 record.
- Added `snap_to_s1_form()` + hard-fail invariant check for `states.csv` canonicals; no violations found. `cities.csv` has no variant→canonical column, so it doesn't apply there.
- Re-ran `mine_resources.py`; committed all resource/code changes.
- Moved hand-written findings from `noise_analysis.md` to `noise_findings.md`.

Left (not done this session, per user's stop request):
- ~8 unit tests (Indic script, legal suffixes, state forms, house numbers) — not written.
- Before/after char-trigram TF-IDF "true S1 at rank 1" comparison on a 100k-row sample — not run.

Next: run the full-data normalisation command below, then unit tests + TF-IDF check.
