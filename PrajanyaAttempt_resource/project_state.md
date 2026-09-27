# Project State — decision & iteration log

This log is appended after every decision or iteration. `context.md` is the static description of the pipeline; this file records **when** each piece was decided, **why**, **what was observed**, and **whether it helped**.

Timestamps are UTC from the container clock. For D01–D23 they were backfilled on 2026-09-25 12:23 from the modification times of the files each step produced (±5 min). From D24 onward they are written when the entry is made. Entries with a letter suffix (D02a, …) were added during the backfill so that numbering of earlier entries, which later entries refer to, stays unchanged.

**Update protocol:** after every decision, experiment or code change, (1) append an entry with ID, timestamp, decision, why, setup, observed results and verdict; (2) update "Current status"; (3) tick or add items under "Pending". `context.md` is never edited; if a decision here supersedes something in it, the entry says so explicitly.

**Verdict key:** ✅ helped / adopted · ❌ did not help / dropped · ⚠️ adopted with caveats · 🔧 fix or tooling · ℹ️ finding, no change

---

## Current status (update on every entry)
- **Stage:** blocking is designed, packaged (`src/`) and validated on held-out entities from state slices. Pruning, the matcher and the decision rule are not started.
- **Latest validated blocking numbers** (mini held-out set, Oregon + Kerala slices):

  | Run | Pair recall | Entity coverage | Oracle F0.5 |
  |---|---|---|---|
  | US | 0.9984 | 1.000 | 0.9996 |
  | India | 0.9961 | 1.000 | 0.9991 |
  | India as unseen country (France proxy, whole partition) | 0.9922 | 0.9994 | 0.9977 |

- **Next action:** environment ready on AWS EC2 `m7i-flex.large` (D28). Smoke passed (D29). Full A passed the gate (D39). C done; France healthy (D40). Full B running with 2 workers × 4 threads since 08:41 UTC. Go/no-go thresholds are in D25. Then build the LightGBM pruning ranker.

---

## Log

### D01 · 2026-09-24 21:55 UTC · Frame the task as S1-anchored retrieval under per-entity F0.5 — ✅
- **Decision:** treat the task as one-to-many retrieval from each S1 entity, not global clustering. Optimise entity coverage and the oracle F0.5, not only pair recall.
- **Why:** the metric is macro-averaged per S1 entity with all-or-nothing outcomes on empty predictions, and S1 is deduplicated.
- **Observed:** worked payoff table. One correct of two scores 0.833; one correct plus one wrong scores 0.556; an empty prediction for an entity with matches scores 0.
- **Helped:** yes. It set the KPIs used everywhere since.

### D02 · 2026-09-24 22:05 UTC · Profile S1 + ground truth — ℹ️ (changed priorities)
- **Observed:**
  - 2,206,821 S1 entities; singleton share 5.6%; mean 3.46 matches; no pool ID in two S1 lists.
  - 178K shared names; "Primary Care Group" appears 253 times; no repeated name+address pair.
- **Decision:** recall and entity coverage get top priority, since 94% of entities have matches. Rely on the exclusivity assumption. Blocking must combine name and address.
- **Helped:** yes. It reversed the earlier "precision first" emphasis for blocking.

### D02a · 2026-09-24 22:05 UTC · Read every TSV with `dtype=str, keep_default_na=False, quoting=3` — 🔧 ✅
- **Why:** default pandas turns empty match lists into NaN, could drop a business literally named "NA", and would interpret quote characters inside names.
- **Helped:** yes. All row counts matched line counts (for example, 2,206,821 S1 rows).

### D03 · 2026-09-24 22:18 UTC · Profile S2/S3 (10.32M records) — ℹ️
- **Observed:**
  - about 2.68M pool records match nothing
  - empty addresses about 3.4%
  - non-ASCII names 15% in S2 and 11% in S3 (Indic scripts plus accents)
  - domain names about 4%
  - 63% of S2 addresses in all caps
- **Decision:** normalization must transliterate, case-fold and handle domains.
- **Environment:** 1 CPU and 3 GB RAM. So all experiments use samples or slices.

### D03a · 2026-09-24 22:21 UTC · Build a 3% development sample — ⚠️
- **Setup:** 3% of S1 entities (seed 0), giving 66,053 entities; all 228,902 of their true pool records; plus 3% of all pool records chosen by `crc32(id) % 10000 < 300`. Total pool 531,329.
- **Why:** 1 CPU and 3 GB RAM make full-scale experiments impossible. Keeping every true match keeps recall measurable.
- **Caveat:** lookalike density is only about 3% of real, so rankings are optimistic. This was corrected by the full-density slices in D07.

### D03b · 2026-09-24 22:22 UTC · Use `anyascii` (ISC) for transliteration, not `unidecode` (GPL) — ✅
- **Why:** licensing cleanliness for the submission package. anyascii also covers every Indic script.
- **Observed:** Devanagari came out lossy ("आनंद वेंचर्स" → "anmd vemcrs"); Malayalam was later found to be badly handled (D08). This motivated the learned dictionary in D10.

### D04 · 2026-09-24 22:22 UTC · Pair-level name × address analysis on a 3% sample — ✅
- **Setup:** 66,053 S1 entities, 228,902 true pairs, a 531K pool (true matches plus a 3% random pool).
- **Observed:**
  - 0 true pairs with differing country labels
  - both name and address ≥85 in 66.4% of pairs; weak name in ~7%, almost all with a strong address; ~0.5% weak on both
  - 99.8% of entities have a strong-name or strong-address match
  - postal codes are nearly absent
  - numbers shared in 83% of pairs
- **Decision:** block within country. Use separate name and address channels plus a combined one. Drop postal-code blocking.
- **Helped:** yes. It set the channel design and the coverage ceiling.

### D05 · 2026-09-24 22:22 UTC · Choose normalization views: compact name, address, skeleton — ✅
- **Why:** domains and concatenations break word-based similarity ("ewapollo.com"). Transliteration variants need a phonetic-like key.
- **Observed:** the skeleton raised the median name similarity of native-script pairs from 87 to 100 (token-set).
- **Helped:** yes. The views are kept, but the skeleton channel is weak on its own (see D06/D07).

### D06 · 2026-09-24 23:09 UTC · First multi-channel blocking prototype (3% sample) — ✅
- **Channels:** combo, name, addr, skel (per source, top-40 stored) plus rev top-3.
- **Observed** (pair recall / entity coverage / candidates per S1):

  | Channel | Pair recall | Entity coverage | Candidates per S1 |
  |---|---|---|---|
  | combo k=10 | 0.9732 | 0.9974 | 20 |
  | name k=10 | 0.8033 | 0.9565 | 20 |
  | addr k=10 | 0.9146 | 0.9895 | 20 |
  | skel k=10 | 0.7496 | 0.9500 | 20 |
  | rev top-1 | 0.9871 | 0.9987 | 8 |
  | union "mid" | 0.9947 | 0.9995 | 36.4 |

- **Helped:** yes. The combined channel and reverse retrieval dominate, and the union beats every single channel.
- **Caveat:** the sample has only 3% distractor density, which led to D07.

### D06a · 2026-09-24 22:40 UTC · Run long jobs detached and resumable — 🔧 ✅
- **Why:** each tool call is limited to 300 s, and the first blocking job was killed along with the command that launched it.
- **Decision:** start jobs with `setsid nohup … &`, and write one parquet file per (country, source, channel) so that finished parts are skipped on restart.
- **Helped:** yes. Every later long run completed.

### D07 · 2026-09-24 23:30 UTC · Validate at full density on state slices (Oregon, Kerala) — ✅
- **Why:** the sample understated competition. The slices take all S1 in a state plus all pool records mentioning it.
- **Slices:** Oregon has 30,532 S1 and 142,907 pool records. Kerala has 35,482 S1 and 166,123 pool records.
- **Observed:**

  | Channel | Oregon pair recall | Kerala pair recall |
  |---|---|---|
  | combo k=10 | 0.9904 | 0.9498 |
  | name k=10 | 0.9041 | 0.7077 |
  | rev top-1 | 0.9901 (4.7 cand) | 0.9801 (4.7 cand) |
  | union mid | 0.9970 (28.4 cand) | 0.9926 (31.6 cand) |

  Name-only recall was higher inside a state than nationwide (0.77 vs 0.69 at k=3), which is evidence for geographic partitioning.
- **Decision:** reverse top-2 becomes the backbone. Partition geographically. India is the harder country.
- **Helped:** yes.

### D07a · 2026-09-24 23:13 UTC · How the slices were built — ℹ️
- **Setup:**
  - S1 in the slice: country matches and the address matches a state pattern (`\bOR\b|oregon`; `\bKL\b|kerala|കേരളം`).
  - Pool in the slice: records of that country whose address matches the pattern, plus every true match of the slice's S1 entities.
- **Observed:** only 95.0% (Oregon) and 96.1% (Kerala) of true pool records match the pattern. The other ~5% were included by ID, so slice recall is optimistic for unkeyed records. That case was measured separately in D11 and D13.
- **Note:** hand-written patterns were used *only* to build the test beds. The pipeline learns its keys (D12).

### D08 · 2026-09-24 23:31 UTC · Miss analysis of the union — ℹ️
- **Observed:**
  - **Kerala:** 906 of 123,133 pairs missed; 16 entities have nothing found; 97% of missed pairs have a found sibling; 47% of missed pairs have Malayalam names; 30% have an empty address.
  - **Oregon:** 315 of 106,004 missed; 9 entities have nothing found; 82% of misses have an empty address with a weakened name.
  - anyascii handles Malayalam badly (ലിമിറ്റഡ് → "limirrd").
- **Decision:** test cluster expansion (D09) and a learned transliteration dictionary (D10).

### D09 · 2026-09-25 09:45 (Kerala) / 10:08 (Oregon) UTC · Pool-cluster (S2↔S3) expansion — ❌ (optional only)
- **Setup:** pool→pool combo top-8, similarity ≥ 0.3. Expand from strong seeds (combo top-2 or rev top-1) using thresholds 0.5–0.8 and 2 or 4 neighbours.
- **Observed:** Kerala 0.9926 → 0.9949 (+2.4 candidates). Oregon 0.9970 → 0.9971.
- **Helped:** marginally in India, not at all in the US. Siblings are found, but the missed records aren't similar enough to them.
- **Status:** not implemented in `block.py`.

### D10 · 2026-09-25 09:51–10:07 UTC · Learned Indic→Latin transliteration dictionary — ✅ (after two fixes)
- **Attempt 1** 🔧: the `\w` tokenizer split words at combining vowel signs, giving 1.5% token coverage. Fixed by splitting only on whitespace and ASCII punctuation.
- **Attempt 2** 🔧: held out all of Kerala. Coverage was 0.3%, because Malayalam occurs almost only in Kerala, so the test was unfair. Replaced by splitting Kerala entities in half.
- **Attempt 3:** learned from the non-held-out half. **100% coverage** of the held-out half's Indic tokens, from about 1,347 tokens.
- **Recall on held-out Kerala entities:**

  | | Pair recall | Entity coverage | Missed pairs |
  |---|---|---|---|
  | Union without dictionary | 0.9924 | 0.9993 | 475 |
  | Union with dictionary | 0.9956 | 0.9998 | 276 (−42%) |

  The combined channel alone went from 4,744 misses to 1,923.
- **Helped:** yes. This is the biggest single improvement.

### D10a · 2026-09-25 09:52 UTC · pyarrow rejects `\u` escapes in raw-string regexes — 🔧
- **Observed:** `str.contains(r'[\u0900-\u0DFF]')` raised `ArrowInvalid` (pandas with the pyarrow string backend).
- **Fix:** use non-raw strings so that Python expands the escapes. `common.INDIC` does this.

### D11 · 2026-09-25 10:16 UTC · Test the partition fallback: nationwide name-only search — ❌
- **Why:** in the slices, about 5% of true pool records had no state token and had been included by ID, which is optimistic.
- **Observed:** 5,278 Oregon records without a state token, 96% with empty addresses. Rank of the true S1 among 1.32M US S1 by name: top-1 0.60, top-3 0.74, top-10 0.82. The combined-score version ran out of memory.
- **Helped:** no. Too weak as a fallback, so it's replaced by D13.

### D12 · 2026-09-25 10:27 UTC · Learned partition keys + multi-key assignment — ✅
- **Setup:** aliases learned from the 3% sample; each pool component maps to the S1 key it co-occurs with in true pairs.
- **Observed:**
  - 12,599 aliases; 42 US and 48 India keys
  - S1 without a key: 0.33% US, 0.03% India; pool without a key: 4.9% US, 3.6% India
  - 0.6% of keyed true pairs disagreed on key when single-key
  - in Oregon: single-key assignment would mismatch 0.17% of true pairs, multi-key 0%
- **Decision:** use multi-key assignment. Unkeyed records get their own channel.
- **Helped:** yes. Cross-partition loss went to zero.

### D13 · 2026-09-25 10:27 UTC · Unkeyed-pool forward channel — ✅
- **Setup:** Oregon S1 search the nationwide US unkeyed pool (276,400 records, 80% with empty addresses) by name.
- **Observed:** top-1 0.644, top-3 0.789, top-10 0.868, top-20 0.909. Better than nationwide reverse search (0.82 at top-10).
- **Decision:** adopt with k=20.
- **Helped:** yes.

### D14 · 2026-09-25 10:35 UTC · Sibling-name queries into the unkeyed pool — ❌
- **Observed:** k=10 went from 0.867 to 0.891, at 4.5 extra queries per S1. S1-only search at k=20 already gives 0.909.
- **Helped:** not cost-effective, so dropped.

### D15 · 2026-09-25 10:38 UTC · End-to-end projection for Oregon — ℹ️
- **Observed:** keyed records within partitions reach 0.9994. With the unkeyed channel at k=20: projected pair recall 0.9952, entity coverage 0.9997, all-matches 0.983, about 48 candidates per S1.

### D16 · 2026-09-25 10:40 UTC · Runtime estimate — ⚠️
- **Observed:** 138 partitions; multi-key inflation 1.022; about 15.6 CPU-hours in partitions plus about 4.6 for the unkeyed channel. Maharashtra takes about 213 minutes, Texas about 102.
- **Decision:** accept the cost for now. Parallelism or second-level splitting of large partitions is a pending item.

### D17 · 2026-09-25 10:50 (cache fix 2026-09-24 23:28) UTC · Blocking evaluator in submission format — 🔧 ✅
- `evaluate_blocking.py` reports pair recall, entity coverage, full recall, oracle F0.5, candidates per S1 and reduction ratio.
- **Observed on Oregon union:** pair recall 0.997, coverage 0.9997, oracle F0.5 0.999, 28.4 candidates, reduction ratio 0.99980, 9 entities with nothing found, a maximum list of 248 (which motivated a cap).
- 🔧 **Tooling bug fixed:** a cached `cand_all.pq` was reused across slices, producing recall above 1. The cache is now per tag.

### D18 · 2026-09-25 11:07 UTC · Package pipeline (`src/`) and validate on held-out entities — ✅
- **Contents:** `make_split`, `learn_artifacts` (dictionary and aliases with exclusions), `block` (modes, channels, unkeyed and orphan channels, cap, outputs, statistics) and `evaluate_blocking`.
- **Mini set:** Oregon + Kerala, 66,014 S1 entities, 309,030 pool records, 10% held out (6,501).
- **Observed:**

  | Held-out | Pair recall | Coverage | Full recall | Oracle F0.5 | Candidates per S1 |
  |---|---|---|---|---|---|
  | US | 0.9984 | 1.000 | 0.9943 | 0.9996 | 48.2 |
  | India | 0.9961 | 1.000 | 0.9874 | 0.9991 | 50.5 |

  Reduction ratio about 0.99984. Zero entities with nothing found.
- **Helped:** yes. This is the reference baseline.

### D18a · 2026-09-25 10:57 UTC · Caveat: artifacts learned on the mini set produce city-level keys — ⚠️
- **Observed:** with Oregon as the only US state in the mini set, cities pass the 0.2% "frequent component" threshold. The result was 20 US partitions instead of 1, and 67 Kerala partitions.
- **Impact:** mini-set results are still valid, since keys were learned without leakage and multi-key assignment protects recall. Full-data learning will produce state-level keys, as in D12 (42 US, 48 India).

### D19 · 2026-09-25 11:20 UTC · Leave-India-out (France proxy), whole-country partition — ✅
- **Setup:** artifacts learned with India excluded, so India has no aliases and no dictionary; one partition of 35,482 S1.
- **Observed:** India pair recall 0.9922, coverage 0.9994, oracle F0.5 0.9977, 31.6 candidates. The label-free signal: top-1 combo p10 dropped from 0.82 to 0.731.
- **Helped:** confirms the whole-country mode works for an unseen country. The top-1 statistics can detect trouble without labels.

### D20 · 2026-09-25 11:27 UTC · Leave-India-out, label-free component keys (large unseen country) — ⚠️
- **Setup:** forced with `--max-partition 10000`, giving 102 partitions.
- **Observed:** India pair recall 0.9777, oracle F0.5 0.9933, candidate lists at the 60 cap (p50 = p90 = max = 60), 6.4% of pool records unkeyed, 1.9% of entities with top-1 combo below 0.5.
- **Diagnosis:** the best-score cap was binding, which led to D21.

### D21 · 2026-09-25 11:45 UTC · Channel-floor cap replaces best-score cap — ✅
- **Setup:** same run with cap 1000, then cap policies replayed offline.
- **Observed** (India, unseen country, component keys):

  | Policy | Pair recall | Oracle F0.5 | Candidates |
  |---|---|---|---|
  | No cap | 0.9877 | 0.9962 | 86.7 |
  | Best-score cap 60 | 0.9777 | 0.9933 | 60 |
  | **Channel-floor cap 60** | **0.9867** | **0.9959** | 60 |
  | Channel-floor cap 100 | 0.9877 | 0.9962 | 84.7 |

  US results are unchanged under every policy.
- **Decision:** the channel-floor cap is the default in `block.py`.
- **Helped:** yes, +0.9 points of recall at the same budget.

### D22 · 2026-09-25 11:45 UTC · Whole-country mode up to 300K S1 (was 150K) — ✅
- **Why:** whole-country mode beat label-free component keys even without a cap (0.9922 vs 0.9877). Component keys can't tell spelling variants (KL/Keralam/കേരളം) are the same, so the Kerala component partition held only 52K of 166K pool records.
- **Decision:** `--max-partition` default is 300000. Component keys are a fallback for very large unseen countries only.
- **Helped:** expected yes for France. Not re-measured separately.

### D23 · 2026-09-25 11:51 UTC · Crash test on a tiny subset; documentation — 🔧
- **Observed:** 3,000 S1 entities, a sparse pool, the code runs end to end. The near-perfect scores are meaningless because the pool had very few distractors.
- **Added:** `README.md` (steps A: validation, B: France proxy, C: test), `requirements.txt`, `context.md`, and this file.

---

### D24 · 2026-09-25 12:23 UTC · Documentation completed and log backfilled — 🔧
- **Decision:**
  - Backfill UTC timestamps on D01–D23 from artifact modification times.
  - Add missing entries D02a, D03a, D03b, D06a, D07a, D10a, D18a.
  - Add the update protocol.
  - Add to `context.md` (§12–§15, file map moved to §16): the reading rules, development test beds, a measured-results reference, compute, and the model shortlist.
- **Why:** the user asked for an exhaustive static context file plus a timestamped log of every decision.
- **Helped:** documentation only; no pipeline change.

### D25 · 2026-09-25 12:40 UTC · Run order and go/no-go thresholds for the blocking stage — ✅ (decision)
- **Decision:** the blocking stage is accepted, and pruning/LightGBM work starts, only if the full-scale validation meets all of these:
  - US and India held-out: pair recall ≥ 0.99, entity coverage ≥ 0.998, oracle F0.5 ≥ 0.995, and per-source recall ≥ 0.99.
  - Leave-India-out (France proxy): India pair recall ≥ 0.97 and oracle F0.5 ≥ 0.99.
  - Test statistics: all test S1 entities present, and zero empty lists (or close to zero).
  - France: `top1_combo_p10` within 0.10 of US/India, `share_top1_below_0_5` ≤ 0.03, and `cand_p90` below the cap.
- **Why:** these are the values measured on slices, minus a margin for full-scale density. Below them, blocking rather than matching would limit the score.
- **Status:** the LightGBM pruning ranker is not implemented yet. It is the next step after this gate passes.

### D26 · 2026-09-25 16:52 UTC · Agent handoff package: parallel partitions, subset builder, runner, CLAUDE.md — ✅
- **Decision:**
  - Add `--workers` to `block.py`: a ProcessPoolExecutor over partitions, submitted largest first.
  - Add `src/make_subset.py` to build smoke-test subsets.
  - Add `run_blocking.sh` with `dry | smoke | A | B | C`, auto-detecting `student_resource(s)/dataset`, logging to `runs/logs/`.
  - Write test outputs to `runs/test/`. `output/` is reserved for the final submission files from later stages.
  - Add a project-root `CLAUDE.md` for the Claude Code agent (setup tasks, complete algorithm explanation, rules, gate).
  - Rewrite `README.md` for the layout.
- **Why:** the user will install the pipeline through a Claude Code agent and run it later. The full run is about 20 CPU-hours, so parallelism matters.
- **Observed:**
  - Tiny subset: `--workers 2` output is byte-identical to `--workers 1`.
  - Mock project layout (`student_resources/`): `dry` resolved every path; `smoke` ran all 5 steps end to end.
  - The subset builder produced 3,276 S1 / 11,461 true pool records from the mini set.
- **Helped:** yes (tooling). The algorithm and its defaults are unchanged.

### D27 · 2026-09-25 17:09 UTC · Install check (CLAUDE.md T1–T8), run from Claude Cowork without a shell on the user's machine — 🔧 ⚠️
- **Decision:** verify the package with no change to the algorithm or its defaults. Steps that need a shell on the user's machine are handed to the user as commands.
- **Why:** CLAUDE.md §3. This session could list, read and write files in the project folder but could **not** run commands on the user's machine (no local shell). The data files (up to 509 MB) are also above the 400 MB per-file transfer limit, so they could not be copied into the session.
- **Setup:**
  - T1: listed `student_resource/dataset/`. All 7 files are present: train_source1 210 MB, train_source2 489 MB, train_source3 504 MB, train_ground_truth 127 MB, test_source1 175 MB, test_source2 509 MB, test_source3 506 MB. Headers and `wc -l` were **not** checked (they need a local shell). The user runs this.
  - T2: the tree matches §2. `artifacts/`, `runs/` and `output/` don't exist yet; the runs create them. The exec bit on `run_blocking.sh` could not be checked, so the user runs `chmod +x`.
  - T3: in a separate Linux container (Python 3.11.15), a fresh venv installed every pin in `requirements.txt` exactly (pandas 3.0.2, numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0, sparse_dot_topn 1.2.0, anyascii 0.3.3, pyarrow 25.0.1), and the import check passed. `requirements.txt` is unchanged. The venv on the user's machine is still to be created.
  - T4: `python -m py_compile src/*.py` passed. Also checked: `Normalizer` with a Malayalam dictionary token, French accents, the domain strip, `skeleton`, `components`, and the non-raw `INDIC` regex on a `string[pyarrow]` series (no ArrowInvalid).
  - T5: `./run_blocking.sh dry` on a copy of the layout with header-only TSVs auto-detected `student_resource/dataset` and printed all A/B/C/smoke commands. Every dataset path resolved. The same relative paths exist on the user's machine (confirmed by listing).
  - T6: `nproc` and `free -g` were not readable (no local shell). The user reports them. Rule: `WORKERS = max(1, min(cores, floor(RAM_GB/6)))`.
  - T7: smoke test not run. It needs the full training files on the user's machine. It is gated on the user saying yes and running `./run_blocking.sh smoke` locally.
- **Observed:** no errors in anything that could be run. No code, parameter or requirements change.
- **Verdict:** 🔧 package verified statically. ⚠️ T1 headers and row counts, T3 local venv, T6 hardware and T7 smoke are left for the user.

### D28 · 2026-09-25 18:00 UTC · Install on AWS EC2 (CLAUDE.md T1–T6) — 🔧
- **Decision:** run the pipeline on an AWS EC2 instance instead of the laptop. Install with no change to the algorithm, defaults or `requirements.txt`.
- **Why:** the user wants the long runs (A/B/C) on a cloud machine. This session has a local shell and connects to the instance over SSH.
- **Setup:**
  - Instance: `m7i-flex.large` (2 vCPU, 7.6 GiB RAM), Ubuntu 24.04.4 LTS (kernel 6.17 aws), Python 3.12.3, 30 GB gp3 root (27 GB free), region ap-south-1. Security group allows SSH from the user's IP only. Added an 8 GB `/swapfile` (in fstab) as out-of-memory headroom. apt: python3-venv, tmux, rsync.
  - Layout: `~/amlc/{CLAUDE.md, student_resource/, code/business_entity_resolution/}`. The package was copied with rsync (without the local `.venv`, `runs/`, `artifacts/`). The dataset was copied with rsync (2.4 GB, 19.5 min).
  - T1: all 7 TSVs present, headers correct. `wc -l` (header included): train_source1 2,206,822; train_source2 5,034,617; train_source3 5,285,604; train_ground_truth 2,206,822; test_source1 1,732,545; test_source2 4,887,274; test_source3 5,082,317.
  - T3: `.venv` on the instance installed every pin exactly (pandas 3.0.2, numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0, sparse_dot_topn 1.2.0, anyascii 0.3.3, pyarrow 25.0.1). The import check passed.
  - T4: `py_compile src/*.py` passed. `chmod +x run_blocking.sh`.
  - T5: `WORKERS=1 ./run_blocking.sh dry` resolved `/home/ubuntu/amlc/student_resource/dataset/{train,test}`. Every path exists.
  - T6: 2 cores, 7 GB → `WORKERS = max(1, min(2, floor(7/6))) = 1`.
- **Observed:** no errors. `run_blocking.sh` calls bare `python`, and the instance has no system `python`, so every run must first `source .venv/bin/activate`. No script change.
- **Verdict:** 🔧 environment ready. T7 smoke test is pending the user's yes.

### D29 · 2026-09-25 18:05 UTC · Smoke test on EC2 (CLAUDE.md T7) — 🔧 ✅
- **Decision:** run `./run_blocking.sh smoke` on the instance after the user said yes. Parameters unchanged.
- **Setup:** `m7i-flex.large`, `WORKERS=1`, venv activated, inside tmux session `smoke`. Console log: `runs_smoke_console.log`. Step logs: `runs/logs/S0–S4`.
- **Observed:**
  - All 5 steps finished (EXIT=0). Wall time about 1–2 min (blocking 16 s).
  - Subset: 10,968 S1, 37,547 true pool records. Learned: 706 translit tokens; aliases US 1,065, India 1,571. Modes: both `learned`.
  - `candidate_pairs.tsv`: 10,968 rows, one per smoke S1, all S1 covered. Empty lists 0. IDs are all `S2-`/`S3-`. No duplicates within a list. List size mean 50.2, max 60.
  - Eval (not meaningful, since the pool has almost no distractors): pair recall 0.9992, entity coverage 1.0, oracle F0.5 0.9997. S2 0.9994 / S3 0.9989.
- **Verdict:** 🔧 ✅ smoke passed. Ready for A/B/C with `WORKERS=1`.

### D30 · 2026-09-25 19:10 UTC · Move to EC2 `r5.2xlarge`; start the A → B → C chain — 🔧
- **Decision:** re-install on a new instance, `r5.2xlarge` (8 vCPU, 62 GiB RAM, Ubuntu 24.04.4, Python 3.12.3, 29 GB root, ap-south-1). The user chose it for C's runtime and memory. Run A, B and C in sequence with `WORKERS=8`. Parameters, defaults and `requirements.txt` unchanged.
- **Why:** on `m7i-flex.large`, `WORKERS=1` gave an estimated 15+ h for C, with a memory risk in final assembly (D28/D29). The rule gives `WORKERS = min(8, floor(62/6)) = 8`.
- **Setup:**
  - 8 GB swap. Venv installed every pin exactly. `py_compile` OK. `dry` resolved all paths.
  - Dataset uploaded with rsync. The first attempt broke after 7 min when the user's mobile IP changed, and it was resumed with `--partial` plus retries. **md5 of all 7 TSVs matches the local copy.** Row counts are the same as in D28.
  - Chain: tmux session `blocking` runs `run_blocking.sh A; B; C` with `WORKERS=8`, stopping on the first non-zero exit. Timestamps go to `runs/logs/chain.log`, console output to `runs/logs/console_{A,B,C}.log`. The tmux session `memlog` writes `free -g` every 60 s to `runs/logs/mem.log`.
- **Observed:** A started at 19:08:13 UTC.
- **Verdict:** 🔧 running. Results will be logged in the next entries.

### D31 · 2026-09-25 19:18 UTC · Run B alongside A; guard so the chain doesn't re-run B — 🔧
- **Decision:** at the user's request, start B now in tmux `sideB` with `WORKERS=4`, while A3 runs with 8. No code or parameter change.
- **Why:** A3 was in its single-threaded load phase (load 1.0 on 8 cores). B needs only `artifacts/heldout_ids.txt` from A, which already exists.
- **Setup:**
  - The chain (pid 4805) cannot be killed without SIGHUP-ing A.
  - `runs/guard.sh` (tmux `guard`) polls every 0.2 s. When the chain spawns its own `run_blocking.sh B`, the guard kills only that process, so the chain ends. After both A and side B have finished, and only if `END A rc=0`, the guard starts C in tmux `runC` with `WORKERS=8`.
  - The chain's B start truncates `B2_learn.log` via tee, so tmux `keepB2` copies side-B's log to `runs/logs/B2_learn.side.log` first.
  - Status: A2 done — translit 1,347; aliases US 10,836, India 21,474, vs about 12,600 total in CLAUDE.md, to be checked against the A3 partition count.
- **Observed:** side B started at 19:17:25 UTC.
- **Verdict:** 🔧 running.

### D32 · 2026-09-25 19:20 UTC · Start C alongside A and B with `WORKERS=4` — 🔧
- **Decision:** at the user's request, start C now in tmux `runC` with `WORKERS=4`. Restart the guard as v2: it kills any chain-spawned B or C and no longer launches C itself. No code or parameter change.
- **Why:** only 2 of 8 cores were busy (A3 and B2 in single-threaded phases). C's wall time is bounded by its largest partition (single core, about 3.5 h), so starting it early moves the finish forward. 4 workers are enough to finish the other partitions within that time and limit the peak memory of three concurrent runs.
- **Setup:** running at 19:20: A3 (8 workers), B3 (4 workers; B2 finished), C1 → C2 (4 workers). The memory log continues. **Risk:** 16 worker processes on 8 cores, and C's final assembly holds all test candidates in memory. There is 8 GB of swap. The local monitor alerts on step ends, errors, available memory < 8 GB, and OOM kills.
- **Verdict:** 🔧 running.

### D33 · 2026-09-25 21:35 UTC · Run A: stats step crashed; India has no in-scope entities in sample mode — ⚠️
- **Observed:**
  - A3 (8 workers) ran for 2 h 20 min. All 16 sampled partitions and the unkeyed channel finished, and `candidate_pairs.tsv`, `candidates_detail.parquet` and `eval_ids_used.txt` were written. It then crashed in the label-free stats block (`block.py:234`, `int(nc.quantile(.9))` → `ValueError: cannot convert float NaN to integer`), so `blocking_stats.json` was not written, the chain stopped with rc=1, and A4 did not run.
  - **Cause:** India had 0 S1 in scope. Its 8 sampled partitions were all tiny city-level keys (jaipur 79 S1, madurai 16, kozhikode 14, …, tirumalagiri 2). An S1 is in scope only if all its keys ran, and India S1 also carry their state key, so none qualified. An empty `nc` gives a NaN quantile.
  - Related: full-data learning gives 32,310 aliases and 173 partitions (CLAUDE.md: about 12,600 and 138). City-level keys are present alongside the state keys, as in D18a on full data. Uniform partition sampling therefore mostly picks tiny partitions.
  - Timings (8 workers, 3 concurrent runs): US tx 135K S1 took about 57 min after nc finished, and the unkeyed channel took about 2.5 min.
- **Action:** ran A4 manually on the written outputs, with the exact command from the script. Log: `runs/logs/A4_eval.log`.
- **A4 result (US only, 37,304 held-out S1):** pair_recall 0.9954, entity_coverage 0.9997, entity_full 0.9836, oracle_f05 0.9986, cand/S1 49.0, S2 0.9955 / S3 0.9953. List size p50/p90/p99/max 49/54/60/60. 0 empty lists. 11 non-singletons with no true match. **US passes every A gate. India was not measured.**
- **Expected in B:** India's 8 sampled `components` partitions are also district/city components (west delhi, vadodara, bhopal, …). India will likely again have 0 in scope, giving the same crash and no India row. B's outputs are still written before the stats step.
- **C:** not affected. In test mode all S1 are in scope, so no country is empty.
- **Verdict:** ⚠️ The stats-block crash is a bug (sample mode + a country with no in-scope S1). The validation design needs a decision before India can be measured. Pending the user's decision. No code changed.

### D34 · 2026-09-25 21:40 UTC · Run B: same stats crash; B4 run manually — US only, India not measured — ⚠️
- **Observed:** B3 (4 workers, about 2 h 10 min) crashed with the same `ValueError` as D33, after writing `runs/loco/` outputs. India (`components` mode, 8 sampled district/city partitions) had 0 S1 in scope.
- **B4 (manual, same command as the script):** US only, 38,365 S1: pair_recall 0.9954, entity_coverage 0.9996, oracle_f05 0.9986, S2 0.9953 / S3 0.9954. 13 non-singletons with no true match. **The France proxy (India row) was not measured**, so the B gate cannot be evaluated.
- **Verdict:** ⚠️ Pending the user's choice among the D33 options. C keeps running.

### D35 · 2026-09-25 21:45 UTC · Stats-crash fix; rerun A and B over all partitions (`SAMPLE_PARTITIONS=0`) — 🔧
- **Decision (user-approved):**
  - `block.py` stats loop: a country with 0 in-scope S1 is recorded as `{mode, s1: 0, pool}` and skipped instead of crashing. No algorithm or parameter change.
  - Rerun A over all partitions with `WORKERS=4`, alongside C. Then rerun B over all partitions, auto-started when C ends (tmux `fullB`, `runs/startB.sh`): 8 workers if full A is done, else 4.
- **Why:** D33/D34. Sample mode gave 0 India S1 in scope, so neither India recall nor the France proxy was measured. All partitions put every S1 in scope.
- **Setup:**
  - Sample-8 results preserved in `runs/sample8/{val,loco,logs,heldout_ids.txt}`.
  - A1 regenerated `heldout_ids.txt`, byte-identical to the previous one (seed 42), so B stays consistent.
  - Full A started 21:44:24 UTC.
- **Estimate:** full A about 5–6 h (4 workers, sharing CPU with C; bounded by the largest partition, Maharashtra, about 3.5 h single-core). Full B about 4–5 h after C. Everything done in about 10–11 h.
- **Risk:** A's and C's final assemblies could overlap (memory). If so, pause one with `kill -STOP`.
- **Verdict:** 🔧 running.

### D36 · 2026-09-26 06:10 UTC · Full A and C are down to one giant partition each; runtime far above estimate — ⚠️
- **Observed (06:07 UTC):**
  - C2: 147/148 partitions done. Test Maharashtra (180,787 S1 / 1,024,688 pool) finished at 36,605 s, after about 9.8 CPU-h (CLAUDE.md estimated about 3.5 h for training Maharashtra). The remaining partition is **France `whole`** (259,452 S1 / 1,434,993 pool). Its worker has run for 10.3 CPU-h.
  - Full A3: 172/173 partitions done. The one remaining worker (8.0 CPU-h so far) is very likely training Maharashtra, the largest India key; delhi at 124K S1 took about 4 h.
  - Load is 2.0: 6 of 8 cores are idle. Memory: 40 GB used, 21 GB available, no swap, no OOM. No errors.
  - The SSH watcher lost contact around 05:22–06:07 UTC (user's IP/SG). No run was affected.
- **Estimate:** cost scales roughly with S1 × pool. Test Maharashtra is 1.85e11 for 9.8 CPU-h. France is 3.7e11, so about 20 CPU-h and **about 9 h more**. Training Maharashtra (~2.5–3e11) is about 14–16 CPU-h, so **about 6–8 h more**. The CLAUDE.md runtime estimates were about 3× too low for the largest partitions.
- **Implication for B:** in full B, India uses `components` mode, and its large state components will be similar giants, so **full B could take 15+ h** with the current single-threaded search. `sp_matmul_topn` is called without `n_threads` (`block.py:41`). Multithreading would only change speed, not results.
- **Verdict:** ⚠️ Pending the user's decision on B (keep the queued full B, add threading first, or use a size-aware sample).

### D37 · 2026-09-26 06:25 UTC · Opt-in multithreaded top-k search (`--threads`); full B queued after A and C — 🔧 ✅
- **Decision (user-approved):**
  - `block.py`: new `--threads` flag (default 1). `topk()` passes `n_threads` to `sp_matmul_topn` when > 1. With the default, the call is unchanged. The flag is threaded through `run_partition` and `name_search` (unkeyed/orphan).
  - `run_blocking.sh`: `THREADS` env (default 1) is passed as `--threads` to A3/B3/C2. The script was replaced by a new file (rsync rename), so the running full A's shell keeps the old inode.
  - Full B waits for both full A and C to finish, then runs with `WORKERS=2 THREADS=4 SAMPLE_PARTITIONS=0` (tmux `fullB`, `runs/startB2.sh`). The earlier `startB.sh` starter was stopped.
- **Why:** D36. The giant partitions are single-core (6 of 8 cores idle), and full B's India `components` giants were estimated at 15+ h.
- **Benchmark** (`/tmp/bench_threads.py` on the instance; real test partition US/`or`, 15,218 S1 / 85,346 pool, built with the C1 aliases):
  - 1 thread: 66 s total, of which top-k 57 s (88%).
  - 4 threads: 27 s total, of which top-k 19 s. **2.4× faster.**
  - **Output identical** (870,463 rows, sorted frame `equals`).
- **Choice:** 2 workers × 4 threads fills 8 cores and gives each giant about 2.5–3× speed. The critical path is the largest India component, about 16 CPU-h / 2.9, so **full B takes about 5–6 h** (vs 15+).
- **Verdict:** 🔧 ✅ adopted as speed-only. Algorithm and defaults unchanged. The already-running full A and C use single-threaded code.

### D38 · 2026-09-26 08:08 UTC · Full A assembly memory spike: extra swap, A protected from the OOM killer; disk nearly full — ⚠️ 🔧
- **Observed:**
  - Full A's last partition (India maharashtra, 196,920 S1 / 892,337 pool) finished at 33,209 s (about 07:00 UTC), much earlier than the D36 estimate. Unkeyed channel: US 266,168 pool, orphan 3,677; India 129,392 pool, orphan 352.
  - Final assembly (pivot over all hits for 2.2M S1) pushed A's main RSS to **48.7 GB** (peaks at 08:03 and 08:06). Available memory hit about 0.5 GB. C's 3 idle pool workers held about 29 GB RSS (partly shared) while France runs.
- **Actions:**
  - Tried a 16 GB `/swapfile2`. fallocate failed with ENOSPC and **briefly filled the disk to 100%**. The file was removed and the fstab line reverted (disk back to 13 GB free).
  - Added a 4 GB `/swapfile2` instead (total swap 12 GB; about 6.7 GB used at the peak).
  - Set `oom_score_adj=-500` on A's main process (pid 89050) so an OOM would not take the nearly finished run.
- **Result:** no OOM kill. `candidates_detail.parquet` written (583 MB so far at 08:06).
- **Disk warning:** the 29 GB root had about 8.8 GB free before A's outputs. Full-data outputs are about 3–4 GB per run (A, C, B), plus the swap files. **Before full B, the EBS volume must grow** (for example to 60 GB, online), or space must be freed (remove `/swapfile2` after C; prune `runs/sample8` parquets).
- **Lesson:** full-data assembly needs about 50 GB for 2.2M S1. Don't run it concurrently with another full run's idle pool, and size the disk for about 4 GB per run.
- **Verdict:** ⚠️ 🔧 A survived. Disk expansion pending with the user.

### D39 · 2026-09-26 08:15 UTC · Full A (all partitions): every A gate passes; India pair recall at the threshold — ✅ ⚠️
- **Setup:** `SAMPLE_PARTITIONS=0`, `WORKERS=4` (single-threaded code), alongside C. 21:44 → 08:13 UTC (10.5 h). 173 partitions; all 2,206,821 S1 and all 220,140 held-out S1 in scope.
- **A4:**

  | | S1 | pair_recall | entity_coverage | entity_full | oracle_f05 | cand/S1 |
  |---|---|---|---|---|---|---|
  | ALL | 220,140 | 0.9926 | 0.9996 | 0.9742 | 0.9979 | 49.9 |
  | India | 87,964 | **0.9902** | 0.9995 | 0.9654 | 0.9972 | 51.4 |
  | US | 132,176 | 0.9942 | 0.9997 | 0.9800 | 0.9983 | 49.0 |

  Recall by source: S2 0.9940, S3 0.9913. List size p50/p90/p99/max 49/57/60/60. 0 empty lists. **82** non-singletons with no true match.
- **A3 stats:**
  - US: learned, 57 partitions, pool unkeyed 4.3%, cand mean 48.9 / p90 55, top-1 combo p10 0.812 / p50 0.944, share below 0.5 = 0.0046.
  - India: learned, 116 partitions, pool unkeyed 3.1%, cand 51.4 / p90 59, top-1 combo p10 0.815 / p50 0.968, share below 0.5 = 0.0026.
- **Gate (D25):** US ✅ (0.9942 / 0.9997 / 0.9983). India ✅ (0.9902 / 0.9995 / 0.9972). S2/S3 ✅. India pair recall clears the 0.99 threshold by only 0.0002, and S3 recall trails S2.
- **vs sample-8 US (D33):** pair recall 0.9954 → 0.9942 on full US. The full data has more competition (larger partitions, TX/NY/CA all in scope).
- **Implication:** the blocking ceiling is about 0.998 oracle F0.5 (India 0.9972). Against a 0.986 leaderboard, India recall (lists hit the 60 cap: p90 59) is the main place to gain. Candidates: cap 100 (D21: +0.1 pt on unseen), pool-cluster expansion (D09: +0.2 pt India), and error analysis of the 82 zero-hit entities.
- **Disk:** outputs 4.1 GB (tsv 1.45 GB, parquet 2.64 GB). Root has 4.9 GB free. C needs about 3 GB, and B about 4 GB → expansion needed (D38).
- **Verdict:** ✅ A gate passed. ⚠️ India margin is thin.

### D40 · 2026-09-26 08:45 UTC · C (test set) finished; label-free gate: France healthy, India lists at the cap. Full B started — ✅ ⚠️
- **Setup:** C with `WORKERS=4` (single-threaded code), 19:19 → 08:40 UTC (13.4 h). 148 partitions (US 45, India 102, France 1 = `whole`). All 1,732,544 test S1 in scope. 2,043 orphan S1.
- **Format check:** `runs/test/candidate_pairs.tsv` has 1,732,544 rows, one per test S1, same S1 set as `test_source1`. 0 empty lists. Only `S2-`/`S3-` IDs. No duplicates within a list. Outputs: tsv 1.10 GB, parquet 2.02 GB.
- **Stats:**

  | | mode | S1 | pool | parts | pool_unkeyed | cand mean/p90/max | empty | top1 p10 / p50 | share <0.5 |
  |---|---|---|---|---|---|---|---|---|---|
  | US | learned | 663,106 | 3,817,031 | 45 | 3.6% | 48.3 / 55 / 60 | 0 | 0.822 / 0.944 | 0.0037 |
  | India | learned | 809,986 | 4,717,565 | 102 | 2.6% | 51.5 / **60** / 60 | 0 | 0.829 / 0.968 | 0.0016 |
  | France | whole | 259,452 | 1,434,993 | 1 | 0 | 37.0 / 45 / 60 | 0 | 0.853 / 0.955 | 0.0002 |

- **C gate (D25):**
  - empty lists ≈ 0 ✅ everywhere.
  - cand_p90 < 60: US ✅, France ✅, **India = 60 ⚠️** (at the cap; A's India p90 was 59).
  - France: top-1 p10 0.853 vs US 0.822 / India 0.829, within 0.10 ✅ and in fact higher. Share <0.5 = 0.0002 ≤ 0.03 ✅. Mode `whole` ✅.
- **Reading:** test US/India stats closely match A's (top-1 p10 0.812/0.815 on train), so A's recall should carry over. France looks at least as well-separated as the training countries. India lists saturating the 60 cap ties in with India's thin pair recall (D39), so a larger cap for India is the obvious lever to test.
- **Full B:** started automatically at 08:40:58 UTC with `WORKERS=2 THREADS=4 SAMPLE_PARTITIONS=0`.
- **Disk:** removed `/swapfile2` (memory now 59 GB available), leaving 6.0 GB free. B's outputs are about 4 GB. Tight but fits. EBS expansion still recommended.
- **Verdict:** ✅ C done and valid. ⚠️ India cand_p90 at the cap.

### D41 · 2026-09-26 09:45 UTC · Stage 2: LightGBM pruning ranker trained on full A — ✅
- **Context:**
  - The user shared `strategy.md` (project root). Amazon update of Sep 25: **the candidate count per S1 counts toward the final ranking**, and smaller sets rank higher.
  - Deadline: Sep 27 23:59 IST.
  - **The user allows `indic-transliteration` and similar libraries** (overrides CLAUDE.md rule 1 for such libraries). Not used yet.
  - Decided: keep our blocking (no recompute), adopt the strategy's later-stage ideas (small candidate sets, model-guided compression, matcher feature list, validation discipline), skip its FAISS / phonetic / numeric re-blocking.
  - B lowered to nice 19 (all threads) at the user's request, so stage-2/3 work gets CPU first.
- **Code:**
  - `src/prune_ranker.py` (`fit` / `apply`).
  - `src/decision.py`: metric plus the per-S1 expected-F0.5 rule plus exclusivity.
  - `src/eval_decision.py`: real F0.5 on held-out for a model and policy.
  - `lightgbm==4.7.0` (MIT) added to requirements.
- **Features (label-free, no country):**
  - s_/r_ for 7 channels, best, floor, is_s3, n_chan.
  - S1 context: q_rank, q_n, q_max, q_gap.
  - Pool context over all 110M rows: c_rank, c_n, c_gap, mutual best.
- **Setup:**
  - Full A parquet: 110,178,937 rows.
  - Train on 10% of non-held-out S1 (hash; 9.94M rows, 6.9% positive), with a 10% early-stopping split by S1.
  - Evaluate on all 220,140 held-out S1 (10.99M rows).
  - binary objective, lr 0.05, 127 leaves, 4 threads. Best iteration 867. About 26 min total.
- **Gain importance:** r_rev 0.63, s_rev 0.20, best 0.04, c_gap 0.04, c_rank 0.02. The reverse channel (exclusivity) dominates.
- **Held-out, list size vs ceiling:**

  | policy | cand/S1 | pair_recall | coverage | oracle F0.5 | India oracle | US oracle |
  |---|---|---|---|---|---|---|
  | top-5 | 5.0 | 0.9175 | 0.9993 | 0.9860 | 0.9856 | 0.9863 |
  | top-8 | 8.0 | 0.9861 | 0.9995 | 0.9966 | 0.9961 | 0.9969 |
  | **top-10** | **10.0** | 0.9899 | 0.9995 | **0.9972** | 0.9967 | 0.9976 |
  | top-15 | 15.0 | 0.9917 | 0.9996 | 0.9976 | 0.9970 | 0.9980 |
  | thr 0.001 (max 20) | 9.2 | 0.9914 | 0.9995 | 0.9975 | 0.9968 | 0.9980 |
  | thr 0.01 (max 20) | 5.9 | 0.9877 | 0.9993 | 0.9963 | 0.9956 | 0.9968 |
  | all (blocking) | 49.9 | 0.9926 | 0.9996 | 0.9979 | 0.9972 | 0.9983 |

- **Reading:** 5× fewer candidates (top-10) costs 0.0007 of oracle F0.5. Variable-length lists (threshold) beat fixed top-K at the same average size: 9.2 → 0.9975 vs top-10 → 0.9972; 5.9 → 0.9963 vs top-5 → 0.9860.
- **Verdict:** ✅ Ranker adopted. The list-size policy is to be picked together with the matcher; likely a threshold policy of about 6–9 candidates.

### D42 · 2026-09-26 11:45 UTC · Stage 3: LightGBM matcher. Held-out real F0.5 0.9681 (ranker-only baseline 0.8669) — ✅
- **Baseline** (`eval_decision.py`, ranker p as matcher, 220,140 held-out S1): best is expF05 bias=2.0 → **F0.5 0.8669** (US 0.8636, India 0.8720). top-1 only gives 0.648.
- **Matcher** (`src/matcher.py fit`):
  - Pairs = ranker top-10 with p ≥ 0.001 → 4.77M pairs (7.72/S1) for 617,858 S1 (20% of non-held-out plus all held-out).
  - Features: 30 ranker features, 7 ranker-p context features, 23 string features (rapidfuzz 3.14.6, MIT, added to requirements). Common name tokens are learned from the split (top 150 by document frequency).
  - Train: 3.07M pairs (44.4% positive), lr 0.05, 127 leaves, 6 threads. **Ran all 3000 rounds** (logloss still falling slowly: 0.07137 → 0.07131 over the last 200). About 75 min, sharing CPU with test prep.
- **Gain importance:** ranker p 0.56, **num_jacc 0.10**, p_gap 0.10, num_first_eq 0.03, then name similarities. Address-number overlap is the top string signal (lookalikes at house 21 vs 22, 105 vs 107).
- **Held-out real F0.5:**

  | policy | F0.5 | pred/S1 | India | US |
  |---|---|---|---|---|
  | all p ≥ 0.5 | 0.9653 | 3.38 | 0.9674 | 0.9640 |
  | all p ≥ 0.7 | 0.9672 | 3.29 | 0.9695 | 0.9657 |
  | expF05 bias=1.0 | 0.9680 | 3.28 | 0.9700 | 0.9667 |
  | **expF05 bias=1.3 + excl** | **0.9681** | 3.28 | 0.9702 | 0.9668 |

  The ceiling of these lists (oracle) is 0.997, so the remaining 0.029 is in the matcher and decision stage. Exclusivity inside the held-out set adds little (it does not see competition from non-held-out S1).
- **Ops:** B paused (SIGSTOP) during the memory crunch. My first pause command matched its own shell (pgrep pattern); the stuck shell was killed and B paused correctly. Disk grown to 58 GB (user expanded EBS; growpart + resize2fs).
- **Verdict:** ✅ Matcher adopted for the first submission (policy expF05 bias 1.3 + exclusivity). Next: error analysis, more rounds / more training data.

### D43 · 2026-09-26 12:55 UTC · First submission files built and validated (expected about 0.968) — ✅
- **Pipeline:**
  - `matcher.py prep` on test (cached features): ranker scored 83,301,646 test candidates and pruned them to 14,756,405 pairs (8.52/S1). String features took about 78 min, sharing CPU with the matcher fit.
  - `matcher.py apply` → `runs/match_test/pairs.parquet` (13 min, 3000-tree model).
  - `make_submission.py --bias 1.3 --excl --cand-k 10` → `output/`.
- **Output:**
  - `matching_results.tsv`: 1,732,544 rows. 3.36 matches/S1. **5.22% empty** (training singleton share is 5.6%). France 3.42 / 4.8% empty; India 3.35 / 5.3%; US 3.34 / 5.3%.
  - `candidate_pairs.tsv`: 8.52 candidates/S1, 0 empty (France 9.28, India 8.60, US 8.11). Matches are a subset.
- **Validator:** `validate_submission.py` PASS, including `--check-ids`.
- **Ops notes:**
  - The make_submission/validator SSH session dropped (empty log), but the files were complete. Re-ran the validator directly.
  - B was found running again (not paused) at about 12:48. The cause is unclear, possibly SIGCONT from the killed stuck shell. It runs at nice 19, so it's left running.
  - Test prep finished at 11:55 unnoticed (no watcher), costing about 25 min.
- **Verdict:** ✅ Ready to submit (submission #1).

### D44 · 2026-09-26 13:05 UTC · Git repository at the project root; history reconstructed; matcher v2 checkpoint — 🔧
- **Decision (user request):** version the project with local git, committing at every valuable checkpoint from now on.
- **Setup:**
  - `git init` at the project root.
  - `.gitignore` excludes the dataset, `output/*.tsv` (about 300 MB; recorded with checksums in `output/SUBMISSIONS.md`), venvs, `runs/`, `artifacts/`.
  - Repo-local identity: prajanya <mishasood1234@gmail.com>.
- **Reconstructed history:**
  - 6 commits dated from this log: D26–27 blocking package, D35 stats fix, D37 threads, D41 ranker, D42 matcher v1, D43 first submission.
  - Earlier file versions come from the pre-edit backups (`block.py`, `run_blocking.sh`, `matcher.py`), or by removing the logged change (D35 fix, requirements lines).
  - Each reconstructed commit says so in its message.
- **Checkpoint:** matcher v2 (sibling features, feature cache, lr/rounds flags, live progress) committed while its full fit runs. Results go in the next entry.
- **Verdict:** 🔧

### D45 · 2026-09-26 13:20 UTC · Matcher v2 (sibling features): held-out F0.5 0.9734 (v1 0.9681) — ✅
- **Setup:**
  - Same pairs as D42 (ranker top-10, p ≥ 0.001; 4.77M pairs; train 20% of non-held-out S1).
  - Features: v1 plus 7 sibling features.
  - lr 0.1, early stop 50: **best iteration 1145** (1,500 s fit in total, vs 6,700 s for v1).
  - Train features cached at `runs/match_val2/train_features.parquet`.
- **Gain importance:** p 0.60, **n_same_num1 0.062, n_same_nums 0.058** (new), p_gap 0.050, num_jacc 0.050. sib_num / sib_a are smaller (0.006 / 0.005). Counting candidates that share address numbers is the strongest new signal.
- **Held-out F0.5 (220,140 S1):** **expF05 bias 1.3 + excl → 0.9734** (India 0.9746, US 0.9726). All p ≥ 0.7 gives 0.9728. **+0.0053 over v1.**
- **Test:**
  - `matcher.py addsib` added the sibling features to the cached test prep in 9 min → `runs/match_test/features_sib.parquet`.
  - Submission 2 build (apply + make_submission + validator) is running into `output_v2/`.
- **Verdict:** ✅ v2 adopted for submission 2.

### D46 · 2026-09-26 13:35 UTC · Submission 2 built; the test set has about 23% more pool records per S1 than training — ✅ ℹ️
- **Leaderboard, submission 1:** **0.955 public** (held-out 0.968).
- **Submission 2** (matcher v2):
  - `output_v2/`; validator PASS with `--check-ids`.
  - **3.92 matches/S1** (US 4.13, India 3.78, France 3.84), 4.56% empty. Submission 1 had 3.36 (US 3.34, India 3.35, France 3.42) and 5.22% empty.
- **Finding:** pool records per S1 are train US 4.67 / India 4.68 vs **test US 5.76 / India 5.82 / France 5.53** (+23%). In training, about 74% of pool records belong to some S1 (3.46 / 4.67). If that holds, test S1 average about **4.2 true matches**.
  - This suggests submission 1 under-predicted, which is a likely part of the 0.955 vs 0.968 gap in every country, not only France.
  - v2's sibling features (records of one business share address numbers) pick up more of them.
- **Implications:**
  - Features and decisions calibrated on training list statistics (p_sum, p_n, the shortlist top-10 cap) may be conservative on test.
  - Entities with many matches may be cut by the top-10 shortlist. To evaluate.
- **Leave-India-out matcher experiment** (`exp_loco_matcher.py`) is running, to separate the unseen-country effect.

### D47 · 2026-09-26 13:40 UTC · Leave-India-out matcher: unseen-country loss is about −0.031 — ℹ️
- **Setup:** `exp_loco_matcher.py` on the cached v2 training features. Trained on US pairs only (1.82M; best iteration 675); scored all held-out S1. The ranker p and artifacts in the features still saw India, so this is optimistic for France.
- **Result (bias 1.3 + excl):** India **0.9432** (vs 0.9746 when India is in training, **−0.031**); US 0.9730 (unchanged).
- **Reading:** the matcher generalizes imperfectly to a new country. France (15% of test) likely loses ≥ 0.03, about ≥ 0.005 overall. Together with the more-matches-per-S1 shift (D46), this explains the 0.955 vs 0.968 gap.
- **Next:** use this as the France test bed (train US → score India) for country-robust changes: within-country rank-normalized features, dropping features whose distribution is country-specific, calibration per country.

### D48 · 2026-09-26 14:05 UTC · Submission 2 scored 0.886 (worse). Test-set distribution shift in count features — ❌ ℹ️
- **Leaderboard:** submission 2 (v2, held-out 0.9734) → **0.886 public**, vs submission 1's 0.955. **The D46 hypothesis was wrong**: the extra test pool records are mostly distractors (lookalikes and neighbours), not extra true matches. v2's 3.92 matches/S1 on test were largely false positives.
- **Shift, held-out (train) → test feature means:**
  - Pairs sharing an address number with another candidate (`n_same_num1 ≥ 1`): 42.5% → **57.2%**.
  - `c_n` 57.2 → **35.9**; `c_rank` 8.1 → **5.1**.
  - `a_tset` 0.65 → 0.73; `sib_a` 0.63 → 0.71; `num_first_eq` 0.50 → 0.43.
  - Name similarity, `num_jacc`, `best` and `p_max` are stable.
- **Reading:**
  - The test set has denser address neighbourhoods and fewer S1 per pool record.
  - v2's top new features (`n_same_num1/nums`) treat neighbours as siblings on test.
  - Absolute count features (c_n, c_rank, q_n, p_n, p_sum) depend on dataset sizes, so held-out validation from the training distribution cannot detect this.
- **Decisions:**
  - Submission 1 stays the best.
  - Before any submission: (a) test predicted matches/S1 must be close to held-out; (b) adversarial validation (train vs test pairs); drop features it relies on.
  - Next, v3: matcher without shift-prone count features; the ranker too if needed.

### D49 · 2026-09-26 14:55 UTC · Test-like validation (hidden S1) built; partly reproduces the test shift. Matcher v3a started — ℹ️ 🔧
- **Code:**
  - `prune_ranker.load(hide=)` / `hidden_s1`: hide S1 as absent, recompute S1/pool context and reverse ranks.
  - `testlike_eval.py`.
  - `decoy.py`: pair-intrinsic decoy features.
  - `matcher.py`: `adddecoy`, `--drop`, `--hide-permille`.
  - Committed `a1115e9`.
- **Results (220,140 held-out S1; bias 1.3 + excl):**

  | setting | v1 | v2 |
  |---|---|---|
  | nothing hidden (sanity) | 0.9680 | 0.9733 (both reproduce D42/D45) |
  | 21% of non-held-out S1 hidden | 0.9630 | 0.9645 |

- **Shift reproduced?**
  - Shortlist per S1: 7.7 → 8.65 (test 8.52) ✅.
  - c_n: 57 → 40 (test 36) ✅.
  - Share of pairs with `n_same_num1 ≥ 1`: 42.5% → 42.1% (test **57.2%**) ❌.
  - a_tset: 0.648 → 0.643 (test 0.731) ❌.
  - v2 predictions per S1: 3.35 (test 3.92) ❌.
- **Reading:**
  - Hiding S1 reproduces the competition/orphan shift but **not the same-address shift**. Test lists have many more candidates at the same or a similar address, and v2's count features read them as siblings.
  - The test-like set does **not** reproduce the leaderboard ordering (v2 ≥ v1 there), so it is not a sufficient gate on its own.
- **Decisions:**
  - Matcher v3a = v1 features + decoy features, **no sibling features**, trained at hide 21% (`runs/c1_match_v3a`, this chat, prefix `c1_`).
  - **Upload gate:** better than v1 on held-out *and* test-like, *and* test predictions/S1 ≤ v1's 3.36 (+ small margin).
- **Parallel chats:** another chat runs `tune_decision.py` / `diag_v2_test.py` on the same instance. This chat uses `c1_`-prefixed sessions and paths and does not touch the other chat's work.

### D50 · 2026-09-26 16:35 UTC · Matcher v3a (decoy features, trained at test-like density) → submission 3 built — ✅ (c1)
- **Model** (`runs/c1_match_v3a`):
  - Features: v1 features + `decoy.py` block; sibling features dropped.
  - Trained with 21% of non-held-out S1 hidden (hide 210): 534,147 S1, 4.62M pairs; lr 0.1, 4 threads.
  - Train features cached at `train_features_h210.parquet`.
- **Top new features by gain:** extra_tok, cand_hn_eq_anchor, hn_logdiff, hn_in_cand, extra_common, legal_rel.
- **Held-out F0.5 (bias 1.3 + excl):**

  | set | v1 | v2 | **v3a** |
  |---|---|---|---|
  | normal held-out | 0.9680 | 0.9733 | **0.9788** (India 0.9785, US 0.9789) |
  | test-like, hide 21% | 0.9630 | 0.9645 | **0.9771** |

- **Test gate:** 3.34 predictions/S1 (v1 3.36, v2 3.92) ✅; 5.48% empty.
  - vs submission 1: 97% of its pairs kept; 84–87% of S1 lists identical in every country.
  - France 3.36/S1 (5.2% empty), India 3.33, US 3.34.
- **Validator:** PASS (`--check-ids`). Output `output_v3/`, md5 `3cbacfcc…`.
- **Ops:**
  - Memory reached 4 GB free with the other chat's `c2_v3` fit (20 GB) running.
  - Stopped this chat's run B (fullB, France-proxy blocking, not needed for submissions).
  - Test `adddecoy` peaked at 36 GB RSS. Chunk it next time.

### D51 · 2026-09-26 17:00 UTC · Submission 3 (matcher v3a) scored **0.971** public (best; v1 0.955, v2 0.886) — ✅
- **Held-out → leaderboard:** v1 0.968 → 0.955 (−0.013); v3a 0.9788 → 0.971 (−0.008). The gap narrowed. The test-like score (0.9771) was closer to the leaderboard than plain held-out.
- **Confirms:**
  - pair-intrinsic decoy features transfer to test;
  - training at test-like density helps;
  - dropping the sibling count features was right;
  - the test-prediction-count gate (≤ v1) is a useful safety check.
- **Next:** see `context.md` §6.

### D52 · 2026-09-26 19:50 UTC · Stage-2 set-level matcher: +0.0022 held-out, +0.0024 test-like → submission 4 building — ✅ (c1)
- **Setup** (`src/c1_stage2.py`, `runs/c1_stage2`):
  - Out-of-fold v3a probabilities (2 folds by S1) for 2.72M non-held pairs of the hide-210 training cache.
  - Set-level features from the other candidates of the same S1. Safe variant: m1 rank/top/top2/gap, name/number agreement with the top candidate, best name similarity with m1 ≥ 0.5 candidates, m1-weighted agreement, best m1 in the same-number group. Full variant adds count features.
  - LightGBM lr 0.1: safe best iteration 299, full 333.
- **Importance:** safe: m1 0.92, m1_gap 0.023, then small set features. Full: m1 0.74, m1_gap 0.19.
- **Results (bias 1.3 + excl):**

  | model | held-out h0 | test-like h210 | pred/S1 |
  |---|---|---|---|
  | v3a | 0.9788 | 0.9771 | 3.32 |
  | **stage 2 safe** | **0.9810** | **0.9795** | 3.33 |
  | stage 2 full | 0.9813 | 0.9798 | 3.33 |

- **Decision:** ship **safe** (full is only +0.0003 and relies on count features, the D48 failure mode).
- Submission 4: `c1_stage2.py apply` on test → `output_v4/`, gated on test predictions per S1 and the validator.
- **Also done:** IDF table `runs/c1_idf/idf.parquet` (24.2M records; 2.33M name / 1.14M address tokens; e.g. "private" 1.97, "hotel" 6.22 in names).

### D53 · 2026-09-26 22:50 UTC · Matcher v4 (IDF + Soft-TFIDF + 50% S1): held-out 0.9829, test-like 0.9815 → decision 1 passed — ✅ (c1)
- **Setup:**
  - `matcher.py fit --hide-permille 210 --train-permille 500 --idf runs/c1_idf/idf.parquet`, sibling features dropped.
  - 1.01M S1 → shortlist 8.74M pairs (8.65/S1); 6.84M training pairs (39.6% positive).
  - lr 0.1, 4 threads, best iteration 1550. Build + fit took 1 h 46 min.
- **New features** (`idf_feats.py`):
  - 8 fixed-corpus IDF features.
  - `soft_tfidf_n` / `soft_tfidf_a`: Soft-TFIDF with JW ≥ 0.9 token matching.
  - `cand_rarity` and `soft_tfidf_a` are in the top 10 by gain.
- **Results (bias 1.3 + excl):**

  | model | held-out | test-like | pred/S1 |
  |---|---|---|---|
  | v3a | 0.9788 | 0.9771 | 3.32 |
  | stage 2 on v3a (sub 4) | 0.9810 | 0.9795 | 3.33 |
  | **v4** | **0.9829** | **0.9815** | 3.33 |

- **Next:**
  - Phase B: stage 2 (safe) on v4 → `runs/c1_v4s2`.
  - In parallel: test IDF features + v4 test scoring (`runs/c1_v4/test_pairs_v4.parquet`).
- **Ops:** added a 4 GB `/swapfile_c1` as a buffer while the other chat ran four jobs; remove it after the heavy steps.

### D54 · 2026-09-27 00:12 UTC · Stage 2 on v4: held-out 0.9846, test-like 0.9835 (decision 2 passed); submission 5 building — ✅ (c1)
- **Stage 2 (safe) on v4** (`runs/c1_v4s2`, OOF on 6.84M pairs):

  | model | held-out | test-like | pred/S1 |
  |---|---|---|---|
  | v4 alone | 0.9829 | 0.9815 | 3.33 |
  | **stage 2 on v4** | **0.9846** (+0.0017) | **0.9835** (+0.0020) | 3.35 |

- **Backup submission** `output_v5a/` (v4 alone): validator PASS, 3.34 matches/S1, md5 `9c63a961…`, downloaded locally.
- **Submission 5** (stage 2 on v4) → `output_v5/` is building.
- **Ops:** SSH to 43.204.218.248 timed out around 23:35; the instance kept running (uptime unchanged) and is now reachable at **43.204.231.50**. All c1 jobs had completed. Removed `/swapfile_c1` (disk had fallen to 2.6 GB free; now 6.6 GB).

### D55 · 2026-09-27 00:20 UTC · Disk cleanup of this chat's superseded files (6.6 → 12 GB free) — 🔧 (c1)
- **Removed** (all created by this chat, superseded, not needed by the current pipeline):
  - `runs/sample8/` (sample-8 A/B runs, superseded by full A);
  - `runs/val/candidate_pairs.tsv` (full-A blocking lists; evaluation done, the parquet is kept);
  - `runs/match_test/features.parquet` (superseded by `features_sib.parquet`, which is kept because the other chat used it);
  - `runs/match_test/pairs.parquet` and `pairs_v2.parquet` (scored pairs of submissions 1 and 2; the outputs are kept locally);
  - `/tmp` smoke-test files.
- **Not touched:** `c2_*` folders, shared caches, `c1_match_v3a/`, `c1_v4*`, `testlike/`.

### D56 · 2026-09-27 00:55 UTC · Phase D (decoy-weighted stage 1): no gain → dropped. Submission 5 stays the best — ❌ (c1)
- **Setup:** `c1_decoyweight.py` on the v4 training cache.
  - "Decoy" negatives = hn_rel 5, OR extra words + non-equal number, OR agrees with the anchor but not the S1.
  - That is 1.96M of 4.13M non-held negatives (47%), weighted 1.8× or 3.0×. Same features as v4; 4 threads each, in parallel.
- **Results (bias 1.3 + excl):**

  | model | held-out | test-like | pred/S1 |
  |---|---|---|---|
  | v4 | 0.9829 | 0.9815 | 3.33 |
  | v4d w=1.8 | 0.9828 | 0.9816 | 3.33 |
  | v4d w=3.0 | 0.9826 | 0.9814 | 3.33 |

- **Reading:** within noise at 1.8×, slightly worse at 3.0×. The matcher already separates these lookalikes using the decoy/IDF features; re-weighting only shifts the prior, which the expected-F0.5 decision rule already absorbs. Adds no information.
- **Decision:** drop phase D (per plan). Submission 5 (stage 2 on v4) remains the best validated file.

### D57 · 2026-09-27 01:50 UTC · User decisions on research_improvements.md; overnight plan (base = submission 5a) — ℹ️ (c1)
- **User decisions:**
  - No leaderboard uploads tonight.
  - **No French self-training** (rules risk).
  - **Candidate shrink allowed** if it costs ≤ 0.0005 F0.5.
  - **No merge with the other chat's models.**
  - Build on **submission 5a (v4, no stage 2)**: stage 2 raised test predictions/S1 +0.025 vs +0.012 on held-out, so its test transfer is unconfirmed without uploads.
- **Adopted from research_improvements.md:**
  - hurdle decision rule with a learned list-level q0 (`src/c1_hurdle.py`; cross-fitted on held-out halves; h0 and h210);
  - candidate shrink (later);
  - v4r retrain continues (it keeps the 10% held-out, so it stays validated).
- **Rejected:**
  - cross-entity joint assignment (exclusivity separates per record, greedy is already optimal);
  - merge / self-training (user);
  - importance and decoy weighting (D56).
- **Gate for submission 6:** better than the base on h0 and h210 (US and India); test predictions/S1 ≤ 3.34 (5a); validator passes.

### D58 · 2026-09-27 02:10 UTC · Hurdle rule negligible (dropped); transductive self-training now allowed (user) — ℹ️ (c1)
- **Hurdle rule** (`c1_hurdle.py`, learned q0, cross-fitted on held-out halves) on v4:
  - h0: 0.9829 → 0.9830; h210: 0.9815 → 0.9817 (best at q0 × 0.8).
  - Below the +0.0005 bar, so **not adopted**.
  - q0 mean 0.0574 vs true singleton share 0.0569: v4 at test-like density is already calibrated for the empty decision.
- **User update:** using test data for training (transductive) is allowed; nothing in the problem statement forbids it. This supersedes D57's "no self-training".
- **Plan** (`src/c1_selftrain.py`):
  1. `loco` validates the method on our v4 features (US only → pseudo-label India's non-held pairs at p ≥ 0.97 / ≤ 0.03 → retrain → India held-out).
  2. If it helps, `full`: training pairs + pseudo-labelled test pairs (all countries, including France), with h0/h210 gates for US/India, then the test build.
- **Running:** candidate-shrink grid (`c1_shrink.py`), v4r retrain R1.

### D59 · 2026-09-27 02:20 UTC · Candidate shrink: ranker threshold 0.005 adopted for the final build (−15% candidates, −0.0002) — ✅ (c1)
- **Setup** (`c1_shrink.py` on v4): shortlist = ranker top-10 with p ≥ thr (top-1 always kept). Ranker-p list features recomputed on the smaller lists; v4 re-scores; real F0.5 (bias 1.3 + excl).
- **Results:**

  | thr | cand/S1 h210 | h0 F0.5 | h210 F0.5 | h210 oracle |
  |---|---|---|---|---|
  | 0.001 (current) | 8.65 | 0.9829 | 0.9815 | 0.9968 |
  | 0.002 | 8.11 | 0.9828 | 0.9814 | 0.9968 |
  | 0.003 | 7.77 | 0.9827 | 0.9814 | 0.9967 |
  | **0.005** | **7.31** | **0.9827** | **0.9813** | 0.9966 |
  | 0.01 | 6.67 | 0.9825 | 0.9810 | 0.9962 |

- **Decision:** 0.005, within the user's ≤ 0.0005 cost rule (−0.0002 on both sets). 0.01 was rejected (−0.0005, at the limit).
- Apply in the final build: the relisted shortlist is exactly what the matcher scores, and `candidate_pairs.tsv` = that list.

### D60 · 2026-09-27 03:15 UTC · Self-training (transductive) hurts on the France proxy → dropped; submission 6 = v4 + shrink building — ❌/🔧 (c1)
- **`c1_selftrain.py loco`** (v4 features; train US → pseudo-label India's non-held pairs → retrain):

  | variant | India h0 | India h210 | pred/S1 h0 |
  |---|---|---|---|
  | US only | **0.9298** | **0.9255** | 3.367 |
  | pseudo p ≥ 0.97 / ≤ 0.03, w = 1 (acc 98.6%, 87.5% kept) | 0.9265 | 0.9232 | 3.384 |
  | pseudo p ≥ 0.995 / ≤ 0.005, w = 0.5 (acc 99.5%, 76.9% kept) | 0.9289 | 0.9251 | 3.377 |

- **Reading:** the few wrong pseudo-labels are confident lookalike false positives; training on them pushes predictions up (the v2 direction). c2's +0.010 (C12, v3m features) does not transfer to v4's IDF/decoy features.
- **Decision:** no self-training.
- **Submission 6 (in progress)** = matcher v4 (= 5a) on the shortlist shrunk at ranker threshold 0.005 (D59), bias 1.3 + excl → `output_v6/`.
  - `c1_shrink.py --apply-test` relists test and recomputes the ranker-p list features.
  - Will be replaced by v4r + shrink if R1 beats v4 on h0 and h210.

### D61 · 2026-09-27 03:35 UTC · Submission 6 built: v4 + candidate shrink, 7.21 cand/S1 (−15%), 99% identical lists to 5a — ✅ (c1)
- Test shortlist relisted at ranker threshold 0.005: 12.49M pairs (7.21/S1; France 9.28 → 7.09, India 8.60 → 7.33, US 8.11 → 7.10).
- v4 re-scored on the relisted lists → 3.35 matches/S1 (5a 3.34), 5.6% empty.
- **Identical lists to 5a:** 98.3% France, 99.2% India, 99.2% US.
- Validator PASS (`--check-ids`, and again locally). md5 `c925d169…`; `output_v6/` is downloaded locally.
- Expected F0.5 ≈ 5a − 0.0002 (D59), with a 15% smaller `candidate_pairs.tsv` for the size criterion.
- **Current recommended final file.** v4r (R1) may replace the model.

### D62 · 2026-09-27 03:50 UTC · Matcher v4r (v4 features, 100% non-held S1): +0.0005 held-out, +0.0006 test-like → submission 7 building — ✅ (c1)
- **Setup:**
  - `matcher.py fit --train-permille 1000 --hide-permille 210 --idf …`, sibling features dropped, 8 threads.
  - 1.79M S1 → 15.48M shortlist pairs; best iteration 1762; ~2 h 45 min in total.
- **Results (bias 1.3 + excl):**

  | model | h0 | h210 | pred/S1 |
  |---|---|---|---|
  | v4 | 0.9829 | 0.9815 | 3.33 |
  | **v4r** | **0.9834** | **0.9821** | 3.34 |

- **Submission 7** = v4r on the shortlist shrunk at 0.005 (D59) → `output_v7/`. It supersedes submission 6 if it passes the test gate.

### D63 · 2026-09-27 04:05 UTC · Submission 7 built (v4r + shrink) — recommended final file — ✅ (c1)
- 12.49M test pairs (7.21 cand/S1); 3.35 matches/S1 (France 3.34, India 3.35, US 3.36); 5.5% empty.
- 94–97% identical lists to submission 6.
- Validator PASS (instance with `--check-ids`, and locally). md5 `330913f3…`; `output_v7/` is downloaded.
- **Recommendation:** upload submission 7. It is the best validated model (h0 0.9834, h210 0.9821) with the smaller candidate file.
- **Fallbacks:** 6 (v4 + shrink), 5a (v4), 5 (stage 2 on v4, not chosen: test count rose 2× more than on held-out).
- **Next:** packaging from ~12:30 UTC. Copy `output_v7/` into `output/` for the zip.

### D64 · 2026-09-27 08:45 UTC · Confidence checks for submission 7: strongly positive — ✅ (c1)
- **A. Paired bootstrap** (220,140 held-out S1, 1000 resamples), submission-7 model (v4r + shrink):

  | vs | h0 | h210 |
  |---|---|---|
  | v3a | +0.0045 [+0.0043, +0.0048] | +0.0049 [+0.0046, +0.0051] |
  | v4 | +0.0004 [+0.0002, +0.0006] | +0.0005 [+0.0003, +0.0006] |

- **B. Label-free test estimate of what sub 7 changes vs sub 3** (h210 bucket precision by house-number relation × address-similarity decile, re-weighted to the test mix):
  - **added:** 82,469 pairs, est. precision **0.861**;
  - **removed:** 68,846 pairs, est. precision **0.391**;
  - net ≈ +44K correct, −30K false matches;
  - stress test (−10 points added, +10 removed): still ≈ +29K / −15K.
- **Recommendation:** upload submission 7 first (expected ~0.975–0.977); submission 5 second to decide on stage 2.

### D65 · 2026-09-27 08:55 UTC · Submission 7 scored **0.978** (best; sub 3 0.971). Stage 2 on v4r launched — ✅ (c1)
- **Held-out → leaderboard:** 0.9833 → 0.978 (gap −0.005; was −0.013 for v1, −0.008 for v3a). The IDF/Soft-TFIDF + data + shrink changes transfer to test.
- **User target:** ≥ 0.988. Assessment: needs held-out ≈ 0.992–0.993 (removing > 50% of the remaining errors). Very unlikely before the deadline; realistic ≈ 0.980–0.981.
- **Launched:**
  - stage 2 (safe) on v4r, with training, h0 and test lists relisted at 0.005 (`runs/c1_v4rs2`) → submission 8 (`output_v8/`) if it passes the gates;
  - bagging (2 seeds) continues at nice 19.
- **Asked the user** to upload submission 5 to check stage 2's test transfer.

---

## Pending
- [x] Header/`wc -l` check, venv install, hardware → WORKERS=1 on EC2 (D28). - [x] Smoke on the instance passed (D29).
- [x] Decided (D35): stats crash fixed; A and B rerun over all partitions. - [x] Full A → gate passed (D39). - [ ] Full B → gate. - [ ] Expand EBS volume before B (D38).
- [ ] User runs README A/B/C at full scale and reports `evaluate_blocking.py` tables and `blocking_stats.json`.
- [ ] Tune the budget and cap from full-scale results.
- [x] Parallelize partitions (D26). - [x] Multithreaded top-k inside partitions (D37). - [ ] Split partitions over ~50K S1 by a second-level key (runtime only).
- [x] Pruning ranker trained (D41). - [ ] Choose list policy; apply to test → `output/candidate_pairs.tsv`.
- [ ] (Optional) cluster expansion for India (D09) if full-scale India recall is below target.
- [x] Matcher + decision rule + exclusivity (D42); first submission built (D43). - [ ] Sibling features + faster retrain (2nd submission).
