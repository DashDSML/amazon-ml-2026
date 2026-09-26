# Status (2026-09-27, commit fcc3116)

Blocking v2 done and frozen at floor=0.5 alpha=0.95 m=3: fix A3 (rarity-weighted retrieval rank, PRE_CAP_V2=80), fix S1/S2 (key evidence added to stage-1 score), fix A1+B (composite P/N keys; P backed by a train-fit global name-token DF table, `dataset/cache/blocking_v2/name_token_df.parquet`). Fastloop (20k S1): oracle 0.9778->0.9898, recall 0.9671, cands/S1 5.81->6.01; grid picked (0.5,0.95,3) at oracle 0.9893 / 5.37 cands/S1 as the frozen point.

`features.py`/`model.py`/`decide.py`/`make_submission.py` all take `--version {v1,v2}` (default v1), reading/writing `dataset/cache/blocking_{version}/`; v1 untouched (`lb-2` depends on it -- verified only against the v2 tree, using temporary fake data, all cleaned up). `decide.py`'s oracle ceiling is now computed live from the holdout's own candidates instead of a hardcoded v1 number.

Running overnight (not run by assistant): full blocking_v2 train+test at the frozen op point.

Morning, in order:
```powershell
python code\business_entity_resolution\src\features.py --split train --version v2
python code\business_entity_resolution\src\features.py --split test --version v2
python code\business_entity_resolution\src\model.py --version v2
python code\business_entity_resolution\src\decide.py --version v2
python code\business_entity_resolution\src\make_submission.py --version v2 --method model
python utils\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir dataset\test
```

Next: compare v2's real full-scale F0.5/cands-per-S1 against v1's `lb-2` (CV 0.9541, LB 0.943) before deciding on `lb-3`.
