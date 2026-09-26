# Status (2026-09-26, commit c21ad23)

Done:
- Blocking v1 tuned (oracle F0.5 0.9778, 5.82 cands/S1 train; 7.85 test) and a rank/score rule submission (train F0.5 0.8388), per prior session.
- New: `features.py` (29 pairwise features/pair, chunked rapidfuzz), `model.py` (LightGBM, folds 0-3 train / fold 4 early-stop, then full retrain at best_iteration), `decide.py` (one-owner top-probability-per-record rule, threshold tuned on fold-4 holdout via evaluate.py's real macro F0.5). `make_submission.py` now takes `--method {rule,model}`.
- All three verified only on `--smoke 20000` samples (fast, illustrative numbers, not representative): model f05=0.9472 vs blocking oracle ceiling 0.9778 on that tiny holdout. Not logged to experiments/log.csv (removed smoke rows before commit).

Not run (full data, per token-economy rules — commands below are for you to run):
```powershell
.venv\Scripts\Activate.ps1
python code\business_entity_resolution\src\features.py --split train
python code\business_entity_resolution\src\features.py --split test
python code\business_entity_resolution\src\model.py
python code\business_entity_resolution\src\decide.py
python code\business_entity_resolution\src\make_submission.py --method model
```

Next: run the above, compare model vs rule F0.5/cands-per-S1, decide which becomes the submission.
