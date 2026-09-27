"""Entity-centric decision layer (NonStandard.md T2 gate + T4 additional-match rule + T6
one-owner), an alternative to decide.py's record-centric "each record picks its own top S1"
rule.

Why: decide.py's fold-4 CV report showed singleton F0.5 (0.9512) BELOW non-singleton F0.5
(0.9635) -- backwards from the usual pattern, and exactly the symptom T2 describes: "an
entity with a crowd of mediocre candidates looks like it has a match when it does not."
decide.py decides per RECORD ("which S1, if any, do I belong to?"); this decides per S1
ENTITY ("do I have any match, and if so how many?"), which is what the metric (macro F0.5
per S1) actually rewards. Per-entity decisions don't automatically respect one-owner (the
same record could be accepted by two different entities), so T6 is applied as a final
reconciliation pass.

Decision, per S1 entity, candidates sorted by probability descending:
  1. T2 gate: accept nothing from this entity if its best candidate's probability < t_gate.
  2. T4: the top-1 candidate is accepted once the gate passes; every additional candidate r
     (r>=1) also needs prob_r >= t2 (a separate, typically stricter threshold) AND
     prob_r >= alpha * best_prob (relative to this entity's best).
  3. T6: for every record accepted by more than one entity, keep it only under the entity
     where its probability is highest.

Probabilities are calibrated first (T5, isotonic regression fit on the fold-4 holdout's own
out-of-fold predictions -- never seen in training) before any of this: NonStandard.md T1/T4
both warn that expected-F/gate decisions on uncalibrated GBDT probabilities choose the wrong
cutoff. A first pass without calibration confirmed this in practice (see experiments/log.csv
"decide_entity.py version=v1..." before this fix): the entity-centric rule landed within noise
of decide.py's plain record-centric result (0.9543 vs 0.9541) and was individually *worse* on
both singleton and non-singleton -- not an improvement, so it was not adopted uncalibrated.

t_gate is swept first (with t2/alpha effectively disabled, i.e. no additional matches),
then t2 x alpha is swept holding t_gate fixed -- both against real macro F0.5 (official
evaluate.py) on the fold-4 holdout only (G1: never test). Adopted only if it beats
decide.py's record-centric result on the same holdout.

    python decide_entity.py --version v2
"""

from __future__ import annotations

import argparse
import json

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

import cv_harness as ev
import evaluate as official
from data import load_ground_truth_pairs
from decide import HOLDOUT_FOLD, _map_to_ids, id_maps, oracle_predictions
from features import FEATURE_COLS
from model import add_fold, load_features, model_dir

T_GATE_GRID = [round(0.05 * i, 3) for i in range(1, 20)]  # 0.05 .. 0.95
T2_GRID = [round(0.05 * i, 3) for i in range(1, 20)]
ALPHA_GRID = [0.70, 0.80, 0.90, 0.95, 1.00]


def entity_frame(df: pl.DataFrame, prob: np.ndarray) -> pl.DataFrame:
    """(s1, src, qi, prob, _r) -- _r is this candidate's rank within its S1 entity, by prob desc."""
    return (
        df.select("s1", "src", "qi")
        .with_columns(prob=pl.Series(prob))
        .sort(["s1", "prob"], descending=[False, True])
        .with_columns(_r=pl.int_range(pl.len()).over("s1"))
    )


def decide(ent: pl.DataFrame, t_gate: float, t2: float, alpha: float) -> pl.DataFrame:
    best = ent.filter(pl.col("_r") == 0).select("s1", pl.col("prob").alias("_best"))
    d = ent.join(best, on="s1")
    keep = (pl.col("_best") >= t_gate) & (
        (pl.col("_r") == 0)
        | ((pl.col("prob") >= t2) & (pl.col("prob") >= alpha * pl.col("_best")))
    )
    accepted = d.filter(keep)
    # T6 one-owner: a record accepted by >1 entity survives only under its best-probability one
    accepted = (
        accepted.sort(["src", "qi", "prob"], descending=[False, False, True])
        .with_columns(_rr=pl.int_range(pl.len()).over(["src", "qi"]))
        .filter(pl.col("_rr") == 0)
        .drop("_rr")
    )
    return accepted.select("s1", "src", "qi")


def score(ent, s1_ids, q_ids, truth, universe, t_gate, t2, alpha) -> float:
    pred = _map_to_ids(decide(ent, t_gate, t2, alpha), s1_ids, q_ids)
    return official.macro_f05(truth, pred, universe)


def main(version: str = "v2") -> None:
    df = add_fold(load_features(None, version))
    ho = df.filter(pl.col("fold") == HOLDOUT_FOLD)
    booster = lgb.Booster(model_file=str(model_dir(version) / "lgbm_cv.txt"))
    raw_prob = booster.predict(ho.select(FEATURE_COLS).to_numpy())
    is_true = ho["is_true"].to_numpy()

    iso = IsotonicRegression(out_of_bounds="clip")
    prob = iso.fit_transform(raw_prob, is_true.astype(np.float64))
    print(f"[decide_entity] calibrated (isotonic, T5): raw mean={raw_prob.mean():.4f} "
          f"-> calibrated mean={prob.mean():.4f}, pos_rate={is_true.mean():.4f}")
    ent = entity_frame(ho, prob)

    s1_ids, q_ids = id_maps("train")
    truth = load_ground_truth_pairs()
    universe = ho.select("s1").unique().join(s1_ids, on="s1")["entity_id"]
    oracle_f05 = official.macro_f05(truth, oracle_predictions(ho, s1_ids, q_ids), universe)
    print(f"[decide_entity] version={version}  fold-{HOLDOUT_FOLD} holdout oracle ceiling={oracle_f05:.4f}")

    print("=== stage 1: sweep t_gate (top-1 per entity only, no additional matches) ===")
    best_tg, best_f1 = None, -1.0
    for tg in T_GATE_GRID:
        f = score(ent, s1_ids, q_ids, truth, universe, tg, t2=1.01, alpha=1.01)
        print(f"  t_gate={tg:.2f}  f05={f:.4f}")
        if f > best_f1:
            best_f1, best_tg = f, tg
    print(f"-> best t_gate={best_tg:.2f}  f05={best_f1:.4f}")

    print(f"\n=== stage 2: sweep t2 x alpha for additional matches (t_gate fixed at {best_tg:.2f}) ===")
    best_t2, best_alpha, best_f2 = None, None, best_f1
    for t2 in [t for t in T2_GRID if t >= best_tg]:
        for alpha in ALPHA_GRID:
            f = score(ent, s1_ids, q_ids, truth, universe, best_tg, t2, alpha)
            if f > best_f2:
                best_f2, best_t2, best_alpha = f, t2, alpha
    if best_t2 is not None:
        print(f"-> best t2={best_t2:.2f} alpha={best_alpha:.2f}  f05={best_f2:.4f} "
              f"(vs top-1-only {best_f1:.4f})")
    else:
        print(f"-> no (t2, alpha) beat top-1-only ({best_f1:.4f}); keeping additional-match rule off")
        best_t2, best_alpha = 1.01, 1.01

    pred = decide(ent, best_tg, best_t2, best_alpha)
    pred_ids = _map_to_ids(pred, s1_ids, q_ids)
    scores = official.per_entity_scores(truth, pred_ids, universe)
    s1_country = s1_ids.select(pl.col("entity_id").alias("source1_entity_id"), "country")
    report = official.score_report(scores, s1_country)
    print(f"\n--- report: entity-centric (t_gate={best_tg:.2f}, t2={best_t2:.2f}, alpha={best_alpha:.2f}) ---")
    print(f"overall f05={report['overall']:.4f}  oracle ceiling={oracle_f05:.4f}")
    print("by country:")
    print(report["by_country"])
    print("singleton vs non-singleton:")
    print(report["by_bucket"])

    md = model_dir(version)
    joblib.dump(iso, md / "isotonic_calibrator.joblib")
    (md / "decision_entity.json").write_text(json.dumps(
        {
            "t_gate": best_tg, "t2": best_t2, "alpha": best_alpha,
            "f05": report["overall"], "holdout_fold": HOLDOUT_FOLD,
            "ceiling": oracle_f05, "version": version,
        },
        indent=2,
    ))
    print(f"\nsaved -> {md / 'decision_entity.json'}")

    ev.log_experiment(
        f"decide_entity.py version={version} t_gate={best_tg:.2f} t2={best_t2:.2f} alpha={best_alpha:.2f}",
        {"f05": report["overall"], "ceiling": oracle_f05},
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v1", "v2", "v2r"], default="v2")
    args = ap.parse_args()
    main(args.version)
