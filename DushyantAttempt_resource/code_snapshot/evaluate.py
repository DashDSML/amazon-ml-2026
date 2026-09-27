"""Macro F0.5 per Source 1 entity: the competition metric.

Per S1 entity with n_true true matches, n_pred predicted IDs and tp correct:
    F0.5 = 1.25 * P * R / (0.25 * P + R) = 1.25 * tp / (0.25 * n_true + n_pred)
with the special cases
    n_true == 0 and n_pred == 0 -> 1.0   (correctly empty singleton)
    otherwise tp == 0           -> 0.0
The score is averaged over every S1 entity in ``s1_ids`` (singletons included).

All inputs are long-form polars frames with columns
``source1_entity_id`` and ``match_id`` (null/absent rows mean "no matches").
Duplicates are ignored (set semantics).
"""

from __future__ import annotations

import polars as pl

S1 = "source1_entity_id"
M = "match_id"
BETA2 = 0.25


def _pair_set(pairs: pl.DataFrame) -> pl.DataFrame:
    return pairs.select(S1, M).filter(pl.col(M).is_not_null() & (pl.col(M) != "")).unique()


def per_entity_scores(
    truth_pairs: pl.DataFrame,
    pred_pairs: pl.DataFrame,
    s1_ids: pl.Series | None = None,
) -> pl.DataFrame:
    """One row per S1 entity: n_true, n_pred, tp, precision, recall, f05.

    ``s1_ids`` is the evaluation universe (defaults to every S1 in truth_pairs,
    including singleton rows with a null match). Predictions for S1 IDs outside
    the universe are ignored.
    """
    if s1_ids is None:
        s1_ids = truth_pairs.get_column(S1)
    universe = pl.DataFrame({S1: s1_ids.cast(pl.String)}).unique()

    truth = _pair_set(truth_pairs)
    pred = _pair_set(pred_pairs).join(universe, on=S1, how="semi")

    n_true = truth.group_by(S1).len("n_true")
    n_pred = pred.group_by(S1).len("n_pred")
    tp = truth.join(pred, on=[S1, M], how="inner").group_by(S1).len("tp")

    out = (
        universe.join(n_true, on=S1, how="left")
        .join(n_pred, on=S1, how="left")
        .join(tp, on=S1, how="left")
        .with_columns(pl.col("n_true", "n_pred", "tp").fill_null(0).cast(pl.Int64))
    )
    return out.with_columns(
        precision=pl.when(pl.col("n_pred") > 0).then(pl.col("tp") / pl.col("n_pred")),
        recall=pl.when(pl.col("n_true") > 0).then(pl.col("tp") / pl.col("n_true")),
        f05=pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0))
        .then(1.0)
        .when(pl.col("tp") == 0)
        .then(0.0)
        .otherwise(
            (1 + BETA2) * pl.col("tp") / (BETA2 * pl.col("n_true") + pl.col("n_pred"))
        ),
    ).sort(S1)


def macro_f05(
    truth_pairs: pl.DataFrame,
    pred_pairs: pl.DataFrame,
    s1_ids: pl.Series | None = None,
) -> float:
    return float(per_entity_scores(truth_pairs, pred_pairs, s1_ids)["f05"].mean())


def score_report(
    scores: pl.DataFrame,
    s1_country: pl.DataFrame | None = None,
) -> dict[str, pl.DataFrame | float]:
    """Breakdowns required by CLAUDE.md: overall, per country, singleton vs not.

    ``scores`` comes from ``per_entity_scores``; ``s1_country`` has columns
    source1_entity_id, country.
    """
    s = scores.with_columns(
        bucket=pl.when(pl.col("n_true") == 0)
        .then(pl.lit("singleton"))
        .otherwise(pl.lit("non_singleton"))
    )
    agg = [pl.len().alias("n_s1"), pl.col("f05").mean().alias("macro_f05")]
    report: dict[str, pl.DataFrame | float] = {
        "overall": float(s["f05"].mean()),
        "by_bucket": s.group_by("bucket").agg(agg).sort("bucket"),
    }
    if s1_country is not None:
        s = s.join(s1_country.select(S1, "country"), on=S1, how="left")
        report["by_country"] = s.group_by("country").agg(agg).sort("country")
        report["by_country_bucket"] = (
            s.group_by("country", "bucket").agg(agg).sort("country", "bucket")
        )
    return report


def _self_test() -> None:
    """Checks against the worked examples in challengecontext.md section 4."""

    def f(truth: list[str], pred: list[str]) -> float:
        t = pl.DataFrame({S1: ["a"] * max(len(truth), 1), M: truth or [None]}, schema={S1: pl.String, M: pl.String})
        p = pl.DataFrame({S1: ["a"] * len(pred), M: pred}, schema={S1: pl.String, M: pl.String})
        return macro_f05(t, p)

    cases = [
        ([], [], 1.0),
        ([], ["x"], 0.0),
        (["x"], [], 0.0),
        (["x"], ["x"], 1.0),
        (["x", "y"], ["x"], 0.833),
        (["x", "y", "z"], ["x"], 0.714),
        (["x", "y", "z"], ["x", "y"], 0.909),
        (["w", "x", "y", "z"], ["x", "y"], 0.833),
        (["x"], ["x", "q"], 0.556),
        (["x"], ["x", "q", "r"], 0.385),
        (["x"], ["q"], 0.0),
    ]
    for truth, pred, want in cases:
        got = f(truth, pred)
        assert abs(got - want) < 5e-4, (truth, pred, got, want)

    # Macro average and universe handling: b is a singleton with no prediction.
    t = pl.DataFrame({S1: ["a", "a", "b"], M: ["x", "y", None]})
    p = pl.DataFrame({S1: ["a", "a", "zzz"], M: ["x", "x", "q"]})  # dup + out-of-universe
    assert abs(macro_f05(t, p) - (0.8333333 + 1.0) / 2) < 1e-6
    print("evaluate.py self-test passed")


if __name__ == "__main__":
    _self_test()
