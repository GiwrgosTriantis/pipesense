import numpy as np
import pandas as pd

from pipesense import model as M

PARAMS = {
    "numeric_features": ["age_years", "diameter_mm", "log_length", "num_breaks_5yr", "dist_to_center_m", "neighbours_500m"],
    "categorical_features": ["material", "soil_type", "pressure_zone"],
    "n_estimators": 50,
    "max_depth": 2,
    "random_state": 7,
    "block_size_m": 1000,
    "holdout_pct": 20,
}


def test_holdout_is_deterministic_and_independent_of_row_order(features_df):
    a = M.holdout_mask(features_df, "fairport", 1000, 20)
    shuffled = features_df.sample(frac=1, random_state=1)
    b = M.holdout_mask(shuffled, "fairport", 1000, 20)
    assert a.equals(b.loc[a.index])
    assert 0.05 < a.mean() < 0.4


def test_holdout_is_whole_blocks(features_df):
    mask = M.holdout_mask(features_df, "fairport", 1000, 20)
    blocks = M.spatial_blocks(features_df, 1000)
    assert (mask.groupby(blocks).nunique() == 1).all()


def test_model_pipeline_handles_missing_values_and_unseen_categories(features_df):
    model = M.build_model(PARAMS).fit(M.model_input(features_df, PARAMS), M.labels(features_df))
    new = features_df.head(5).copy()
    new.loc[new.index[0], "material"] = "OTHER"  # never seen in training
    new.loc[new.index[1], "diameter_mm"] = np.nan
    scores = model.predict_proba(M.model_input(new, PARAMS))[:, 1]
    assert np.isfinite(scores).all() and len(scores) == 5


def test_model_input_does_not_need_the_label(features_df):
    x = M.model_input(features_df.drop(columns="has_leak"), PARAMS)
    assert "has_leak" not in x.columns


def test_precision_at_frac():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    scores = np.arange(10)[::-1].astype(float)  # first rows ranked highest
    assert M.precision_at_frac(y, scores, 0.2) == 0.5


def test_evaluate_and_bootstrap_shapes(features_df):
    y = M.labels(features_df)
    rng = np.random.default_rng(0)
    good = y * 0.5 + rng.uniform(0, 0.5, len(y))
    noise = rng.uniform(0, 1, len(y))
    ev = M.evaluate(y, good, 0.1)
    assert {"average_precision", "roc_auc", "precision_at_10pct", "brier"} <= set(ev)
    boot = M.paired_bootstrap_ap_diff(y, good, noise, n_samples=200)
    assert boot["p_better"] > 0.95 and boot["ci90_low"] > 0


def test_baseline_ranks_by_age_with_median_fill():
    df = pd.DataFrame({"age_years": [10.0, np.nan, 30.0]})
    assert list(M.baseline_scores(df)) == [10.0, 20.0, 30.0]
