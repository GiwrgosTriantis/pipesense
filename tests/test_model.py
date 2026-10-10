import numpy as np
import pandas as pd

from pipesense import model as M

from conftest import synthetic_features

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


class _Memorizer:
    """Stand-in model that scores 1.0 for any row it was fitted on and 0.0 for anything else."""

    def fit(self, x, y):
        self.seen = {tuple(map(repr, row)) for row in x.itertuples(index=False)}
        return self

    def predict_proba(self, x):
        hit = np.array([tuple(map(repr, row)) in self.seen for row in x.itertuples(index=False)], dtype=float)
        return np.column_stack([1 - hit, hit])


def test_no_pipe_is_scored_by_a_model_that_trained_on_it(features_df):
    hold = M.holdout_mask(features_df, "fairport", 1000.0, 20)
    train_df = features_df[~hold]
    model, oof = M.fit_cross_fitted(train_df, PARAMS, build=lambda params: _Memorizer())
    x_train = M.model_input(train_df, PARAMS)
    assert model.full.predict_proba(x_train)[:, 1].all()  # the memoriser works: in-sample scores are 1

    far = synthetic_features(n=20, seed=3).assign(x_m=50_000.0)  # new pipes in blocks nobody trained on
    pipes = pd.concat([features_df, far], ignore_index=True)
    scores, source = model.score_pipes(M.model_input(pipes, PARAMS), M.spatial_blocks(pipes, 1000.0))

    assert (scores == 0).all() and (oof == 0).all()  # every score came from a model that never saw the row
    is_train = np.r_[(~hold).to_numpy(), np.zeros(len(far), dtype=bool)]
    assert (source[is_train] == "oof").all() and (source[~is_train] == "full").all()
    assert len(model.fold_models) == 5 and set(model.block_fold.values()) == set(range(5))


def test_cross_fit_full_model_is_the_plain_model_the_gate_evaluates(features_df):
    model, oof = M.fit_cross_fitted(features_df, PARAMS)
    x, y = M.model_input(features_df, PARAMS), M.labels(features_df)
    plain = M.build_model(PARAMS).fit(x, y)
    assert np.array_equal(model.predict_proba(x), plain.predict_proba(x))
    assert set(model.block_fold) == set(M.spatial_blocks(features_df, 1000.0))
    assert 0 < oof.min() and oof.max() < 1
