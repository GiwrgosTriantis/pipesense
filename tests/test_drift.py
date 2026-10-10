import json

import numpy as np
import pandas as pd

from pipesense import drift
from pipesense import model as M

from conftest import synthetic_features

NUMERIC = ["age_years", "diameter_mm", "log_length", "num_breaks_5yr", "dist_to_center_m", "neighbours_500m"]
CATEGORICAL = ["material", "soil_type", "pressure_zone"]
SPEC = {"numeric_features": NUMERIC, "categorical_features": CATEGORICAL}


def _x(df):
    return M.model_input(df, SPEC)


def _profile(df):
    return drift.reference_profile(_x(df), NUMERIC, CATEGORICAL, bins=10)


def test_profile_is_json_and_fractions_sum_to_one(features_df):
    profile = json.loads(json.dumps(_profile(features_df)))
    assert profile["rows"] == len(features_df) and set(profile["features"]) == set(NUMERIC + CATEGORICAL)
    age = profile["features"]["age_years"]
    assert len(age["edges"]) == 9 and len(age["fractions"]) == 11  # 10 bins + missing
    assert abs(age["fractions"][-1] - 0.05) < 1e-9                 # the fixture blanks 5% of ages
    for ref in profile["features"].values():
        total = sum(ref["fractions"]) if ref["type"] == "numeric" else sum(ref["fractions"].values())
        assert abs(total - 1) < 1e-9


def test_psi_is_zero_on_the_reference_and_small_on_a_fresh_sample(features_df):
    profile = _profile(features_df)
    same = drift.psi_report(profile, _x(features_df), threshold=0.2)
    assert max(same["psi"].values()) < 1e-6 and same["status"] == "ok" and same["drifted"] == []
    fresh = drift.psi_report(profile, _x(synthetic_features(seed=1)), threshold=0.2)
    assert max(fresh["psi"].values()) < 0.1 and fresh["status"] == "ok"


def test_shifted_numeric_feature_is_flagged(features_df):
    shifted = features_df.copy()
    shifted["age_years"] = shifted["age_years"] + 30
    report = drift.psi_report(_profile(features_df), _x(shifted), threshold=0.2)
    assert report["status"] == "drift" and report["drifted"] == ["age_years"]
    assert report["psi"]["age_years"] > 0.2


def test_more_missing_values_count_as_drift(features_df):
    gappy = features_df.copy()
    gappy.loc[gappy.sample(frac=0.4, random_state=0).index, "diameter_mm"] = np.nan
    report = drift.psi_report(_profile(features_df), _x(gappy), threshold=0.2)
    assert "diameter_mm" in report["drifted"]


def test_unseen_categories_count_as_drift(features_df):
    renamed = features_df.copy()
    renamed.loc[renamed["material"] == "CI", "material"] = "Cast-Iron"  # a new export spelling
    report = drift.psi_report(_profile(features_df), _x(renamed), threshold=0.2)
    assert report["drifted"] == ["material"]


def test_constant_and_discrete_features_do_not_break_binning():
    df = pd.DataFrame({"a": [5.0] * 50, "b": [0.0] * 45 + [1.0] * 5, "c": ["x"] * 50})
    profile = drift.reference_profile(df, ["a", "b"], ["c"], bins=10)
    assert profile["features"]["a"]["edges"] == [5.0]
    report = drift.psi_report(profile, df, threshold=0.2)
    assert all(v < 1e-6 for v in report["psi"].values())


def test_psi_matches_the_formula():
    expected, actual = [0.5, 0.5], [0.25, 0.75]
    by_hand = (0.25 - 0.5) * np.log(0.25 / 0.5) + (0.75 - 0.5) * np.log(0.75 / 0.5)
    assert abs(drift.psi(expected, actual) - by_hand) < 1e-12
