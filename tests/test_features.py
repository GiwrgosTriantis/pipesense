import numpy as np
import pandas as pd
import pytest

from pipesense.errors import DataValidationError
from pipesense.features import (
    UNKNOWN_MATERIAL,
    QualityReport,
    build_feature_table,
    drop_bad_records,
    normalize_materials,
)


def test_normalize_materials_maps_all_spellings(raw_sample):
    df, n_unknown = normalize_materials(raw_sample)
    assert set(df["material"]) == {"CI", "PVC", "STL"}
    assert n_unknown == 0


def test_normalize_materials_does_not_mutate_input(raw_sample):
    before = raw_sample["material"].tolist()
    normalize_materials(raw_sample)
    assert raw_sample["material"].tolist() == before


def test_unknown_and_missing_materials_become_other_and_are_counted(raw_sample):
    raw = raw_sample.copy()
    raw.loc[0, "material"] = "Cast-Iron"
    raw.loc[3, "material"] = None
    df, n_unknown = normalize_materials(raw)
    assert n_unknown == 2
    assert df.loc[0, "material"] == UNKNOWN_MATERIAL and df.loc[3, "material"] == UNKNOWN_MATERIAL


def test_drop_bad_records_removes_exact_duplicates_conflicts_and_bad_lengths(raw_sample):
    df, _ = normalize_materials(raw_sample)  # "Cast Iron" and "CI" rows of FP-1 become identical
    report = QualityReport()
    out = drop_bad_records(df, report)
    assert list(out["pipe_id"]) == ["FP-1", "FP-2", "FP-3"]
    assert report.exact_duplicates == 1
    assert report.conflicting_duplicate_rows == 2  # FP-5 has two different lengths: dropped, not guessed
    assert report.invalid_length == 1


def test_missing_values_are_kept_for_the_model_to_impute(raw_sample):
    raw = pd.concat([raw_sample] * 1, ignore_index=True)
    df, report = build_feature_table(raw, as_of_year=2026, max_dropped_frac=0.9)
    assert df["age_years"].isna().sum() == 1  # FP-2 has no install year
    assert report.missing_install_year == 1


def test_age_is_measured_from_the_export_date(raw_sample):
    df, _ = build_feature_table(raw_sample, as_of_year=2000, max_dropped_frac=0.9)
    row = df.set_index("pipe_id").loc["FP-1"]
    assert row["age_years"] == 50


def test_missing_required_column_is_a_validation_error(raw_sample):
    with pytest.raises(DataValidationError, match="material"):
        build_feature_table(raw_sample.drop(columns="material"), as_of_year=2026)


def test_too_many_dropped_rows_fails_the_job(raw_sample):
    with pytest.raises(DataValidationError, match="dropped"):
        build_feature_table(raw_sample, as_of_year=2026, max_dropped_frac=0.1)


def test_too_many_unknown_materials_fails_the_job(raw_sample):
    raw = raw_sample.copy()
    raw["material"] = "unobtainium"
    with pytest.raises(DataValidationError, match="unrecognised materials"):
        build_feature_table(raw, as_of_year=2026, max_dropped_frac=0.9)


def test_fairport_feature_table_matches_review_numbers(fairport_raw):
    df, report = build_feature_table(fairport_raw, as_of_year=2026)
    assert report.rows_in == 3015 and report.rows_out == 2992
    assert report.exact_duplicates == 15 and report.invalid_length == 8
    assert report.conflicting_duplicate_rows == 0 and report.unknown_material == 0
    assert df["pipe_id"].is_unique
    assert {"dist_to_center_m", "neighbours_500m", "x_m", "y_m", "age_years", "log_length"} <= set(df.columns)
    # derived UTM zone for Fairport (~94.8W) is zone 15: x within the zone's false-easting range
    assert df["x_m"].between(166_000, 834_000).all()
    assert np.isfinite(df["dist_to_center_m"]).all()
