"""The only tests existing. They currently run on demand."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from prepare_data import drop_bad_records, impute_missing, normalize_materials


def _sample_df():
    return pd.DataFrame(
        {
            "pipe_id": ["FP-1", "FP-1", "FP-2", "FP-3", "FP-4"],
            "material": ["Cast Iron", "cast iron", "CI", "pvc", "PVC"],
            "install_year": [1950.0, 1950.0, None, 1990.0, 1995.0],
            "diameter_mm": [150.0, 150.0, 200.0, None, 200.0],
            "length_m": [100.0, 100.0, 80.0, 50.0, -1.0],
        }
    )


def test_normalize_materials_maps_spellings():
    df = normalize_materials(_sample_df())
    assert set(df["material"]) == {"CI", "PVC"}


def test_drop_bad_records_removes_duplicates_and_bad_lengths():
    df = drop_bad_records(_sample_df())
    assert list(df["pipe_id"]) == ["FP-1", "FP-2", "FP-3"]


def test_impute_missing_fills_with_material_median():
    df = impute_missing(normalize_materials(_sample_df()))
    assert df["install_year"].notna().all()
    assert df["diameter_mm"].notna().all()
