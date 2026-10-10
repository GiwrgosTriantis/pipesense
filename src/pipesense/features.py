"""Cleaning and feature engineering for one utility's pipe inventory.

These are pure functions on DataFrames: no I/O, no environment variables, no global state.
Changes from the starter kit (each one is a review finding in DECISIONS.md):
* validation raises DataValidationError instead of `assert` (F8)
* pipe age is measured from the export's `as_of` date, not a hard-coded CURRENT_YEAR
* the metric CRS is derived per utility (estimate_utm_crs), not hard-coded to UTM 15N
* unknown materials map to OTHER and are counted; too many fail the job instead of passing silently
* values outside plausible bounds are counted; too many in one column fail the job as a likely unit
  mismatch (inches instead of mm, km instead of m), instead of producing confident nonsense
* conflicting duplicates (same pipe_id, different values) are dropped and counted, not resolved
  by row order
* missing values are left as NaN. Imputation is fitted inside the model pipeline, so its state
  is versioned with the model (M5), and a missing-value indicator keeps the signal (M6).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd

from pipesense.errors import DataValidationError

MATERIAL_MAP = {
    "cast iron": "CI",
    "ci": "CI",
    "asbestos cement": "AC",
    "ac": "AC",
    "steel": "STL",
    "stl": "STL",
    "ductile iron": "DI",
    "di": "DI",
    "pvc": "PVC",
    "hdpe": "HDPE",
}
UNKNOWN_MATERIAL = "OTHER"

# What a unit mismatch in each column usually looks like, for the error message.
UNIT_HINTS = {
    "diameter_mm": "diameter_mm must be millimetres: inches are ~25x smaller, metres ~1000x smaller",
    "length_m": "length_m must be metres: kilometres are 1000x smaller",
    "install_year": "install_year must be a 4-digit calendar year, not an age or a 2-digit year",
    "lat": "lat/lon must be WGS84 degrees, not projected coordinates in metres",
    "lon": "lat/lon must be WGS84 degrees, not projected coordinates in metres",
}

REQUIRED_COLUMNS = [
    "pipe_id",
    "material",
    "install_year",
    "diameter_mm",
    "length_m",
    "lon",
    "lat",
    "pressure_zone",
    "soil_type",
    "num_breaks_5yr",
]
LABEL = "has_leak"


@dataclass
class QualityReport:
    rows_in: int = 0
    exact_duplicates: int = 0
    conflicting_duplicate_rows: int = 0
    invalid_length: int = 0
    missing_coordinates: int = 0
    unknown_material: int = 0
    missing_install_year: int = 0
    missing_diameter: int = 0
    out_of_range: dict = field(default_factory=dict)
    rows_out: int = 0

    @property
    def dropped_frac(self) -> float:
        return 0.0 if self.rows_in == 0 else 1 - self.rows_out / self.rows_in

    def as_dict(self) -> dict:
        return {**asdict(self), "dropped_frac": round(self.dropped_frac, 4)}


def validate_schema(df: pd.DataFrame) -> None:
    if df.empty:
        raise DataValidationError("raw pipe table is empty")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"raw export is missing required columns: {missing}")


def check_plausible_ranges(df: pd.DataFrame, bounds: dict, max_frac: float) -> dict[str, int]:
    """Count values outside [low, high] per column. Fail if any column has more than max_frac of them.

    Missing values are not counted here (they have their own handling). A few implausible values are
    tolerated and reported; many of them in one column almost always mean the column has the wrong unit.
    """
    counts, problems = {}, []
    for col, (low, high) in bounds.items():
        values = pd.to_numeric(df[col], errors="coerce")
        present = values.notna()
        outside = present & ~values.between(low, high)
        counts[col] = int(outside.sum())
        frac = outside.sum() / max(int(present.sum()), 1)
        if frac > max_frac:
            hint = UNIT_HINTS.get(col, f"check the unit of {col}")
            problems.append(f"{col}: {int(outside.sum())} of {int(present.sum())} values ({frac:.1%}) outside "
                            f"[{low:g}, {high:g}], median {values[present].median():g}. Likely a unit mismatch: {hint}")
    if problems:
        raise DataValidationError(
            f"implausible values (limit {max_frac:.0%} per column): " + "; ".join(problems))
    return counts


def normalize_materials(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Map the GIS export's spellings to canonical codes. Returns (copy, number of unknown rows)."""
    df = df.copy()
    material = df["material"].astype("string").str.strip().str.lower()
    mapped = material.map(MATERIAL_MAP)
    n_unknown = int(mapped.isna().sum())
    df["material"] = mapped.fillna(UNKNOWN_MATERIAL).astype(str)
    return df, n_unknown


def drop_bad_records(df: pd.DataFrame, report: QualityReport | None = None) -> pd.DataFrame:
    """Drop exact duplicates, pipes with conflicting duplicate records, and impossible rows."""
    report = report if report is not None else QualityReport()
    n = len(df)
    df = df.drop_duplicates()
    report.exact_duplicates = n - len(df)

    conflicting = df["pipe_id"].duplicated(keep=False)
    report.conflicting_duplicate_rows = int(conflicting.sum())
    df = df[~conflicting]

    bad_length = ~(df["length_m"] > 0)  # also catches NaN
    report.invalid_length = int(bad_length.sum())
    df = df[~bad_length]

    no_coords = df["lon"].isna() | df["lat"].isna()
    report.missing_coordinates = int(no_coords.sum())
    return df[~no_coords].reset_index(drop=True)


def add_geo_features(df: pd.DataFrame, neighbour_radius_m: float = 500.0) -> pd.DataFrame:
    """Distance to the network centroid and pipe density, in the utility's own UTM zone.

    Both features depend on the whole inventory. That is consistent because every job processes
    a utility's full inventory, but see DECISIONS.md (known debt) for network growth.
    """
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
    gdf = gdf.to_crs(gdf.estimate_utm_crs())

    center = gdf.geometry.union_all().centroid
    gdf["dist_to_center_m"] = gdf.geometry.distance(center).round(1)

    buffers = gpd.GeoDataFrame(
        {"pipe_id": gdf["pipe_id"]}, geometry=gdf.geometry.buffer(neighbour_radius_m), crs=gdf.crs
    )
    joined = gpd.sjoin(buffers, gdf[["geometry"]], predicate="intersects")
    gdf["neighbours_500m"] = gdf["pipe_id"].map(joined.groupby("pipe_id").size() - 1).fillna(0).astype(int)

    # metric coordinates are kept for spatial blocking in training
    gdf["x_m"] = gdf.geometry.x.round(1)
    gdf["y_m"] = gdf.geometry.y.round(1)
    return pd.DataFrame(gdf.drop(columns="geometry"))


def add_model_features(df: pd.DataFrame, as_of_year: int) -> pd.DataFrame:
    df = df.copy()
    df["age_years"] = as_of_year - df["install_year"]  # NaN stays NaN; imputed inside the model
    df["log_length"] = np.log1p(df["length_m"])
    return df


def build_feature_table(
    raw: pd.DataFrame,
    as_of_year: int,
    max_unknown_material_frac: float = 0.05,
    max_dropped_frac: float = 0.2,
    neighbour_radius_m: float = 500.0,
    bounds: dict | None = None,
    max_out_of_range_frac: float = 0.02,
) -> tuple[pd.DataFrame, QualityReport]:
    validate_schema(raw)
    report = QualityReport(rows_in=len(raw))
    if bounds:
        bounds = {col: tuple(b) for col, b in bounds.items()}
        if "install_year" in bounds:  # a pipe cannot be installed after the export was taken
            low, high = bounds["install_year"]
            bounds["install_year"] = (low, min(high, as_of_year))
        report.out_of_range = check_plausible_ranges(raw, bounds, max_out_of_range_frac)

    df, n_unknown = normalize_materials(raw)
    report.unknown_material = n_unknown
    df = drop_bad_records(df, report)
    report.rows_out = len(df)
    report.missing_install_year = int(df["install_year"].isna().sum())
    report.missing_diameter = int(df["diameter_mm"].isna().sum())

    if report.rows_out == 0:
        raise DataValidationError("no valid pipes left after cleaning")
    if report.dropped_frac > max_dropped_frac:
        raise DataValidationError(
            f"cleaning dropped {report.dropped_frac:.1%} of rows (limit {max_dropped_frac:.0%}): {report.as_dict()}"
        )
    if n_unknown / len(raw) > max_unknown_material_frac:
        raise DataValidationError(
            f"{n_unknown} rows ({n_unknown / len(raw):.1%}) have unrecognised materials "
            f"(limit {max_unknown_material_frac:.0%}); extend MATERIAL_MAP or fix the export"
        )
    if LABEL in df.columns and not df[LABEL].dropna().isin([0, 1]).all():
        raise DataValidationError(f"{LABEL} must be 0/1")

    df = add_geo_features(df, neighbour_radius_m)
    df = add_model_features(df, as_of_year)
    return df, report
