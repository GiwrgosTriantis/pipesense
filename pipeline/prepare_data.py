"""Stage 1: clean the raw pipe inventory and build the model feature table.

Reads data/pipes.csv (raw GIS export from the utility), cleans it, adds
geospatial features and writes features.gpkg into the shared work dir for the
training and prediction stages.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd

from common import get_data_dir, get_work_dir, log_time, print_banner, read_table, write_status

CURRENT_YEAR = 2026
METRIC_CRS = "EPSG:26915"  # UTM 15N
NEIGHBOUR_RADIUS_M = 500

MATERIAL_MAP = {
    "cast iron": "CI",
    "asbestos cement": "AC",
    "steel": "STL",
    "ductile iron": "DI",
    "pvc": "PVC",
    "hdpe": "HDPE",
}


def normalize_materials(df):
    """Map the many spellings coming out of the GIS export to canonical codes."""
    assert "material" in df.columns, "material column is missing from the raw export"
    material = df["material"].str.strip().str.lower()
    df["material"] = material.map(MATERIAL_MAP).fillna(material.str.upper())
    return df


def drop_bad_records(df):
    """Remove duplicates and physically impossible rows."""
    assert not df.empty, "raw pipe table is empty"
    n_before = len(df)
    df = df.drop_duplicates(subset="pipe_id", keep="first")
    df = df[df["length_m"] > 0]
    print(f"Dropped {n_before - len(df)} duplicate/invalid rows")
    return df


def impute_missing(df):
    """Fill missing install years and diameters with per-material medians."""
    assert df["material"].notna().all(), "materials must be normalized before imputation"
    for col in ["install_year", "diameter_mm"]:
        df[col] = df[col].fillna(df.groupby("material")[col].transform("median"))
    return df


def add_geo_features(df):
    """Build a GeoDataFrame and add distance-to-center and pipe-density features."""
    assert {"lon", "lat"}.issubset(df.columns), "lon/lat columns are missing"
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
    ).to_crs(METRIC_CRS)

    center = gdf.geometry.union_all().centroid
    gdf["dist_to_center_m"] = gdf.geometry.distance(center).round(1)

    buffers = gpd.GeoDataFrame(
        {"pipe_id": gdf["pipe_id"]}, geometry=gdf.geometry.buffer(NEIGHBOUR_RADIUS_M), crs=METRIC_CRS
    )
    joined = gpd.sjoin(buffers, gdf[["geometry"]], predicate="intersects")
    gdf["neighbours_500m"] = gdf["pipe_id"].map(joined.groupby("pipe_id").size() - 1)
    return gdf


def add_model_features(gdf):
    """Derive the numeric features the model trains on."""
    gdf["age_years"] = CURRENT_YEAR - gdf["install_year"]
    gdf["log_length"] = np.log1p(gdf["length_m"])
    return gdf


@log_time("prepare_data")
def run():
    """Run the full preparation stage."""
    raw_path = os.path.join(get_data_dir(), "pipes.csv")
    print_banner(f"Preparing data from {raw_path}")
    df = read_table(raw_path)

    df = normalize_materials(df)
    df = drop_bad_records(df)
    df = impute_missing(df)
    gdf = add_geo_features(df)
    gdf = add_model_features(gdf)

    out_path = os.path.join(get_work_dir(), "features.gpkg")
    gdf.to_file(out_path, driver="GPKG")
    print(f"Wrote {len(gdf)} pipes to {out_path}")


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        write_status("FAILED", error=str(exc))
        raise
