"""Stage 3: score every pipe with the trained model and export the risk table.

Reads features.gpkg and model.pkl from the shared work dir and writes
predictions.csv (one risk score per pipe, sorted highest first).
"""

import os
import pickle

import geopandas as gpd

from common import get_work_dir, log_time, print_banner, write_status
from train_model import build_matrix

TOP_N = 25


@log_time("predict")
def run():
    """Run the full prediction stage."""
    work_dir = get_work_dir()
    assert os.path.exists(os.path.join(work_dir, "model.pkl")), "model.pkl not found — run train_model.py first"
    print_banner("Scoring all pipes")

    gdf = gpd.read_file(os.path.join(work_dir, "features.gpkg"))
    with open(os.path.join(work_dir, "model.pkl"), "rb") as f:
        bundle = pickle.load(f)

    x, _, columns = build_matrix(gdf)
    x = x.reindex(columns=bundle["columns"], fill_value=0)
    gdf["leak_risk"] = bundle["model"].predict_proba(x)[:, 1].round(4)

    out = gdf[["pipe_id", "material", "age_years", "length_m", "leak_risk"]].sort_values(
        "leak_risk", ascending=False
    )
    out_path = os.path.join(work_dir, "predictions.csv")
    out.to_csv(out_path, index=False)

    print(f"Wrote {len(out)} predictions to {out_path}")
    print(f"Top {TOP_N} riskiest pipes:")
    print(out.head(TOP_N).to_string(index=False))


if __name__ == "__main__":
    try:
        run()
        write_status("SUCCESS")
    except Exception as exc:
        write_status("FAILED", error=str(exc))
        raise
