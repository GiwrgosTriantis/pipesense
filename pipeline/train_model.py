"""Stage 2: train the leak-risk model on the prepared feature table.

Reads features.gpkg from the shared work dir, trains a gradient boosting
classifier and saves model.pkl + a feature importance plot for the analysts.
"""

import os
import pickle
import time

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from common import get_work_dir, log_time, print_banner, write_status

RANDOM_STATE = 7
NUMERIC_FEATURES = [
    "age_years",
    "diameter_mm",
    "log_length",
    "num_breaks_5yr",
    "dist_to_center_m",
    "neighbours_500m",
]
CATEGORICAL_FEATURES = ["material", "soil_type", "pressure_zone"]
TARGET = "has_leak"


def build_matrix(gdf):
    """One-hot encode categoricals and return X, y and the final column list."""
    assert TARGET in gdf.columns, f"{TARGET} column is missing from the feature table"
    import pandas as pd

    x = pd.get_dummies(gdf[NUMERIC_FEATURES + CATEGORICAL_FEATURES], columns=CATEGORICAL_FEATURES)
    print(x)
    return x, gdf[TARGET], list(x.columns)


def plot_importances(model, columns, out_path):
    """Save a horizontal bar chart of the top 15 feature importances."""
    import pandas as pd

    imp = pd.Series(model.feature_importances_, index=columns).nlargest(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.sort_values().plot.barh(ax=ax)
    ax.set_title("Leak model — feature importances")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


@log_time("train_model")
def run():
    """Run the full training stage."""
    work_dir = get_work_dir()
    features_path = os.path.join(work_dir, "features.gpkg")
    assert os.path.exists(features_path), f"{features_path} not found — run prepare_data.py first"
    print_banner(f"Training on {features_path}")
    gdf = gpd.read_file(features_path)

    x, y, columns = build_matrix(gdf)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    model = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=RANDOM_STATE)
    model.fit(x_train, y_train)

    scores = model.predict_proba(x_test)[:, 1]
    print(f"ROC AUC: {roc_auc_score(y_test, scores):.3f}")
    print(f"Average precision: {average_precision_score(y_test, scores):.3f}")

    with open(os.path.join(work_dir, "model.pkl"), "wb") as f:
        pickle.dump({"model": model, "columns": columns}, f)
    plot_importances(model, columns, os.path.join(work_dir, "feature_importances.png"))
    print(f"Saved model.pkl and feature_importances.png to {work_dir}")


if __name__ == "__main__":
    start = time.time()
    try:
        run()
    except Exception as exc:
        elapsed = time.time() - start
        print(f"Training failed after {elapsed:.0f}s")
        write_status("FAILED", error=str(exc))
        raise
