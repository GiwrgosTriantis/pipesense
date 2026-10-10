"""Stage: train a candidate model and register it. It does NOT serve it; only the gate can.

Inputs:  prepare/features.parquet (must contain has_leak)
Outputs: model.json (registered version + metrics), metrics.json, plus an MLflow run with the model
         (a CrossFitModel: the full model, one model per spatial CV fold and the block->fold map),
         spatial-CV metrics, holdout metrics, age-rule baseline metrics, permutation importances and
         the drift reference (drift/reference.json) that scoring jobs compare their data against.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from pipesense import drift
from pipesense import model as M
from pipesense import registry
from pipesense.errors import DataValidationError
from pipesense.features import LABEL
from pipesense.log import get_logger, kv
from pipesense.runner import StageContext, read_marker
from pipesense.stages.io import read_parquet, write_json

log = get_logger(__name__)

# Models are saved with skops, not pickle: loading cannot execute arbitrary code. skops still
# makes us list the non-default types the model contains. These were reviewed. Tree is accepted
# because models are only ever loaded from our own registry, written by this stage.
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.compose._column_transformer._RemainderColsList",
    "sklearn.tree._tree.Tree",
    "pipesense.model.CrossFitModel",
]


def _importances(model, x: pd.DataFrame, y: np.ndarray, out_dir: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imp = permutation_importance(model, x, y, scoring="average_precision", n_repeats=5, random_state=0)
    table = pd.DataFrame(
        {"feature": x.columns, "ap_drop_mean": imp.importances_mean, "ap_drop_std": imp.importances_std}
    ).sort_values("ap_drop_mean", ascending=False)
    table.to_csv(os.path.join(out_dir, "permutation_importance.csv"), index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    t = table.iloc[::-1]
    ax.barh(t["feature"], t["ap_drop_mean"], xerr=t["ap_drop_std"])
    ax.set_xlabel("drop in holdout average precision when shuffled")
    ax.set_title("Leak model: permutation importance (holdout)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "permutation_importance.png"))
    plt.close(fig)
    return table.iloc[0]["feature"]


def run(ctx: StageContext) -> dict:
    params = ctx.params
    df = read_parquet(ctx.storage, ctx.key("prepare", "features.parquet"))
    if LABEL not in df.columns or df[LABEL].isna().any():
        raise DataValidationError(f"training needs a complete '{LABEL}' column for utility '{ctx.utility}'")

    hold = M.holdout_mask(df, ctx.utility, float(params["block_size_m"]), int(params["holdout_pct"]))
    train_df, hold_df = df[~hold], df[hold]
    if hold_df[LABEL].sum() < 10 or train_df[LABEL].sum() < 10:
        raise DataValidationError("too few leaking pipes to train and evaluate (need >= 10 on each side)")

    # The spatial CV folds are also the scoring models: each serves the training blocks it held out.
    crossfit, oof = M.fit_cross_fitted(train_df, params)
    cv = {"cv_average_precision": float(average_precision_score(M.labels(train_df), oof)),
          "cv_folds": len(crossfit.fold_models)}
    model = crossfit.full
    x_train = M.model_input(train_df, params)
    reference = drift.reference_profile(x_train, params["numeric_features"], params["categorical_features"],
                                        bins=int(params.get("drift_bins", 10)))

    x_hold, y_hold = M.model_input(hold_df, params), M.labels(hold_df)
    scores = model.predict_proba(x_hold)[:, 1]
    k = float(params["top_k_frac"])
    holdout = M.evaluate(y_hold, scores, k)
    baseline = M.evaluate(y_hold, M.baseline_scores(hold_df), k)
    log.info("candidate evaluated", extra=kv(holdout_ap=round(holdout["average_precision"], 4),
                                              baseline_ap=round(baseline["average_precision"], 4), **cv))

    prepare_fp = (read_marker(replace(ctx, stage="prepare")) or {}).get("fingerprint", "unknown")
    feature_spec = {"numeric_features": params["numeric_features"], "categorical_features": params["categorical_features"]}
    name = registry.model_name(ctx.utility)
    c = registry.client(ctx.settings.mlflow_tracking_uri)
    mlflow.set_experiment(registry.train_experiment(ctx.utility))
    with mlflow.start_run(run_name=f"train-{ctx.job_id}") as run, tempfile.TemporaryDirectory() as d:
        mlflow.set_tags({"run_type": "train", "utility": ctx.utility, "job_id": ctx.job_id,
                         "data_fingerprint": prepare_fp, "git_sha": ctx.settings.git_sha,
                         "image_tag": ctx.settings.image_tag, "pipeline_version": ctx.config.version,
                         "holdout_spec": M.holdout_spec(params)})
        mlflow.log_params({k_: (json.dumps(v) if isinstance(v, list) else v) for k_, v in params.items()})
        mlflow.log_params({"n_train": len(train_df), "n_holdout": len(hold_df)})
        mlflow.log_metrics({**{f"holdout_{k_}": v for k_, v in holdout.items()},
                            **{f"baseline_{k_}": v for k_, v in baseline.items()},
                            "cv_average_precision": cv["cv_average_precision"]})
        top_feature = _importances(model, x_hold, y_hold, d)
        mlflow.log_artifacts(d, artifact_path="evaluation")
        mlflow.log_dict(reference, registry.DRIFT_REFERENCE)
        info = mlflow.sklearn.log_model(
            crossfit,
            name="model",
            registered_model_name=name,
            serialization_format="skops",
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
            signature=infer_signature(x_hold, scores),
            input_example=x_hold.head(3),
        )
        run_id = run.info.run_id

    ref = registry.ModelRef(name, str(info.registered_model_version), run_id)
    for tag, value in {
        "lifecycle": "candidate", "utility": ctx.utility, "job_id": ctx.job_id,
        "holdout_spec": M.holdout_spec(params), "holdout_average_precision": f"{holdout['average_precision']:.4f}",
        "feature_spec": json.dumps(feature_spec), "data_fingerprint": prepare_fp,
        "git_sha": ctx.settings.git_sha, "pipeline_version": ctx.config.version,
    }.items():
        c.set_model_version_tag(name, ref.version, tag, value)

    result = {"model": ref.as_dict(), "holdout": holdout, "baseline": baseline, "cv": cv, "top_feature": top_feature}
    write_json(ctx.storage, ctx.out("metrics.json"), result)
    write_json(ctx.storage, ctx.out("model.json"), ref.as_dict())
    return result

