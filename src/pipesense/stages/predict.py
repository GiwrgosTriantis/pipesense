"""Stage: score the utility's inventory with the champion model and record exactly which model did it.

Inputs:  prepare/features.parquet, the `champion` alias in the MLflow registry
Outputs: predictions.csv (customer-facing, one row per pipe, with model name and version on every row)
         manifest.json (lineage: model version and run, data fingerprint, image, git sha)
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np

from pipesense import model as M
from pipesense import registry
from pipesense.errors import PermanentError
from pipesense.features import LABEL
from pipesense.log import get_logger, kv
from pipesense.runner import StageContext, read_marker
from pipesense.stages.io import read_parquet, write_json

log = get_logger(__name__)


def run(ctx: StageContext) -> dict:
    df = read_parquet(ctx.storage, ctx.key("prepare", "features.parquet"))
    c = registry.client(ctx.settings.mlflow_tracking_uri)
    ref = registry.require_champion(c, ctx.utility)  # resolve the alias ONCE, then use the version
    spec = json.loads(registry.version_tags(c, ref)["feature_spec"])
    model = registry.load_model(ref)

    scores = model.predict_proba(M.model_input(df, spec))[:, 1]
    if not np.isfinite(scores).all():
        raise PermanentError(f"model {ref.uri} produced non-finite scores")

    out = df[["pipe_id", "material", "age_years", "length_m"]].copy()
    out["leak_risk"] = scores.round(4)
    out["risk_rank"] = out["leak_risk"].rank(ascending=False, method="first").astype(int)
    if LABEL in df.columns:
        # a recorded leak is already known to the utility; flagged so it can filter to unknown risk
        out["known_leak"] = df[LABEL].astype(int)
    out["model_name"] = ref.name
    out["model_version"] = ref.version
    out["job_id"] = ctx.job_id
    out = out.sort_values("risk_rank")
    ctx.storage.write_bytes(ctx.out("predictions.csv"), out.to_csv(index=False).encode())

    prepare_marker = read_marker(replace(ctx, stage="prepare")) or {}
    manifest = {
        "job_id": ctx.job_id,
        "utility": ctx.utility,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": ref.as_dict(),
        "data_fingerprint": prepare_marker.get("fingerprint"),
        "data_as_of": prepare_marker.get("summary", {}).get("as_of"),
        "rows": int(len(out)),
        "pipeline_version": ctx.config.version,
        "image_tag": ctx.settings.image_tag,
        "git_sha": ctx.settings.git_sha,
        "outputs": {"predictions": ctx.storage.uri(ctx.out("predictions.csv"))},
    }
    write_json(ctx.storage, ctx.out("manifest.json"), manifest)
    log.info("predictions written", extra=kv(rows=len(out), model_version=ref.version))
    return {"rows": int(len(out)), "model": ref.as_dict()}
