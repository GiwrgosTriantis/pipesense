"""Stage: clean the raw export and build the feature table.

Inputs:  raw/<utility>/pipes.csv, raw/<utility>/meta.json ({"as_of": "YYYY-MM-DD"})
Outputs: features.parquet, quality.json
"""

from __future__ import annotations

import io
import json
from datetime import date

import pandas as pd

from pipesense.errors import DataValidationError, MissingPrerequisiteError
from pipesense.features import build_feature_table
from pipesense.log import get_logger, kv
from pipesense.runner import StageContext
from pipesense.stages.io import write_json, write_parquet

log = get_logger(__name__)
RAW_FILES = ("pipes.csv", "meta.json")


def input_checksums(ctx: StageContext) -> dict[str, str]:
    out = {}
    for name in RAW_FILES:
        try:
            out[name] = ctx.storage.checksum(ctx.raw(name))
        except FileNotFoundError as exc:
            raise MissingPrerequisiteError(
                f"no {name} for utility '{ctx.utility}' at {ctx.storage.uri(ctx.raw(name))}"
            ) from exc
    return out


def _as_of_year(meta: dict) -> int:
    try:
        return date.fromisoformat(meta["as_of"]).year
    except (KeyError, TypeError, ValueError) as exc:
        raise DataValidationError(f"meta.json needs an ISO 'as_of' date, got {meta!r}") from exc


def run(ctx: StageContext) -> dict:
    meta = json.loads(ctx.storage.read_bytes(ctx.raw("meta.json")))
    as_of_year = _as_of_year(meta)
    try:
        raw = pd.read_csv(io.BytesIO(ctx.storage.read_bytes(ctx.raw("pipes.csv"))))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"cannot parse pipes.csv: {exc}") from exc

    p = ctx.params
    features, report = build_feature_table(
        raw,
        as_of_year=as_of_year,
        max_unknown_material_frac=float(p.get("max_unknown_material_frac", 0.05)),
        max_dropped_frac=float(p.get("max_dropped_frac", 0.2)),
        neighbour_radius_m=float(p.get("neighbour_radius_m", 500)),
    )
    write_parquet(ctx.storage, ctx.out("features.parquet"), features)
    write_json(ctx.storage, ctx.out("quality.json"), report.as_dict())
    log.info("features written", extra=kv(rows=len(features), **report.as_dict()))
    return {"rows": len(features), "as_of": meta["as_of"], "quality": report.as_dict()}
