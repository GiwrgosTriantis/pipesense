"""Thin wrapper over the MLflow registry. One registered model per utility: leak-risk-<utility>.

* Candidates are registered with tag lifecycle=candidate. Only the gate (or a logged manual
  promote) moves the `champion` alias.
* Each gate decision is its own MLflow run (experiment leak-risk/<utility>/gate), with metrics,
  checks, decision tags and a JSON report. The decision is also tagged on the model version.
  MLflow keeps no alias history, so these runs are the audit trail.
* Scoring resolves `champion` to a concrete version once, then loads that version. If the alias
  moves mid-job, the job still records exactly the model it used.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from pipesense.errors import MissingPrerequisiteError, TransientError

CHAMPION = "champion"


def model_name(utility: str) -> str:
    return f"leak-risk-{utility}"


def train_experiment(utility: str) -> str:
    return f"leak-risk/{utility}/train"


def gate_experiment(utility: str) -> str:
    return f"leak-risk/{utility}/gate"


@dataclass(frozen=True)
class ModelRef:
    name: str
    version: str
    run_id: str

    @property
    def uri(self) -> str:
        return f"models:/{self.name}/{self.version}"

    def as_dict(self) -> dict:
        return {"name": self.name, "version": self.version, "run_id": self.run_id, "uri": self.uri}


def client(tracking_uri: str) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def get_champion(c: MlflowClient, utility: str) -> ModelRef | None:
    name = model_name(utility)
    try:
        mv = c.get_model_version_by_alias(name, CHAMPION)
    except MlflowException as exc:
        if exc.error_code in {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}:
            return None
        raise TransientError(f"MLflow error resolving champion: {exc}") from exc
    return ModelRef(name, str(mv.version), mv.run_id)


def require_champion(c: MlflowClient, utility: str) -> ModelRef:
    ref = get_champion(c, utility)
    if ref is None:
        raise MissingPrerequisiteError(
            f"no '{CHAMPION}' model for utility '{utility}': run `pipesense run train --utility {utility}` first"
        )
    return ref


def version_tags(c: MlflowClient, ref: ModelRef) -> dict:
    return dict(c.get_model_version(ref.name, ref.version).tags)


def load_model(ref: ModelRef):
    import mlflow.sklearn

    return mlflow.sklearn.load_model(ref.uri)


def record_gate_decision(
    c: MlflowClient,
    utility: str,
    candidate: ModelRef,
    champion: ModelRef | None,
    report: dict,
    actor: str,
    job_id: str,
) -> str:
    """Log the decision as an MLflow run, tag the version and, if promoted, move the alias."""
    decision = report["decision"]
    mlflow.set_experiment(gate_experiment(utility))
    with mlflow.start_run(run_name=f"gate-{candidate.name}-v{candidate.version}") as run:
        mlflow.set_tags(
            {
                "run_type": "promotion_gate",
                "utility": utility,
                "job_id": job_id,
                "decision": decision,
                "actor": actor,
                "candidate_version": candidate.version,
                "champion_version_before": champion.version if champion else "none",
            }
        )
        for scope in ("candidate", "champion", "baseline"):
            for k, v in (report.get("metrics", {}).get(scope) or {}).items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"{scope}_{k}", float(v))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gate_report.json")
            with open(path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            mlflow.log_artifact(path)
        gate_run_id = run.info.run_id

    c.set_model_version_tag(candidate.name, candidate.version, "gate_decision", decision)
    c.set_model_version_tag(candidate.name, candidate.version, "gate_run_id", gate_run_id)
    if decision in {"PROMOTED", "MANUAL_PROMOTION"}:
        c.set_registered_model_alias(candidate.name, CHAMPION, candidate.version)
        c.set_model_version_tag(candidate.name, candidate.version, "lifecycle", "champion")
        c.set_model_version_tag(
            candidate.name, candidate.version, "promoted_at", datetime.now(timezone.utc).isoformat()
        )
        if champion and champion.version != candidate.version:
            c.set_model_version_tag(champion.name, champion.version, "lifecycle", "retired")
            c.set_model_version_tag(champion.name, champion.version, "superseded_by", candidate.version)
    return gate_run_id


def list_versions(c: MlflowClient, utility: str) -> list[dict]:
    name = model_name(utility)
    try:
        aliases = {}
        for alias, version in (c.get_registered_model(name).aliases or {}).items():
            aliases.setdefault(str(version), []).append(alias)
    except MlflowException:
        return []
    rows = []
    for mv in c.search_model_versions(f"name='{name}'"):
        rows.append(
            {
                "version": int(mv.version),
                "aliases": ",".join(aliases.get(str(mv.version), [])),
                "lifecycle": mv.tags.get("lifecycle", ""),
                "gate_decision": mv.tags.get("gate_decision", ""),
                "holdout_ap": mv.tags.get("holdout_average_precision", ""),
                "trained_job": mv.tags.get("job_id", ""),
                "run_id": mv.run_id,
            }
        )
    return sorted(rows, key=lambda r: r["version"])
