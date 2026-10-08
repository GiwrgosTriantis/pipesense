"""Configuration.

There are two kinds of configuration, kept apart on purpose:

* Environment settings (Settings) say WHERE the pipeline runs: the artifact store, MLflow, the
  status backend and the queue. They come from environment variables, so the same image runs
  in dev and prod-like setups unchanged. AWS endpoints use boto3's standard
  AWS_ENDPOINT_URL_S3 / AWS_ENDPOINT_URL_SQS, which are simply unset on real AWS.
* Pipeline parameters (PipelineConfig) say WHAT the pipeline does: features, hyperparameters
  and gate thresholds. They live in a versioned TOML file and are part of every stage
  fingerprint, so changing them invalidates cached stage outputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipesense.errors import PermanentError

DEFAULT_PIPELINE_CONFIG = "config/pipeline.toml"


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    env: str
    artifact_root: str
    mlflow_tracking_uri: str
    pipeline_config_path: str
    status_backend: str = "file"
    database_url: str | None = None
    queue_name: str = "pipesense-jobs"
    dlq_name: str = "pipesense-jobs-dlq"
    max_attempts: int = 3
    visibility_timeout_s: int = 120
    allow_fault_injection: bool = False
    log_format: str = "json"
    image_tag: str = "unknown"
    git_sha: str = "unknown"

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "Settings":
        e = os.environ if environ is None else environ
        missing = [k for k in ("PIPESENSE_ARTIFACT_ROOT", "MLFLOW_TRACKING_URI") if not e.get(k)]
        if missing:
            raise PermanentError(f"missing required environment variables: {', '.join(missing)}")
        settings = cls(
            env=e.get("PIPESENSE_ENV", "dev"),
            artifact_root=e["PIPESENSE_ARTIFACT_ROOT"],
            mlflow_tracking_uri=e["MLFLOW_TRACKING_URI"],
            pipeline_config_path=e.get("PIPESENSE_PIPELINE_CONFIG", DEFAULT_PIPELINE_CONFIG),
            status_backend=e.get("PIPESENSE_STATUS_BACKEND", "file"),
            database_url=e.get("PIPESENSE_DATABASE_URL") or None,
            queue_name=e.get("PIPESENSE_QUEUE_NAME", "pipesense-jobs"),
            dlq_name=e.get("PIPESENSE_DLQ_NAME", "pipesense-jobs-dlq"),
            max_attempts=int(e.get("PIPESENSE_MAX_ATTEMPTS", "3")),
            visibility_timeout_s=int(e.get("PIPESENSE_VISIBILITY_TIMEOUT_S", "120")),
            allow_fault_injection=_bool(e.get("PIPESENSE_ALLOW_FAULT_INJECTION")),
            log_format=e.get("PIPESENSE_LOG_FORMAT", "json"),
            image_tag=e.get("PIPESENSE_IMAGE_TAG", "unknown"),
            git_sha=e.get("PIPESENSE_GIT_SHA", "unknown"),
        )
        if settings.status_backend not in {"file", "postgres"}:
            raise PermanentError(f"unknown PIPESENSE_STATUS_BACKEND={settings.status_backend!r}")
        if settings.status_backend == "postgres" and not settings.database_url:
            raise PermanentError("PIPESENSE_STATUS_BACKEND=postgres needs PIPESENSE_DATABASE_URL")
        return settings


@dataclass(frozen=True)
class PipelineConfig:
    """Parsed config/pipeline.toml. `section()` hands each stage only its own block."""

    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        p = Path(path)
        if not p.exists():
            raise PermanentError(f"pipeline config not found: {p}")
        with p.open("rb") as f:
            return cls(raw=tomllib.load(f))

    @property
    def version(self) -> str:
        """Bumped by hand when stage logic changes meaning; invalidates cached outputs."""
        return str(self.raw.get("pipeline", {}).get("version", "0"))

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name, {}))

    def fingerprint(self, name: str) -> str:
        payload = json.dumps(self.section(name), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]
