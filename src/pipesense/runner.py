"""Stage runner: runs a pipeline's stages in order and skips the ones that already finished.

A stage's whole contract is (utility, job_id, config) -> objects under
runs/<utility>/<job_id>/<stage>/. When those objects are complete, the stage writes _SUCCESS.json
LAST. That marker is the commit point: a stage that dies midway leaves no marker and reruns, and
partial files are simply overwritten.

The marker stores a fingerprint of everything the output depends on: pipeline version, the
stage's config section, the checksums of its external inputs and the fingerprints of its upstream
stages. A marker counts only if its fingerprint matches, so a changed raw file or config
re-runs that stage and everything downstream of it, and nothing else.

Nothing here is tied to an orchestrator. `run_pipeline(..., only="train")` runs one stage after
checking that its upstream markers are valid. That is the entry point an Airflow task, a
Step Functions state or an AWS Batch job would call.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from pipesense.config import PipelineConfig, Settings
from pipesense.errors import MissingPrerequisiteError, PermanentError, PipelineError, TransientError
from pipesense.log import get_logger, kv, log_context
from pipesense.status import StatusStore
from pipesense.storage import Storage

log = get_logger(__name__)
MARKER = "_SUCCESS.json"


@dataclass(frozen=True)
class StageContext:
    utility: str
    job_id: str
    stage: str
    storage: Storage
    settings: Settings
    config: PipelineConfig

    def key(self, stage: str, name: str) -> str:
        return f"runs/{self.utility}/{self.job_id}/{stage}/{name}"

    def out(self, name: str) -> str:
        return self.key(self.stage, name)

    def raw(self, name: str) -> str:
        return f"raw/{self.utility}/{name}"

    @property
    def params(self) -> dict:
        return self.config.section(self.stage)


@dataclass(frozen=True)
class Stage:
    name: str
    fn: Callable[[StageContext], dict]
    upstream: tuple[str, ...] = ()
    config_sections: tuple[str, ...] = ()
    external_inputs: Callable[[StageContext], dict[str, str]] | None = None


@dataclass
class RunResult:
    pipeline: str
    utility: str
    job_id: str
    executed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    summaries: dict[str, dict] = field(default_factory=dict)


def _registry() -> dict[str, Stage]:
    from pipesense.stages import STAGES

    return STAGES


def _pipelines() -> dict[str, list[str]]:
    from pipesense.stages import PIPELINES

    return PIPELINES


def fingerprint(stage: Stage, ctx: StageContext, upstream_fps: dict[str, str]) -> str:
    payload = {
        "stage": stage.name,
        "pipeline_version": ctx.config.version,
        "config": {s: ctx.config.fingerprint(s) for s in stage.config_sections},
        "upstream": {u: upstream_fps[u] for u in stage.upstream},
        "inputs": stage.external_inputs(ctx) if stage.external_inputs else {},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def read_marker(ctx: StageContext) -> dict | None:
    try:
        return json.loads(ctx.storage.read_bytes(ctx.out(MARKER)))
    except FileNotFoundError:
        return None


def run_pipeline(
    pipeline: str,
    utility: str,
    job_id: str,
    settings: Settings,
    config: PipelineConfig,
    storage: Storage,
    status: StatusStore,
    force: bool = False,
    only: str | None = None,
    fault_stage: str | None = None,
    attempt: int | None = None,
) -> RunResult:
    pipelines, stages = _pipelines(), _registry()
    if pipeline not in pipelines:
        raise PermanentError(f"unknown pipeline {pipeline!r}; choose from {sorted(pipelines)}")
    order = pipelines[pipeline]
    if only is not None and only not in order:
        raise PermanentError(f"stage {only!r} is not part of pipeline {pipeline!r} ({order})")

    result = RunResult(pipeline, utility, job_id)
    fps: dict[str, str] = {}
    with log_context(job_id=job_id, utility=utility, pipeline=pipeline):
        status.record(job_id, utility, "RUNNING", attempt=attempt, detail={"pipeline": pipeline})
        for name in order:
            stage = stages[name]
            ctx = StageContext(utility, job_id, name, storage, settings, config)
            with log_context(stage=name):
                fp = fingerprint(stage, ctx, fps)
                marker = read_marker(ctx)
                valid = marker is not None and marker.get("fingerprint") == fp
                before_target = only is not None and order.index(name) < order.index(only)
                after_target = only is not None and order.index(name) > order.index(only)

                if after_target:
                    break
                if before_target:
                    if not valid:
                        raise MissingPrerequisiteError(
                            f"stage {name!r} has no valid output for job {job_id}; run it before {only!r}"
                        )
                    fps[name] = fp
                    continue
                if valid and not force:
                    log.info("stage already complete, skipping", extra=kv(fingerprint=fp[:12]))
                    status.record(job_id, utility, "STAGE_SKIPPED", stage=name, attempt=attempt)
                    result.skipped.append(name)
                    result.summaries[name] = marker.get("summary", {})
                    fps[name] = fp
                    continue

                status.record(job_id, utility, "STAGE_STARTED", stage=name, attempt=attempt)
                log.info("stage started")
                started = time.monotonic()
                try:
                    if fault_stage == name:
                        raise TransientError(f"injected fault in stage {name!r} (dev fault injection)")
                    summary = stage.fn(ctx) or {}
                except PipelineError as exc:
                    status.record(job_id, utility, "STAGE_FAILED", stage=name, attempt=attempt, error=str(exc))
                    raise
                except Exception as exc:
                    status.record(job_id, utility, "STAGE_FAILED", stage=name, attempt=attempt, error=repr(exc))
                    raise
                duration = round(time.monotonic() - started, 2)
                storage.write_bytes(
                    ctx.out(MARKER),
                    json.dumps(
                        {
                            "stage": name,
                            "fingerprint": fp,
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                            "duration_s": duration,
                            "image_tag": settings.image_tag,
                            "git_sha": settings.git_sha,
                            "summary": summary,
                        },
                        indent=2,
                        default=str,
                    ).encode(),
                )
                log.info("stage succeeded", extra=kv(duration_s=duration))
                status.record(job_id, utility, "STAGE_SUCCEEDED", stage=name, attempt=attempt,
                              detail={"duration_s": duration})
                result.executed.append(name)
                result.summaries[name] = summary
                fps[name] = fp
    return result
