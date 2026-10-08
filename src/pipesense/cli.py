"""Command-line entry point: the one interface every orchestrator uses.

    pipesense run score --utility fairport --job-id job-42      # a customer job
    pipesense run train --utility fairport                       # candidate model + promotion gate
    pipesense stage predict --pipeline score --utility fairport --job-id job-42
    pipesense promote --utility fairport --version 3 --reason "rollback after bad v4"
    pipesense models --utility fairport                          # registry audit view
    pipesense seed                                               # load data/raw into the artifact store

Exit codes: 0 success, 2 permanent failure (do not retry), 75 transient failure (retry).
"""

from __future__ import annotations

import argparse
import getpass
import json
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from pipesense.config import PipelineConfig, Settings
from pipesense.errors import EXIT_OK, EXIT_TRANSIENT, PipelineError
from pipesense.log import get_logger, kv, log_context, setup_logging
from pipesense.status import FileStatusStore, StatusStore
from pipesense.storage import Storage, open_storage

log = get_logger("pipesense")


def make_status_store(settings: Settings, storage: Storage) -> StatusStore:
    if settings.status_backend == "postgres":
        from pipesense.status_pg import PostgresStatusStore

        return PostgresStatusStore(settings.database_url)
    return FileStatusStore(storage)


def _default_job_id(pipeline: str) -> str:
    return f"{pipeline}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"


def cmd_run(args, settings: Settings) -> int:
    from pipesense.runner import run_pipeline

    config = PipelineConfig.load(settings.pipeline_config_path)
    storage = open_storage(settings.artifact_root)
    status = make_status_store(settings, storage)
    pipeline = args.pipeline
    job_id = args.job_id or _default_job_id(pipeline)
    only = getattr(args, "stage_name", None)
    with log_context(job_id=job_id, utility=args.utility):
        try:
            result = run_pipeline(pipeline, args.utility, job_id, settings, config, storage, status,
                                  force=args.force, only=only)
        except PipelineError as exc:
            log.error("job failed", extra=kv(error=str(exc), retryable=exc.retryable))
            status.record(job_id, args.utility, "FAILED", error=str(exc), detail={"retryable": exc.retryable})
            return exc.exit_code
        except Exception as exc:  # a bug: retryable by default, dead-lettered after max attempts
            log.exception("job crashed")
            status.record(job_id, args.utility, "FAILED", error=repr(exc), detail={"retryable": True})
            return EXIT_TRANSIENT
        if only is None:
            status.record(job_id, args.utility, "SUCCEEDED")
        print(json.dumps({"job_id": job_id, "executed": result.executed, "skipped": result.skipped,
                          "summaries": result.summaries}, indent=2, default=str))
    return EXIT_OK


def cmd_promote(args, settings: Settings) -> int:
    from pipesense import registry

    c = registry.client(settings.mlflow_tracking_uri)
    name = registry.model_name(args.utility)
    mv = c.get_model_version(name, str(args.version))
    candidate = registry.ModelRef(name, str(mv.version), mv.run_id)
    champion = registry.get_champion(c, args.utility)
    report = {"decision": "MANUAL_PROMOTION", "reason": args.reason, "actor": args.actor,
              "champion_before": champion.as_dict() if champion else None, "candidate": candidate.as_dict()}
    run_id = registry.record_gate_decision(c, args.utility, candidate, champion, report, args.actor,
                                           job_id=f"manual-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    log.info("manual promotion recorded", extra=kv(version=candidate.version, gate_run_id=run_id))
    print(json.dumps({"champion": candidate.as_dict(), "audit_run_id": run_id}, indent=2))
    return EXIT_OK


def cmd_models(args, settings: Settings) -> int:
    from pipesense import registry

    rows = registry.list_versions(registry.client(settings.mlflow_tracking_uri), args.utility)
    print(json.dumps(rows, indent=2))
    return EXIT_OK


def cmd_seed(args, settings: Settings) -> int:
    """Copy local raw exports into the artifact store (creates the bucket on S3-compatible stores)."""
    storage = open_storage(settings.artifact_root)
    if hasattr(storage, "bucket"):
        try:
            storage.client.head_bucket(Bucket=storage.bucket)
        except Exception:
            storage.client.create_bucket(Bucket=storage.bucket)
    source = Path(args.source)
    copied = 0
    for path in sorted(source.rglob("*")):
        if path.is_file():
            key = f"raw/{path.relative_to(source).as_posix()}"
            storage.write_bytes(key, path.read_bytes())
            copied += 1
    log.info("seeded raw data", extra=kv(files=copied, root=settings.artifact_root))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipesense", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run a whole pipeline, skipping stages that already completed")
    r.add_argument("pipeline", choices=["score", "train"])
    r.add_argument("--utility", required=True)
    r.add_argument("--job-id")
    r.add_argument("--force", action="store_true", help="ignore success markers and recompute")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("stage", help="run exactly one stage (what an external orchestrator calls)")
    s.add_argument("stage_name")
    s.add_argument("--pipeline", required=True, choices=["score", "train"])
    s.add_argument("--utility", required=True)
    s.add_argument("--job-id", required=True)
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_run)

    pr = sub.add_parser("promote", help="manually set the champion (rollback / override), audited")
    pr.add_argument("--utility", required=True)
    pr.add_argument("--version", required=True, type=int)
    pr.add_argument("--reason", required=True)
    pr.add_argument("--actor", default=getpass.getuser())
    pr.set_defaults(func=cmd_promote)

    m = sub.add_parser("models", help="list model versions, aliases and gate decisions")
    m.add_argument("--utility", required=True)
    m.set_defaults(func=cmd_models)

    sd = sub.add_parser("seed", help="copy data/raw into the artifact store")
    sd.add_argument("--source", default="data/raw")
    sd.set_defaults(func=cmd_seed)
    return p


def _exit_on_sigterm(signum, frame):
    # As PID 1 in a container, Python ignores SIGTERM unless a handler is installed, so
    # `docker stop` would wait and then SIGKILL. Raise instead: no marker is written for the
    # interrupted stage, so the retried job resumes there.
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
    except PipelineError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return exc.exit_code
    setup_logging(settings.log_format)
    return args.func(args, settings)


if __name__ == "__main__":
    sys.exit(main())
