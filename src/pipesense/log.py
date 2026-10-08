"""Structured logging. Every line carries job_id / utility / stage, so logs from hundreds of
concurrent jobs stay searchable in CloudWatch or any other aggregator."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from contextlib import contextmanager
from datetime import datetime, timezone

_context: contextvars.ContextVar[dict] = contextvars.ContextVar("pipesense_log_context", default={})


@contextmanager
def log_context(**fields):
    token = _context.set({**_context.get(), **{k: v for k, v in fields.items() if v is not None}})
    try:
        yield
    finally:
        _context.reset(token)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            **_context.get(),
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ctx = " ".join(f"{k}={v}" for k, v in {**_context.get(), **getattr(record, "fields", {})}.items())
        line = f"{record.levelname:<7} {record.name}: {record.getMessage()}" + (f"  [{ctx}]" if ctx else "")
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def setup_logging(fmt: str = "json", level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("botocore", "boto3", "urllib3", "mlflow", "alembic", "matplotlib", "pyogrio", "fiona"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def kv(**fields) -> dict:
    """Usage: log.info("wrote features", extra=kv(rows=123))."""
    return {"fields": fields}
