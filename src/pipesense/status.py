"""Job status: an append-only event log plus a current state per job.

States: QUEUED -> RUNNING -> (RETRYING -> RUNNING)* -> SUCCEEDED | FAILED | DEAD_LETTERED.
Stage-level events (STAGE_STARTED / STAGE_SKIPPED / STAGE_SUCCEEDED / STAGE_FAILED) are kept in
the same log.

Backends:
* file (dev): runs/<utility>/<job_id>/status.json in the artifact store, written atomically.
  During migration it is also what the legacy NFS monitor can keep polling.
* postgres (prod-like): job_events table plus a job_status view (see status_pg.py).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Protocol

from pipesense.storage import Storage

TERMINAL = {"SUCCEEDED", "FAILED", "DEAD_LETTERED"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class StatusStore(Protocol):
    def record(
        self,
        job_id: str,
        utility: str,
        state: str,
        stage: str | None = None,
        attempt: int | None = None,
        error: str | None = None,
        detail: dict | None = None,
    ) -> None: ...


class FileStatusStore:
    def __init__(self, storage: Storage):
        self.storage = storage

    @staticmethod
    def key(utility: str, job_id: str) -> str:
        return f"runs/{utility}/{job_id}/status.json"

    def get(self, utility: str, job_id: str) -> dict | None:
        try:
            return json.loads(self.storage.read_bytes(self.key(utility, job_id)))
        except FileNotFoundError:
            return None

    def record(self, job_id, utility, state, stage=None, attempt=None, error=None, detail=None) -> None:
        doc = self.get(utility, job_id) or {"job_id": job_id, "utility": utility, "events": []}
        event = {"at": now(), "state": state, "stage": stage, "attempt": attempt, "error": error, "detail": detail}
        doc["events"].append({k: v for k, v in event.items() if v is not None})
        if not state.startswith("STAGE_"):
            doc["state"] = state
            doc["error"] = error
        doc["stage"] = stage or doc.get("stage")
        if attempt is not None:
            doc["attempt"] = attempt
        doc["updated_at"] = event["at"]
        self.storage.write_bytes(self.key(utility, job_id), json.dumps(doc, indent=2).encode())


class NullStatusStore:
    def record(self, *args, **kwargs) -> None:
        return None
