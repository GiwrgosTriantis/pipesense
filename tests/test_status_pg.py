"""Runs against a real Postgres when PIPESENSE_TEST_DATABASE_URL is set; skipped otherwise."""

import os
import uuid

import pytest

DSN = os.environ.get("PIPESENSE_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="set PIPESENSE_TEST_DATABASE_URL to run Postgres tests")


def test_job_view_tracks_state_last_stage_and_history():
    from pipesense.status_pg import PostgresStatusStore
    from pipesense.status_server import render

    store = PostgresStatusStore(DSN)
    job = f"job-{uuid.uuid4().hex[:8]}"
    store.record(job, "fairport", "QUEUED", detail={"pipeline": "score"})
    store.record(job, "fairport", "RUNNING", attempt=1)
    store.record(job, "fairport", "STAGE_STARTED", stage="prepare", attempt=1)
    store.record(job, "fairport", "STAGE_FAILED", stage="prepare", attempt=1, error="boom")
    store.record(job, "fairport", "RETRYING", attempt=1, error="boom", detail={"retry_in_s": 10})

    row = next(j for j in store.jobs() if j["job_id"] == job)
    assert (row["state"], row["stage"], row["last_event"], row["attempt"], row["events"]) == \
        ("RETRYING", "prepare", "RETRYING", 1, 5)
    events = store.events(job)
    assert [e["state"] for e in events][-2:] == ["STAGE_FAILED", "RETRYING"]
    assert job in render(store.jobs(), events, job)
