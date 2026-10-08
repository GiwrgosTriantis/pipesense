"""Postgres job-status backend (prod-like profile). Append-only events; current state is derived."""

from __future__ import annotations

import json

import psycopg
from psycopg.rows import dict_row

SCHEMA = """
CREATE TABLE IF NOT EXISTS job_events (
    id       BIGSERIAL PRIMARY KEY,
    job_id   TEXT        NOT NULL,
    utility  TEXT        NOT NULL,
    state    TEXT        NOT NULL,
    stage    TEXT,
    attempt  INTEGER,
    error    TEXT,
    detail   JSONB,
    at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS job_events_job_id ON job_events (job_id, id);
"""

JOBS_QUERY = """
WITH job_level AS (
    SELECT DISTINCT ON (job_id) job_id, utility, state, attempt, error
    FROM job_events WHERE state NOT LIKE 'STAGE\\_%%' ORDER BY job_id, id DESC
), latest AS (
    SELECT DISTINCT ON (job_id) job_id, state AS last_event, at AS updated_at
    FROM job_events ORDER BY job_id, id DESC
), last_stage AS (
    SELECT DISTINCT ON (job_id) job_id, stage
    FROM job_events WHERE stage IS NOT NULL ORDER BY job_id, id DESC
), created AS (
    SELECT job_id, min(at) AS created_at, count(*) AS events FROM job_events GROUP BY job_id
)
SELECT j.job_id, j.utility, j.state, j.attempt, j.error, s.stage, l.last_event, l.updated_at, c.created_at, c.events
FROM job_level j
JOIN latest l USING (job_id)
JOIN created c USING (job_id)
LEFT JOIN last_stage s USING (job_id)
ORDER BY l.updated_at DESC LIMIT %s
"""


class PostgresStatusStore:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self._schema_ready = False

    def _connect(self):
        conn = psycopg.connect(self.dsn, autocommit=True, connect_timeout=5, row_factory=dict_row)
        if not self._schema_ready:
            with conn.transaction():
                conn.execute("SELECT pg_advisory_xact_lock(4242)")  # concurrent first starts
                conn.execute(SCHEMA)
            self._schema_ready = True
        return conn

    def record(self, job_id, utility, state, stage=None, attempt=None, error=None, detail=None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO job_events (job_id, utility, state, stage, attempt, error, detail) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (job_id, utility, state, stage, attempt, error, json.dumps(detail) if detail else None),
            )

    def jobs(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            return list(conn.execute(JOBS_QUERY, (limit,)).fetchall())

    def events(self, job_id: str) -> list[dict]:
        with self._connect() as conn:
            return list(
                conn.execute(
                    "SELECT at, state, stage, attempt, error, detail FROM job_events WHERE job_id = %s ORDER BY id",
                    (job_id,),
                ).fetchall()
            )
