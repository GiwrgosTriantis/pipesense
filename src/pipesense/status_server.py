"""Tiny read-only status page over the Postgres job log (stdlib only, no web framework).

    GET /               HTML table, refreshes itself every 2 s
    GET /api/jobs       latest state per job (JSON)
    GET /api/jobs/<id>  full event history of one job (JSON)
    GET /healthz
"""

from __future__ import annotations

import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pipesense.log import get_logger, kv
from pipesense.status_pg import PostgresStatusStore

log = get_logger(__name__)

COLOURS = {"SUCCEEDED": "#1a7f37", "FAILED": "#cf222e", "DEAD_LETTERED": "#8250df",
           "RETRYING": "#bf8700", "PREVIOUS_ATTEMPT_LOST": "#bc4c00", "RUNNING": "#0969da", "QUEUED": "#57606a"}


def render(jobs: list[dict], selected: list[dict] | None, job_id: str | None) -> str:
    rows = "".join(
        f"<tr><td><a href='/?job={html.escape(j['job_id'])}'>{html.escape(j['job_id'])}</a></td>"
        f"<td>{html.escape(j['utility'])}</td>"
        f"<td><b style='color:{COLOURS.get(j['state'], '#000')}'>{j['state']}</b></td>"
        f"<td>{html.escape(j['stage'] or '')}</td><td>{html.escape(j['last_event'])}</td>"
        f"<td>{j['attempt'] or ''}</td><td>{j['updated_at']:%H:%M:%S}</td>"
        f"<td class=err>{html.escape((j['error'] or '')[:140])}</td></tr>"
        for j in jobs
    )
    detail = ""
    if selected is not None:
        ev = "".join(
            f"<tr><td>{e['at']:%H:%M:%S.%f}</td><td>{e['state']}</td><td>{html.escape(e['stage'] or '')}</td>"
            f"<td>{e['attempt'] or ''}</td><td class=err>{html.escape(e['error'] or '')}</td>"
            f"<td>{html.escape(json.dumps(e['detail']) if e['detail'] else '')}</td></tr>"
            for e in selected
        )
        detail = (f"<h2>Events for {html.escape(job_id or '')}</h2><table><tr><th>at</th><th>event</th>"
                  f"<th>stage</th><th>attempt</th><th>error</th><th>detail</th></tr>{ev}</table>")
    return f"""<!doctype html><html><head><meta charset=utf-8><meta http-equiv=refresh content=2>
<title>PipeSense jobs</title><style>
body{{font:14px system-ui,sans-serif;margin:24px;color:#1f2328}} table{{border-collapse:collapse;width:100%;margin-bottom:24px}}
td,th{{border-bottom:1px solid #d0d7de;padding:6px 8px;text-align:left;vertical-align:top}} .err{{color:#cf222e;font-size:12px}}
</style></head><body><h1>PipeSense jobs</h1><p>Live from the Postgres job log; refreshes every 2 s.</p>
<table><tr><th>job</th><th>utility</th><th>state</th><th>stage</th><th>last event</th><th>attempt</th>
<th>updated (UTC)</th><th>error</th></tr>{rows}</table>{detail}</body></html>"""


def serve(store: PostgresStatusStore, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: str, ctype: str) -> None:
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            from urllib.parse import parse_qs, urlparse

            url = urlparse(self.path)
            try:
                if url.path == "/healthz":
                    return self._send(200, "ok", "text/plain")
                if url.path == "/api/jobs":
                    return self._send(200, json.dumps(store.jobs(), default=str), "application/json")
                if url.path.startswith("/api/jobs/"):
                    return self._send(200, json.dumps(store.events(url.path.rsplit("/", 1)[1]), default=str),
                                      "application/json")
                if url.path == "/":
                    job = parse_qs(url.query).get("job", [None])[0]
                    return self._send(200, render(store.jobs(), store.events(job) if job else None, job),
                                      "text/html; charset=utf-8")
                return self._send(404, "not found", "text/plain")
            except Exception as exc:
                log.warning("status page error", extra=kv(error=repr(exc)))
                return self._send(503, f"status store unavailable: {exc}", "text/plain")

        def log_message(self, *args):
            return

    log.info("status page listening", extra=kv(port=port))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
