"""Queue worker: the local miniature of the AWS design (SQS -> container -> status).

Message: {"job_id": "job-42", "utility": "fairport"} plus optional "pipeline" ("score" by
default, or "train") and, in dev only, "inject_fault": "<stage>".

Failure policy (the decision is documented in DECISIONS.md):
* Permanent error (bad data, unknown utility, no champion): no retry. Status FAILED and the
  message goes to the DLQ with the reason. Retrying cannot fix the input and would only delay
  telling the customer.
* Transient error or bug: the message is left on the queue with a growing visibility delay
  (10 s, 20 s, 40 s ...). The next attempt resumes after the last stage that finished (success
  markers). After PIPESENSE_MAX_ATTEMPTS the status is DEAD_LETTERED and the message moves to
  the DLQ.
* A worker that dies mid-job (OOM kill, SIGKILL, lost host) never deletes the message, so SQS
  redelivers it once the visibility timeout expires. Every attempt that ends normally writes an
  outcome record next to the job's outputs. When a message arrives with a receive count showing
  earlier attempts that have no record, those attempts are recorded as PREVIOUS_ATTEMPT_LOST. If
  the receive count exceeds PIPESENSE_MAX_ATTEMPTS, the job is dead-lettered at once with that
  reason, so it ends DEAD_LETTERED instead of staying RUNNING forever. The queue's own redrive
  policy (maxReceiveCount above PIPESENSE_MAX_ATTEMPTS) remains the backstop.
* Long jobs: a heartbeat thread keeps extending the message's visibility while the job runs.
"""

from __future__ import annotations

import json
import re
import signal
import threading
from dataclasses import dataclass

import boto3

from pipesense.config import PipelineConfig, Settings
from pipesense.errors import PermanentError, PipelineError
from pipesense.log import get_logger, kv, log_context
from pipesense.runner import run_pipeline
from pipesense.status import StatusStore, now
from pipesense.storage import Storage

log = get_logger(__name__)
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")  # job_id and utility become storage paths
BACKOFF_BASE_S = 10
BACKOFF_MAX_S = 900
LOST_ATTEMPT = "worker died without reporting, e.g. OOM kill or SIGKILL"


@dataclass(frozen=True)
class JobRequest:
    job_id: str
    utility: str
    pipeline: str = "score"
    inject_fault: str | None = None

    @classmethod
    def parse(cls, body: str) -> "JobRequest":
        try:
            data = json.loads(body)
            req = cls(job_id=str(data["job_id"]), utility=str(data["utility"]),
                      pipeline=str(data.get("pipeline", "score")), inject_fault=data.get("inject_fault"))
        except (ValueError, KeyError, TypeError) as exc:
            raise PermanentError(f"invalid job message: {exc!r}") from exc
        for name, value in (("job_id", req.job_id), ("utility", req.utility)):
            if not ID_PATTERN.match(value):
                raise PermanentError(f"invalid {name} {value!r}: must match {ID_PATTERN.pattern}")
        if req.pipeline not in {"score", "train"}:
            raise PermanentError(f"invalid pipeline {req.pipeline!r}")
        return req


def backoff_seconds(attempt: int) -> int:
    return min(BACKOFF_BASE_S * 2 ** (attempt - 1), BACKOFF_MAX_S)


class Heartbeat:
    """Extends the message's visibility while a job runs, so long jobs are not redelivered mid-run."""

    def __init__(self, sqs, queue_url: str, receipt: str, timeout_s: int):
        self.sqs, self.queue_url, self.receipt, self.timeout_s = sqs, queue_url, receipt, timeout_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.timeout_s / 3):
            try:
                self.sqs.change_message_visibility(QueueUrl=self.queue_url, ReceiptHandle=self.receipt,
                                                   VisibilityTimeout=self.timeout_s)
            except Exception as exc:  # the job keeps running; worst case it is redelivered and resumes
                log.warning("heartbeat failed", extra=kv(error=repr(exc)))

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)


class SafeStatus:
    """Status is for humans. A status-store outage must not fail or retry a customer job."""

    def __init__(self, inner: StatusStore):
        self.inner = inner

    def record(self, *args, **kwargs) -> None:
        try:
            self.inner.record(*args, **kwargs)
        except Exception as exc:
            log.warning("status write failed", extra=kv(error=repr(exc)))


class Worker:
    def __init__(self, settings: Settings, config: PipelineConfig, storage: Storage, status: StatusStore, sqs=None):
        self.settings, self.config, self.storage = settings, config, storage
        self.status = SafeStatus(status)
        self.sqs = sqs or boto3.client("sqs")
        self.queue_url = self.sqs.get_queue_url(QueueName=settings.queue_name)["QueueUrl"]
        self.dlq_url = self.sqs.get_queue_url(QueueName=settings.dlq_name)["QueueUrl"]
        self.stopping = False

    def request_stop(self, *_):
        log.info("stop requested: finishing the current job, taking no new ones")
        self.stopping = True

    def run_forever(self, max_messages: int | None = None) -> int:
        handled = 0
        while not self.stopping and (max_messages is None or handled < max_messages):
            resp = self.sqs.receive_message(
                QueueUrl=self.queue_url, MaxNumberOfMessages=1, WaitTimeSeconds=10,
                VisibilityTimeout=self.settings.visibility_timeout_s, AttributeNames=["ApproximateReceiveCount"],
            )
            for msg in resp.get("Messages", []):
                self.handle(msg)
                handled += 1
        return handled

    def _dead_letter(self, msg: dict, req: JobRequest | None, state: str, error: str, attempt: int) -> None:
        self.sqs.send_message(
            QueueUrl=self.dlq_url, MessageBody=msg["Body"],
            MessageAttributes={
                "final_state": {"DataType": "String", "StringValue": state},
                "error": {"DataType": "String", "StringValue": error[:1000]},
                "attempts": {"DataType": "Number", "StringValue": str(attempt)},
            },
        )
        self.sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=msg["ReceiptHandle"])
        if req is not None:
            self.status.record(req.job_id, req.utility, state, attempt=attempt, error=error,
                               detail={"dead_letter_queue": self.settings.dlq_name})

    @staticmethod
    def _attempt_key(req: JobRequest, attempt: int) -> str:
        return f"runs/{req.utility}/{req.job_id}/attempts/{attempt}.json"

    def _close_attempt(self, req: JobRequest, attempt: int, outcome: str) -> None:
        """Record how an attempt ended. A later attempt that finds no record knows its worker died."""
        try:
            self.storage.write_bytes(self._attempt_key(req, attempt),
                                     json.dumps({"attempt": attempt, "outcome": outcome, "at": now()}).encode())
        except Exception as exc:
            log.warning("could not record attempt outcome", extra=kv(error=repr(exc), outcome=outcome))

    def _lost_attempts(self, req: JobRequest, attempt: int) -> list[int]:
        lost = []
        for earlier in range(1, attempt):
            try:
                if not self.storage.exists(self._attempt_key(req, earlier)):
                    lost.append(earlier)
            except Exception as exc:  # an unreadable store is not evidence of a crash
                log.warning("could not check attempt outcome", extra=kv(error=repr(exc), earlier_attempt=earlier))
        return lost

    def handle(self, msg: dict) -> str:
        attempt = int(msg.get("Attributes", {}).get("ApproximateReceiveCount", "1"))
        try:
            req = JobRequest.parse(msg["Body"])
        except PermanentError as exc:
            log.error("rejecting invalid message", extra=kv(error=str(exc)))
            self._dead_letter(msg, None, "FAILED", str(exc), attempt)
            return "FAILED"

        with log_context(job_id=req.job_id, utility=req.utility, attempt=attempt):
            lost = self._lost_attempts(req, attempt)
            for earlier in lost:
                log.error("previous attempt lost", extra=kv(lost_attempt=earlier, error=LOST_ATTEMPT))
                self.status.record(req.job_id, req.utility, "PREVIOUS_ATTEMPT_LOST", attempt=earlier,
                                   error=LOST_ATTEMPT)
                self._close_attempt(req, earlier, "LOST")  # reported once, not again by later attempts
            if attempt > self.settings.max_attempts:
                error = (f"{LOST_ATTEMPT} (attempt {', '.join(map(str, lost))}); " if lost else "") + \
                    f"received {attempt} times, more than PIPESENSE_MAX_ATTEMPTS={self.settings.max_attempts}"
                log.error("attempts used up by lost workers, dead-lettering", extra=kv(error=error))
                self._close_attempt(req, attempt, "DEAD_LETTERED")
                self._dead_letter(msg, req, "DEAD_LETTERED", error, attempt)
                return "DEAD_LETTERED"

            fault = req.inject_fault if (self.settings.allow_fault_injection and attempt == 1) else None
            try:
                with Heartbeat(self.sqs, self.queue_url, msg["ReceiptHandle"], self.settings.visibility_timeout_s):
                    run_pipeline(req.pipeline, req.utility, req.job_id, self.settings, self.config, self.storage,
                                 self.status, fault_stage=fault, attempt=attempt)
            except Exception as exc:
                retryable = exc.retryable if isinstance(exc, PipelineError) else True
                error = str(exc) if isinstance(exc, PipelineError) else repr(exc)
                if not retryable:
                    log.error("permanent failure, not retrying", extra=kv(error=error))
                    self._close_attempt(req, attempt, "FAILED")
                    self._dead_letter(msg, req, "FAILED", error, attempt)
                    return "FAILED"
                if attempt >= self.settings.max_attempts:
                    log.error("retries exhausted", extra=kv(error=error))
                    self._close_attempt(req, attempt, "DEAD_LETTERED")
                    self._dead_letter(msg, req, "DEAD_LETTERED", error, attempt)
                    return "DEAD_LETTERED"
                delay = backoff_seconds(attempt)
                log.warning("transient failure, will retry", extra=kv(error=error, retry_in_s=delay))
                self.status.record(req.job_id, req.utility, "RETRYING", attempt=attempt, error=error,
                                   detail={"retry_in_s": delay, "max_attempts": self.settings.max_attempts})
                self._close_attempt(req, attempt, "RETRYING")  # before the message can be seen again
                self.sqs.change_message_visibility(QueueUrl=self.queue_url, ReceiptHandle=msg["ReceiptHandle"],
                                                   VisibilityTimeout=delay)
                return "RETRYING"

            self.status.record(req.job_id, req.utility, "SUCCEEDED", attempt=attempt)
            self._close_attempt(req, attempt, "SUCCEEDED")
            self.sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=msg["ReceiptHandle"])
            log.info("job succeeded")
            return "SUCCEEDED"


def install_signal_handlers(worker: Worker) -> None:
    signal.signal(signal.SIGTERM, worker.request_stop)
    signal.signal(signal.SIGINT, worker.request_stop)
