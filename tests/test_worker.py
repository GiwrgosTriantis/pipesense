"""The queue miniature against a mocked SQS: retry + resume, fail fast, dead-lettering."""

import json
from dataclasses import replace

import boto3
import pytest
from moto import mock_aws

from pipesense import worker as W
from pipesense.errors import PermanentError, TransientError
from pipesense.status import FileStatusStore

from conftest import train_and_promote


@pytest.fixture
def queue(env, monkeypatch):
    with mock_aws():
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
        monkeypatch.setattr(W, "BACKOFF_BASE_S", 0)  # retry immediately in tests
        sqs = boto3.client("sqs")
        sqs.create_queue(QueueName="pipesense-jobs")
        sqs.create_queue(QueueName="pipesense-jobs-dlq")
        settings = replace(env["settings"], allow_fault_injection=True, max_attempts=3, visibility_timeout_s=30)
        worker = W.Worker(settings, env["config"], env["storage"], FileStatusStore(env["storage"]), sqs=sqs)

        def submit(**body):
            sqs.send_message(QueueUrl=worker.queue_url, MessageBody=json.dumps(body))

        def receive(url):
            return sqs.receive_message(QueueUrl=url, MaxNumberOfMessages=10, MessageAttributeNames=["All"],
                                       VisibilityTimeout=0).get("Messages", [])

        yield {"sqs": sqs, "worker": worker, "submit": submit, "receive": receive}


def _status(env, job):
    return FileStatusStore(env["storage"]).get("fairport" if "fair" in job else "brokenville", job)


def test_job_message_validation():
    assert W.JobRequest.parse('{"job_id": "job-1", "utility": "fairport"}').pipeline == "score"
    for bad in ['not json', '{"utility": "fairport"}', '{"job_id": "../etc", "utility": "fairport"}',
                '{"job_id": "j", "utility": "fairport", "pipeline": "drop-tables"}']:
        with pytest.raises(PermanentError):
            W.JobRequest.parse(bad)


def test_backoff_grows_and_is_capped():
    assert [W.backoff_seconds(a) for a in (1, 2, 3)] == [10, 20, 40]
    assert W.backoff_seconds(20) == W.BACKOFF_MAX_S


def test_transient_fault_is_retried_and_resumes_after_finished_stages(env, queue):
    train_and_promote(env)
    queue["submit"](job_id="fair-1", utility="fairport", inject_fault="predict")
    assert queue["worker"].run_forever(max_messages=2) == 2

    states = [(e["state"], e.get("stage"), e.get("attempt")) for e in _status(env, "fair-1")["events"]]
    assert ("STAGE_SUCCEEDED", "prepare", 1) in states
    assert ("STAGE_FAILED", "predict", 1) in states and ("RETRYING", None, 1) in states
    assert ("STAGE_SKIPPED", "prepare", 2) in states  # attempt 2 resumes, prepare not redone
    assert states[-1] == ("SUCCEEDED", None, 2)
    assert queue["receive"](queue["worker"].queue_url) == [] and queue["receive"](queue["worker"].dlq_url) == []


def test_bad_data_fails_fast_to_the_dlq_without_retry(env, queue):
    queue["submit"](job_id="broken-1", utility="brokenville")
    assert queue["worker"].run_forever(max_messages=1) == 1
    status = _status(env, "broken-1")
    assert status["state"] == "FAILED" and "material" in status["error"]
    assert not any(e["state"] == "RETRYING" for e in status["events"])
    dlq = queue["receive"](queue["worker"].dlq_url)
    assert len(dlq) == 1 and dlq[0]["MessageAttributes"]["final_state"]["StringValue"] == "FAILED"
    assert queue["receive"](queue["worker"].queue_url) == []


def test_persistent_transient_failure_is_dead_lettered_after_max_attempts(env, queue, monkeypatch):
    def always_fails(*args, **kwargs):
        raise TransientError("S3 throttling")

    monkeypatch.setattr(W, "run_pipeline", always_fails)
    queue["submit"](job_id="fair-2", utility="fairport")
    queue["worker"].run_forever(max_messages=3)
    status = _status(env, "fair-2")
    assert [e["state"] for e in status["events"]] == ["RETRYING", "RETRYING", "DEAD_LETTERED"]
    dlq = queue["receive"](queue["worker"].dlq_url)
    assert dlq[0]["MessageAttributes"]["attempts"]["StringValue"] == "3"


def test_invalid_message_goes_straight_to_dlq(env, queue):
    queue["sqs"].send_message(QueueUrl=queue["worker"].queue_url, MessageBody="{oops")
    queue["worker"].run_forever(max_messages=1)
    assert len(queue["receive"](queue["worker"].dlq_url)) == 1


def test_fault_injection_is_ignored_unless_enabled(env, queue):
    train_and_promote(env)
    queue["worker"].settings = replace(queue["worker"].settings, allow_fault_injection=False)
    queue["submit"](job_id="fair-3", utility="fairport", inject_fault="predict")
    queue["worker"].run_forever(max_messages=1)
    assert _status(env, "fair-3")["state"] == "SUCCEEDED"


def test_status_store_outage_does_not_fail_the_job(env, queue):
    class Broken:
        def record(self, *a, **k):
            raise ConnectionError("postgres down")

    train_and_promote(env)
    queue["worker"].status = W.SafeStatus(Broken())
    queue["submit"](job_id="fair-4", utility="fairport")
    queue["worker"].run_forever(max_messages=1)
    assert env["storage"].exists("runs/fairport/fair-4/predict/_SUCCESS.json")
    assert queue["receive"](queue["worker"].queue_url) == []
