import boto3
import pytest
from moto import mock_aws

from pipesense.errors import PermanentError
from pipesense.storage import LocalStorage, S3Storage, open_storage


def test_local_roundtrip_and_listing(tmp_path):
    s = LocalStorage(tmp_path)
    s.write_bytes("runs/u/j/prepare/a.txt", b"hello")
    assert s.read_bytes("runs/u/j/prepare/a.txt") == b"hello"
    assert s.exists("runs/u/j/prepare/a.txt") and not s.exists("runs/u/j/prepare/b.txt")
    assert s.list("runs/u/j") == ["runs/u/j/prepare/a.txt"]
    assert len(s.checksum("runs/u/j/prepare/a.txt")) == 64


def test_local_write_leaves_no_temp_files(tmp_path):
    s = LocalStorage(tmp_path)
    s.write_bytes("x/y.bin", b"1")
    s.write_bytes("x/y.bin", b"2")
    assert [p.name for p in (tmp_path / "x").iterdir()] == ["y.bin"]


def test_local_rejects_path_traversal(tmp_path):
    with pytest.raises(PermanentError):
        LocalStorage(tmp_path / "root").write_bytes("../escape.txt", b"x")


def test_open_storage_picks_backend(tmp_path):
    assert isinstance(open_storage(f"file://{tmp_path}"), LocalStorage)
    with pytest.raises(PermanentError):
        open_storage("ftp://nope")


@mock_aws
def test_s3_roundtrip_with_prefix():
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="pipesense")
    s = S3Storage("pipesense", "/artifacts/", client=client)
    s.write_bytes("runs/u/j/x.json", b"{}")
    assert s.read_bytes("runs/u/j/x.json") == b"{}"
    assert s.exists("runs/u/j/x.json") and not s.exists("runs/u/j/missing")
    assert s.list("runs/u") == ["runs/u/j/x.json"]
    assert s.uri("a") == "s3://pipesense/artifacts/a"
    with pytest.raises(FileNotFoundError):
        s.read_bytes("nope")


@mock_aws
def test_seed_creates_buckets_and_uploads_raw_exports(monkeypatch):
    from pipesense.cli import main

    from conftest import REPO

    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("PIPESENSE_ARTIFACT_ROOT", "s3://pipesense/artifacts")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://unused:5000")
    monkeypatch.setenv("PIPESENSE_LOG_FORMAT", "text")
    assert main(["seed", "--source", str(REPO / "data" / "raw"), "--extra-bucket", "mlflow-artifacts"]) == 0
    client = boto3.client("s3")
    assert {b["Name"] for b in client.list_buckets()["Buckets"]} == {"pipesense", "mlflow-artifacts"}
    keys = S3Storage("pipesense", "artifacts", client=client).list("raw/fairport")
    assert keys == ["raw/fairport/meta.json", "raw/fairport/pipes.csv"]
