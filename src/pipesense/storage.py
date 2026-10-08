"""Artifact storage: one small interface, a local backend and an S3 backend.

Stages only see keys such as "runs/fairport/job-1/prepare/features.parquet". The backend is chosen
by PIPESENSE_ARTIFACT_ROOT (file:///data or s3://bucket/prefix). That is the whole difference
between the dev and prod-like setups, and it is what lets a stage move to another orchestrator
unchanged.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from pipesense.errors import PermanentError, TransientError


class Storage(Protocol):
    def read_bytes(self, key: str) -> bytes: ...
    def write_bytes(self, key: str, data: bytes) -> None: ...
    def exists(self, key: str) -> bool: ...
    def checksum(self, key: str) -> str: ...
    def list(self, prefix: str) -> list[str]: ...
    def uri(self, key: str) -> str: ...


class LocalStorage:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root.resolve() not in path.parents and path != self.root.resolve():
            raise PermanentError(f"key escapes the artifact root: {key}")
        return path

    def read_bytes(self, key: str) -> bytes:
        path = self._path(key)
        if not path.exists():
            raise FileNotFoundError(self.uri(key))
        return path.read_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        # write to a temp file in the same directory, then rename: readers never see half a file
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def checksum(self, key: str) -> str:
        return hashlib.sha256(self.read_bytes(key)).hexdigest()

    def list(self, prefix: str) -> list[str]:
        base = self._path(prefix) if prefix else self.root
        if not base.exists():
            return []
        return sorted(
            str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file() and not p.name.startswith(".")
        )

    def uri(self, key: str) -> str:
        return f"file://{self._path(key)}"


class S3Storage:
    """S3 or any S3-compatible store. Single-object PUTs are atomic, so no temp-file dance."""

    def __init__(self, bucket: str, prefix: str = "", client=None):
        import boto3

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        # endpoint comes from AWS_ENDPOINT_URL_S3 when set (local SeaweedFS); unset on real AWS
        self.client = client or boto3.client("s3")

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def _call(self, fn, **kwargs):
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            return fn(Bucket=self.bucket, **kwargs)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"NoSuchKey", "404", "NotFound"}:
                raise FileNotFoundError(self.uri(kwargs.get("Key", ""))) from exc
            if code in {"AccessDenied", "NoSuchBucket", "403"}:
                raise PermanentError(f"S3 {code} for s3://{self.bucket}/{kwargs.get('Key', '')}") from exc
            raise TransientError(f"S3 error {code}") from exc
        except BotoCoreError as exc:
            raise TransientError(f"S3 connection error: {exc}") from exc

    def read_bytes(self, key: str) -> bytes:
        return self._call(self.client.get_object, Key=self._key(key))["Body"].read()

    def write_bytes(self, key: str, data: bytes) -> None:
        self._call(self.client.put_object, Key=self._key(key), Body=data)

    def exists(self, key: str) -> bool:
        try:
            self._call(self.client.head_object, Key=self._key(key))
            return True
        except FileNotFoundError:
            return False

    def checksum(self, key: str) -> str:
        # content hash, not the ETag: multipart ETags are not content hashes
        return hashlib.sha256(self.read_bytes(key)).hexdigest()

    def list(self, prefix: str) -> list[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        keys: list[str] = []
        strip = len(self.prefix) + 1 if self.prefix else 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._key(prefix)):
            keys.extend(obj["Key"][strip:] for obj in page.get("Contents", []))
        return sorted(keys)

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{self._key(key)}"


def open_storage(root_uri: str) -> Storage:
    parsed = urlparse(root_uri)
    if parsed.scheme == "file":
        return LocalStorage(parsed.path)
    if parsed.scheme == "s3":
        return S3Storage(bucket=parsed.netloc, prefix=parsed.path)
    raise PermanentError(f"unsupported PIPESENSE_ARTIFACT_ROOT scheme: {root_uri!r}")
