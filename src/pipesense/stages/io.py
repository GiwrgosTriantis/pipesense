"""Serialisation helpers shared by stages."""

from __future__ import annotations

import io
import json

import pandas as pd

from pipesense.errors import MissingPrerequisiteError
from pipesense.storage import Storage


def write_parquet(storage: Storage, key: str, df: pd.DataFrame) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    storage.write_bytes(key, buf.getvalue())


def read_parquet(storage: Storage, key: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(io.BytesIO(storage.read_bytes(key)))
    except FileNotFoundError as exc:
        raise MissingPrerequisiteError(f"missing upstream output {storage.uri(key)}") from exc


def write_json(storage: Storage, key: str, obj) -> None:
    storage.write_bytes(key, json.dumps(obj, indent=2, default=str).encode())


def read_json(storage: Storage, key: str):
    try:
        return json.loads(storage.read_bytes(key))
    except FileNotFoundError as exc:
        raise MissingPrerequisiteError(f"missing upstream output {storage.uri(key)}") from exc
