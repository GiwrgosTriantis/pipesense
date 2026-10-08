import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipesense.config import PipelineConfig, Settings
from pipesense.runner import run_pipeline
from pipesense.status import FileStatusStore
from pipesense.storage import LocalStorage

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    """Small raw export with the defects the cleaning stage must handle."""
    return pd.DataFrame(
        {
            "pipe_id": ["FP-1", "FP-1", "FP-2", "FP-3", "FP-4", "FP-5", "FP-5"],
            "material": ["Cast Iron", "CI", "CI", "pvc", "PVC", "steel", "steel"],
            "install_year": [1950.0, 1950.0, None, 1990.0, 1995.0, 1960.0, 1960.0],
            "diameter_mm": [150.0, 150.0, 200.0, None, 200.0, 100.0, 100.0],
            "length_m": [100.0, 100.0, 80.0, 50.0, -1.0, 30.0, 31.0],
            "lon": [-94.80, -94.80, -94.81, -94.82, -94.83, -94.84, -94.84],
            "lat": [39.30, 39.30, 39.31, 39.32, 39.33, 39.34, 39.34],
            "pressure_zone": ["A", "A", "B", "C", "A", "B", "B"],
            "soil_type": ["clay", "clay", "loam", "sand", "rock", "clay", "clay"],
            "num_breaks_5yr": [1, 1, 0, 0, 2, 0, 0],
            "has_leak": [1, 1, 0, 0, 1, 0, 0],
        }
    )


@pytest.fixture
def fairport_raw() -> pd.DataFrame:
    return pd.read_csv(REPO / "data" / "raw" / "fairport" / "pipes.csv")


def synthetic_features(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Feature table with a learnable signal (old cast iron leaks), for fast model tests."""
    rng = np.random.default_rng(seed)
    material = rng.choice(["CI", "PVC", "DI"], n)
    age = rng.uniform(5, 120, n)
    logit = -4 + 0.04 * age + 1.0 * (material == "CI")
    df = pd.DataFrame(
        {
            "pipe_id": [f"P-{i}" for i in range(n)],
            "material": material,
            "soil_type": rng.choice(["clay", "sand"], n),
            "pressure_zone": rng.choice(["A", "B"], n),
            "age_years": age,
            "diameter_mm": rng.choice([100.0, 150.0, 200.0], n),
            "log_length": np.log1p(rng.uniform(10, 300, n)),
            "num_breaks_5yr": rng.poisson(0.3, n).astype(float),
            "dist_to_center_m": rng.uniform(0, 8000, n),
            "neighbours_500m": rng.integers(0, 100, n),
            "x_m": rng.uniform(0, 12000, n),
            "y_m": rng.uniform(0, 12000, n),
            "length_m": rng.uniform(10, 300, n),
            "has_leak": (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int),
        }
    )
    df.loc[df.sample(frac=0.05, random_state=seed).index, "age_years"] = np.nan
    return df


@pytest.fixture
def features_df() -> pd.DataFrame:
    return synthetic_features()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # MLflow writes local artifacts relative to cwd
    root = tmp_path / "artifacts"
    shutil.copytree(REPO / "data" / "raw", root / "raw")
    cfg = (REPO / "config" / "pipeline.toml").read_text().replace("n_estimators = 200", "n_estimators = 40")
    cfg = cfg.replace("bootstrap_samples = 1000", "bootstrap_samples = 200")
    (tmp_path / "pipeline.toml").write_text(cfg)
    settings = Settings.from_env(
        {
            "PIPESENSE_ARTIFACT_ROOT": f"file://{root}",
            "MLFLOW_TRACKING_URI": f"sqlite:///{tmp_path / 'mlflow.db'}",
            "PIPESENSE_PIPELINE_CONFIG": str(tmp_path / "pipeline.toml"),
            "PIPESENSE_IMAGE_TAG": "test",
            "PIPESENSE_GIT_SHA": "abc123",
        }
    )
    for k, v in {"PIPESENSE_ARTIFACT_ROOT": settings.artifact_root, "MLFLOW_TRACKING_URI": settings.mlflow_tracking_uri,
                 "PIPESENSE_PIPELINE_CONFIG": settings.pipeline_config_path, "PIPESENSE_LOG_FORMAT": "text"}.items():
        monkeypatch.setenv(k, v)  # for tests that go through the CLI
    storage = LocalStorage(root)
    config = PipelineConfig.load(settings.pipeline_config_path)

    def run(pipeline, job_id, **kw):
        return run_pipeline(pipeline, "fairport", job_id, settings, config, storage, FileStatusStore(storage), **kw)

    return {"settings": settings, "storage": storage, "run": run, "root": root, "config": config}


def train_and_promote(env, job="train-1"):
    """Train v1 and make it champion via the audited manual path (gate outcome irrelevant here)."""
    from pipesense.cli import main

    env["run"]("train", job)
    assert main(["promote", "--utility", "fairport", "--version", "1", "--reason", "test", "--actor", "pytest"]) == 0
