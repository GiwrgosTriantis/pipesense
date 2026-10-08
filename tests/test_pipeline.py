"""End-to-end tests on the real Fairport data, against a real (sqlite) MLflow registry."""

import json
import shutil

import pandas as pd
import pytest

from pipesense import registry
from pipesense.config import PipelineConfig, Settings
from pipesense.errors import MissingPrerequisiteError, TransientError
from pipesense.runner import run_pipeline
from pipesense.status import FileStatusStore
from pipesense.storage import LocalStorage

from conftest import REPO


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # MLflow writes local artifacts relative to cwd
    root = tmp_path / "artifacts"
    shutil.copytree(REPO / "data" / "raw", root / "raw")
    cfg = (REPO / "config" / "pipeline.toml").read_text().replace("n_estimators = 200", "n_estimators = 40")
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


def _champion(env):
    return registry.get_champion(registry.client(env["settings"].mlflow_tracking_uri), "fairport")


def test_score_without_champion_fails_permanently_and_keeps_prepare(env):
    with pytest.raises(MissingPrerequisiteError, match="no 'champion'"):
        env["run"]("score", "job-1")
    assert env["storage"].exists("runs/fairport/job-1/prepare/_SUCCESS.json")
    status = FileStatusStore(env["storage"]).get("fairport", "job-1")
    assert [e["state"] for e in status["events"]][-1] == "STAGE_FAILED"


def test_train_registers_candidate_but_does_not_serve_it(env):
    result = env["run"]("train", "train-1")
    ref = result.summaries["train"]["model"]
    assert ref["version"] == "1"
    c = registry.client(env["settings"].mlflow_tracking_uri)
    tags = c.get_model_version(ref["name"], "1").tags
    assert tags["lifecycle"] == "candidate" and tags["job_id"] == "train-1"
    assert json.loads(tags["feature_spec"])["categorical_features"] == ["material", "soil_type", "pressure_zone"]
    holdout, baseline = result.summaries["train"]["holdout"], result.summaries["train"]["baseline"]
    assert holdout["n"] > 400 and holdout["positives"] >= 50
    assert 0 < baseline["average_precision"] < 1


def _train_and_promote(env, job="train-1"):
    from pipesense.cli import main

    env["run"]("train", job)
    assert main(["promote", "--utility", "fairport", "--version", "1", "--reason", "test", "--actor", "pytest"]) == 0


def test_resume_after_failure_skips_finished_stages(env):
    with pytest.raises(MissingPrerequisiteError):
        env["run"]("score", "job-2")
    _train_and_promote(env)
    result = env["run"]("score", "job-2")
    assert result.skipped == ["prepare"] and result.executed == ["predict"]


def test_injected_transient_fault_then_retry_resumes(env):
    _train_and_promote(env)
    with pytest.raises(TransientError):
        env["run"]("score", "job-3", fault_stage="predict")
    result = env["run"]("score", "job-3")
    assert result.skipped == ["prepare"] and result.executed == ["predict"]


def test_changed_raw_data_invalidates_prepare_and_downstream(env):
    _train_and_promote(env)
    env["run"]("score", "job-4")
    raw = env["root"] / "raw" / "fairport" / "pipes.csv"
    df = pd.read_csv(raw)
    df.loc[0, "length_m"] = df.loc[0, "length_m"] + 1
    df.to_csv(raw, index=False)
    result = env["run"]("score", "job-4")
    assert result.executed == ["prepare", "predict"]


def test_single_stage_requires_upstream_marker(env):
    with pytest.raises(MissingPrerequisiteError, match="run it before"):
        env["run"]("score", "job-5", only="predict")


def test_predictions_record_model_version_and_lineage(env):
    _train_and_promote(env)
    env["run"]("score", "job-6")
    preds = pd.read_csv(env["root"] / "runs/fairport/job-6/predict/predictions.csv")
    assert len(preds) == 2992 and preds["pipe_id"].is_unique
    assert set(preds["model_version"]) == {1} and set(preds["model_name"]) == {"leak-risk-fairport"}
    assert preds["leak_risk"].between(0, 1).all() and preds["risk_rank"].tolist() == list(range(1, 2993))
    manifest = json.loads((env["root"] / "runs/fairport/job-6/predict/manifest.json").read_text())
    assert manifest["model"]["version"] == "1" and manifest["model"]["run_id"]
    assert manifest["data_fingerprint"] and manifest["git_sha"] == "abc123"


def test_scoring_does_not_need_labels(env):
    _train_and_promote(env)
    raw = env["root"] / "raw" / "fairport" / "pipes.csv"
    pd.read_csv(raw).drop(columns="has_leak").to_csv(raw, index=False)
    env["run"]("score", "job-7")
    preds = pd.read_csv(env["root"] / "runs/fairport/job-7/predict/predictions.csv")
    assert "known_leak" not in preds.columns and len(preds) == 2992
