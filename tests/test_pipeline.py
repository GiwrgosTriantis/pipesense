"""End-to-end tests on the real Fairport data, against a real (sqlite) MLflow registry."""

import json
import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from pipesense import model as M
from pipesense import registry
from pipesense.errors import DataValidationError, MissingPrerequisiteError, TransientError
from pipesense.status import FileStatusStore

from conftest import train_and_promote


def _champion(env):
    return registry.get_champion(registry.client(env["settings"].mlflow_tracking_uri), "fairport")


def test_score_without_champion_fails_permanently_and_keeps_prepare(env):
    with pytest.raises(MissingPrerequisiteError, match="no 'champion'"):
        env["run"]("score", "job-1")
    assert env["storage"].exists("runs/fairport/job-1/prepare/_SUCCESS.json")
    status = FileStatusStore(env["storage"]).get("fairport", "job-1")
    assert [e["state"] for e in status["events"]][-1] == "STAGE_FAILED"


def test_train_registers_a_version_and_the_gate_decides_with_an_audit_trail(env):
    result = env["run"]("train", "train-1")
    ref = result.summaries["train"]["model"]
    assert ref["version"] == "1"
    c = registry.client(env["settings"].mlflow_tracking_uri)
    tags = c.get_model_version(ref["name"], "1").tags
    assert tags["job_id"] == "train-1" and tags["holdout_spec"].startswith("spatial-block:1000")
    gate = result.summaries["gate"]
    assert tags["gate_decision"] == gate["decision"] and tags["gate_run_id"] == gate["gate_run_id"]
    audit = c.get_run(gate["gate_run_id"])
    assert audit.data.tags["decision"] == gate["decision"] and audit.data.tags["actor"] == "gate:auto"
    report = json.loads((env["root"] / "runs/fairport/train-1/gate/decision.json").read_text())
    assert report["holdout"]["n"] > 400 and report["bootstrap"]["candidate_vs_baseline"]["ci90_low"] < 1
    champion = _champion(env)
    if gate["decision"] == "PROMOTED":
        assert champion.version == "1" and tags["lifecycle"] == "champion"
    else:
        assert champion is None and tags["lifecycle"] == "candidate"
    assert json.loads(tags["feature_spec"])["categorical_features"] == ["material", "soil_type", "pressure_zone"]
    holdout, baseline = result.summaries["train"]["holdout"], result.summaries["train"]["baseline"]
    assert holdout["n"] > 400 and holdout["positives"] >= 50
    assert 0 < baseline["average_precision"] < 1


def test_resume_after_failure_skips_finished_stages(env):
    with pytest.raises(MissingPrerequisiteError):
        env["run"]("score", "job-2")
    train_and_promote(env)
    result = env["run"]("score", "job-2")
    assert result.skipped == ["prepare"] and result.executed == ["predict"]


def test_injected_transient_fault_then_retry_resumes(env):
    train_and_promote(env)
    with pytest.raises(TransientError):
        env["run"]("score", "job-3", fault_stage="predict")
    result = env["run"]("score", "job-3")
    assert result.skipped == ["prepare"] and result.executed == ["predict"]


def test_changed_raw_data_invalidates_prepare_and_downstream(env):
    train_and_promote(env)
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
    train_and_promote(env)
    env["run"]("score", "job-6")
    preds = pd.read_csv(env["root"] / "runs/fairport/job-6/predict/predictions.csv")
    assert len(preds) == 2992 and preds["pipe_id"].is_unique
    assert set(preds["model_version"]) == {1} and set(preds["model_name"]) == {"leak-risk-fairport"}
    assert preds["leak_risk"].between(0, 1).all() and preds["risk_rank"].tolist() == list(range(1, 2993))
    manifest = json.loads((env["root"] / "runs/fairport/job-6/predict/manifest.json").read_text())
    assert manifest["model"]["version"] == "1" and manifest["model"]["run_id"]
    assert manifest["data_fingerprint"] and manifest["git_sha"] == "abc123"


def test_scoring_does_not_need_labels(env):
    train_and_promote(env)
    raw = env["root"] / "raw" / "fairport" / "pipes.csv"
    pd.read_csv(raw).drop(columns="has_leak").to_csv(raw, index=False)
    env["run"]("score", "job-7")
    preds = pd.read_csv(env["root"] / "runs/fairport/job-7/predict/predictions.csv")
    assert "known_leak" not in preds.columns and len(preds) == 2992


def test_retraining_unchanged_data_does_not_replace_the_champion(env):
    train_and_promote(env)
    result = env["run"]("train", "train-2")
    assert result.summaries["train"]["model"]["version"] == "2"
    assert result.summaries["gate"]["decision"] == "REJECTED"
    assert any("G2_ap_improves_on_champion" in r for r in result.summaries["gate"]["failed"])
    assert _champion(env).version == "1"


def test_predict_reports_drift_against_the_models_training_data(env):
    train_and_promote(env)
    env["run"]("score", "job-8")
    manifest = json.loads((env["root"] / "runs/fairport/job-8/predict/manifest.json").read_text())
    report = manifest["drift"]
    assert report["status"] == "ok" and report["threshold"] == 0.2 and report["drifted"] == []
    assert set(report["psi"]) == {"age_years", "diameter_mm", "log_length", "num_breaks_5yr", "dist_to_center_m",
                                  "neighbours_500m", "material", "soil_type", "pressure_zone"}
    assert report["reference_rows"] < report["rows"] == 2992  # reference = training blocks only


def test_shifted_inventory_logs_a_drift_warning(env, caplog):
    train_and_promote(env)
    raw = env["root"] / "raw" / "fairport" / "pipes.csv"
    df = pd.read_csv(raw)
    df["install_year"] = df["install_year"] - 40  # every pipe 40 years older than the model has seen
    df.to_csv(raw, index=False)
    predict_log = logging.getLogger("pipesense.stages.predict")
    predict_log.addHandler(caplog.handler)  # the CLI's setup_logging replaced the root handlers
    try:
        result = env["run"]("score", "job-9")
    finally:
        predict_log.removeHandler(caplog.handler)
    assert result.summaries["predict"]["drift"] == "drift"
    assert "age_years" in result.summaries["predict"]["drifted_features"]
    warning = next(r for r in caplog.records if r.getMessage() == "feature drift above threshold")
    assert warning.fields["psi"]["age_years"] > 0.2
    manifest = json.loads((env["root"] / "runs/fairport/job-9/predict/manifest.json").read_text())
    assert manifest["drift"]["drifted"][0] == "age_years"


def test_model_without_drift_reference_still_scores(env, monkeypatch):
    train_and_promote(env)
    monkeypatch.setattr(registry, "load_drift_reference", lambda ref: None)  # e.g. a champion from before drift
    env["run"]("score", "job-10")
    manifest = json.loads((env["root"] / "runs/fairport/job-10/predict/manifest.json").read_text())
    assert manifest["drift"]["status"] == "no_reference"


def test_score_job_never_uses_a_model_that_trained_on_the_pipe(env):
    train_and_promote(env)
    env["run"]("score", "job-11")
    preds = pd.read_csv(env["root"] / "runs/fairport/job-11/predict/predictions.csv").set_index("pipe_id")
    features = pd.read_parquet(env["root"] / "runs/fairport/job-11/prepare/features.parquet")
    params = env["config"].section("train")
    model = registry.load_model(_champion(env))
    assert isinstance(model, M.CrossFitModel)

    # Independently re-derive which rows each model trained on, the way the train stage split them.
    hold = M.holdout_mask(features, "fairport", float(params["block_size_m"]), int(params["holdout_pct"]))
    train_df = features[~hold]
    blocks = M.spatial_blocks(train_df, float(params["block_size_m"]))
    trained_on = {"full": set(train_df["pipe_id"])}
    for k, (tr, te) in enumerate(GroupKFold(int(params["cv_folds"])).split(train_df, groups=blocks)):
        trained_on[k] = set(train_df["pipe_id"].iloc[tr])
        assert {model.block_fold[b] for b in blocks.iloc[te]} == {k}  # stored map matches the split

    all_blocks = M.spatial_blocks(features, model.block_size_m)
    x = M.model_input(features, json.loads(registry.version_tags(
        registry.client(env["settings"].mlflow_tracking_uri), _champion(env))["feature_spec"]))
    for i, pipe in enumerate(features["pipe_id"]):
        k = model.block_fold.get(all_blocks.iloc[i], "full")
        assert pipe not in trained_on[k]
        assert preds.loc[pipe, "score_source"] == ("full" if k == "full" else "oof")
        scorer = model.full if k == "full" else model.fold_models[k]
        if i % 50 == 0:  # the published score really came from that model
            assert preds.loc[pipe, "leak_risk"] == round(scorer.predict_proba(x.iloc[[i]])[0, 1], 4)

    manifest = json.loads((env["root"] / "runs/fairport/job-11/predict/manifest.json").read_text())
    assert manifest["score_sources"] == {"oof": int((~hold).sum()), "full": int(hold.sum())}


def test_champion_from_before_cross_fitting_scores_everything_with_one_model(env, monkeypatch):
    train_and_promote(env)
    load = registry.load_model
    monkeypatch.setattr(registry, "load_model", lambda ref: load(ref).full)  # a plain sklearn Pipeline
    env["run"]("score", "job-12")
    preds = pd.read_csv(env["root"] / "runs/fairport/job-12/predict/predictions.csv")
    assert set(preds["score_source"]) == {"full"} and np.isfinite(preds["leak_risk"]).all()


def test_export_in_inches_is_refused_by_prepare_using_the_configured_bounds(env):
    raw = env["root"] / "raw" / "fairport" / "pipes.csv"
    df = pd.read_csv(raw)
    df["diameter_mm"] = (df["diameter_mm"] / 25.4).round(1)
    df.to_csv(raw, index=False)
    with pytest.raises(DataValidationError, match="diameter_mm: .*inches"):
        env["run"]("score", "job-13")
    assert not env["storage"].exists("runs/fairport/job-13/prepare/_SUCCESS.json")


def test_gate_report_has_material_slices_and_gates_only_slices_with_enough_leaks(env):
    train_and_promote(env)
    env["run"]("train", "train-3")
    report = json.loads((env["root"] / "runs/fairport/train-3/gate/decision.json").read_text())
    slices = report["slices"]
    assert slices["by"] == "material" and slices["min_leaks"] == 30
    groups = slices["groups"]
    assert {g: s["positives"] for g, s in groups.items()} == {"AC": 27, "CI": 107, "DI": 11, "HDPE": 0, "PVC": 7, "STL": 7}
    assert [g for g, s in groups.items() if s["gated"]] == ["CI"]
    assert groups["HDPE"]["candidate"]["average_precision"] is None
    for name in ("candidate", "champion", "baseline"):
        assert set(groups["CI"][name]) == {"average_precision", "precision_at_10pct"}
    g4 = [c for c in report["checks"] if c["name"].startswith("G4")]
    assert [(c["name"], c["passed"], c["value"]) for c in g4] == [("G4_slice_ap_not_worse[CI]", True, 0.0)]
