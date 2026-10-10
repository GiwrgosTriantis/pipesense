"""Stage: promotion gate. Decides whether the new candidate replaces the champion, and records why.

Inputs:  train/model.json, prepare/features.parquet, the current champion (if any)
Outputs: decision.json, plus an MLflow run in leak-risk/<utility>/gate holding the full report.
         The candidate version is tagged with the decision. On PROMOTED the `champion` alias moves.

Candidate and champion are scored on the SAME spatial holdout, which neither model trained on
(see model.holdout_mask). The rules live in gate.decide() and config/pipeline.toml [gate].
The report also breaks AP and precision@k down per material (slices) for candidate, champion and age rule.
"""

from __future__ import annotations

import json

import numpy as np

from pipesense import gate, registry
from pipesense import model as M
from pipesense.log import get_logger, kv
from pipesense.runner import StageContext
from pipesense.stages.io import read_json, read_parquet, write_json

log = get_logger(__name__)


def _score(c, ref: registry.ModelRef, df) -> np.ndarray:
    spec = json.loads(registry.version_tags(c, ref)["feature_spec"])
    return registry.load_model(ref).predict_proba(M.model_input(df, spec))[:, 1]


def run(ctx: StageContext) -> dict:
    rules = ctx.params
    train_params = ctx.config.section("train")
    k = float(rules["top_k_frac"])
    n_boot = int(rules.get("bootstrap_samples", 1000))

    m = read_json(ctx.storage, ctx.key("train", "model.json"))
    candidate = registry.ModelRef(m["name"], str(m["version"]), m["run_id"])
    df = read_parquet(ctx.storage, ctx.key("prepare", "features.parquet"))
    hold = df[M.holdout_mask(df, ctx.utility, float(train_params["block_size_m"]), int(train_params["holdout_pct"]))]
    y = M.labels(hold)

    c = registry.client(ctx.settings.mlflow_tracking_uri)
    champion = registry.get_champion(c, ctx.utility)

    cand_scores = _score(c, candidate, hold)
    sane = len(cand_scores) == len(hold) and bool(np.isfinite(cand_scores).all()) and \
        bool((cand_scores >= 0).all() and (cand_scores <= 1).all())
    base_scores = M.baseline_scores(hold)
    metrics = {"candidate": M.evaluate(y, cand_scores, k), "baseline": M.evaluate(y, base_scores, k), "champion": None}
    bootstrap = {"candidate_vs_baseline": M.paired_bootstrap_ap_diff(y, cand_scores, base_scores, n_boot)}

    comparable, champ_scores = True, None
    if champion is not None:
        spec_now = M.holdout_spec(train_params)
        spec_champion = registry.version_tags(c, champion).get("holdout_spec")
        comparable = spec_champion == spec_now
        if comparable:
            champ_scores = _score(c, champion, hold)
            metrics["champion"] = M.evaluate(y, champ_scores, k)
            bootstrap["candidate_vs_champion"] = M.paired_bootstrap_ap_diff(y, cand_scores, champ_scores, n_boot)

    slice_by = rules.get("slice_by", "material")
    slices = M.slice_metrics(y, hold[slice_by].to_numpy(), k,
                             {"candidate": cand_scores, "champion": champ_scores, "baseline": base_scores})
    for s in slices.values():
        s["gated"] = s["positives"] >= int(rules["slice_min_leaks"]) and s["champion"] is not None

    decision = gate.decide(
        candidate=metrics["candidate"],
        baseline=metrics["baseline"],
        champion=metrics["champion"] if champion is not None else None,
        rules=rules,
        candidate_sane=sane,
        champion_comparable=comparable,
        slices=slices,
    )
    report = {
        **decision.as_dict(),
        "utility": ctx.utility,
        "job_id": ctx.job_id,
        "candidate": candidate.as_dict(),
        "champion_before": champion.as_dict() if champion else None,
        "holdout": {"spec": M.holdout_spec(train_params), "n": int(len(hold)), "positives": int(y.sum())},
        "rules": rules,
        "metrics": metrics,
        "bootstrap": bootstrap,
        "slices": {"by": slice_by, "min_leaks": int(rules["slice_min_leaks"]),
                   "max_ap_drop": float(rules["slice_max_ap_drop"]), "groups": slices},
    }
    gate_run_id = registry.record_gate_decision(c, ctx.utility, candidate, champion, report,
                                                actor="gate:auto", job_id=ctx.job_id)
    report["gate_run_id"] = gate_run_id
    write_json(ctx.storage, ctx.out("decision.json"), report)
    log.info("gate decision", extra=kv(decision=decision.decision, candidate_version=candidate.version,
                                        champion_version=champion.version if champion else None,
                                        failed=decision.reasons))
    return {"decision": decision.decision, "failed": decision.reasons, "gate_run_id": gate_run_id,
            "candidate_version": candidate.version}
