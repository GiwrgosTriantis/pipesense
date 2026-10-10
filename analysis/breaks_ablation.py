"""One-off ablation for A2 / M4: how much of the model's lift over the age rule needs num_breaks_5yr?

Trains the champion's model with and without num_breaks_5yr on the same spatial training blocks and
evaluates both, plus the age rule, on the same holdout the gate uses. Nothing is registered, no gate
runs, config/pipeline.toml is only read. Uses the pipeline's own feature, split, model and metric code,
so the full model reproduces the champion's holdout numbers exactly (checked below).

Usage, from the repo root, in the pipeline image (no local Python needed):
    docker run --rm -v "$PWD":/repo:ro --entrypoint python pipesense:dev /repo/analysis/breaks_ablation.py
or locally:  uv run python analysis/breaks_ablation.py [utility]
"""

import json
import sys
import tomllib
from datetime import date
from pathlib import Path

import pandas as pd

from pipesense import model as M
from pipesense.features import build_feature_table

REPO = Path(__file__).resolve().parents[1]
utility = sys.argv[1] if len(sys.argv) > 1 else "fairport"
cfg = tomllib.loads((REPO / "config" / "pipeline.toml").read_text())
prep, train, gate = cfg["prepare"], cfg["train"], cfg["gate"]

raw_dir = REPO / "data" / "raw" / utility
as_of = date.fromisoformat(json.loads((raw_dir / "meta.json").read_text())["as_of"]).year
df, _ = build_feature_table(
    pd.read_csv(raw_dir / "pipes.csv"), as_of_year=as_of,
    max_unknown_material_frac=float(prep["max_unknown_material_frac"]),
    max_dropped_frac=float(prep["max_dropped_frac"]), neighbour_radius_m=float(prep["neighbour_radius_m"]),
)
hold = M.holdout_mask(df, utility, float(train["block_size_m"]), int(train["holdout_pct"]))
train_df, hold_df = df[~hold], df[hold]
y = M.labels(hold_df)
k, n_boot = float(gate["top_k_frac"]), int(gate["bootstrap_samples"])


def fit_and_score(params: dict):
    model = M.build_model(params).fit(M.model_input(train_df, params), M.labels(train_df))
    return model.predict_proba(M.model_input(hold_df, params))[:, 1]


without = {**train, "numeric_features": [f for f in train["numeric_features"] if f != "num_breaks_5yr"]}
scores = {
    "full model": fit_and_score(train),
    "without num_breaks_5yr": fit_and_score(without),
    "age rule": M.baseline_scores(hold_df),
}
metrics = {name: M.evaluate(y, s, k) for name, s in scores.items()}

print(f"{utility}: {len(train_df)} training pipes, holdout {len(hold_df)} pipes / {int(y.sum())} leaks, "
      f"bootstrap {n_boot} paired resamples (as in the gate)\n")
print(f"{'':24s} {'AP':>6s} {'P@10%':>6s} {'Brier':>6s}   AP lift over age rule [90% CI]   P(lift > 0)")
for name, m in metrics.items():
    brier = f"{m['brier']:.3f}" if "brier" in m else "   n/a"  # the age rule's scores are not probabilities
    line = f"{name:24s} {m['average_precision']:6.3f} {m['precision_at_10pct']:6.3f} {brier:>6s}"
    if name != "age rule":
        b = M.paired_bootstrap_ap_diff(y, scores[name], scores["age rule"], n_boot)
        line += f"   {b['mean']:+.3f} [{b['ci90_low']:+.3f}, {b['ci90_high']:+.3f}]          {b['p_better']:.2f}"
    print(line)
d = M.paired_bootstrap_ap_diff(y, scores["full model"], scores["without num_breaks_5yr"], n_boot)
print(f"\nfull minus without num_breaks_5yr: AP {d['mean']:+.3f} [{d['ci90_low']:+.3f}, {d['ci90_high']:+.3f}]")

if utility == "fairport":  # the full model must be the champion's model, or the comparison is meaningless
    full = metrics["full model"]
    assert round(full["average_precision"], 3) == 0.495 and round(full["brier"], 3) == 0.112, full
