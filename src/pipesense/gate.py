"""Promotion gate rules. A pure function: metrics in, decision out, so it is unit-testable.

What winning means for a candidate model (thresholds live in config/pipeline.toml [gate]):

  G0 sane        scores every holdout pipe, in [0, 1], with no NaN
  G1 useful      holdout AP >= age-rule AP + min_lift_over_baseline, and precision@top-k is no worse
                 than the age rule
  G2 better      (only when a champion exists) holdout AP >= champion AP + min_improvement,
                 precision@top-k drops by at most max_precision_at_k_drop, and Brier rises by at
                 most max_brier_increase
  G3 comparable  the champion was evaluated on the same holdout definition. If not, there is no
                 automatic decision: the candidate is REJECTED and needs a manual promote.
  G4 no slice    (only when a comparable champion exists) in every slice (material) with at least
     regression  slice_min_leaks holdout leaks, AP drops by at most slice_max_ap_drop vs the champion.
                 Smaller slices are reported but not gated: with a handful of leaks their AP is noise.

A candidate that only ties the champion is rejected, so retraining unchanged data does not churn
the serving model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

PROMOTED = "PROMOTED"
REJECTED = "REJECTED"
MANUAL = "MANUAL_PROMOTION"


@dataclass
class Check:
    name: str
    passed: bool
    value: float | str | None
    threshold: float | str | None
    detail: str = ""


@dataclass
class Decision:
    decision: str
    checks: list[Check] = field(default_factory=list)

    @property
    def promote(self) -> bool:
        return self.decision == PROMOTED

    @property
    def reasons(self) -> list[str]:
        return [f"{c.name}: {c.detail}" for c in self.checks if not c.passed]

    def as_dict(self) -> dict:
        return {"decision": self.decision, "checks": [asdict(c) for c in self.checks], "failed": self.reasons}


def _pk(metrics: dict, top_k_frac: float) -> float:
    return metrics[f"precision_at_{int(top_k_frac * 100)}pct"]


def _slice_checks(slices: dict, rules: dict) -> list[Check]:
    min_leaks, max_drop = int(rules["slice_min_leaks"]), float(rules["slice_max_ap_drop"])
    checks = []
    for group, s in slices.items():
        if s["positives"] < min_leaks or not s.get("champion"):
            continue
        drop = s["champion"]["average_precision"] - s["candidate"]["average_precision"]
        checks.append(Check(f"G4_slice_ap_not_worse[{group}]", drop <= max_drop, round(drop, 4), max_drop,
                            f"AP on {group} ({s['positives']} holdout leaks) drops by {drop:.4f} > {max_drop}"))
    return checks


def decide(
    candidate: dict,
    baseline: dict,
    champion: dict | None,
    rules: dict,
    candidate_sane: bool = True,
    champion_comparable: bool = True,
    slices: dict | None = None,
) -> Decision:
    k = float(rules["top_k_frac"])
    checks = [
        Check("G0_sane_scores", candidate_sane, str(candidate_sane), "True",
              "" if candidate_sane else "candidate produced NaN/out-of-range scores or missed rows"),
    ]

    lift = candidate["average_precision"] - baseline["average_precision"]
    need = float(rules["min_lift_over_baseline"])
    checks.append(Check("G1_ap_beats_age_rule", lift >= need, round(lift, 4), need,
                        f"AP lift over age rule {lift:+.4f} < {need}"))
    pk_lift = _pk(candidate, k) - _pk(baseline, k)
    checks.append(Check("G1_precision_at_k_vs_age_rule", pk_lift >= 0, round(pk_lift, 4), 0.0,
                        f"precision@{k:.0%} is {pk_lift:+.4f} vs age rule"))

    if champion is not None:
        checks.append(Check("G3_champion_comparable", champion_comparable, str(champion_comparable), "True",
                            "champion was evaluated on a different holdout definition; promote manually"))
        if champion_comparable:
            imp = candidate["average_precision"] - champion["average_precision"]
            need_imp = float(rules["min_improvement_over_champion"])
            checks.append(Check("G2_ap_improves_on_champion", imp >= need_imp, round(imp, 4), need_imp,
                                f"AP improvement over champion {imp:+.4f} < {need_imp}"))
            pk_drop = _pk(champion, k) - _pk(candidate, k)
            max_drop = float(rules["max_precision_at_k_drop"])
            checks.append(Check("G2_precision_at_k_not_worse", pk_drop <= max_drop, round(pk_drop, 4), max_drop,
                                f"precision@{k:.0%} drops by {pk_drop:.4f} > {max_drop}"))
            brier_up = candidate["brier"] - champion["brier"]
            max_up = float(rules["max_brier_increase"])
            checks.append(Check("G2_calibration_not_worse", brier_up <= max_up, round(brier_up, 4), max_up,
                                f"Brier score rises by {brier_up:.4f} > {max_up}"))
            if slices:
                checks.extend(_slice_checks(slices, rules))

    return Decision(PROMOTED if all(c.passed for c in checks) else REJECTED, checks)
