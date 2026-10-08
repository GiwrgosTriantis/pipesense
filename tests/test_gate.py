from pipesense.gate import PROMOTED, REJECTED, decide

RULES = {
    "top_k_frac": 0.10,
    "min_lift_over_baseline": 0.0,
    "min_improvement_over_champion": 0.005,
    "max_precision_at_k_drop": 0.02,
    "max_brier_increase": 0.01,
}


def m(ap, p10, brier=0.11):
    return {"average_precision": ap, "precision_at_10pct": p10, "brier": brier}


def test_first_model_must_beat_the_age_rule():
    assert decide(m(0.50, 0.60), m(0.48, 0.50), None, RULES).decision == PROMOTED
    d = decide(m(0.47, 0.60), m(0.48, 0.50), None, RULES)
    assert d.decision == REJECTED and any("G1_ap_beats_age_rule" in r for r in d.reasons)


def test_precision_at_k_must_not_lose_to_the_age_rule():
    d = decide(m(0.50, 0.45), m(0.48, 0.50), None, RULES)
    assert d.decision == REJECTED and any("precision_at_k_vs_age_rule" in r for r in d.reasons)


def test_tie_with_champion_is_rejected_no_churn():
    d = decide(m(0.50, 0.60), m(0.48, 0.50), m(0.50, 0.60), RULES)
    assert d.decision == REJECTED and any("G2_ap_improves_on_champion" in r for r in d.reasons)


def test_clear_improvement_is_promoted():
    assert decide(m(0.53, 0.61), m(0.48, 0.50), m(0.50, 0.60), RULES).decision == PROMOTED


def test_better_ap_but_worse_calibration_is_rejected():
    d = decide(m(0.53, 0.61, brier=0.13), m(0.48, 0.50), m(0.50, 0.60, brier=0.11), RULES)
    assert d.decision == REJECTED and any("calibration" in r for r in d.reasons)


def test_better_ap_but_worse_top_decile_is_rejected():
    d = decide(m(0.53, 0.55), m(0.48, 0.50), m(0.50, 0.60), RULES)
    assert d.decision == REJECTED and any("precision_at_k_not_worse" in r for r in d.reasons)


def test_incomparable_champion_requires_manual_promotion():
    d = decide(m(0.60, 0.70), m(0.48, 0.50), m(0.50, 0.60), RULES, champion_comparable=False)
    assert d.decision == REJECTED and any("G3_champion_comparable" in r for r in d.reasons)


def test_insane_scores_are_rejected():
    d = decide(m(0.60, 0.70), m(0.48, 0.50), None, RULES, candidate_sane=False)
    assert d.decision == REJECTED


def test_report_lists_every_check():
    d = decide(m(0.53, 0.61), m(0.48, 0.50), m(0.50, 0.60), RULES)
    names = [c["name"] for c in d.as_dict()["checks"]]
    assert names == [
        "G0_sane_scores", "G1_ap_beats_age_rule", "G1_precision_at_k_vs_age_rule", "G3_champion_comparable",
        "G2_ap_improves_on_champion", "G2_precision_at_k_not_worse", "G2_calibration_not_worse",
    ]
