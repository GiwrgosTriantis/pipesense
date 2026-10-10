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


SLICE_RULES = {**RULES, "slice_min_leaks": 30, "slice_max_ap_drop": 0.05}


def s(positives, cand_ap, champ_ap):
    return {"positives": positives, "candidate": {"average_precision": cand_ap},
            "champion": {"average_precision": champ_ap}}


def test_regression_on_a_large_slice_is_rejected_even_when_overall_ap_improves():
    slices = {"CI": s(107, 0.50, 0.58), "AC": s(27, 0.40, 0.40)}
    d = decide(m(0.53, 0.61), m(0.48, 0.50), m(0.50, 0.60), SLICE_RULES, slices=slices)
    assert d.decision == REJECTED and d.reasons == [
        "G4_slice_ap_not_worse[CI]: AP on CI (107 holdout leaks) drops by 0.0800 > 0.05"]


def test_small_slices_are_reported_but_not_gated():
    slices = {"CI": s(107, 0.58, 0.57), "PVC": s(7, 0.10, 0.60)}  # PVC collapses, but 7 leaks is noise
    d = decide(m(0.53, 0.61), m(0.48, 0.50), m(0.50, 0.60), SLICE_RULES, slices=slices)
    assert d.decision == PROMOTED
    assert [c.name for c in d.checks if c.name.startswith("G4")] == ["G4_slice_ap_not_worse[CI]"]


def test_slice_drop_within_tolerance_passes_and_no_champion_means_no_slice_checks():
    d = decide(m(0.53, 0.61), m(0.48, 0.50), m(0.50, 0.60), SLICE_RULES, slices={"CI": s(107, 0.54, 0.58)})
    assert d.decision == PROMOTED
    first = decide(m(0.53, 0.61), m(0.48, 0.50), None, SLICE_RULES, slices={"CI": s(107, 0.20, 0.58)})
    assert first.decision == PROMOTED and not any(c.name.startswith("G4") for c in first.checks)
