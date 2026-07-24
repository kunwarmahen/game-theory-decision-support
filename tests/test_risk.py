"""Tests for risk preferences (exponential/CARA utility and certainty equivalents)."""
import math

import pytest

from models import Analysis, NodeType, TreeNode
from game_theory import (
    _certainty_equivalent,
    compute_expected_values,
    process_analysis,
    risk_tolerance_for,
)


def gamble_vs_certainty(sure_thing=0.0, up=100.0, down=-100.0, p=0.5):
    """Root choice between a coin-flip and a guaranteed payoff of the same EV."""
    return [
        TreeNode(id="root", label="Choose", type=NodeType.decision,
                 children=["gamble", "safe"]),
        TreeNode(id="gamble", label="Gamble", type=NodeType.chance,
                 children=["win", "lose"]),
        TreeNode(id="win", label="Win", type=NodeType.outcome, probability=p, payoff=up),
        TreeNode(id="lose", label="Lose", type=NodeType.outcome,
                 probability=1 - p, payoff=down),
        TreeNode(id="safe", label="Safe", type=NodeType.outcome, payoff=sure_thing),
    ]


def by_id(nodes):
    return {n.id: n for n in nodes}


# --- The utility transform itself -----------------------------------------

def test_certainty_equivalent_below_mean_when_risk_averse():
    ce = _certainty_equivalent([(0.5, 100.0), (0.5, -100.0)], R=50.0)
    assert ce < 0.0  # mean is 0; a risk-averse decider values the flip below it


def test_certainty_equivalent_above_mean_when_risk_seeking():
    assert _certainty_equivalent([(0.5, 100.0), (0.5, -100.0)], R=-50.0) > 0.0


def test_certain_outcome_is_worth_its_face_value():
    assert math.isclose(_certainty_equivalent([(1.0, 42.0)], R=10.0), 42.0, abs_tol=1e-6)


def test_high_tolerance_approaches_risk_neutral():
    # As R grows the transform flattens and the CE converges on the mean.
    ce = _certainty_equivalent([(0.5, 100.0), (0.5, -100.0)], R=1e7)
    assert math.isclose(ce, 0.0, abs_tol=1e-2)


def test_extreme_payoff_to_tolerance_ratio_does_not_overflow():
    # Without a log-sum-exp shift exp(1e5) overflows and the CE collapses.
    ce = _certainty_equivalent([(0.5, 1e5), (0.5, -1e5)], R=1.0)
    assert math.isfinite(ce)
    assert ce < 0


def test_certainty_equivalent_of_empty_lottery():
    assert _certainty_equivalent([], R=10.0) == 0.0


# --- Scaling risk aversion to the payoffs in play --------------------------

def test_risk_neutral_has_no_tolerance():
    assert risk_tolerance_for(gamble_vs_certainty(), 0.0) is None


def test_no_tolerance_when_payoffs_are_identical():
    flat = [TreeNode(id=f"n{i}", label="x", payoff=5.0) for i in range(3)]
    assert risk_tolerance_for(flat, 1.0) is None


def test_tolerance_scales_with_payoff_spread():
    # Same aversion, payoffs 10x larger -> tolerance 10x larger, so the *relative*
    # penalty is unchanged. Payoff scale is arbitrary per analysis, so it must be.
    small = risk_tolerance_for(gamble_vs_certainty(up=10, down=-10), 0.5)
    large = risk_tolerance_for(gamble_vs_certainty(up=100, down=-100), 0.5)
    assert math.isclose(large, small * 10, rel_tol=1e-9)


def test_stronger_aversion_lowers_tolerance():
    assert risk_tolerance_for(gamble_vs_certainty(), 1.0) < risk_tolerance_for(
        gamble_vs_certainty(), 0.25
    )


# --- Backward induction under risk preference ------------------------------

def test_risk_neutral_is_indifferent_and_reports_no_ce():
    nodes, label, ev, ce = compute_expected_values(gamble_vs_certainty(), 0.0)
    assert ev == 0.0            # both branches have EV 0
    assert ce is None           # nothing to report when risk-neutral
    assert all(n.certainty_equivalent is None for n in nodes)


def test_risk_aversion_switches_the_recommendation_to_the_sure_thing():
    # EV is a tie; only risk preference can break it.
    nodes, label, ev, ce = compute_expected_values(gamble_vs_certainty(), 0.6)
    assert label == "Safe"
    assert by_id(nodes)["safe"].is_optimal
    assert not by_id(nodes)["gamble"].is_optimal
    assert math.isclose(ce, 0.0)  # the chosen branch is certain, so CE == payoff


def test_risk_seeking_switches_the_recommendation_to_the_gamble():
    nodes, label, ev, ce = compute_expected_values(gamble_vs_certainty(), -0.6)
    assert label == "Gamble"
    assert by_id(nodes)["gamble"].is_optimal
    assert ce > 0.0  # a risk-seeker values the flip above its mean of 0


def test_risk_averse_decider_gives_up_expected_value_for_safety():
    # The gamble has the higher EV, but a risk-averse decider still declines it.
    nodes = gamble_vs_certainty(sure_thing=10.0, up=100.0, down=-70.0)
    _, label, ev, ce = compute_expected_values(nodes, 1.0)
    assert label == "Safe"
    assert ev == 10.0        # EV *of the chosen path*, not the abandoned gamble
    assert ce == 10.0


def test_ce_sits_below_ev_on_a_risky_chosen_path():
    # Only one branch, so the gamble is taken regardless; the CE prices its risk.
    nodes = [
        TreeNode(id="root", label="Gamble", type=NodeType.chance, children=["a", "b"]),
        TreeNode(id="a", label="Up", type=NodeType.outcome, probability=0.5, payoff=100),
        TreeNode(id="b", label="Down", type=NodeType.outcome, probability=0.5, payoff=0),
    ]
    _, _, ev, ce = compute_expected_values(nodes, 0.8)
    assert math.isclose(ev, 50.0)
    assert 0.0 < ce < 50.0
    # The risk premium is what the decider would pay to avoid the uncertainty.
    assert ev - ce > 0


def test_certainty_equivalents_are_annotated_on_every_node():
    nodes, _, _, _ = compute_expected_values(gamble_vs_certainty(), 0.5)
    assert all(n.certainty_equivalent is not None for n in nodes)
    # Leaves are certain, so their CE is exactly their payoff.
    leaves = by_id(nodes)
    assert leaves["win"].certainty_equivalent == 100.0
    assert leaves["safe"].certainty_equivalent == 0.0


def test_nested_chance_nodes_fold_consistently():
    # CARA's recursive property: a two-stage 50/50 over {100, 0} must price the
    # same as the equivalent single-stage lottery with the same leaves.
    nested = [
        TreeNode(id="root", label="Stage 1", type=NodeType.chance, children=["s2", "c"]),
        TreeNode(id="s2", label="Stage 2", type=NodeType.chance, probability=0.5,
                 children=["a", "b"]),
        TreeNode(id="a", label="A", type=NodeType.outcome, probability=0.5, payoff=100),
        TreeNode(id="b", label="B", type=NodeType.outcome, probability=0.5, payoff=0),
        TreeNode(id="c", label="C", type=NodeType.outcome, probability=0.5, payoff=100),
    ]
    flat = [
        TreeNode(id="root", label="One shot", type=NodeType.chance,
                 children=["a", "b", "c"]),
        TreeNode(id="a", label="A", type=NodeType.outcome, probability=0.25, payoff=100),
        TreeNode(id="b", label="B", type=NodeType.outcome, probability=0.25, payoff=0),
        TreeNode(id="c", label="C", type=NodeType.outcome, probability=0.5, payoff=100),
    ]
    _, _, _, nested_ce = compute_expected_values(nested, 0.7)
    _, _, _, flat_ce = compute_expected_values(flat, 0.7)
    assert math.isclose(nested_ce, flat_ce, rel_tol=1e-9)


def test_recomputing_clears_stale_certainty_equivalents():
    nodes = gamble_vs_certainty()
    compute_expected_values(nodes, 0.5)
    assert any(n.certainty_equivalent is not None for n in nodes)
    compute_expected_values(nodes, 0.0)
    assert all(n.certainty_equivalent is None for n in nodes)


# --- Through process_analysis ---------------------------------------------

def test_process_analysis_threads_risk_aversion_through():
    a = process_analysis(
        Analysis(decision_tree=gamble_vs_certainty()), risk_aversion=0.6
    )
    assert a.risk_aversion == 0.6
    assert a.optimal_decision == "Safe"
    assert a.optimal_certainty_equivalent is not None


def test_process_analysis_defaults_to_risk_neutral():
    a = process_analysis(Analysis(decision_tree=gamble_vs_certainty()))
    assert a.risk_aversion == 0.0
    assert a.optimal_certainty_equivalent is None


@pytest.mark.parametrize("aversion", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_every_slider_position_produces_a_finite_recommendation(aversion):
    _, label, ev, ce = compute_expected_values(gamble_vs_certainty(), aversion)
    assert label in {"Safe", "Gamble"}
    assert math.isfinite(ev)
    assert ce is None or math.isfinite(ce)
