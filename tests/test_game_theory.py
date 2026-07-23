"""Tests for the decision-tree engine: normalization, sanitizing, backward induction."""
import math

import pytest

from models import Analysis, NodeType, Outcome, TreeNode
from game_theory import (
    compute_expected_values,
    infer_node_types,
    normalize_outcomes,
    normalize_tree_probabilities,
    process_analysis,
    reconnect_orphan_roots,
    sanitize_tree,
)


def node(id, type="outcome", children=None, probability=None, payoff=None, label=None):
    return TreeNode(
        id=id, label=label or id, type=type, children=children or [],
        probability=probability, payoff=payoff,
    )


# --------------------------------------------------------------------------- #
# normalize_outcomes
# --------------------------------------------------------------------------- #
def test_normalize_outcomes_sums_to_one():
    # Valid probabilities (<=1) that don't yet sum to 1 -> rescaled to 0.5 each.
    outs = [Outcome(description="a", probability=0.6, recommendation="x"),
            Outcome(description="b", probability=0.6, recommendation="y")]
    normalize_outcomes(outs)
    assert math.isclose(sum(o.probability for o in outs), 1.0)
    assert all(math.isclose(o.probability, 0.5) for o in outs)


def test_normalize_outcomes_all_zero_becomes_equal():
    outs = [Outcome(description="a", probability=0, recommendation="x"),
            Outcome(description="b", probability=0, recommendation="y"),
            Outcome(description="c", probability=0, recommendation="z")]
    normalize_outcomes(outs)
    assert all(math.isclose(o.probability, 1 / 3) for o in outs)


def test_normalize_outcomes_empty():
    assert normalize_outcomes([]) == []


# --------------------------------------------------------------------------- #
# normalize_tree_probabilities
# --------------------------------------------------------------------------- #
def test_normalize_tree_probabilities_siblings_sum_to_one():
    nodes = [
        node("root", "chance", ["a", "b"]),
        node("a", "outcome", probability=0.6, payoff=10),
        node("b", "outcome", probability=0.2, payoff=5),
    ]
    normalize_tree_probabilities(nodes)
    a = next(n for n in nodes if n.id == "a")
    b = next(n for n in nodes if n.id == "b")
    assert math.isclose(a.probability + b.probability, 1.0)
    assert math.isclose(a.probability, 0.75)  # 0.6 / (0.6 + 0.2)


def test_normalize_leaves_probability_less_children_untouched():
    # A decision node's children carry no probability; they must stay None.
    nodes = [
        node("root", "decision", ["a", "b"]),
        node("a", "outcome", payoff=1),
        node("b", "outcome", payoff=2),
    ]
    normalize_tree_probabilities(nodes)
    assert all(n.probability is None for n in nodes if n.id in ("a", "b"))


# --------------------------------------------------------------------------- #
# sanitize_tree
# --------------------------------------------------------------------------- #
def test_sanitize_merges_duplicate_ids():
    nodes = [
        node("x", "outcome", []),                       # bare duplicate
        node("x", "decision", ["y"], label="Longer X"),  # informative duplicate
        node("y", "outcome", [], payoff=5),
    ]
    out = sanitize_tree(nodes)
    ids = [n.id for n in out]
    assert ids.count("x") == 1
    x = next(n for n in out if n.id == "x")
    # Prefers the structural type, children, and longer label.
    assert x.type == NodeType.decision
    assert x.children == ["y"]
    assert x.label == "Longer X"


def test_sanitize_drops_dangling_and_self_refs():
    nodes = [
        node("root", "decision", ["a", "ghost", "root"]),  # ghost missing, self-loop
        node("a", "outcome", [], payoff=1),
    ]
    out = sanitize_tree(nodes)
    root = next(n for n in out if n.id == "root")
    assert root.children == ["a"]


# --------------------------------------------------------------------------- #
# reconnect_orphan_roots
# --------------------------------------------------------------------------- #
def test_reconnect_orphan_roots_single_root_unchanged():
    nodes = [node("root", "decision", ["a"]), node("a", "outcome", [], payoff=1)]
    out = reconnect_orphan_roots(nodes)
    root = next(n for n in out if n.id == "root")
    assert root.children == ["a"]


def test_reconnect_orphan_roots_attaches_extra_root():
    nodes = [
        node("root", "decision", ["a"]),
        node("a", "outcome", [], payoff=1),
        node("orphan", "decision", ["b"]),   # not referenced anywhere
        node("b", "outcome", [], payoff=2),
    ]
    out = reconnect_orphan_roots(nodes)
    root = next(n for n in out if n.id == "root")
    assert "orphan" in root.children


# --------------------------------------------------------------------------- #
# compute_expected_values (backward induction)
# --------------------------------------------------------------------------- #
def test_leaf_value_is_payoff():
    nodes = [node("root", "outcome", [], payoff=42)]
    compute_expected_values(nodes)
    assert nodes[0].expected_value == 42


def test_chance_node_is_probability_weighted():
    nodes = [
        node("root", "chance", ["good", "bad"]),
        node("good", "outcome", [], probability=0.7, payoff=100),
        node("bad", "outcome", [], probability=0.3, payoff=0),
    ]
    nodes = normalize_tree_probabilities(nodes)
    compute_expected_values(nodes)
    root = next(n for n in nodes if n.id == "root")
    assert math.isclose(root.expected_value, 70.0)


def test_decision_node_takes_max_and_marks_only_optimal_path():
    nodes = [
        node("root", "decision", ["hi", "lo"]),
        node("hi", "outcome", [], payoff=100),
        node("lo", "outcome", [], payoff=10),
    ]
    _, label, ev = compute_expected_values(nodes)
    root = next(n for n in nodes if n.id == "root")
    hi = next(n for n in nodes if n.id == "hi")
    lo = next(n for n in nodes if n.id == "lo")
    assert ev == 100 and label == "hi"
    assert root.is_optimal and hi.is_optimal
    assert not lo.is_optimal  # the losing branch must NOT be highlighted


def test_optimal_path_does_not_leak_into_unchosen_decision():
    # root -> {A (decision), B (decision)}; root picks the better of A/B, and only
    # that subtree's best child should be marked.
    nodes = [
        node("root", "decision", ["A", "B"]),
        node("A", "decision", ["a1", "a2"]),
        node("a1", "outcome", [], payoff=95),
        node("a2", "outcome", [], payoff=-50),
        node("B", "decision", ["b1"]),
        node("b1", "outcome", [], payoff=80),
    ]
    compute_expected_values(nodes)
    ids = {n.id: n for n in nodes}
    assert ids["A"].is_optimal and ids["a1"].is_optimal
    assert not ids["B"].is_optimal      # B (EV 80) loses to A (EV 95)
    assert not ids["b1"].is_optimal     # so its child isn't on the optimal path


def test_chance_children_missing_probability_use_equal_weight():
    nodes = [
        node("root", "chance", ["a", "b"]),
        node("a", "outcome", [], payoff=10),
        node("b", "outcome", [], payoff=20),
    ]
    compute_expected_values(nodes)
    root = next(n for n in nodes if n.id == "root")
    assert math.isclose(root.expected_value, 15.0)  # 0.5*10 + 0.5*20


def test_recompute_clears_stale_optimal_path():
    # Simulates the sensitivity flow: compute, change a payoff, recompute. The
    # previously-optimal branch must no longer be flagged.
    nodes = [
        node("root", "decision", ["hi", "lo"]),
        node("hi", "outcome", [], payoff=100),
        node("lo", "outcome", [], payoff=10),
    ]
    compute_expected_values(nodes)
    ids = {n.id: n for n in nodes}
    assert ids["hi"].is_optimal and not ids["lo"].is_optimal

    # Flip the payoffs: "lo" is now the better branch.
    ids["hi"].payoff = 5
    _, label, ev = compute_expected_values(nodes)
    assert ev == 10 and label == "lo"
    assert ids["lo"].is_optimal
    assert not ids["hi"].is_optimal  # stale flag must be cleared


def test_cycle_guard_terminates():
    nodes = [
        node("root", "decision", ["a"]),
        node("a", "chance", ["root"]),  # malformed cycle back to root
    ]
    # Should not hang / recurse infinitely.
    compute_expected_values(nodes)


# --------------------------------------------------------------------------- #
# infer_node_types
# --------------------------------------------------------------------------- #
def test_gamble_mislabeled_as_decision_is_reclassified():
    # "Push hard" is really a gamble: its branches carry probabilities.
    nodes = [
        node("root", "decision", ["push"]),
        node("push", "decision", ["win", "lose"]),
        node("win", "outcome", [], probability=0.5, payoff=100),
        node("lose", "outcome", [], probability=0.5, payoff=-100),
    ]
    _, warnings = infer_node_types(nodes)
    push = next(n for n in nodes if n.id == "push")
    assert push.type == NodeType.chance
    assert any("chance event" in w for w in warnings)


def test_reclassified_gamble_no_longer_ignores_downside():
    # The whole point: as a "decision" this returns max(100, -100) = 100.
    # As a chance node it must return the probability-weighted 0.
    raw = [
        node("root", "decision", ["push"]),
        node("push", "decision", ["win", "lose"]),
        node("win", "outcome", [], probability=0.5, payoff=100),
        node("lose", "outcome", [], probability=0.5, payoff=-100),
    ]
    a = process_analysis(Analysis(decision_tree=raw))
    assert a.optimal_expected_value == 0.0   # not 100
    assert a.warnings


def test_root_decision_keeps_type_and_drops_spurious_probabilities():
    nodes = [
        node("root", "decision", ["a", "b"]),
        node("a", "outcome", [], probability=0.6, payoff=10),
        node("b", "outcome", [], probability=0.4, payoff=20),
    ]
    _, warnings = infer_node_types(nodes)
    root = next(n for n in nodes if n.id == "root")
    assert root.type == NodeType.decision          # root stays a choice
    assert all(n.probability is None for n in nodes if n.id in ("a", "b"))
    assert any("choices you control" in w for w in warnings)


def test_childless_non_outcome_becomes_outcome():
    nodes = [node("root", "decision", ["x"]), node("x", "chance", [], payoff=5)]
    _, warnings = infer_node_types(nodes)
    assert next(n for n in nodes if n.id == "x").type == NodeType.outcome
    assert any("final outcome" in w for w in warnings)


def test_chance_without_probabilities_is_flagged_not_changed():
    nodes = [
        node("root", "chance", ["a", "b"]),
        node("a", "outcome", [], payoff=10),
        node("b", "outcome", [], payoff=20),
    ]
    _, warnings = infer_node_types(nodes)
    assert next(n for n in nodes if n.id == "root").type == NodeType.chance
    assert any("equally likely" in w for w in warnings)


def test_well_formed_tree_produces_no_warnings():
    nodes = [
        node("root", "decision", ["gamble", "safe"]),
        node("gamble", "chance", ["win", "lose"]),
        node("win", "outcome", [], probability=0.5, payoff=100),
        node("lose", "outcome", [], probability=0.5, payoff=0),
        node("safe", "outcome", [], payoff=40),
    ]
    _, warnings = infer_node_types(nodes)
    assert warnings == []


# --------------------------------------------------------------------------- #
# process_analysis integration
# --------------------------------------------------------------------------- #
def test_process_analysis_cleans_messy_tree():
    raw = [
        node("root", "decision", ["d1", "d2"]),
        node("d1", "decision", ["w", "l"]),
        node("w", "outcome", [], payoff=95),
        node("l", "outcome", [], payoff=-50),
        node("d2", "outcome", []),               # duplicate, bare
        node("d2", "decision", ["s", "ghost"]),  # duplicate + dangling ref
        node("s", "outcome", [], payoff=80),
    ]
    a = process_analysis(Analysis(decision_tree=raw))
    ids = [n.id for n in a.decision_tree]
    assert len(ids) == len(set(ids))                      # de-duplicated
    child_ids = {c for n in a.decision_tree for c in n.children}
    roots = [n.id for n in a.decision_tree if n.id not in child_ids]
    assert roots == ["root"]                               # single root
    assert all(c in set(ids) for n in a.decision_tree for c in n.children)  # no dangling
    assert a.optimal_decision == "d1" and a.optimal_expected_value == 95
