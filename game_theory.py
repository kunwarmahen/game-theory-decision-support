"""
Game-theory engine: probability normalization and expected-value computation.

The headline feature here is `compute_expected_values`, which runs backward
induction over the decision tree instead of trusting the LLM to pick the "best"
outcome. This gives a deterministic, defensible Expected Value for every node and
identifies the EV-maximizing path.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from models import (
    Analysis,
    NashEquilibrium,
    NodeType,
    Outcome,
    PayoffMatrix,
    TreeNode,
)

logger = logging.getLogger(__name__)

_EPS = 1e-9


def normalize_outcomes(outcomes: List[Outcome]) -> List[Outcome]:
    """Scale outcome probabilities so they sum to 1.0."""
    if not outcomes:
        return outcomes

    total = sum(o.probability for o in outcomes)
    if total > 0:
        for o in outcomes:
            o.probability = o.probability / total
    else:
        equal = 1.0 / len(outcomes)
        for o in outcomes:
            o.probability = equal
    return outcomes


def normalize_tree_probabilities(nodes: List[TreeNode]) -> List[TreeNode]:
    """Normalize sibling probabilities so each parent's children sum to 1.0.

    Only children that carry a probability participate in normalization (a
    decision node's children are choices, not chance branches, and are left
    alone).
    """
    lookup: Dict[str, TreeNode] = {n.id: n for n in nodes}

    for node in nodes:
        prob_children = [
            lookup[c]
            for c in node.children
            if c in lookup and lookup[c].probability is not None
        ]
        if not prob_children:
            continue

        total = sum(c.probability or 0.0 for c in prob_children)
        if total > 0:
            for c in prob_children:
                c.probability = (c.probability or 0.0) / total
        else:
            equal = 1.0 / len(prob_children)
            for c in prob_children:
                c.probability = equal
    return nodes


def sanitize_tree(nodes: List[TreeNode]) -> List[TreeNode]:
    """Make a messy LLM-produced tree well-formed.

    Local models sometimes emit duplicate ids (with conflicting definitions) or
    reference children that don't exist. We:
    1. Merge duplicate-id nodes into one, preferring the more informative values.
    2. Drop child references that don't resolve to a node, plus self-loops.
    """
    merged: Dict[str, TreeNode] = {}
    order: List[str] = []
    for n in nodes:
        if n.id not in merged:
            merged[n.id] = n
            order.append(n.id)
            continue
        # Merge into the existing node, keeping the most informative fields.
        existing = merged[n.id]
        if not existing.children and n.children:
            existing.children = n.children
        if existing.payoff is None and n.payoff is not None:
            existing.payoff = n.payoff
        if existing.probability is None and n.probability is not None:
            existing.probability = n.probability
        # Prefer a structural type (decision/chance) over a plain outcome.
        if existing.type == NodeType.outcome and n.type != NodeType.outcome:
            existing.type = n.type
        if len(n.label) > len(existing.label):
            existing.label = n.label
        logger.info("Merged duplicate tree node id '%s'", n.id)

    deduped = [merged[i] for i in order]
    valid_ids = set(merged)
    for n in deduped:
        cleaned = [c for c in n.children if c in valid_ids and c != n.id]
        if len(cleaned) != len(n.children):
            logger.info("Dropped %d dangling/self child ref(s) on '%s'",
                        len(n.children) - len(cleaned), n.id)
        n.children = cleaned
    return deduped


def _find_roots(nodes: List[TreeNode]) -> List[TreeNode]:
    """Return nodes that are never referenced as a child (tree roots)."""
    child_ids = {c for n in nodes for c in n.children}
    roots = [n for n in nodes if n.id not in child_ids]
    # Fallback: an explicit "root" id, else the first node.
    if not roots and nodes:
        roots = [next((n for n in nodes if n.id == "root"), nodes[0])]
    return roots


def reconnect_orphan_roots(nodes: List[TreeNode]) -> List[TreeNode]:
    """Repair disconnected trees so there is a single connected root.

    LLMs sometimes emit alternative strategy branches without linking them to the
    initial decision, leaving floating subtrees. We attach those extra roots as
    children of the primary root (preferring an id of "root", otherwise the first
    decision node) so they render cleanly and participate in the EV comparison.
    """
    roots = _find_roots(nodes)
    if len(roots) <= 1:
        return nodes

    primary = (
        next((r for r in roots if r.id == "root"), None)
        or next((r for r in roots if r.type == NodeType.decision), None)
        or roots[0]
    )
    for r in roots:
        if r.id != primary.id and r.id not in primary.children:
            primary.children.append(r.id)
            logger.info("Reconnected orphan subtree '%s' under root '%s'", r.id, primary.id)
    return nodes


def compute_expected_values(
    nodes: List[TreeNode],
) -> Tuple[List[TreeNode], Optional[str], Optional[float]]:
    """Backward induction over the decision tree.

    Node value rules:
    - outcome/leaf: payoff (default 0.0)
    - chance node:  sum(child.probability * child.value)
    - decision node: max(child.value); the maximizing child is marked optimal

    Returns the annotated nodes plus the optimal first-decision label and its EV.
    """
    lookup: Dict[str, TreeNode] = {n.id: n for n in nodes}
    visiting: set[str] = set()
    best_child: Dict[str, str] = {}  # decision id -> its EV-maximizing child id

    def value_of(node_id: str) -> float:
        node = lookup.get(node_id)
        if node is None:
            return 0.0
        if node_id in visiting:  # cycle guard against malformed trees
            logger.warning("Cycle detected at node '%s'; treating as leaf", node_id)
            return node.payoff or 0.0
        visiting.add(node_id)

        real_children = [c for c in node.children if c in lookup]

        if not real_children:
            # Terminal node: value is its payoff (probability-weighting is applied
            # by the parent chance node, not here).
            node.expected_value = node.payoff if node.payoff is not None else 0.0
            visiting.discard(node_id)
            return node.expected_value

        if node.type == NodeType.decision:
            best_val = float("-inf")
            chosen = None
            for cid in real_children:
                cval = value_of(cid)
                if cval > best_val:
                    best_val, chosen = cval, cid
            node.expected_value = best_val if chosen is not None else 0.0
            if chosen is not None:
                best_child[node_id] = chosen
            visiting.discard(node_id)
            return node.expected_value

        # chance node (or anything with children that isn't a decision):
        # expected value = weighted sum over children.
        total = 0.0
        for cid in real_children:
            child = lookup[cid]
            prob = child.probability if child.probability is not None else (
                1.0 / len(real_children)
            )
            total += prob * value_of(cid)
        node.expected_value = total
        visiting.discard(node_id)
        return node.expected_value

    # First pass: compute expected values for every node.
    roots = _find_roots(nodes)
    primary_root: Optional[TreeNode] = None
    optimal_ev: Optional[float] = None
    for root in roots:
        ev = value_of(root.id)
        if optimal_ev is None or ev > optimal_ev:
            optimal_ev, primary_root = ev, root

    # Second pass: mark ONLY the optimal path from the chosen root. At decisions we
    # follow the single best child; at chance nodes every child is a possible
    # consequence of the chosen strategy, so all are on-path.
    marking: set[str] = set()

    def mark(node_id: str) -> None:
        if node_id in marking:
            return
        marking.add(node_id)
        node = lookup.get(node_id)
        if node is None:
            return
        node.is_optimal = True
        if node.type == NodeType.decision:
            if node_id in best_child:
                mark(best_child[node_id])
        else:
            for cid in node.children:
                if cid in lookup:
                    mark(cid)

    optimal_label: Optional[str] = None
    if primary_root is not None:
        mark(primary_root.id)
        if primary_root.type == NodeType.decision and primary_root.id in best_child:
            optimal_label = lookup[best_child[primary_root.id]].label
        else:
            optimal_label = primary_root.label

    return nodes, optimal_label, optimal_ev


def compute_nash_equilibria(matrix: PayoffMatrix) -> List[NashEquilibrium]:
    """Compute Nash equilibria of a two-player normal-form game.

    - Pure-strategy equilibria for any n x m game, via mutual best response.
    - Mixed-strategy equilibrium for 2 x 2 games (where a pure NE may not exist,
      e.g. Matching Pennies), by making each player indifferent.
    """
    rows = matrix.row_strategies
    cols = matrix.col_strategies
    if not rows or not cols or not matrix.cells:
        return []

    # Build payoff lookup: (row_strategy, col_strategy) -> (row_payoff, col_payoff)
    pay: Dict[tuple, tuple] = {
        (c.row_strategy, c.col_strategy): (c.row_payoff, c.col_payoff)
        for c in matrix.cells
    }
    # Require a fully specified matrix.
    if any((r, c) not in pay for r in rows for c in cols):
        logger.warning("Payoff matrix is incomplete; skipping Nash computation")
        return []

    equilibria: List[NashEquilibrium] = []

    # --- Pure-strategy equilibria -----------------------------------------
    for r in rows:
        for c in cols:
            rp, cp = pay[(r, c)]
            # Row player can't do better by switching rows (column c fixed).
            row_best = max(pay[(r2, c)][0] for r2 in rows)
            # Column player can't do better by switching columns (row r fixed).
            col_best = max(pay[(r, c2)][1] for c2 in cols)
            if rp >= row_best - _EPS and cp >= col_best - _EPS:
                equilibria.append(NashEquilibrium(
                    kind="pure",
                    profile=f"{r} / {c}",
                    row_strategy=r, col_strategy=c,
                    row_payoff=rp, col_payoff=cp,
                    description=(
                        f"Neither player can improve by unilaterally deviating: "
                        f"{matrix.player_row or 'Row'} plays '{r}', "
                        f"{matrix.player_col or 'Column'} plays '{c}'."
                    ),
                ))

    # --- Mixed-strategy equilibrium for 2x2 games -------------------------
    if len(rows) == 2 and len(cols) == 2:
        mixed = _mixed_2x2(rows, cols, pay, matrix)
        if mixed is not None:
            equilibria.append(mixed)

    return equilibria


def _mixed_2x2(rows, cols, pay, matrix) -> Optional[NashEquilibrium]:
    """Interior mixed-strategy NE of a 2x2 game, if one exists in (0,1)."""
    r0, r1 = rows
    c0, c1 = cols
    # Row payoffs a[i][j], column payoffs b[i][j].
    a = [[pay[(r0, c0)][0], pay[(r0, c1)][0]], [pay[(r1, c0)][0], pay[(r1, c1)][0]]]
    b = [[pay[(r0, c0)][1], pay[(r0, c1)][1]], [pay[(r1, c0)][1], pay[(r1, c1)][1]]]

    # q = P(column plays c0) that makes ROW indifferent between r0 and r1.
    den_q = a[0][0] - a[0][1] - a[1][0] + a[1][1]
    # p = P(row plays r0) that makes COLUMN indifferent between c0 and c1.
    den_p = b[0][0] - b[0][1] - b[1][0] + b[1][1]
    if abs(den_q) < _EPS or abs(den_p) < _EPS:
        return None  # degenerate; no interior mixed NE

    q = (a[1][1] - a[0][1]) / den_q
    p = (b[1][1] - b[1][0]) / den_p
    if not (_EPS < p < 1 - _EPS and _EPS < q < 1 - _EPS):
        return None  # not a valid interior mix (a pure NE governs instead)

    # Expected payoffs under the mix.
    row_ev = sum(
        (p if i == 0 else 1 - p) * (q if j == 0 else 1 - q) * a[i][j]
        for i in (0, 1) for j in (0, 1)
    )
    col_ev = sum(
        (p if i == 0 else 1 - p) * (q if j == 0 else 1 - q) * b[i][j]
        for i in (0, 1) for j in (0, 1)
    )
    return NashEquilibrium(
        kind="mixed",
        profile=(
            f"{matrix.player_row or 'Row'}: {p:.0%} {r0} / {1-p:.0%} {r1}  |  "
            f"{matrix.player_col or 'Column'}: {q:.0%} {c0} / {1-q:.0%} {c1}"
        ),
        row_mix=[round(p, 4), round(1 - p, 4)],
        col_mix=[round(q, 4), round(1 - q, 4)],
        row_payoff=round(row_ev, 4),
        col_payoff=round(col_ev, 4),
        description=(
            "Mixed-strategy equilibrium: each player randomizes to keep the other "
            "indifferent, so neither can gain by changing their mix."
        ),
    )


def process_analysis(analysis: Analysis) -> Analysis:
    """Normalize probabilities and compute expected values in place."""
    analysis.outcomes = normalize_outcomes(analysis.outcomes)

    if analysis.decision_tree:
        analysis.decision_tree = sanitize_tree(analysis.decision_tree)
        analysis.decision_tree = reconnect_orphan_roots(analysis.decision_tree)
        analysis.decision_tree = normalize_tree_probabilities(analysis.decision_tree)
        (
            analysis.decision_tree,
            analysis.optimal_decision,
            analysis.optimal_expected_value,
        ) = compute_expected_values(analysis.decision_tree)

    # Nash equilibria (only when the model provided an applicable payoff matrix).
    if analysis.payoff_matrix and analysis.payoff_matrix.applicable:
        analysis.nash_equilibria = compute_nash_equilibria(analysis.payoff_matrix)

    return analysis
