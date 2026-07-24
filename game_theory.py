"""
Game-theory engine: probability normalization and expected-value computation.

The headline feature here is `compute_expected_values`, which runs backward
induction over the decision tree instead of trusting the LLM to pick the "best"
outcome. This gives a deterministic, defensible Expected Value for every node and
identifies the EV-maximizing path.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple  # noqa: F401

from models import (
    Analysis,
    DominanceFinding,
    NashEquilibrium,
    NodeType,
    Outcome,
    ParetoCell,
    PayoffMatrix,
    TreeNode,
)

logger = logging.getLogger(__name__)

_EPS = 1e-9

# How sharply the risk-aversion slider bites: at full aversion the risk-tolerance
# constant is half the payoff spread. Tuned so the whole slider is useful — a
# steeper curve pushes even mild settings into refusing every gamble, which
# leaves most of the range indistinguishable.
_RISK_CURVATURE = 2.0


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


def infer_node_types(nodes: List[TreeNode]) -> Tuple[List[TreeNode], List[str]]:
    """Correct mislabeled node types and report what was changed.

    Models frequently type a gamble as a "decision". That is not cosmetic: a
    decision node is resolved with max() during backward induction, so a
    mislabeled risky branch has its downside silently ignored and the analysis
    becomes over-optimistic. Branch probabilities are the tell — per the prompt
    spec, chance branches carry probabilities and choices do not.

    Rules:
    1. A non-root "decision" whose children carry probabilities is really a
       chance node -> reclassify.
    2. A root "decision" whose children carry probabilities keeps its type (the
       root is the choice being analysed); the spurious probabilities are dropped.
    3. A "decision"/"chance" node with no children is a terminal outcome.
    4. A "chance" node with no branch probabilities is left alone but flagged,
       since expected value falls back to equal weighting.
    """
    warnings: List[str] = []
    lookup = {n.id: n for n in nodes}
    root_ids = {r.id for r in _find_roots(nodes)}

    for node in nodes:
        children = [lookup[c] for c in node.children if c in lookup]

        # Rule 3: no children -> terminal outcome.
        if not children and node.type != NodeType.outcome:
            warnings.append(
                f"'{node.label}' was typed {node.type.value} but has no branches; "
                "treated as a final outcome."
            )
            node.type = NodeType.outcome
            continue

        if not children:
            continue

        probabilistic = [c for c in children if c.probability is not None]

        if node.type == NodeType.decision and probabilistic:
            if node.id in root_ids:
                # Rule 2: the root is the decision under analysis; its branches are
                # choices, so the probabilities are spurious.
                for c in probabilistic:
                    c.probability = None
                warnings.append(
                    f"Ignored branch probabilities under '{node.label}': its branches "
                    "are choices you control, not chance events."
                )
            else:
                # Rule 1: probabilities mean chance, not choice.
                node.type = NodeType.chance
                warnings.append(
                    f"'{node.label}' was typed decision but its branches carry "
                    "probabilities; treated as a chance event so its downside is "
                    "weighted rather than ignored."
                )
        elif node.type == NodeType.chance and not probabilistic:
            # Rule 4: flag only - compute_expected_values falls back to equal weights.
            warnings.append(
                f"Chance event '{node.label}' has no branch probabilities; "
                "assuming each branch is equally likely."
            )

    return nodes, warnings


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


def risk_tolerance_for(nodes: List[TreeNode], risk_aversion: float) -> Optional[float]:
    """Convert a dimensionless risk appetite into a risk-tolerance constant R.

    Payoffs are on an arbitrary per-analysis scale, so an absolute R would mean
    nothing to the user. We scale it to the spread of the payoffs actually in
    play: at risk_aversion = 1, R is half that spread, which prices a coin-flip
    over the full range at roughly 57% of its expected value.

    Returns None when there is no risk to price (no payoffs, or all identical),
    or when the decider is risk-neutral.
    """
    if abs(risk_aversion) < _EPS:
        return None
    payoffs = [n.payoff for n in nodes if n.payoff is not None]
    if not payoffs:
        return None
    spread = max(payoffs) - min(payoffs)
    if spread <= _EPS:
        return None
    return spread / (risk_aversion * _RISK_CURVATURE)


def _certainty_equivalent(branches: List[Tuple[float, float]], R: float) -> float:
    """The guaranteed value a decider would accept in place of a gamble.

    Uses exponential (constant absolute risk aversion) utility
    u(x) = 1 - e^(-x/R). CARA is what makes this usable inside backward
    induction: folding certainty equivalents up the tree gives the same answer
    as evaluating the whole compound lottery at once, so every node's number
    stays in payoff units and remains directly comparable to its EV.

    R > 0 is risk-averse (CE below the mean), R < 0 risk-seeking.

    The sum is computed with a log-sum-exp shift; without it a large
    payoff-to-R ratio overflows exp() and the whole branch collapses to -inf.
    """
    if not branches:
        return 0.0
    exponents = [-value / R for _, value in branches]
    shift = max(exponents)
    total = sum(p * math.exp(e - shift) for (p, _), e in zip(branches, exponents))
    if total <= 0:  # degenerate weights; fall back to the plain mean
        return sum(p * v for p, v in branches)
    return -R * (shift + math.log(total))


def compute_expected_values(
    nodes: List[TreeNode],
    risk_aversion: float = 0.0,
) -> Tuple[List[TreeNode], Optional[str], Optional[float], Optional[float]]:
    """Backward induction over the decision tree.

    Node value rules:
    - outcome/leaf: payoff (default 0.0)
    - chance node:  sum(child.probability * child.value)
    - decision node: max(child.value); the maximizing child is marked optimal

    With `risk_aversion` non-zero each node also gets a certainty equivalent,
    and decisions are made on *that* rather than on raw expected value — a
    risk-averse decider will refuse a gamble whose EV is higher. The expected
    value reported alongside is the EV of the path risk preference actually
    chose, so the two numbers describe the same recommendation.

    Returns the annotated nodes, the optimal first-decision label, its EV, and
    its certainty equivalent (None when risk-neutral).
    """
    lookup: Dict[str, TreeNode] = {n.id: n for n in nodes}
    visiting: set[str] = set()
    best_child: Dict[str, str] = {}  # decision id -> its value-maximizing child id
    R = risk_tolerance_for(nodes, risk_aversion)

    # Clear any previous results so recomputation (e.g. after the user edits an
    # assumption) can't leave a stale optimal path or expected value behind.
    for n in nodes:
        n.expected_value = None
        n.certainty_equivalent = None
        n.is_optimal = False

    def record(node: TreeNode, ev: float, ce: float) -> Tuple[float, float]:
        node.expected_value = ev
        node.certainty_equivalent = ce if R is not None else None
        return ev, ce

    def value_of(node_id: str) -> Tuple[float, float]:
        """Return (expected value, certainty equivalent) for a node."""
        node = lookup.get(node_id)
        if node is None:
            return 0.0, 0.0
        if node_id in visiting:  # cycle guard against malformed trees
            logger.warning("Cycle detected at node '%s'; treating as leaf", node_id)
            return (node.payoff or 0.0,) * 2
        visiting.add(node_id)

        real_children = [c for c in node.children if c in lookup]

        if not real_children:
            # Terminal node: value is its payoff (probability-weighting is applied
            # by the parent chance node, not here). A certain payoff is worth its
            # face value at any risk appetite.
            payoff = node.payoff if node.payoff is not None else 0.0
            result = record(node, payoff, payoff)
            visiting.discard(node_id)
            return result

        if node.type == NodeType.decision:
            best: Optional[Tuple[float, float]] = None
            chosen = None
            for cid in real_children:
                cev, cce = value_of(cid)
                # Choose on the risk-adjusted value; when risk-neutral it equals the EV.
                if best is None or cce > best[1]:
                    best, chosen = (cev, cce), cid
            result = record(node, *(best if chosen is not None else (0.0, 0.0)))
            if chosen is not None:
                best_child[node_id] = chosen
            visiting.discard(node_id)
            return result

        # chance node (or anything with children that isn't a decision):
        # expected value = weighted sum over children.
        branches: List[Tuple[float, float]] = []  # (probability, child CE)
        total = 0.0
        for cid in real_children:
            child = lookup[cid]
            prob = child.probability if child.probability is not None else (
                1.0 / len(real_children)
            )
            cev, cce = value_of(cid)
            total += prob * cev
            branches.append((prob, cce))
        ce = _certainty_equivalent(branches, R) if R is not None else total
        result = record(node, total, ce)
        visiting.discard(node_id)
        return result

    # First pass: compute values for every node.
    roots = _find_roots(nodes)
    primary_root: Optional[TreeNode] = None
    optimal_ev: Optional[float] = None
    optimal_ce: Optional[float] = None
    best_root_value: Optional[float] = None
    for root in roots:
        ev, ce = value_of(root.id)
        if best_root_value is None or ce > best_root_value:
            best_root_value, optimal_ev, optimal_ce, primary_root = ce, ev, ce, root

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

    return nodes, optimal_label, optimal_ev, (optimal_ce if R is not None else None)


def _payoff_table(matrix: PayoffMatrix) -> Optional[Dict[tuple, tuple]]:
    """(row_strategy, col_strategy) -> (row_payoff, col_payoff), or None.

    Returns None unless the matrix is fully specified, since every analysis
    below (equilibria, dominance, Pareto) needs a payoff for every combination.
    """
    rows, cols = matrix.row_strategies, matrix.col_strategies
    if not rows or not cols or not matrix.cells:
        return None

    pay: Dict[tuple, tuple] = {
        (c.row_strategy, c.col_strategy): (c.row_payoff, c.col_payoff)
        for c in matrix.cells
    }
    if any((r, c) not in pay for r in rows for c in cols):
        logger.warning("Payoff matrix is incomplete; skipping game analysis")
        return None
    return pay


def compute_nash_equilibria(matrix: PayoffMatrix) -> List[NashEquilibrium]:
    """Compute Nash equilibria of a two-player normal-form game.

    - Pure-strategy equilibria for any n x m game, via mutual best response.
    - Mixed-strategy equilibrium for 2 x 2 games (where a pure NE may not exist,
      e.g. Matching Pennies), by making each player indifferent.
    """
    pay = _payoff_table(matrix)
    if pay is None:
        return []
    rows = matrix.row_strategies
    cols = matrix.col_strategies

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


def _num(v: float) -> str:
    """Format a payoff without a trailing '.0' on whole numbers."""
    return f"{v:g}"


def find_dominated_strategies(matrix: PayoffMatrix) -> List[DominanceFinding]:
    """Find strategies a rational player should never play.

    Nash equilibria say a cell is stable but not *why*. Dominance does: in the
    Prisoner's Dilemma, Defect beats Cooperate against every opponent choice, so
    the tragic equilibrium follows from each player's own arithmetic without any
    reasoning about the other.

    Strict dominance: `t` pays more than `s` against *every* opponent strategy.
    Weak dominance: `t` is never worse and is strictly better somewhere — real
    but a softer argument, so it's reported separately.

    At most one finding per dominated strategy (strict preferred over weak),
    because "never play s" is the useful conclusion, not every route to it.
    """
    pay = _payoff_table(matrix)
    if pay is None:
        return []

    rows, cols = matrix.row_strategies, matrix.col_strategies
    findings: List[DominanceFinding] = []

    # payoff_at(own, other) reads the mover's own payoff, so one routine serves
    # both players.
    for player, own, others, payoff_at in (
        ("row", rows, cols, lambda s, o: pay[(s, o)][0]),
        ("col", cols, rows, lambda s, o: pay[(o, s)][1]),
    ):
        if len(own) < 2 or not others:
            continue
        player_name = (matrix.player_row if player == "row" else matrix.player_col) or (
            "Row" if player == "row" else "Column"
        )
        for s in own:
            best: Optional[DominanceFinding] = None
            for t in own:
                if t == s:
                    continue
                diffs = [payoff_at(t, o) - payoff_at(s, o) for o in others]
                if all(d > _EPS for d in diffs):
                    kind = "strict"
                elif all(d >= -_EPS for d in diffs) and any(d > _EPS for d in diffs):
                    kind = "weak"
                else:
                    continue
                if best is not None and not (kind == "strict" and best.kind == "weak"):
                    continue
                qualifier = (
                    "always does better" if kind == "strict" else "never does worse"
                )
                best = DominanceFinding(
                    player=player,
                    player_name=player_name,
                    strategy=s,
                    dominated_by=t,
                    kind=kind,
                    description=(
                        f"{player_name} {qualifier} playing '{t}' than '{s}', "
                        f"whatever the other player does"
                        + ("." if kind == "strict" else ", and sometimes better.")
                    ),
                )
                if kind == "strict":
                    break  # can't do better than a strict dominator
            if best is not None:
                findings.append(best)

    return findings


def find_pareto_efficient(matrix: PayoffMatrix) -> List[ParetoCell]:
    """Pure-strategy outcomes that can't be improved for one player without
    costing the other.

    The contrast with the equilibrium is the point: when the stable outcome is
    *not* on this list, the players are trapped somewhere they'd both leave if
    they could commit.
    """
    pay = _payoff_table(matrix)
    if pay is None:
        return []

    cells = [(r, c) for r in matrix.row_strategies for c in matrix.col_strategies]
    efficient: List[ParetoCell] = []
    for r, c in cells:
        rp, cp = pay[(r, c)]
        dominated = any(
            orp >= rp - _EPS
            and ocp >= cp - _EPS
            and (orp > rp + _EPS or ocp > cp + _EPS)
            for orp, ocp in (pay[(r2, c2)] for r2, c2 in cells)
        )
        if not dominated:
            efficient.append(
                ParetoCell(row_strategy=r, col_strategy=c, row_payoff=rp, col_payoff=cp)
            )
    return efficient


def describe_game(
    matrix: PayoffMatrix,
    equilibria: List[NashEquilibrium],
    dominated: List[DominanceFinding],
    pareto: List[ParetoCell],
) -> List[str]:
    """Plain-language conclusions drawn from the computed results.

    Two observations worth surfacing: a player who has one strategy that beats
    all their others (the choice is settled before any strategic reasoning), and
    an equilibrium that both players would trade away if they could.
    """
    notes: List[str] = []
    rows, cols = matrix.row_strategies, matrix.col_strategies

    # A strategy is dominant when every alternative is strictly dominated by it.
    for player, own in (("row", rows), ("col", cols)):
        strict = [d for d in dominated if d.player == player and d.kind == "strict"]
        if len(strict) != len(own) - 1 or not strict:
            continue
        winners = {d.dominated_by for d in strict}
        if len(winners) != 1:
            continue
        winner = winners.pop()
        if winner in {d.strategy for d in strict}:
            continue
        notes.append(
            f"{strict[0].player_name} has a strictly dominant strategy: '{winner}' "
            f"pays more than every alternative no matter what the other player does."
        )

    # An equilibrium both players would leave if they could move together.
    pure = [e for e in equilibria if e.kind == "pure"]
    pareto_keys = {(p.row_strategy, p.col_strategy) for p in pareto}
    inefficient = [e for e in pure if (e.row_strategy, e.col_strategy) not in pareto_keys]
    for eq in inefficient:
        better = [
            p
            for p in pareto
            if p.row_payoff >= (eq.row_payoff or 0) - _EPS
            and p.col_payoff >= (eq.col_payoff or 0) - _EPS
        ]
        if not better:
            continue
        # The clearest illustration: the alternative that gains both players most.
        alt = max(
            better,
            key=lambda p: (p.row_payoff - (eq.row_payoff or 0))
            + (p.col_payoff - (eq.col_payoff or 0)),
        )
        # A Pareto improvement only requires *one* player to gain, so don't claim
        # both do unless both actually do.
        row_gain = alt.row_payoff - (eq.row_payoff or 0)
        col_gain = alt.col_payoff - (eq.col_payoff or 0)
        row_name = matrix.player_row or "Row"
        col_name = matrix.player_col or "Column"
        if row_gain > _EPS and col_gain > _EPS:
            who = "both players do better"
        else:
            gainer = row_name if row_gain > _EPS else col_name
            who = f"{gainer} does better, and the other is no worse,"
        notes.append(
            f"The equilibrium '{eq.profile}' ({_num(eq.row_payoff or 0)}, "
            f"{_num(eq.col_payoff or 0)}) is not Pareto efficient: {who} "
            f"at '{alt.row_strategy} / {alt.col_strategy}' "
            f"({_num(alt.row_payoff)}, {_num(alt.col_payoff)}), but neither can move "
            f"there alone — it takes a binding agreement, or repetition and trust."
        )

    return notes


def process_analysis(analysis: Analysis, risk_aversion: float = 0.0) -> Analysis:
    """Normalize probabilities and compute expected values in place.

    `risk_aversion` runs from -1 (risk-seeking) through 0 (risk-neutral, the
    default) to 1 (strongly risk-averse); see `compute_expected_values`.
    """
    analysis.outcomes = normalize_outcomes(analysis.outcomes)
    analysis.risk_aversion = risk_aversion

    if analysis.decision_tree:
        analysis.decision_tree = sanitize_tree(analysis.decision_tree)
        analysis.decision_tree = reconnect_orphan_roots(analysis.decision_tree)
        # Correct mislabeled node types before any values are computed, so a gamble
        # typed as a "decision" doesn't get max()'d and lose its downside.
        analysis.decision_tree, type_warnings = infer_node_types(analysis.decision_tree)
        analysis.warnings = type_warnings
        analysis.decision_tree = normalize_tree_probabilities(analysis.decision_tree)
        (
            analysis.decision_tree,
            analysis.optimal_decision,
            analysis.optimal_expected_value,
            analysis.optimal_certainty_equivalent,
        ) = compute_expected_values(analysis.decision_tree, risk_aversion)

    # Normal-form analysis (only when the model provided an applicable matrix).
    if analysis.payoff_matrix and analysis.payoff_matrix.applicable:
        matrix = analysis.payoff_matrix
        analysis.nash_equilibria = compute_nash_equilibria(matrix)
        analysis.dominated_strategies = find_dominated_strategies(matrix)
        analysis.pareto_efficient = find_pareto_efficient(matrix)
        analysis.game_notes = describe_game(
            matrix,
            analysis.nash_equilibria,
            analysis.dominated_strategies,
            analysis.pareto_efficient,
        )

    return analysis
