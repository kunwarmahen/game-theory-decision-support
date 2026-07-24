"""Tests for dominance, Pareto efficiency, and the notes drawn from them."""
from models import MatrixCell, PayoffMatrix
from game_theory import (
    compute_nash_equilibria,
    describe_game,
    find_dominated_strategies,
    find_pareto_efficient,
)


def matrix(rows, cols, cells, applicable=True):
    return PayoffMatrix(
        applicable=applicable, player_row="R", player_col="C",
        row_strategies=rows, col_strategies=cols,
        cells=[MatrixCell(row_strategy=r, col_strategy=c, row_payoff=rp, col_payoff=cp)
               for (r, c, rp, cp) in cells],
    )


def prisoners_dilemma():
    return matrix(
        ["Cooperate", "Defect"], ["Cooperate", "Defect"],
        [("Cooperate", "Cooperate", 3, 3), ("Cooperate", "Defect", 0, 5),
         ("Defect", "Cooperate", 5, 0), ("Defect", "Defect", 1, 1)],
    )


def pairs(findings, player=None):
    return {
        (f.strategy, f.dominated_by, f.kind)
        for f in findings
        if player is None or f.player == player
    }


def cells_of(pareto):
    return {(p.row_strategy, p.col_strategy) for p in pareto}


# --- Dominance ------------------------------------------------------------

def test_prisoners_dilemma_defect_strictly_dominates():
    d = find_dominated_strategies(prisoners_dilemma())
    # Cooperate is strictly dominated for both players; Defect never is.
    assert pairs(d, "row") == {("Cooperate", "Defect", "strict")}
    assert pairs(d, "col") == {("Cooperate", "Defect", "strict")}


def test_weak_dominance_flagged_separately():
    # "Up" ties on Left and wins on Right -> weakly, not strictly, dominant.
    m = matrix(
        ["Up", "Down"], ["Left", "Right"],
        [("Up", "Left", 2, 0), ("Up", "Right", 3, 0),
         ("Down", "Left", 2, 0), ("Down", "Right", 1, 0)],
    )
    assert pairs(find_dominated_strategies(m), "row") == {("Down", "Up", "weak")}


def test_ties_everywhere_are_not_dominance():
    # Identical payoffs: neither strategy is ever better, so nothing is dominated.
    m = matrix(
        ["Up", "Down"], ["Left", "Right"],
        [("Up", "Left", 1, 1), ("Up", "Right", 2, 1),
         ("Down", "Left", 1, 1), ("Down", "Right", 2, 1)],
    )
    assert find_dominated_strategies(m) == []


def test_strict_dominator_preferred_over_weak():
    # "Low" is weakly dominated by "Mid" (ties on Left) and strictly by "High".
    m = matrix(
        ["Low", "Mid", "High"], ["Left", "Right"],
        [("Low", "Left", 1, 0), ("Low", "Right", 1, 0),
         ("Mid", "Left", 1, 0), ("Mid", "Right", 2, 0),
         ("High", "Left", 3, 0), ("High", "Right", 3, 0)],
    )
    low = [f for f in find_dominated_strategies(m) if f.strategy == "Low"]
    assert len(low) == 1
    assert (low[0].dominated_by, low[0].kind) == ("High", "strict")


def test_no_dominance_in_matching_pennies():
    m = matrix(
        ["Heads", "Tails"], ["Heads", "Tails"],
        [("Heads", "Heads", 1, -1), ("Heads", "Tails", -1, 1),
         ("Tails", "Heads", -1, 1), ("Tails", "Tails", 1, -1)],
    )
    assert find_dominated_strategies(m) == []


def test_non_square_matrix_dominance():
    # Column has three options; "Bad" loses to "Good" against every row.
    m = matrix(
        ["Up", "Down"], ["Good", "Bad", "Mid"],
        [("Up", "Good", 0, 5), ("Up", "Bad", 0, 1), ("Up", "Mid", 0, 3),
         ("Down", "Good", 0, 4), ("Down", "Bad", 0, 0), ("Down", "Mid", 0, 2)],
    )
    d = pairs(find_dominated_strategies(m), "col")
    assert ("Bad", "Good", "strict") in d
    assert ("Mid", "Good", "strict") in d


def test_incomplete_matrix_returns_empty():
    m = matrix(["A", "B"], ["X", "Y"], [("A", "X", 1, 1)])
    assert find_dominated_strategies(m) == []
    assert find_pareto_efficient(m) == []


def test_single_strategy_player_has_no_dominance():
    m = matrix(["Only"], ["X", "Y"], [("Only", "X", 1, 2), ("Only", "Y", 1, 1)])
    assert pairs(find_dominated_strategies(m), "row") == set()


# --- Pareto efficiency ----------------------------------------------------

def test_prisoners_dilemma_equilibrium_is_not_pareto_efficient():
    m = prisoners_dilemma()
    pareto = cells_of(find_pareto_efficient(m))
    # (Defect, Defect) pays (1,1) — worse for both than (3,3).
    assert ("Defect", "Defect") not in pareto
    assert pareto == {
        ("Cooperate", "Cooperate"), ("Cooperate", "Defect"), ("Defect", "Cooperate"),
    }


def test_pareto_excludes_cells_beaten_on_one_side_only():
    # (A,Y) pays (1,5); (A,X) pays (2,5) — better for row, equal for column.
    m = matrix(
        ["A", "B"], ["X", "Y"],
        [("A", "X", 2, 5), ("A", "Y", 1, 5),
         ("B", "X", 0, 0), ("B", "Y", 0, 0)],
    )
    assert cells_of(find_pareto_efficient(m)) == {("A", "X")}


def test_all_cells_efficient_in_zero_sum():
    m = matrix(
        ["Heads", "Tails"], ["Heads", "Tails"],
        [("Heads", "Heads", 1, -1), ("Heads", "Tails", -1, 1),
         ("Tails", "Heads", -1, 1), ("Tails", "Tails", 1, -1)],
    )
    # Nothing can improve one player without costing the other.
    assert len(find_pareto_efficient(m)) == 4


def test_duplicate_optimal_cells_both_efficient():
    m = matrix(
        ["A", "B"], ["X", "Y"],
        [("A", "X", 2, 2), ("A", "Y", 2, 2),
         ("B", "X", 1, 1), ("B", "Y", 1, 1)],
    )
    assert cells_of(find_pareto_efficient(m)) == {("A", "X"), ("A", "Y")}


# --- Narrative notes ------------------------------------------------------

def notes_for(m):
    eqs = compute_nash_equilibria(m)
    return describe_game(m, eqs, find_dominated_strategies(m), find_pareto_efficient(m))


def test_notes_report_dominant_strategy_and_inefficiency():
    notes = notes_for(prisoners_dilemma())
    assert sum("dominant strategy" in n for n in notes) == 2  # one per player
    inefficiency = [n for n in notes if "not Pareto efficient" in n]
    assert len(inefficiency) == 1
    # It should name the outcome both players would prefer.
    assert "Cooperate / Cooperate" in inefficiency[0]


def test_efficient_equilibrium_produces_no_inefficiency_note():
    # Both players' dominant strategies land on the best cell for everyone.
    m = matrix(
        ["Up", "Down"], ["Left", "Right"],
        [("Up", "Left", 5, 5), ("Up", "Right", 4, 0),
         ("Down", "Left", 0, 4), ("Down", "Right", 1, 1)],
    )
    assert not any("not Pareto efficient" in n for n in notes_for(m))


def test_no_dominant_strategy_note_when_dominance_is_partial():
    # Battle of the sexes: no strategy dominates, so no note claims one does.
    m = matrix(
        ["Boxing", "Ballet"], ["Boxing", "Ballet"],
        [("Boxing", "Boxing", 2, 1), ("Boxing", "Ballet", 0, 0),
         ("Ballet", "Boxing", 0, 0), ("Ballet", "Ballet", 1, 2)],
    )
    assert not any("dominant strategy" in n for n in notes_for(m))
