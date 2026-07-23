"""Tests for Nash equilibrium computation on classic normal-form games."""
import math

from models import MatrixCell, PayoffMatrix
from game_theory import compute_nash_equilibria


def matrix(rows, cols, cells, applicable=True):
    return PayoffMatrix(
        applicable=applicable, player_row="R", player_col="C",
        row_strategies=rows, col_strategies=cols,
        cells=[MatrixCell(row_strategy=r, col_strategy=c, row_payoff=rp, col_payoff=cp)
               for (r, c, rp, cp) in cells],
    )


def pures(eqs):
    return {(e.row_strategy, e.col_strategy) for e in eqs if e.kind == "pure"}


def mixed(eqs):
    return [e for e in eqs if e.kind == "mixed"]


def test_prisoners_dilemma_single_pure_ne():
    m = matrix(
        ["Cooperate", "Defect"], ["Cooperate", "Defect"],
        [("Cooperate", "Cooperate", 3, 3), ("Cooperate", "Defect", 0, 5),
         ("Defect", "Cooperate", 5, 0), ("Defect", "Defect", 1, 1)],
    )
    eqs = compute_nash_equilibria(m)
    assert pures(eqs) == {("Defect", "Defect")}
    # Defect strictly dominates, so there is no interior mixed NE.
    assert mixed(eqs) == []


def test_matching_pennies_only_mixed_5050():
    m = matrix(
        ["Heads", "Tails"], ["Heads", "Tails"],
        [("Heads", "Heads", 1, -1), ("Heads", "Tails", -1, 1),
         ("Tails", "Heads", -1, 1), ("Tails", "Tails", 1, -1)],
    )
    eqs = compute_nash_equilibria(m)
    assert pures(eqs) == set()          # no pure NE
    mx = mixed(eqs)
    assert len(mx) == 1
    assert all(math.isclose(p, 0.5) for p in mx[0].row_mix)
    assert all(math.isclose(q, 0.5) for q in mx[0].col_mix)


def test_battle_of_the_sexes_two_pure_one_mixed():
    m = matrix(
        ["Boxing", "Ballet"], ["Boxing", "Ballet"],
        [("Boxing", "Boxing", 2, 1), ("Boxing", "Ballet", 0, 0),
         ("Ballet", "Boxing", 0, 0), ("Ballet", "Ballet", 1, 2)],
    )
    eqs = compute_nash_equilibria(m)
    assert pures(eqs) == {("Boxing", "Boxing"), ("Ballet", "Ballet")}
    mx = mixed(eqs)
    assert len(mx) == 1
    # Row plays Boxing 2/3, Column plays Boxing 1/3 (mixes are rounded to 4 dp).
    assert math.isclose(mx[0].row_mix[0], 2 / 3, abs_tol=1e-3)
    assert math.isclose(mx[0].col_mix[0], 1 / 3, abs_tol=1e-3)


def test_dominant_strategy_coordination():
    # Row's "Up" and Col's "Left" strictly dominate -> unique pure NE.
    m = matrix(
        ["Up", "Down"], ["Left", "Right"],
        [("Up", "Left", 5, 5), ("Up", "Right", 4, 0),
         ("Down", "Left", 0, 4), ("Down", "Right", 1, 1)],
    )
    eqs = compute_nash_equilibria(m)
    assert ("Up", "Left") in pures(eqs)


def test_incomplete_matrix_returns_empty():
    m = matrix(
        ["A", "B"], ["X", "Y"],
        [("A", "X", 1, 1)],  # missing three cells
    )
    assert compute_nash_equilibria(m) == []


def test_not_applicable_or_empty_returns_empty():
    assert compute_nash_equilibria(matrix([], [], [], applicable=True)) == []
    empty = PayoffMatrix(applicable=False)
    assert compute_nash_equilibria(empty) == []


def test_ne_payoffs_reported_for_pure():
    m = matrix(
        ["A", "B"], ["X", "Y"],
        [("A", "X", 3, 2), ("A", "Y", 0, 0),
         ("B", "X", 0, 0), ("B", "Y", 2, 3)],
    )
    eqs = compute_nash_equilibria(m)
    ax = next(e for e in eqs if (e.row_strategy, e.col_strategy) == ("A", "X"))
    assert ax.row_payoff == 3 and ax.col_payoff == 2
