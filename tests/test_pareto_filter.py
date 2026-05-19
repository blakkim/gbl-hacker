"""Unit tests for ``pareto_filter`` (Sub-AC 2.4).

The headline contract this test suite enforces:

* ``pareto_filter(scored_teams)`` returns the **non-dominated** subset
  of the input — every team for which no *other* team is at least as
  good on all three axes AND strictly better on at least one.
* Dominance is the strict Pareto definition (``>=`` on every axis AND
  ``>`` on at least one). Equal scores survive together; neither
  dominates the other.
* Input order is preserved on output. The Pareto frontier has no
  canonical single-axis sort, so input order is the most honest
  default.
* The three axes (``expected_win_rate``, ``worst_case_robustness``,
  ``meta_coverage``) are all "higher is better" and in ``[0.0, 1.0]``.
  Out-of-range or NaN values are rejected at :class:`Score` construction
  time, before they ever reach the frontier math.
* The filter is robust on degenerate inputs (empty, single, all-equal,
  all-on-frontier, all-dominated).
"""

from __future__ import annotations

import math

import pytest

from gbl_hacker.score import (
    CandidateTeam,
    Score,
    ScoredTeam,
    dominates,
    pareto_filter,
)
from gbl_hacker.simulator import ChargedMove, CombatantBuild, FastMove

# --- fixture helpers ------------------------------------------------------
# Build cheap placeholder ``CandidateTeam`` instances. ``pareto_filter``
# does not inspect team contents — only scores — so any well-formed
# ``CandidateTeam`` works. Kept structurally aligned with the other score-
# axis tests for one-glance cross-check.


def _build(species: str) -> CombatantBuild:
    return CombatantBuild(
        species=species,
        max_hp=100,
        fast=FastMove(name="quick", damage=2, energy_gain=8),
        charged=ChargedMove(name="bomb", energy_cost=40, damage=70),
    )


def _team(label: str) -> CandidateTeam:
    """A unique 3-slot team labelled by a human-readable tag.

    The three slots use ``"{label}-lead"`` / ``"{label}-swap"`` /
    ``"{label}-closer"`` so debug output is greppable. ``pareto_filter``
    is score-only so slot identities are cosmetic.
    """

    return CandidateTeam.from_slots(
        [
            _build(f"{label}-lead"),
            _build(f"{label}-swap"),
            _build(f"{label}-closer"),
        ]
    )


def _scored(label: str, ev: float, worst: float, cov: float) -> ScoredTeam:
    return ScoredTeam(
        team=_team(label),
        score=Score(
            expected_win_rate=ev,
            worst_case_robustness=worst,
            meta_coverage=cov,
        ),
    )


# --- Score construction / validation -------------------------------------


def test_score_stores_three_axes_in_canonical_order() -> None:
    s = Score(expected_win_rate=0.6, worst_case_robustness=0.4, meta_coverage=0.7)
    assert s.expected_win_rate == 0.6
    assert s.worst_case_robustness == 0.4
    assert s.meta_coverage == 0.7
    assert s.as_tuple == (0.6, 0.4, 0.7)


def test_score_is_frozen() -> None:
    s = Score(expected_win_rate=0.5, worst_case_robustness=0.5, meta_coverage=0.5)
    with pytest.raises(Exception):  # FrozenInstanceError is a subclass of Exception
        s.expected_win_rate = 0.9  # type: ignore[misc]


@pytest.mark.parametrize("axis", ["expected_win_rate", "worst_case_robustness", "meta_coverage"])
@pytest.mark.parametrize("bad_value", [-0.01, -1.0, 1.01, 99.9])
def test_score_rejects_out_of_range_axis(axis: str, bad_value: float) -> None:
    kwargs: dict[str, float] = {
        "expected_win_rate": 0.5,
        "worst_case_robustness": 0.5,
        "meta_coverage": 0.5,
    }
    kwargs[axis] = bad_value
    with pytest.raises(ValueError, match=axis):
        Score(**kwargs)


@pytest.mark.parametrize("axis", ["expected_win_rate", "worst_case_robustness", "meta_coverage"])
def test_score_rejects_nan(axis: str) -> None:
    kwargs: dict[str, float] = {
        "expected_win_rate": 0.5,
        "worst_case_robustness": 0.5,
        "meta_coverage": 0.5,
    }
    kwargs[axis] = math.nan
    with pytest.raises(ValueError, match="NaN"):
        Score(**kwargs)


def test_score_accepts_boundary_values() -> None:
    Score(expected_win_rate=0.0, worst_case_robustness=0.0, meta_coverage=0.0)
    Score(expected_win_rate=1.0, worst_case_robustness=1.0, meta_coverage=1.0)


# --- dominates() ---------------------------------------------------------


def test_dominates_strict_better_on_all_axes_dominates() -> None:
    a = Score(0.7, 0.6, 0.8)
    b = Score(0.5, 0.4, 0.6)
    assert dominates(a, b) is True
    assert dominates(b, a) is False


def test_dominates_equal_on_two_axes_strict_on_one_dominates() -> None:
    a = Score(0.5, 0.5, 0.8)
    b = Score(0.5, 0.5, 0.6)
    assert dominates(a, b) is True
    assert dominates(b, a) is False


def test_dominates_equal_scores_do_not_dominate_each_other() -> None:
    a = Score(0.5, 0.5, 0.5)
    b = Score(0.5, 0.5, 0.5)
    assert dominates(a, b) is False
    assert dominates(b, a) is False


def test_dominates_is_irreflexive() -> None:
    a = Score(0.7, 0.6, 0.8)
    # A point cannot strictly dominate itself.
    assert dominates(a, a) is False


def test_dominates_mixed_better_worse_neither_dominates() -> None:
    # Classic incomparable pair: each beats the other on one axis.
    a = Score(0.7, 0.4, 0.5)
    b = Score(0.5, 0.6, 0.5)
    assert dominates(a, b) is False
    assert dominates(b, a) is False


def test_dominates_one_axis_better_one_axis_worse_two_equal() -> None:
    a = Score(0.6, 0.5, 0.5)
    b = Score(0.4, 0.5, 0.7)
    # a wins EV, b wins coverage, robustness tied → incomparable.
    assert dominates(a, b) is False
    assert dominates(b, a) is False


# --- pareto_filter() — degenerate inputs ---------------------------------


def test_empty_input_returns_empty_list() -> None:
    assert pareto_filter([]) == []


def test_single_input_returns_singleton() -> None:
    only = _scored("solo", 0.5, 0.4, 0.6)
    assert pareto_filter([only]) == [only]


def test_generator_input_is_accepted() -> None:
    teams = [_scored(f"t{i}", 0.5, 0.5, 0.5) for i in range(3)]
    result = pareto_filter(iter(teams))
    # All three are tied (equal scores) — none dominates another.
    assert result == teams


def test_non_scored_team_element_raises_typeerror() -> None:
    with pytest.raises(TypeError, match="ScoredTeam"):
        pareto_filter([("not", "a", "ScoredTeam")])  # type: ignore[list-item]


# --- pareto_filter() — frontier math --------------------------------------


def test_dominated_team_is_dropped() -> None:
    winner = _scored("winner", 0.8, 0.7, 0.9)
    loser = _scored("loser", 0.4, 0.3, 0.2)
    result = pareto_filter([winner, loser])
    assert result == [winner]


def test_dominated_team_is_dropped_regardless_of_input_order() -> None:
    winner = _scored("winner", 0.8, 0.7, 0.9)
    loser = _scored("loser", 0.4, 0.3, 0.2)
    # Reverse input order — dominance result is the same.
    result = pareto_filter([loser, winner])
    assert result == [winner]


def test_two_incomparable_teams_both_survive() -> None:
    # ``a`` wins EV, ``b`` wins robustness; coverage tied. Neither dominates.
    a = _scored("a", 0.7, 0.4, 0.5)
    b = _scored("b", 0.4, 0.7, 0.5)
    result = pareto_filter([a, b])
    assert set(result) == {a, b}
    # Order preserved on input order.
    assert result == [a, b]


def test_three_axis_frontier_with_one_strict_winner_per_axis() -> None:
    # Three teams each dominating on exactly one axis. All three are on the
    # frontier: no team is ≥ another on all axes.
    ev_specialist = _scored("ev", 0.9, 0.2, 0.2)
    worst_specialist = _scored("worst", 0.2, 0.9, 0.2)
    cov_specialist = _scored("cov", 0.2, 0.2, 0.9)
    result = pareto_filter([ev_specialist, worst_specialist, cov_specialist])
    assert set(result) == {ev_specialist, worst_specialist, cov_specialist}


def test_dominated_specialist_is_dropped_from_frontier() -> None:
    # Add a "generalist" that beats every specialist on every axis.
    ev_specialist = _scored("ev", 0.6, 0.2, 0.2)
    worst_specialist = _scored("worst", 0.2, 0.6, 0.2)
    cov_specialist = _scored("cov", 0.2, 0.2, 0.6)
    dominator = _scored("dominator", 0.7, 0.7, 0.7)
    result = pareto_filter(
        [ev_specialist, worst_specialist, cov_specialist, dominator]
    )
    assert result == [dominator]


def test_all_equal_scores_all_survive() -> None:
    # Pareto convention: equal scores do not dominate each other. All three
    # candidates remain on the frontier even though they are score-byte-equal.
    a = _scored("a", 0.5, 0.5, 0.5)
    b = _scored("b", 0.5, 0.5, 0.5)
    c = _scored("c", 0.5, 0.5, 0.5)
    result = pareto_filter([a, b, c])
    assert result == [a, b, c]


def test_output_order_matches_input_order() -> None:
    # Mix of frontier and dominated teams; surviving ones keep input order.
    t1 = _scored("t1", 0.6, 0.3, 0.4)  # frontier (best EV)
    t2 = _scored("t2", 0.2, 0.2, 0.2)  # dominated
    t3 = _scored("t3", 0.3, 0.6, 0.4)  # frontier (best robust)
    t4 = _scored("t4", 0.1, 0.1, 0.1)  # dominated
    t5 = _scored("t5", 0.3, 0.4, 0.7)  # frontier (best coverage)
    result = pareto_filter([t1, t2, t3, t4, t5])
    assert result == [t1, t3, t5]


def test_partial_equality_does_not_block_dominance() -> None:
    # ``b`` is byte-equal to ``a`` on two axes but strictly worse on one →
    # ``a`` dominates ``b``.
    a = _scored("a", 0.5, 0.5, 0.5)
    b = _scored("b", 0.5, 0.5, 0.3)
    result = pareto_filter([a, b])
    assert result == [a]


def test_transitively_dominated_team_is_dropped() -> None:
    # ``a`` dominates ``b``, which dominates ``c``. The filter must drop
    # both ``b`` and ``c`` — it does not stop at "is anything strictly worse".
    a = _scored("a", 0.9, 0.9, 0.9)
    b = _scored("b", 0.6, 0.6, 0.6)
    c = _scored("c", 0.3, 0.3, 0.3)
    result = pareto_filter([a, b, c])
    assert result == [a]


def test_duplicate_team_objects_both_survive_when_equal() -> None:
    # The same scored-team passed twice survives twice (equal does not
    # dominate equal). The filter does not deduplicate by identity — that
    # is a presentation-layer decision.
    a = _scored("a", 0.5, 0.5, 0.5)
    result = pareto_filter([a, a])
    assert result == [a, a]


# --- pareto_filter() — boundary axis values ------------------------------


def test_frontier_admits_axis_endpoints() -> None:
    # A score of (1.0, 1.0, 1.0) dominates everything else; a score of
    # (0.0, 0.0, 0.0) is dominated by everything else with any positive axis.
    perfect = _scored("perfect", 1.0, 1.0, 1.0)
    zeros = _scored("zeros", 0.0, 0.0, 0.0)
    mid = _scored("mid", 0.5, 0.5, 0.5)
    result = pareto_filter([perfect, zeros, mid])
    assert result == [perfect]


def test_two_at_zero_one_at_corner_only_corner_survives() -> None:
    zeros_a = _scored("zeros_a", 0.0, 0.0, 0.0)
    zeros_b = _scored("zeros_b", 0.0, 0.0, 0.0)
    one_axis = _scored("one_axis", 0.1, 0.0, 0.0)
    result = pareto_filter([zeros_a, zeros_b, one_axis])
    assert result == [one_axis]


# --- pareto_correctness regression guard ---------------------------------


def test_pareto_correctness_no_single_axis_collapses_to_winner() -> None:
    """The seed pins ``pareto_correctness`` — output must span the front,
    not collapse onto a single metric.

    Construct a frontier where the EV-maximum team is *not* the robustness
    or coverage maximum. A correct ``pareto_filter`` keeps all three; a
    bug that secretly weighted EV (e.g. ``sort by EV; pick best``) would
    drop the other two.
    """

    ev_max = _scored("ev_max", 0.9, 0.3, 0.3)
    worst_max = _scored("worst_max", 0.5, 0.8, 0.4)
    cov_max = _scored("cov_max", 0.4, 0.4, 0.9)
    result = pareto_filter([ev_max, worst_max, cov_max])
    assert set(result) == {ev_max, worst_max, cov_max}
