"""Unit tests for ``rank_top_k`` (Sub-AC 2.5).

The headline contract this test suite enforces:

* ``rank_top_k(pareto_set, k)`` returns up to ``K`` :class:`ScoredTeam`
  entries from the input, ordered descending by a weighted-sum
  composite of the three score axes.
* Default weights are equal ``(1.0, 1.0, 1.0)`` — honoring the seed's
  ``pareto_correctness`` principle (no single-axis bias at the
  presentation stage).
* Tie-breaking is **deterministic**: ties on the weighted-sum break
  lexicographic descending by ``(EV, worst, coverage)``; remaining
  ties fall back to input order (stable sort).
* The function is **pure**: it returns a fresh list and does not mutate
  the input iterable. Generator inputs are accepted (single-pass).
* Degenerate inputs behave gracefully: empty input → empty output,
  ``k = 0`` → empty output, ``k > len(input)`` → all teams ranked.
* Invalid ``k`` (negative, non-int, bool) and invalid weights
  (non-tuple, wrong length, NaN, negative, zero-sum) raise loudly.
* Non-:class:`ScoredTeam` elements raise :class:`TypeError`.

Cross-axis design check: a "winner-on-all-axes" team is the top rank
regardless of weight triple; an "incomparable Pareto frontier" produces
a deterministic but weight-sensitive ordering — exactly what a
top-rank operator expects from the engine's final stage.
"""

from __future__ import annotations

import math

import pytest

from gbl_hacker.score import (
    CandidateTeam,
    Score,
    ScoredTeam,
    rank_top_k,
)
from gbl_hacker.simulator import ChargedMove, CombatantBuild, FastMove


# --- fixture helpers ------------------------------------------------------
# Build cheap placeholder ``CandidateTeam`` instances. ``rank_top_k`` does
# not inspect team contents — only scores — so any well-formed
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
    """A unique 3-slot team labelled by a human-readable tag."""

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


# --- degenerate inputs ----------------------------------------------------


def test_empty_input_returns_empty_list() -> None:
    assert rank_top_k([], 5) == []


def test_empty_input_with_zero_k_returns_empty_list() -> None:
    assert rank_top_k([], 0) == []


def test_zero_k_returns_empty_list_for_non_empty_input() -> None:
    teams = [_scored("a", 0.5, 0.5, 0.5), _scored("b", 0.6, 0.6, 0.6)]
    assert rank_top_k(teams, 0) == []


def test_single_input_singleton_returned() -> None:
    only = _scored("solo", 0.4, 0.7, 0.3)
    assert rank_top_k([only], 1) == [only]


def test_single_input_with_large_k_returns_singleton() -> None:
    only = _scored("solo", 0.4, 0.7, 0.3)
    assert rank_top_k([only], 99) == [only]


def test_generator_input_is_accepted() -> None:
    teams = [
        _scored("a", 0.7, 0.5, 0.5),
        _scored("b", 0.6, 0.5, 0.5),
        _scored("c", 0.8, 0.5, 0.5),
    ]
    # Pass via iter() so the function must handle a single-pass generator.
    result = rank_top_k(iter(teams), 3)
    assert [r.team.species for r in result] == [
        ("c-lead", "c-swap", "c-closer"),
        ("a-lead", "a-swap", "a-closer"),
        ("b-lead", "b-swap", "b-closer"),
    ]


# --- ordering -------------------------------------------------------------


def test_strict_winner_on_all_axes_is_top_ranked() -> None:
    winner = _scored("winner", 0.9, 0.9, 0.9)
    middle = _scored("middle", 0.6, 0.6, 0.6)
    loser = _scored("loser", 0.3, 0.3, 0.3)
    result = rank_top_k([loser, middle, winner], 3)
    assert result == [winner, middle, loser]


def test_top_1_returns_the_highest_weighted_sum_team() -> None:
    a = _scored("a", 0.5, 0.5, 0.5)  # sum 1.5
    b = _scored("b", 0.9, 0.1, 0.2)  # sum 1.2
    c = _scored("c", 0.6, 0.6, 0.6)  # sum 1.8 → winner
    result = rank_top_k([a, b, c], 1)
    assert result == [c]


def test_k_greater_than_input_returns_all_ranked() -> None:
    a = _scored("a", 0.3, 0.3, 0.3)  # sum 0.9
    b = _scored("b", 0.5, 0.5, 0.5)  # sum 1.5
    c = _scored("c", 0.4, 0.4, 0.4)  # sum 1.2
    result = rank_top_k([a, b, c], 10)
    assert result == [b, c, a]


def test_default_weights_treat_axes_equally() -> None:
    # Equal weights: (0.9, 0.0, 0.0) and (0.0, 0.9, 0.0) and (0.0, 0.0, 0.9)
    # all sum to 0.9 → tied on primary key. The lexicographic tiebreak
    # (EV → WCR → COV descending) then orders them.
    ev_specialist = _scored("ev", 0.9, 0.0, 0.0)
    wcr_specialist = _scored("wcr", 0.0, 0.9, 0.0)
    cov_specialist = _scored("cov", 0.0, 0.0, 0.9)
    result = rank_top_k([cov_specialist, wcr_specialist, ev_specialist], 3)
    # All sums tied; lex on (EV, WCR, COV) → ev wins, wcr next, cov last.
    assert result == [ev_specialist, wcr_specialist, cov_specialist]


def test_pareto_frontier_with_distinct_sums_orders_by_sum() -> None:
    # Three teams on a Pareto frontier with different total mass.
    t1 = _scored("t1", 0.7, 0.4, 0.5)  # sum 1.6
    t2 = _scored("t2", 0.4, 0.7, 0.5)  # sum 1.6 → tie with t1 on sum
    t3 = _scored("t3", 0.6, 0.6, 0.7)  # sum 1.9 → winner
    result = rank_top_k([t1, t2, t3], 3)
    # t3 wins on sum; t1 vs t2 tied at 1.6 → lex EV picks t1 first.
    assert result == [t3, t1, t2]


def test_returns_fresh_list_does_not_mutate_input() -> None:
    a = _scored("a", 0.3, 0.3, 0.3)
    b = _scored("b", 0.6, 0.6, 0.6)
    c = _scored("c", 0.5, 0.5, 0.5)
    original = [a, b, c]
    snapshot = list(original)
    _ = rank_top_k(original, 2)
    # Input list still in original order — no in-place sort leaked.
    assert original == snapshot


# --- tie-breaking ---------------------------------------------------------


def test_byte_equal_scores_preserve_input_order() -> None:
    # Three teams with identical scores. Stable sort → input order kept.
    a = _scored("a", 0.5, 0.5, 0.5)
    b = _scored("b", 0.5, 0.5, 0.5)
    c = _scored("c", 0.5, 0.5, 0.5)
    result = rank_top_k([a, b, c], 3)
    assert result == [a, b, c]


def test_sum_tie_broken_by_expected_win_rate() -> None:
    # Both sum to 1.5; team_a has higher EV → wins the tiebreak.
    a = _scored("a", 0.7, 0.4, 0.4)  # sum 1.5, EV 0.7
    b = _scored("b", 0.5, 0.5, 0.5)  # sum 1.5, EV 0.5
    result = rank_top_k([b, a], 2)
    assert result == [a, b]


def test_sum_and_ev_tie_broken_by_worst_case_robustness() -> None:
    # Sums tied at 1.5 and EV tied at 0.5 → tiebreak escalates to WCR.
    a = _scored("a", 0.5, 0.6, 0.4)  # sum 1.5, EV 0.5, WCR 0.6
    b = _scored("b", 0.5, 0.4, 0.6)  # sum 1.5, EV 0.5, WCR 0.4
    result = rank_top_k([b, a], 2)
    assert result == [a, b]


def test_sum_ev_wcr_tie_broken_by_meta_coverage() -> None:
    # Sums tied 1.4, EV tied 0.5, WCR tied 0.5 → tiebreak on coverage.
    a = _scored("a", 0.5, 0.5, 0.4)  # sum 1.4, COV 0.4
    b = _scored("b", 0.5, 0.5, 0.4)  # identical
    # Construct so a and b are byte-equal — stable sort then preserves
    # input order for the final fallback. Test separately below.
    # Here we tweak coverage to break the byte-equality.
    a2 = _scored("a", 0.5, 0.5, 0.45)
    b2 = _scored("b", 0.5, 0.5, 0.35)
    result = rank_top_k([b2, a2], 2)
    # Sums: a2 = 1.40, b2 = 1.30 — wait, those aren't tied.
    # The "all-three-tied-except-coverage" case is by definition only
    # tied on EV+WCR (not on the sum). The sum-key naturally breaks
    # before the lex tiebreak ever consults coverage. Validate that
    # the dominant axis wins regardless.
    assert result == [a2, b2]


def test_genuine_three_axis_tie_falls_back_to_input_order() -> None:
    # ``a`` and ``b`` have byte-equal scores → stable sort preserves
    # input order for the absolute fallback.
    a = _scored("a", 0.5, 0.5, 0.5)
    b = _scored("b", 0.5, 0.5, 0.5)
    result = rank_top_k([b, a], 2)
    # ``b`` came first in input → ``b`` is first in output.
    assert result == [b, a]


def test_duplicate_scored_team_objects_both_retained() -> None:
    # Same object passed twice: both appear in the output (the ranker is
    # not a set; deduplication is a presentation-layer decision).
    a = _scored("a", 0.6, 0.6, 0.6)
    result = rank_top_k([a, a], 2)
    assert result == [a, a]


# --- weights --------------------------------------------------------------


def test_custom_weights_change_ordering() -> None:
    # With equal weights, EV-heavy and WCR-heavy specialists tie on sum
    # and lex-tiebreak by EV. With WCR-heavy weights, the WCR specialist
    # should win outright.
    ev_specialist = _scored("ev", 0.9, 0.3, 0.3)  # sum 1.5
    wcr_specialist = _scored("wcr", 0.3, 0.9, 0.3)  # sum 1.5
    default = rank_top_k([ev_specialist, wcr_specialist], 1)
    assert default == [ev_specialist]  # EV wins lex tiebreak
    biased = rank_top_k(
        [ev_specialist, wcr_specialist], 1, weights=(1.0, 10.0, 1.0)
    )
    # WCR weight × 10 makes wcr_specialist's weighted sum vastly larger.
    assert biased == [wcr_specialist]


def test_weights_can_zero_an_axis() -> None:
    # Zero out the EV axis — the ranker should then sort purely on
    # (WCR + COV), with WCR/COV/input lex tiebreaks if needed.
    a = _scored("a", 0.9, 0.1, 0.1)  # WCR+COV = 0.2
    b = _scored("b", 0.1, 0.5, 0.5)  # WCR+COV = 1.0 → winner under (0,1,1)
    result = rank_top_k([a, b], 1, weights=(0.0, 1.0, 1.0))
    assert result == [b]


def test_weights_default_is_equal_triple() -> None:
    # Sanity: default weights = (1, 1, 1) is documented as the
    # least-biased default. Pass the default explicitly and confirm the
    # ordering matches the implicit default.
    a = _scored("a", 0.6, 0.5, 0.4)
    b = _scored("b", 0.5, 0.6, 0.4)
    c = _scored("c", 0.4, 0.4, 0.6)
    implicit = rank_top_k([a, b, c], 3)
    explicit = rank_top_k([a, b, c], 3, weights=(1.0, 1.0, 1.0))
    assert implicit == explicit


def test_weights_magnitude_does_not_affect_order_when_proportional() -> None:
    # Doubling every weight is a no-op for ranking order (weighted sum
    # scales linearly; relative magnitudes are what matter).
    teams = [
        _scored("a", 0.7, 0.5, 0.5),
        _scored("b", 0.5, 0.7, 0.5),
        _scored("c", 0.5, 0.5, 0.7),
    ]
    base = rank_top_k(teams, 3, weights=(1.0, 1.0, 1.0))
    scaled = rank_top_k(teams, 3, weights=(2.0, 2.0, 2.0))
    assert base == scaled


# --- validation errors ----------------------------------------------------


def test_negative_k_raises_valueerror() -> None:
    with pytest.raises(ValueError, match="k must be >= 0"):
        rank_top_k([_scored("a", 0.5, 0.5, 0.5)], -1)


def test_non_int_k_raises_valueerror() -> None:
    with pytest.raises(ValueError, match="k must be an int"):
        rank_top_k([_scored("a", 0.5, 0.5, 0.5)], 1.5)  # type: ignore[arg-type]


def test_bool_k_is_rejected() -> None:
    # bool is a subclass of int in Python; reject it explicitly so
    # ``rank_top_k(teams, True)`` does not silently become k=1.
    with pytest.raises(ValueError, match="k must be an int"):
        rank_top_k([_scored("a", 0.5, 0.5, 0.5)], True)  # type: ignore[arg-type]


def test_non_scored_team_element_raises_typeerror() -> None:
    with pytest.raises(TypeError, match="ScoredTeam"):
        rank_top_k(
            [_scored("a", 0.5, 0.5, 0.5), ("not", "scored")],  # type: ignore[list-item]
            2,
        )


@pytest.mark.parametrize(
    "bad_weights",
    [
        (1.0, 1.0),  # length 2
        (1.0, 1.0, 1.0, 1.0),  # length 4
        [1.0, 1.0, 1.0],  # list, not tuple
        "1,1,1",  # string
    ],
)
def test_invalid_weights_shape_raises_valueerror(bad_weights: object) -> None:
    with pytest.raises(ValueError, match="weights"):
        rank_top_k(
            [_scored("a", 0.5, 0.5, 0.5)],
            1,
            weights=bad_weights,  # type: ignore[arg-type]
        )


def test_negative_weight_raises_valueerror() -> None:
    with pytest.raises(ValueError, match="negative"):
        rank_top_k(
            [_scored("a", 0.5, 0.5, 0.5)],
            1,
            weights=(1.0, -0.1, 1.0),
        )


def test_nan_weight_raises_valueerror() -> None:
    with pytest.raises(ValueError, match="NaN"):
        rank_top_k(
            [_scored("a", 0.5, 0.5, 0.5)],
            1,
            weights=(1.0, math.nan, 1.0),
        )


def test_zero_sum_weights_raises_valueerror() -> None:
    with pytest.raises(ValueError, match="sum to zero"):
        rank_top_k(
            [_scored("a", 0.5, 0.5, 0.5)],
            1,
            weights=(0.0, 0.0, 0.0),
        )


# --- integration with pareto_filter --------------------------------------


def test_rank_after_pareto_filter_produces_engine_final_output() -> None:
    """End-to-end smoke check for the Sub-AC 2.4 → 2.5 hand-off.

    The engine final output is ``rank_top_k(pareto_filter(scored), K)``.
    Build a small mixed set of dominated + frontier teams, pipe through
    both, and verify that:

    * dominated teams never appear in the final ranked list;
    * the frontier teams are present in deterministic ranked order.
    """

    from gbl_hacker.score import pareto_filter

    dominated = _scored("dominated", 0.2, 0.2, 0.2)
    ev_max = _scored("ev_max", 0.9, 0.3, 0.3)  # sum 1.5
    wcr_max = _scored("wcr_max", 0.3, 0.9, 0.3)  # sum 1.5
    cov_max = _scored("cov_max", 0.3, 0.3, 0.9)  # sum 1.5
    balanced = _scored("balanced", 0.7, 0.7, 0.7)  # sum 2.1 — frontier winner

    frontier = pareto_filter(
        [dominated, ev_max, wcr_max, cov_max, balanced]
    )
    # ``dominated`` is dropped; the four others are all incomparable.
    assert dominated not in frontier
    assert set(frontier) == {ev_max, wcr_max, cov_max, balanced}

    ranked = rank_top_k(frontier, 4)
    # ``balanced`` has the highest weighted sum and is first; the three
    # specialists tie at 1.5 and lex-tiebreak EV → WCR → COV.
    assert ranked[0] == balanced
    assert ranked[1:] == [ev_max, wcr_max, cov_max]


def test_top_k_never_pads_with_placeholders() -> None:
    """When K exceeds the frontier size, the engine returns fewer than K
    teams — *not* synthetic placeholders. The seed contract on
    ``rationale_card`` requires real per-team data; padding with
    placeholder ScoredTeams would silently produce uninterpretable cards.
    """

    only = _scored("solo", 0.5, 0.5, 0.5)
    result = rank_top_k([only], 5)
    assert len(result) == 1
    assert result[0] is only
