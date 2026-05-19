"""Unit tests for ``select_unfavorable_matchups`` (Sub-AC 3.2).

The headline contract this test suite enforces:

* ``select_unfavorable_matchups(team, results, n=N)`` returns the
  **bottom-N records ranked ascending by ``win_rate``** —
  unfavorable = lower win rate.
* The N-cap is honored: the output length is ``min(n, len(input))``;
  no padding when fewer records exist.
* The ranking is **stable**: ties on ``win_rate`` preserve input order.
* Out-of-range win rates / negative ``n`` are rejected loudly.
* The function does not mutate its input (callers reuse the same list
  to derive both favorable + unfavorable selections side-by-side).
* The favorable and unfavorable selectors must coexist on the same
  data: running both back-to-back returns the *opposite* ends in the
  *opposite* sort order, and neither selector mutates the shared list.

The fixed-fixture ordering + N-cap test is the AC's explicit minimum
deliverable; the rest of the file fences in the documented contract so
the rationale-card renderer can rely on it without re-checking.

The fixture shape mirrors ``test_select_favorable_matchups.py`` line
for line — cross-axis diffing stays a one-glance affair, and the
"reverse-of-favorable on full-input" coexistence test below is only
meaningful if the two fixture helpers are byte-identical.
"""

from __future__ import annotations

import pytest

from gbl_hacker.score import (
    CandidateTeam,
    MetaMatchupResult,
    select_favorable_matchups,
    select_unfavorable_matchups,
)
from gbl_hacker.simulator import (
    ChargedMove,
    CombatantBuild,
    FastMove,
)


# --- fixture helpers ------------------------------------------------------
# Structurally aligned with test_select_favorable_matchups.py so the
# coexistence test below ("favorable + unfavorable on the same list")
# is meaningful and the two test files diff cleanly.


def _build(species: str, *, max_hp: int = 100, fast_damage: int = 2) -> CombatantBuild:
    return CombatantBuild(
        species=species,
        max_hp=max_hp,
        fast=FastMove(name="quick", damage=fast_damage, energy_gain=8),
        charged=ChargedMove(name="bomb", energy_cost=40, damage=70),
    )


def _candidate_team(*species_names: str) -> CandidateTeam:
    if len(species_names) != 3:
        raise AssertionError(
            f"test fixture needs 3 species, got {len(species_names)}"
        )
    return CandidateTeam.from_slots(_build(s) for s in species_names)


def _record(
    opp_prefix: str,
    win_rate: float,
    *,
    usage_pct: float | None = None,
) -> MetaMatchupResult:
    """Build a ``MetaMatchupResult`` with a unique opponent triple."""

    opponent = _candidate_team(
        f"{opp_prefix}-1",
        f"{opp_prefix}-2",
        f"{opp_prefix}-3",
    )
    return MetaMatchupResult(
        opponent=opponent,
        win_rate=win_rate,
        usage_pct=usage_pct,
    )


# ---------------------------------------------------------------------------
# Ordering + N-cap on a fixed fixture — the AC's explicit minimum deliverable
# ---------------------------------------------------------------------------


def test_orders_ascending_by_win_rate_and_caps_at_n() -> None:
    """Fixed fixture: 5 records, ask for top-3 worst → ascending by win_rate.

    Setup (deliberately *unsorted* in the input):
      * opp-A: 0.40
      * opp-B: 0.90
      * opp-C: 0.55
      * opp-D: 0.80
      * opp-E: 0.05

    Expected bottom-3 (the rationale card's "key losses" bullet list):
      1. opp-E (0.05)
      2. opp-A (0.40)
      3. opp-C (0.55)

    A bug that sorted descending would return [B, D, C]; a bug that
    forgot the N-cap would return all five; a bug that lost stability
    would not show up on this fixture (all win rates are distinct) —
    that is fenced by the separate stability test below.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    records = [
        _record("opp-A", 0.40),
        _record("opp-B", 0.90),
        _record("opp-C", 0.55),
        _record("opp-D", 0.80),
        _record("opp-E", 0.05),
    ]
    bottom = select_unfavorable_matchups(team, records, n=3)

    assert len(bottom) == 3, f"N-cap violated: expected 3 records, got {len(bottom)}"

    # Strict ascending order — ordering contract.
    win_rates = [r.win_rate for r in bottom]
    assert win_rates == [0.05, 0.40, 0.55], (
        f"ordering contract violated: expected [0.05, 0.4, 0.55], got {win_rates}"
    )

    # Opponent identity is preserved on the records — the rationale card
    # downstream prints these species names.
    species_bottom = [r.opponent.species for r in bottom]
    assert species_bottom == [
        ("opp-E-1", "opp-E-2", "opp-E-3"),
        ("opp-A-1", "opp-A-2", "opp-A-3"),
        ("opp-C-1", "opp-C-2", "opp-C-3"),
    ]


# ---------------------------------------------------------------------------
# N-cap edge cases
# ---------------------------------------------------------------------------


def test_n_zero_returns_empty_list() -> None:
    """``n = 0`` is a legitimate "select nothing yet" signal, not an error."""

    team = _candidate_team("y-1", "y-2", "y-3")
    records = [_record("opp-A", 0.9), _record("opp-B", 0.5)]
    assert select_unfavorable_matchups(team, records, n=0) == []


def test_n_larger_than_input_returns_everything_in_rank_order() -> None:
    """No padding: contract is "up to N records, in ranked order"."""

    team = _candidate_team("y-1", "y-2", "y-3")
    records = [
        _record("opp-A", 0.8),
        _record("opp-B", 0.2),
    ]
    bottom = select_unfavorable_matchups(team, records, n=10)
    assert len(bottom) == 2
    assert [r.win_rate for r in bottom] == [0.2, 0.8]


def test_empty_input_returns_empty_list() -> None:
    """Empty meta → empty result, regardless of ``n``."""

    team = _candidate_team("y-1", "y-2", "y-3")
    assert select_unfavorable_matchups(team, [], n=5) == []
    assert select_unfavorable_matchups(team, [], n=0) == []


def test_default_n_is_three() -> None:
    """The rationale card convention of "bottom three" is the default."""

    team = _candidate_team("y-1", "y-2", "y-3")
    records = [
        _record("opp-A", 0.1),
        _record("opp-B", 0.2),
        _record("opp-C", 0.3),
        _record("opp-D", 0.4),
        _record("opp-E", 0.5),
    ]
    bottom = select_unfavorable_matchups(team, records)
    assert len(bottom) == 3
    assert [r.win_rate for r in bottom] == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Stability — ties preserve input order
# ---------------------------------------------------------------------------


def test_ties_preserve_input_order() -> None:
    """Stable sort: equal win rates keep the order they were given.

    Three records all at 0.0 (the GBL "auto-loss" floor value). The
    rationale card must render them in the *same* order on repeat
    runs; an unstable sort would let the order drift between runs over
    the same data.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    records = [
        _record("first", 0.0),
        _record("second", 0.0),
        _record("third", 0.0),
    ]
    bottom = select_unfavorable_matchups(team, records, n=3)
    assert [r.opponent.species for r in bottom] == [
        ("first-1", "first-2", "first-3"),
        ("second-1", "second-2", "second-3"),
        ("third-1", "third-2", "third-3"),
    ]


def test_partial_tie_preserves_relative_order_among_ties() -> None:
    """When only some records tie, the tied subset keeps input order.

    Setup:
      * A: 0.9
      * B: 0.3   ← ties with D
      * C: 0.6
      * D: 0.3   ← ties with B
      * E: 0.05

    Expected (ascending by win_rate, B-before-D by stability):
      [E (0.05), B (0.3), D (0.3), C (0.6), A (0.9)]
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    records = [
        _record("A", 0.9),
        _record("B", 0.3),
        _record("C", 0.6),
        _record("D", 0.3),
        _record("E", 0.05),
    ]
    bottom = select_unfavorable_matchups(team, records, n=5)
    assert [r.opponent.species[0] for r in bottom] == [
        "E-1",
        "B-1",
        "D-1",
        "C-1",
        "A-1",
    ]


# ---------------------------------------------------------------------------
# Purity — input is not mutated
# ---------------------------------------------------------------------------


def test_input_list_is_not_mutated() -> None:
    """The caller can reuse its input list after the call (e.g. for the
    favorable pass that lives in the sibling Sub-AC 3.1)."""

    team = _candidate_team("y-1", "y-2", "y-3")
    records = [
        _record("opp-A", 0.4),
        _record("opp-B", 0.9),
        _record("opp-C", 0.5),
    ]
    before = list(records)
    _ = select_unfavorable_matchups(team, records, n=2)
    assert records == before, "input list mutated by select_unfavorable_matchups"


def test_works_on_generators() -> None:
    """The function consumes its iterable exactly once — generators work."""

    team = _candidate_team("y-1", "y-2", "y-3")
    pairs = [("A", 0.8), ("B", 0.1), ("C", 0.5)]
    gen = (_record(name, rate) for name, rate in pairs)
    bottom = select_unfavorable_matchups(team, gen, n=2)
    assert [r.win_rate for r in bottom] == [0.1, 0.5]


# ---------------------------------------------------------------------------
# Usage_pct passthrough — preserved on the returned records
# ---------------------------------------------------------------------------


def test_usage_pct_is_preserved_on_returned_records() -> None:
    """The optional meta-usage weight follows the record through the sort."""

    team = _candidate_team("y-1", "y-2", "y-3")
    records = [
        _record("opp-A", 0.4, usage_pct=10.0),
        _record("opp-B", 0.9, usage_pct=30.0),
        _record("opp-C", 0.5, usage_pct=None),
    ]
    bottom = select_unfavorable_matchups(team, records, n=3)
    by_rate = {r.win_rate: r.usage_pct for r in bottom}
    assert by_rate == {0.4: 10.0, 0.5: None, 0.9: 30.0}


# ---------------------------------------------------------------------------
# Coexistence with select_favorable_matchups — the rationale-card pipeline
# ---------------------------------------------------------------------------


def test_favorable_and_unfavorable_on_full_input_are_reverses() -> None:
    """When ``n == len(records)`` and all win rates are distinct, the
    unfavorable selection is the reverse of the favorable selection.

    The seed's rationale card runs both selectors on the *same* list to
    populate "best wins" + "worst losses" simultaneously. This test
    locks the symmetry-on-full-input invariant, which catches drift
    such as "favorable" accidentally also sorting ascending.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    records = [
        _record("opp-A", 0.20),
        _record("opp-B", 0.95),
        _record("opp-C", 0.45),
        _record("opp-D", 0.70),
    ]
    top = select_favorable_matchups(team, records, n=4)
    bottom = select_unfavorable_matchups(team, records, n=4)
    assert [r.win_rate for r in top] == list(reversed([r.win_rate for r in bottom]))
    # ...and the same records appear in both selections (just reordered).
    assert {r.win_rate for r in top} == {r.win_rate for r in bottom}


def test_favorable_and_unfavorable_share_the_same_list_without_mutating_it() -> None:
    """Running both selectors back-to-back on one list is safe.

    This is the *typical* rationale-card call shape: build the
    per-opponent records once, then ask for both ends. Either selector
    mutating the list would break the other on the second call.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    records = [
        _record("opp-A", 0.40),
        _record("opp-B", 0.90),
        _record("opp-C", 0.55),
        _record("opp-D", 0.80),
        _record("opp-E", 0.05),
    ]
    snapshot_before = list(records)

    top = select_favorable_matchups(team, records, n=2)
    bottom = select_unfavorable_matchups(team, records, n=2)

    assert records == snapshot_before, "input list mutated by selector pair"
    assert [r.win_rate for r in top] == [0.90, 0.80]
    assert [r.win_rate for r in bottom] == [0.05, 0.40]


# ---------------------------------------------------------------------------
# Contract enforcement — N / win_rate / type validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_n", [-1, -5, -100])
def test_negative_n_rejected(bad_n: int) -> None:
    team = _candidate_team("y-1", "y-2", "y-3")
    records = [_record("opp-A", 0.5)]
    with pytest.raises(ValueError, match="n must be >= 0"):
        select_unfavorable_matchups(team, records, n=bad_n)


@pytest.mark.parametrize("bad_n", [1.5, "3", None, [3]])
def test_non_int_n_rejected(bad_n: object) -> None:
    team = _candidate_team("y-1", "y-2", "y-3")
    records = [_record("opp-A", 0.5)]
    with pytest.raises(ValueError, match="n must be an int"):
        select_unfavorable_matchups(team, records, n=bad_n)  # type: ignore[arg-type]


def test_bool_n_is_rejected() -> None:
    """``bool`` is an ``int`` subclass in Python; reject it explicitly."""

    team = _candidate_team("y-1", "y-2", "y-3")
    records = [_record("opp-A", 0.5)]
    with pytest.raises(ValueError, match="n must be an int"):
        select_unfavorable_matchups(team, records, n=True)  # type: ignore[arg-type]


def test_non_record_element_raises_typeerror() -> None:
    team = _candidate_team("y-1", "y-2", "y-3")
    bad_records = [_record("opp-A", 0.5), "not-a-record"]
    with pytest.raises(TypeError, match="MetaMatchupResult"):
        select_unfavorable_matchups(team, bad_records, n=2)  # type: ignore[arg-type]
