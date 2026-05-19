"""Unit tests for asymmetric-shield matchup resolution (Sub-AC 4.2).

PvPoke's isolated-matchup simulator treats shield count as a per-matchup
parameter and tends to assume both sides arrive at the same shield value.
The set-state-aware simulator built in :mod:`gbl_hacker.simulator.matchup`
must consume ``a_shields`` and ``b_shields`` independently — and the
*outcome* of a matchup must demonstrably differ when those values diverge.

The headline test :func:`test_asymmetric_shields_flip_winner` builds two
identical Pokémon, runs them at (2,2), (1,2), and (2,1), and asserts that
the asymmetric (1,2) case produces a *different winner* than the
symmetric (2,2) case. That is the canonical "shield count matters per
side, not per matchup" property the simulator is contracted to express.
"""

from __future__ import annotations

import pytest

from gbl_hacker.simulator import (
    ChargedMove,
    CombatantBuild,
    CombatantState,
    FastMove,
    resolve_matchup,
)

# --- fixtures --------------------------------------------------------------

# Two identical, deterministic mid-tier-shaped combatants. Numbers chosen
# so the matchup resolves cleanly: both sides reach charged at the same
# turns and shield availability is the only difference across runs.
#
# Per-turn economy:
#   fast: 2 damage, +8 energy   -> charged at turn 5 (5 * 8 == 40)
#   charged: 40 energy, 70 damage
#   max HP: 100
#
# Hand-traced terminal turn for all three cases below is turn 15 — the
# numbers are tuned so the winner flips when A loses one shield.


def _make_build(species: str) -> CombatantBuild:
    return CombatantBuild(
        species=species,
        max_hp=100,
        fast=FastMove(name="quick", damage=2, energy_gain=8),
        charged=ChargedMove(name="bomb", energy_cost=40, damage=70),
    )


def _starting_state(build: CombatantBuild, shields: int) -> CombatantState:
    return CombatantState.fresh(build, shields=shields)


# --- the headline asymmetric-vs-symmetric divergence test -----------------


def test_asymmetric_shields_flip_winner() -> None:
    """Asymmetric (1,2) shields produce a different winner than (2,2).

    Identical builds, identical HP/energy at start — the only thing that
    changes between runs is the per-side shield count. The (2,2) baseline
    and (1,2) asymmetric run resolve to *different winners*, which is the
    behavior PvPoke's isolated-matchup model cannot express without an
    external wrapper.
    """

    build_a = _make_build("alpha")
    build_b = _make_build("bravo")

    # Symmetric 2/2 — A wins on the third charged exchange because A
    # gets charged-move priority on the deciding turn.
    sym = resolve_matchup(
        _starting_state(build_a, shields=2),
        _starting_state(build_b, shields=2),
    )
    assert sym.winner == "A", (
        f"sanity: symmetric (2,2) baseline must resolve to A winning; "
        f"got winner={sym.winner!r} hp=({sym.a_terminal_hp},{sym.b_terminal_hp})"
    )

    # Asymmetric A=1, B=2 — A burns through its lone shield earlier and
    # eats a full-damage charged move it cannot block; B wins.
    asym_a_low = resolve_matchup(
        _starting_state(build_a, shields=1),
        _starting_state(build_b, shields=2),
    )
    assert asym_a_low.winner == "B", (
        "Asymmetric (1,2) must NOT collapse to the symmetric outcome — "
        "A's missing shield is exactly what the resolver must surface. "
        f"got winner={asym_a_low.winner!r} hp="
        f"({asym_a_low.a_terminal_hp},{asym_a_low.b_terminal_hp})"
    )

    # The headline contract: asymmetric outcome differs from symmetric.
    assert asym_a_low.winner != sym.winner, (
        "Sub-AC 4.2 contract violation: shield asymmetry produced the "
        "same winner as the symmetric baseline. The resolver is collapsing "
        "the per-side shield count somewhere — this is the PvPoke isolated-"
        "matchup bug this simulator exists to fix."
    )

    # Mirror asymmetry: A=2, B=1 — B burns through its lone shield and
    # gets KO'd before A. Same winner as the symmetric case, but the
    # terminal state diverges (B faints earlier in the turn sequence).
    asym_b_low = resolve_matchup(
        _starting_state(build_a, shields=2),
        _starting_state(build_b, shields=1),
    )
    assert asym_b_low.winner == "A"
    assert (
        asym_b_low.b_terminal_hp == 0
        and asym_b_low.a_terminal_hp > 0
    ), "A=2, B=1 must KO B with A still alive"


# --- pinpoint terminal-state assertions for the (1,2) asymmetric case -----


def test_asymmetric_one_two_terminal_state_matches_hand_trace() -> None:
    """The (A=1, B=2) outcome matches the hand-traced expected values.

    Locks the resolver against silent regressions. Numbers come from a
    hand trace of the deterministic turn loop:

      * Turn 5  — both reach charged. A's shield is consumed (1 -> 0).
                  B's shield is consumed (2 -> 1).
      * Turn 10 — both reach charged again. B still has a shield (1 -> 0).
                  A has no shield: takes 70 damage → HP drops to 9.
      * Turn 15 — fast tick KO's A (HP 1 → 0). B is at 68 HP, full energy.
    """

    build_a = _make_build("alpha")
    build_b = _make_build("bravo")
    result = resolve_matchup(
        _starting_state(build_a, shields=1),
        _starting_state(build_b, shields=2),
    )

    assert result.winner == "B"
    assert result.turns == 15
    assert result.a_terminal_hp == 0
    assert result.b_terminal_hp == 68
    assert result.a_terminal_shields == 0
    assert result.b_terminal_shields == 0
    # Both sides finish the matchup with a full charged ready — neither
    # got to fire it on the final turn because the fast tick KO'd A first.
    assert result.a_terminal_energy == 40
    assert result.b_terminal_energy == 40


# --- API-shape contract: shields are independent --------------------------


@pytest.mark.parametrize(
    "a_shields,b_shields",
    [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1), (2, 2),
    ],
)
def test_resolver_accepts_full_shield_grid(a_shields: int, b_shields: int) -> None:
    """All nine cells of the 0/1/2 x 0/1/2 shield grid resolve without error.

    Demonstrates the API surface accepts per-side shield counts
    independently — there is no shared "shield count" parameter that
    would forcibly couple the two sides.
    """
    build_a = _make_build("alpha")
    build_b = _make_build("bravo")
    result = resolve_matchup(
        _starting_state(build_a, shields=a_shields),
        _starting_state(build_b, shields=b_shields),
    )
    # Terminal shield counts cannot exceed starting counts (shields are
    # only consumed, never gained).
    assert 0 <= result.a_terminal_shields <= a_shields
    assert 0 <= result.b_terminal_shields <= b_shields


def test_shielded_charged_move_passes_one_damage_through() -> None:
    """The "1 damage gets through a shield" GBL rule is honored."""
    build_a = _make_build("alpha")
    build_b = _make_build("bravo")
    result = resolve_matchup(
        _starting_state(build_a, shields=2),
        _starting_state(build_b, shields=2),
    )
    shielded_events = [e for e in result.charged_events if e.shielded]
    assert shielded_events, "expected at least one shielded charged event"
    for event in shielded_events:
        assert event.damage_applied == 1, (
            f"shielded charged event must apply 1 bleed damage; "
            f"got {event.damage_applied}"
        )


def test_zero_shield_charged_move_applies_full_damage() -> None:
    """A defender with 0 shields eats full charged damage."""
    build_a = _make_build("alpha")
    build_b = _make_build("bravo")
    result = resolve_matchup(
        _starting_state(build_a, shields=2),
        _starting_state(build_b, shields=0),
    )
    full_dmg_events = [
        e for e in result.charged_events if not e.shielded and e.attacker == "A"
    ]
    assert full_dmg_events, "expected at least one unshielded charged event from A"
    assert full_dmg_events[0].damage_applied == 70


# --- invalid-input guard --------------------------------------------------


@pytest.mark.parametrize("bad_shields", [-1, 3])
def test_shields_out_of_range_rejected(bad_shields: int) -> None:
    build_a = _make_build("alpha")
    with pytest.raises(ValueError, match="shields"):
        CombatantState(build=build_a, hp=100, energy=0, shields=bad_shields)
