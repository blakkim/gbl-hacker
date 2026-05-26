"""Tests for conservative shield baiting in ``_select_charged_move``.

Top-player guidance: don't bait reflexively. Bait (throw a cheap move to strip
a shield) only when the attacker holds a move threatening enough that the
defender is forced to shield it — lethal or near-lethal. With no such threat,
fire the best honest (highest-DPE) move; a needless bait just gets called.
"""
from __future__ import annotations

from gbl_hacker.build_registry import materialize_build
from gbl_hacker.simulator.matchup import CombatantState, _select_charged_move


def _attacker():
    # Stunfisk: two distinct charged moves (Mud Bomb 45e, Discharge 40e) where
    # the cheapest is NOT the highest-DPE — so bait vs honest are different moves.
    build = materialize_build("マッギョ")
    assert build is not None
    return build


def _defender():
    build = materialize_build("カビゴン")  # bulky Normal — moves land neutral
    assert build is not None
    return build


def test_baits_cheapest_against_a_near_lethal_threat():
    atk_build = _attacker()
    cheapest = min(atk_build.charged_moves, key=lambda m: m.energy_cost)
    attacker = CombatantState(build=atk_build, hp=atk_build.max_hp, energy=100, shields=2)
    defender = CombatantState(build=_defender(), hp=12, energy=0, shields=1)  # near-lethal range

    selected = _select_charged_move(attacker, defender)
    assert selected is cheapest  # strip the shield protecting against the kill


def test_throws_honest_when_no_forcing_threat():
    atk_build = _attacker()
    cheapest = min(atk_build.charged_moves, key=lambda m: m.energy_cost)
    attacker = CombatantState(build=atk_build, hp=atk_build.max_hp, energy=100, shields=2)
    defender = CombatantState(build=_defender(), hp=_defender().max_hp, energy=0, shields=1)  # healthy

    selected = _select_charged_move(attacker, defender)
    # No forcing threat → commit the best honest move, NOT a reflexive bait.
    assert selected is not cheapest


def test_selection_is_shield_threat_dependent():
    """The same matchup baits at low defender HP but not at high — the
    behavioral contrast that proves baiting is conditional."""
    atk_build = _attacker()
    attacker = CombatantState(build=atk_build, hp=atk_build.max_hp, energy=100, shields=2)
    low = CombatantState(build=_defender(), hp=12, energy=0, shields=1)
    high = CombatantState(build=_defender(), hp=_defender().max_hp, energy=0, shields=1)
    assert _select_charged_move(attacker, low) is not _select_charged_move(attacker, high)
