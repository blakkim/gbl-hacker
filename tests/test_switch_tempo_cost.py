"""Tests for the active-switch tempo cost (:func:`_apply_switch_tempo_cost`).

Switching out of a losing matchup is not free: the switcher spends its turn
bringing the new Pokémon in, so the opponent lands one free fast move — dealing
damage to the incoming mon and banking that fast move's energy. A swap policy
that ignored this would over-value switching.
"""
from __future__ import annotations

from gbl_hacker.build_registry import materialize_build
from gbl_hacker.simulator.set_driver import SetSlot, _apply_switch_tempo_cost
from gbl_hacker.simulator.state import ENERGY_CAP


def _slot(species_ja: str) -> SetSlot:
    build = materialize_build(species_ja)
    assert build is not None
    return SetSlot.fresh(build)


def test_incoming_takes_damage_and_opponent_banks_energy():
    incoming = _slot("マッギョ")  # base Stunfisk
    opponent = _slot("ファイアロー")  # lands a free fast move (Incinerate)
    opp_fast = opponent.to_combatant_state().effective_build.fast

    new_inc, new_opp = _apply_switch_tempo_cost(incoming=incoming, opponent=opponent)

    assert new_inc.hp < incoming.hp  # incoming ate the free fast move
    assert new_opp.energy == opponent.energy + opp_fast.energy_gain
    # The free move only hits one way: opponent HP and incoming energy unchanged.
    assert new_opp.hp == opponent.hp
    assert new_inc.energy == incoming.energy


def test_tempo_cost_can_faint_a_one_hp_incoming():
    incoming = _slot("マッギョ").with_(hp=1)
    opponent = _slot("ファイアロー")
    new_inc, _new_opp = _apply_switch_tempo_cost(incoming=incoming, opponent=opponent)
    assert new_inc.hp == 0
    assert new_inc.fainted


def test_opponent_energy_is_capped():
    incoming = _slot("マッギョ")
    opponent = _slot("ファイアロー").with_(energy=ENERGY_CAP - 1)
    _new_inc, new_opp = _apply_switch_tempo_cost(incoming=incoming, opponent=opponent)
    assert new_opp.energy == ENERGY_CAP
