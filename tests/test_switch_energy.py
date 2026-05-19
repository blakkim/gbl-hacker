"""Unit tests for entry-energy carry-over across switches (Sub-AC 4.1).

These tests pin the simulator's contract that the rest of the engine relies
on: a Pokémon's accumulated energy follows it across a switch. The headline
test is :func:`test_incoming_pokemon_retains_pre_accumulated_energy`, which
encodes the exact case PvPoke's isolated-matchup simulator gets wrong —
incoming energy is preserved, not reset to zero.
"""

from __future__ import annotations

import pytest

from gbl_hacker.simulator import (
    ENERGY_CAP,
    SetState,
    SlotState,
    SwitchError,
    apply_switch,
    entry_energy,
)


# --- fixtures --------------------------------------------------------------


def _make_side(
    *,
    lead_energy: int = 0,
    safe_energy: int = 0,
    closer_energy: int = 0,
    lead_hp: int = 120,
    safe_hp: int = 130,
    closer_hp: int = 140,
    lead_shields: int = 2,
    safe_shields: int = 2,
    closer_shields: int = 2,
    on_field: int = 0,
) -> SetState:
    """Build a 3-slot side parametrically so each test reads at a glance."""

    slots = (
        SlotState(
            species="lead", hp=lead_hp, energy=lead_energy, shields=lead_shields
        ),
        SlotState(
            species="safe_swap",
            hp=safe_hp,
            energy=safe_energy,
            shields=safe_shields,
        ),
        SlotState(
            species="closer",
            hp=closer_hp,
            energy=closer_energy,
            shields=closer_shields,
        ),
    )
    return SetState.starting(slots, lead_index=on_field)


# --- the headline carry-over test -----------------------------------------


def test_incoming_pokemon_retains_pre_accumulated_energy() -> None:
    """Switching does not reset the incoming Pokémon's energy.

    This is the canonical state-divergence scenario the seed calls out:
    PvPoke's isolated-matchup simulator restarts each fight at 0 energy.
    Our set-state simulator must instead honor the energy each slot has
    accumulated across the prior matchup(s).

    Scenario:
        - Lead is on field with 40 energy (mid-bait cycle, hasn't fired yet).
        - Safe-swap already accumulated 73 energy in an earlier rotation.
        - We rotate to safe-swap.
    Expected:
        - Safe-swap enters the field with energy == 73 (NOT 0).
        - Lead leaves the field still carrying its 40 energy.
        - Energies are byte-equal before vs after on every other slot.
    """

    pre = _make_side(
        lead_energy=40,
        safe_energy=73,
        closer_energy=12,
        on_field=0,  # lead on field
    )
    assert pre.on_field_index == 0

    # entry_energy is a pure-function preview — must report the real value
    # the incoming Pokémon will arrive with, not 0.
    previewed_entry = entry_energy(pre, incoming_index=1)
    assert previewed_entry == 73, (
        "entry_energy must surface the safe-swap's pre-accumulated energy; "
        "PvPoke's isolated-matchup sim would return 0 here."
    )

    post = apply_switch(pre, incoming_index=1)

    # 1. The incoming Pokémon is on field, carrying its 73 energy.
    assert post.on_field_index == 1
    assert post.slots[1].on_field is True
    assert post.slots[1].energy == 73, (
        "REGRESSION: incoming Pokémon's energy was reset on switch — "
        "this is the exact PvPoke isolated-matchup bug this simulator fixes."
    )

    # 2. The outgoing Pokémon left field but kept its 40 energy.
    assert post.slots[0].on_field is False
    assert post.slots[0].energy == 40

    # 3. The bench Pokémon is unaffected.
    assert post.slots[2].on_field is False
    assert post.slots[2].energy == 12

    # 4. HP and shields preserved across the switch on every slot.
    for before, after in zip(pre.slots, post.slots, strict=True):
        assert before.hp == after.hp
        assert before.shields == after.shields


# --- secondary invariants --------------------------------------------------


def test_switching_back_resumes_outgoing_energy() -> None:
    """A → B → A returns the lead to its original energy (no drift)."""
    pre = _make_side(lead_energy=55, safe_energy=88, on_field=0)
    mid = apply_switch(pre, incoming_index=1)
    back = apply_switch(mid, incoming_index=0)

    assert back.on_field_index == 0
    assert back.slots[0].energy == 55, "lead must resume at its prior energy"
    assert back.slots[1].energy == 88, "safe-swap must retain its energy off field"
    # Identity by value: a A→B→A round trip is a no-op on energy fields.
    assert tuple(s.energy for s in back.slots) == tuple(s.energy for s in pre.slots)


def test_switch_does_not_transfer_energy_between_teammates() -> None:
    """The outgoing slot's energy is NOT shared with the incoming slot."""
    pre = _make_side(lead_energy=99, safe_energy=10, on_field=0)
    post = apply_switch(pre, incoming_index=1)

    assert post.slots[1].energy == 10, (
        "incoming slot must use its own accumulated energy only — "
        "GBL never pools energy across teammates."
    )
    assert post.slots[0].energy == 99


def test_switch_preserves_energy_cap_invariant() -> None:
    """Switching never produces an energy value outside [0, ENERGY_CAP]."""
    pre = _make_side(lead_energy=0, safe_energy=ENERGY_CAP, on_field=0)
    post = apply_switch(pre, incoming_index=1)
    assert 0 <= post.slots[0].energy <= ENERGY_CAP
    assert 0 <= post.slots[1].energy <= ENERGY_CAP
    assert post.slots[1].energy == ENERGY_CAP


def test_state_objects_are_immutable_snapshots() -> None:
    """Each switch returns a fresh state; the input is not mutated."""
    pre = _make_side(lead_energy=30, safe_energy=60, on_field=0)
    snapshot = (
        pre.on_field_index,
        tuple((s.species, s.hp, s.energy, s.shields, s.on_field) for s in pre.slots),
    )
    _ = apply_switch(pre, incoming_index=1)
    after_snapshot = (
        pre.on_field_index,
        tuple((s.species, s.hp, s.energy, s.shields, s.on_field) for s in pre.slots),
    )
    assert snapshot == after_snapshot, "apply_switch must not mutate its input"


# --- illegal-switch guards -------------------------------------------------


def test_switch_to_already_on_field_slot_raises() -> None:
    pre = _make_side(on_field=0)
    with pytest.raises(SwitchError, match="already on field"):
        apply_switch(pre, incoming_index=0)


def test_switch_to_fainted_slot_raises() -> None:
    fainted_safe = SlotState(
        species="safe_swap", hp=0, energy=42, shields=0, alive=False, on_field=False
    )
    slots = (
        SlotState(species="lead", hp=120, energy=20, on_field=True),
        fainted_safe,
        SlotState(species="closer", hp=140, energy=0),
    )
    pre = SetState(slots=slots)
    with pytest.raises(SwitchError, match="fainted"):
        apply_switch(pre, incoming_index=1)


@pytest.mark.parametrize("bad_index", [-1, 3, 99])
def test_switch_to_out_of_range_index_raises(bad_index: int) -> None:
    pre = _make_side(on_field=0)
    with pytest.raises(SwitchError, match="out of range"):
        apply_switch(pre, incoming_index=bad_index)


def test_entry_energy_is_zero_for_fresh_slot() -> None:
    """A slot that has never been on field reports entry_energy == 0.

    This is the *only* case where the set-state simulator agrees with
    PvPoke's "everything starts at zero" assumption — useful as a
    sanity floor and a contrast to the non-zero carry-over test above.
    """
    pre = _make_side(lead_energy=40, safe_energy=0, on_field=0)
    assert entry_energy(pre, incoming_index=1) == 0
