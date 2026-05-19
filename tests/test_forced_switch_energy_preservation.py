"""Unit tests for mid-matchup forced switch with partial energy preservation
(Sub-AC 4.3).

A *forced switch* in GBL Hacker is a mid-matchup transition where the on-field
Pokémon has been actively fighting — it has accumulated energy, taken damage,
and possibly spent shields *during the matchup* — and is then yanked off the
field by the matchup driver (a faint-trigger swap, a tactical surprise switch
fired before a charged move lands, etc).

The contract the simulator must honor — and that PvPoke's isolated-matchup
model does NOT honor:

* The outgoing Pokémon's **mid-matchup residual energy** must be preserved
  byte-for-byte onto its :class:`SlotState`. It is **not** zeroed (the
  PvPoke isolated-matchup bug) and it is **not** "refunded" — neither
  rewound to the slot's pre-matchup energy nor reset to its entry energy.
  Whatever the Pokémon had accumulated in the active fight is what it
  carries into bench storage for the next time it comes back on field.
* The incoming Pokémon brings **its own** previously-stored energy — energy
  is never pooled across teammates.

The headline test is :func:`test_forced_switch_preserves_outgoing_residual_energy`:
it builds a scenario where the residual energy value (57) is distinct from
the zeroed value (0), from the entry energy (20), and from the full-refund
cap (100), and asserts the outgoing slot carries exactly 57. That triple
distinction is what pins down "preserved, neither zeroed nor refunded".
"""

from __future__ import annotations

import pytest

from gbl_hacker.simulator import (
    ENERGY_CAP,
    SetState,
    SlotState,
    SwitchError,
    apply_forced_switch,
)


# --- fixtures -------------------------------------------------------------


def _three_slot_side(
    *,
    lead_entry_energy: int = 20,
    safe_stored_energy: int = 73,
    closer_stored_energy: int = 12,
    lead_hp: int = 120,
    safe_hp: int = 130,
    closer_hp: int = 140,
    lead_shields: int = 2,
    safe_shields: int = 2,
    closer_shields: int = 2,
    on_field: int = 0,
) -> SetState:
    """Construct a 3-slot side with named per-slot HP/energy/shields.

    ``lead_entry_energy`` is the energy the on-field slot brought *into* the
    current matchup. The matchup driver tracks its mid-matchup energy
    separately and hands :func:`apply_forced_switch` a *residual* value
    that is typically different from this entry value.
    """

    slots = (
        SlotState(
            species="lead",
            hp=lead_hp,
            energy=lead_entry_energy,
            shields=lead_shields,
        ),
        SlotState(
            species="safe_swap",
            hp=safe_hp,
            energy=safe_stored_energy,
            shields=safe_shields,
        ),
        SlotState(
            species="closer",
            hp=closer_hp,
            energy=closer_stored_energy,
            shields=closer_shields,
        ),
    )
    return SetState.starting(slots, lead_index=on_field)


# --- headline preservation test ------------------------------------------


def test_forced_switch_preserves_outgoing_residual_energy() -> None:
    """A forced mid-matchup switch preserves the residual energy.

    The outgoing Pokémon's mid-matchup energy is **preserved**: not
    zeroed (PvPoke isolated-matchup behaviour), not refunded back to the
    slot's entry energy, and not snapped to the energy cap.

    Scenario:
        * Lead entered the matchup with 20 energy.
        * Mid-matchup, the matchup driver has tracked it accumulating
          37 more energy → residual energy at the moment of the forced
          switch is **57**.
        * Mid-matchup, lead took 80 HP of damage (120 -> 40) and burned
          one shield (2 -> 1).
        * The matchup driver fires a forced switch to the safe-swap.

    Expected:
        * Outgoing lead slot stores energy == 57 (the residual, byte-equal).
        * Outgoing lead slot stores hp == 40, shields == 1 (the residual
          HP and shield count, also folded in).
        * Outgoing lead is off field, but still alive (hp > 0).
        * Incoming safe-swap enters on field with **its own** stored
          energy == 73, NOT 57 and NOT 0.
        * The bench closer is untouched.

    The 57 / 20 / 0 / 100 distinction is intentional: it rules out any
    of the three plausible bugs in a single assertion bundle:
        * energy == 0   → PvPoke-style zeroing
        * energy == 20  → "refunded" back to entry energy
        * energy == 100 → "refunded" to the cap (no-op switch interpreted
                          as a charged-move cancel)
    Only ``energy == 57`` satisfies the preservation contract.
    """

    pre = _three_slot_side(
        lead_entry_energy=20,
        safe_stored_energy=73,
        closer_stored_energy=12,
        lead_hp=120,
        lead_shields=2,
        on_field=0,
    )
    assert pre.on_field_index == 0
    assert pre.slots[0].energy == 20, "fixture sanity: lead entered with 20"

    residual_energy = 57  # 20 entry + 37 mid-matchup fast-move accumulation
    residual_hp = 40
    residual_shields = 1

    # Triple-distinctness sanity: the residual value must be distinct from
    # zeroed (0), refunded-to-entry (20), and refunded-to-cap (100) so
    # that the post-condition assertions below cannot be satisfied by any
    # of those bugs by accident.
    assert residual_energy not in (0, pre.slots[0].energy, ENERGY_CAP)

    post = apply_forced_switch(
        pre,
        incoming_index=1,
        outgoing_residual_hp=residual_hp,
        outgoing_residual_energy=residual_energy,
        outgoing_residual_shields=residual_shields,
    )

    # --- 1. Outgoing slot carries residual energy verbatim ------------
    assert post.slots[0].energy == residual_energy, (
        "REGRESSION: outgoing slot's mid-matchup energy was not preserved. "
        f"Expected {residual_energy}, got {post.slots[0].energy}. "
        "Preservation contract: energy must be neither zeroed (PvPoke bug) "
        "nor refunded to the slot's entry energy."
    )
    assert post.slots[0].energy != 0, (
        "REGRESSION: outgoing energy was zeroed — this is the exact PvPoke "
        "isolated-matchup bug Sub-AC 4.3 exists to prevent."
    )
    assert post.slots[0].energy != pre.slots[0].energy, (
        "REGRESSION: outgoing energy was 'refunded' back to entry energy. "
        "Mid-matchup energy accumulation must be preserved, not rolled back."
    )
    assert post.slots[0].energy != ENERGY_CAP, (
        "REGRESSION: outgoing energy snapped to cap — forced switch is not "
        "a free energy refund."
    )

    # --- 2. Outgoing slot carries residual HP / shields too -----------
    assert post.slots[0].hp == residual_hp
    assert post.slots[0].shields == residual_shields
    assert post.slots[0].alive is True
    assert post.slots[0].on_field is False

    # --- 3. Incoming slot enters with its own preserved energy --------
    assert post.on_field_index == 1
    assert post.slots[1].on_field is True
    assert post.slots[1].energy == 73, (
        "incoming Pokémon must use its OWN previously-stored energy — "
        "energy is never pooled across teammates."
    )
    # Triple-distinctness on the incoming side: 73 must differ from the
    # outgoing residual (57) and from zero, so we know we didn't
    # accidentally copy the residual onto the incoming slot.
    assert post.slots[1].energy != residual_energy
    assert post.slots[1].energy != 0

    # --- 4. Bench slot untouched --------------------------------------
    assert post.slots[2].on_field is False
    assert post.slots[2].energy == 12
    assert post.slots[2].hp == pre.slots[2].hp
    assert post.slots[2].shields == pre.slots[2].shields


# --- secondary invariants -------------------------------------------------


def test_forced_switch_does_not_mutate_input_state() -> None:
    """The input :class:`SetState` is not mutated; a fresh state is returned."""
    pre = _three_slot_side(lead_entry_energy=10, safe_stored_energy=50)
    snapshot_before = tuple(
        (s.species, s.hp, s.energy, s.shields, s.alive, s.on_field)
        for s in pre.slots
    )
    _ = apply_forced_switch(
        pre,
        incoming_index=1,
        outgoing_residual_hp=80,
        outgoing_residual_energy=44,
        outgoing_residual_shields=1,
    )
    snapshot_after = tuple(
        (s.species, s.hp, s.energy, s.shields, s.alive, s.on_field)
        for s in pre.slots
    )
    assert snapshot_before == snapshot_after, (
        "apply_forced_switch must not mutate its input state."
    )


def test_forced_switch_on_faint_marks_outgoing_dead() -> None:
    """A forced switch triggered by faint (residual_hp == 0) flips alive=False."""
    pre = _three_slot_side(lead_entry_energy=20, safe_stored_energy=33)
    post = apply_forced_switch(
        pre,
        incoming_index=1,
        outgoing_residual_hp=0,
        outgoing_residual_energy=15,  # accumulated some, then fainted
        outgoing_residual_shields=0,
    )
    assert post.slots[0].alive is False
    assert post.slots[0].hp == 0
    # Even on faint, the residual energy is preserved on the slot — this
    # matters because some downstream rationale logic may want to surface
    # "this Pokémon fainted carrying X energy" in its post-mortem.
    assert post.slots[0].energy == 15
    assert post.slots[0].on_field is False
    # Incoming slot took the field as expected.
    assert post.on_field_index == 1
    assert post.slots[1].energy == 33


def test_forced_switch_to_already_on_field_slot_raises() -> None:
    pre = _three_slot_side(on_field=0)
    with pytest.raises(SwitchError, match="already on field"):
        apply_forced_switch(
            pre,
            incoming_index=0,
            outgoing_residual_hp=40,
            outgoing_residual_energy=33,
            outgoing_residual_shields=1,
        )


def test_forced_switch_to_fainted_slot_raises() -> None:
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
        apply_forced_switch(
            pre,
            incoming_index=1,
            outgoing_residual_hp=80,
            outgoing_residual_energy=44,
            outgoing_residual_shields=1,
        )


@pytest.mark.parametrize("bad_index", [-1, 3, 99])
def test_forced_switch_to_out_of_range_index_raises(bad_index: int) -> None:
    pre = _three_slot_side(on_field=0)
    with pytest.raises(SwitchError, match="out of range"):
        apply_forced_switch(
            pre,
            incoming_index=bad_index,
            outgoing_residual_hp=80,
            outgoing_residual_energy=44,
            outgoing_residual_shields=1,
        )


@pytest.mark.parametrize("bad_energy", [-1, ENERGY_CAP + 1, 9999])
def test_forced_switch_rejects_out_of_range_residual_energy(bad_energy: int) -> None:
    """Residual energy outside [0, ENERGY_CAP] is a caller bug, not silently clamped."""
    pre = _three_slot_side(on_field=0)
    with pytest.raises(SwitchError, match="outgoing_residual_energy"):
        apply_forced_switch(
            pre,
            incoming_index=1,
            outgoing_residual_hp=80,
            outgoing_residual_energy=bad_energy,
            outgoing_residual_shields=1,
        )


def test_forced_switch_rejects_shield_gain() -> None:
    """Mid-matchup shields can only decrease; an apparent gain is a logic bug."""
    pre = _three_slot_side(on_field=0, lead_shields=1)
    with pytest.raises(SwitchError, match="exceeds pre-switch shields"):
        apply_forced_switch(
            pre,
            incoming_index=1,
            outgoing_residual_hp=80,
            outgoing_residual_energy=44,
            outgoing_residual_shields=2,  # somehow gained a shield → reject
        )


def test_forced_switch_accepts_zero_energy_residual() -> None:
    """Residual energy == 0 is legal (just-fired-a-charged scenario).

    Important boundary: this is the *only* case where the post-switch
    outgoing energy and a PvPoke-style zeroed energy agree. The test
    ensures the API does not reject 0 — preservation means "preserve
    whatever the residual is", including zero.
    """
    pre = _three_slot_side(lead_entry_energy=80, safe_stored_energy=60, on_field=0)
    post = apply_forced_switch(
        pre,
        incoming_index=1,
        outgoing_residual_hp=70,
        outgoing_residual_energy=0,  # just spent a charged move
        outgoing_residual_shields=2,
    )
    assert post.slots[0].energy == 0
    # Incoming side is unaffected — still 60.
    assert post.slots[1].energy == 60


def test_forced_switch_accepts_full_cap_residual() -> None:
    """Residual energy == ENERGY_CAP is legal (built up to overflow then switched)."""
    pre = _three_slot_side(lead_entry_energy=0, safe_stored_energy=10, on_field=0)
    post = apply_forced_switch(
        pre,
        incoming_index=1,
        outgoing_residual_hp=100,
        outgoing_residual_energy=ENERGY_CAP,
        outgoing_residual_shields=2,
    )
    assert post.slots[0].energy == ENERGY_CAP
    assert post.slots[1].energy == 10
