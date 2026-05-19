"""Switch event handling with per-slot energy carry-over.

GBL switch energy rules (the rules this module enforces):

* A Pokémon's accumulated energy is **per-Pokémon** and persists across
  switches. Switching out does **not** reset energy; switching back in
  resumes from the energy you had when you previously left the field.
* Energy is **not transferred** between teammates. The outgoing Pokémon
  keeps its own energy; the incoming Pokémon brings only the energy *it*
  had accumulated previously (0 if it has never been on the field).
* HP and remaining shields persist across switches for the same reason.
* Energy is capped at :data:`ENERGY_CAP` (100) — but a switch event by
  itself never adds energy, so the cap is just an invariant to assert.

PvPoke's isolated-matchup simulator does **not** model this. It plays each
matchup as a fresh 0-energy fight, so it systematically under-credits leads
that bait shields, accumulate energy, and then switch out to deliver that
energy through the safe-swap or closer. Capturing this carry-over is the
whole point of Sub-AC 4.1.

The simulator's switch policy is intentionally pluggable; this module only
implements the *mechanic*. The decision of *when* to switch is a separate
policy concern handled by the matchup driver.
"""

from __future__ import annotations

from gbl_hacker.simulator.state import (
    ENERGY_CAP,
    SLOTS_PER_SIDE,
    SetState,
    SlotState,
)


class SwitchError(ValueError):
    """Raised when a requested switch violates GBL switch mechanics."""


def apply_switch(state: SetState, incoming_index: int) -> SetState:
    """Execute a switch on ``state``, bringing ``incoming_index`` on field.

    Implements the GBL switch-energy mechanic:

    1. The current on-field slot is moved off field, **keeping** its
       accumulated HP, energy, and shields untouched.
    2. The incoming slot is placed on field, **keeping** its own previously
       accumulated HP, energy, and shields. In particular, if the incoming
       slot had energy ``E > 0`` from an earlier rotation, it re-enters at
       ``E`` — **not** at 0 as PvPoke's isolated matchup sim would assume.
    3. No energy is transferred between teammates. The outgoing slot
       retains its own energy; the incoming slot brings only its own.

    Parameters
    ----------
    state:
        Current set state for one side.
    incoming_index:
        Index in ``[0, SLOTS_PER_SIDE)`` of the slot to bring on field.

    Returns
    -------
    SetState
        New, immutable state with the switch applied.

    Raises
    ------
    SwitchError
        If the switch is illegal — out-of-range index, fainted target,
        already-on-field target, or no live on-field source.
    """

    if not (0 <= incoming_index < SLOTS_PER_SIDE):
        raise SwitchError(
            f"incoming_index out of range: {incoming_index}"
        )

    current_index = state.on_field_index
    if current_index is None:
        raise SwitchError("no slot is currently on field — set is over")

    if current_index == incoming_index:
        raise SwitchError(
            f"slot {incoming_index} is already on field; switch is a no-op"
        )

    incoming = state.slots[incoming_index]
    if not incoming.alive:
        raise SwitchError(
            f"cannot switch to fainted slot {incoming_index} "
            f"(species={incoming.species!r})"
        )

    outgoing = state.slots[current_index]

    # --- carry-over invariants (defense in depth) -----------------------
    # The switch mechanic does not modify HP, energy, or shields on either
    # side. We assert the implementation honors that so a regression
    # cannot silently zero the incoming Pokémon's energy.
    if not (0 <= outgoing.energy <= ENERGY_CAP):
        raise SwitchError(
            f"outgoing energy out of range pre-switch: {outgoing.energy}"
        )
    if not (0 <= incoming.energy <= ENERGY_CAP):
        raise SwitchError(
            f"incoming energy out of range pre-switch: {incoming.energy}"
        )

    new_outgoing = outgoing.with_changes(on_field=False)
    new_incoming = incoming.with_changes(on_field=True)

    new_slots = list(state.slots)
    new_slots[current_index] = new_outgoing
    new_slots[incoming_index] = new_incoming

    new_state = SetState(slots=tuple(new_slots))  # type: ignore[arg-type]

    # --- post-conditions -------------------------------------------------
    # The per-slot energy of every slot must be byte-for-byte identical to
    # before the switch. This is the precise property PvPoke's isolated-
    # matchup sim violates by resetting incoming energy to zero.
    for before, after in zip(state.slots, new_state.slots, strict=True):
        if before.energy != after.energy:
            raise SwitchError(
                "switch mechanic mutated per-slot energy: "
                f"{before.species} {before.energy} -> {after.energy}"
            )
        if before.hp != after.hp:
            raise SwitchError(
                "switch mechanic mutated per-slot hp: "
                f"{before.species} {before.hp} -> {after.hp}"
            )
        if before.shields != after.shields:
            raise SwitchError(
                "switch mechanic mutated per-slot shields: "
                f"{before.species} {before.shields} -> {after.shields}"
            )

    return new_state


def entry_energy(state: SetState, incoming_index: int) -> int:
    """Return the energy the incoming Pokémon would bring on field.

    Pure-function inspector — does not mutate state. Useful for the
    matchup driver and for unit tests that want to assert *what would
    happen* without executing the switch.
    """

    if not (0 <= incoming_index < SLOTS_PER_SIDE):
        raise SwitchError(
            f"incoming_index out of range: {incoming_index}"
        )
    return state.slots[incoming_index].energy


def apply_forced_switch(
    state: SetState,
    incoming_index: int,
    *,
    outgoing_residual_hp: int,
    outgoing_residual_energy: int,
    outgoing_residual_shields: int,
) -> SetState:
    """Execute a forced mid-matchup switch, preserving residual state.

    Sub-AC 4.3 contract — the *mid-matchup* counterpart to
    :func:`apply_switch`:

    * :func:`apply_switch` swaps the ``on_field`` flag with no other
      state mutation. It assumes :class:`SlotState` already reflects
      whatever the slot has accumulated.
    * :func:`apply_forced_switch` is called *mid-matchup* when the
      on-field Pokémon has been actively fighting. The matchup-driver
      has been tracking its HP / energy / shields independently of the
      :class:`SlotState`; on a forced switch event those *residual*
      values must be folded back into the outgoing :class:`SlotState`
      **before** the on_field flag is moved. Otherwise the outgoing
      slot would silently snap back to its pre-matchup energy, which
      is exactly the PvPoke isolated-matchup bug this engine exists
      to fix.

    What "forced switch" means here:

    * In actual GBL, "forced switch" can mean a few things — a
      surprise-switch policy the matchup driver triggers, a faint that
      forces the next slot onto the field, or a player-initiated
      tactical switch fired before a charged move lands. The mechanic
      they share is the same: the on-field Pokémon's mid-matchup
      energy must be preserved (never zeroed, never refunded back to
      its pre-matchup or its starting value) and the incoming
      Pokémon's previously-stored energy must be honored.

    Energy-preservation spec (the core property under test):

    * ``outgoing_residual_energy`` is written verbatim onto the
      outgoing slot. It is **not** zeroed (PvPoke isolated-matchup
      bug) and it is **not** rewound to the slot's pre-matchup energy
      ("refunded"). Whatever the slot had accumulated mid-fight is
      what it keeps.
    * The incoming slot keeps **its own** previously-stored energy.
      No energy is transferred between teammates.

    Parameters
    ----------
    state:
        Current set state for one side.
    incoming_index:
        Index in ``[0, SLOTS_PER_SIDE)`` of the slot to bring on field.
    outgoing_residual_hp:
        The on-field Pokémon's HP at the moment of the forced switch.
        ``0`` is permitted — a forced switch can be a faint trigger.
    outgoing_residual_energy:
        The on-field Pokémon's energy at the moment of the forced
        switch. Stored byte-for-byte on the outgoing slot. Must be in
        ``[0, ENERGY_CAP]``.
    outgoing_residual_shields:
        The on-field Pokémon's remaining shields at the moment of the
        forced switch. Must be in ``[0, MAX_SHIELDS]`` and must not
        exceed the slot's pre-switch shields (shields are only
        consumed, never gained).

    Returns
    -------
    SetState
        New, immutable state with the forced switch applied. The
        outgoing slot carries the residual HP / energy / shields; the
        incoming slot is on field with its own preserved energy.

    Raises
    ------
    SwitchError
        If the switch or residual values are illegal — out-of-range
        index, fainted incoming target, already-on-field target,
        no live on-field source, energy/shield/HP out of range, or
        shields somehow *increased* mid-matchup.
    """

    if not (0 <= incoming_index < SLOTS_PER_SIDE):
        raise SwitchError(
            f"incoming_index out of range: {incoming_index}"
        )

    current_index = state.on_field_index
    if current_index is None:
        raise SwitchError("no slot is currently on field — set is over")

    if current_index == incoming_index:
        raise SwitchError(
            f"slot {incoming_index} is already on field; forced switch is a no-op"
        )

    incoming = state.slots[incoming_index]
    if not incoming.alive:
        raise SwitchError(
            f"cannot force-switch to fainted slot {incoming_index} "
            f"(species={incoming.species!r})"
        )

    outgoing = state.slots[current_index]

    # --- residual-value validation -------------------------------------
    if not (0 <= outgoing_residual_energy <= ENERGY_CAP):
        raise SwitchError(
            "outgoing_residual_energy out of range: "
            f"{outgoing_residual_energy} (must be in [0, {ENERGY_CAP}])"
        )
    if outgoing_residual_hp < 0:
        raise SwitchError(
            f"outgoing_residual_hp must be >= 0: {outgoing_residual_hp}"
        )
    if outgoing_residual_shields < 0:
        raise SwitchError(
            f"outgoing_residual_shields must be >= 0: {outgoing_residual_shields}"
        )
    if outgoing_residual_shields > outgoing.shields:
        # Shields are only consumed mid-matchup, never gained. If the
        # caller hands us a larger value we've found a logic bug
        # upstream — surface it loudly rather than silently coerce.
        raise SwitchError(
            "outgoing_residual_shields exceeds pre-switch shields: "
            f"{outgoing_residual_shields} > {outgoing.shields}"
        )
    # The outgoing slot's *incoming* energy cannot exceed cap (defense
    # in depth — protects the caller from accidentally storing > cap).
    if not (0 <= outgoing.energy <= ENERGY_CAP):
        raise SwitchError(
            f"outgoing pre-switch energy out of range: {outgoing.energy}"
        )
    if not (0 <= incoming.energy <= ENERGY_CAP):
        raise SwitchError(
            f"incoming pre-switch energy out of range: {incoming.energy}"
        )

    # --- fold residuals into outgoing slot, swap on_field --------------
    # CRITICAL: outgoing_residual_energy is preserved *verbatim*. It is
    # not zeroed, not capped to entry energy, not refunded. This is the
    # exact contract Sub-AC 4.3 pins down.
    outgoing_alive = outgoing_residual_hp > 0
    new_outgoing = outgoing.with_changes(
        hp=outgoing_residual_hp,
        energy=outgoing_residual_energy,
        shields=outgoing_residual_shields,
        alive=outgoing_alive,
        on_field=False,
    )
    new_incoming = incoming.with_changes(on_field=True)

    new_slots = list(state.slots)
    new_slots[current_index] = new_outgoing
    new_slots[incoming_index] = new_incoming

    new_state = SetState(slots=tuple(new_slots))  # type: ignore[arg-type]

    # --- post-condition: energy preservation invariant -----------------
    # Make the preservation contract unmissable from the implementation
    # side too. If somebody refactors and silently zeroes the residual
    # energy the post-condition fires before bad state escapes.
    if new_state.slots[current_index].energy != outgoing_residual_energy:
        raise SwitchError(
            "forced switch mutated outgoing residual energy: "
            f"expected {outgoing_residual_energy}, "
            f"got {new_state.slots[current_index].energy}"
        )
    if new_state.slots[incoming_index].energy != incoming.energy:
        raise SwitchError(
            "forced switch mutated incoming slot's stored energy: "
            f"expected {incoming.energy}, "
            f"got {new_state.slots[incoming_index].energy}"
        )

    return new_state


__all__ = [
    "SwitchError",
    "apply_forced_switch",
    "apply_switch",
    "entry_energy",
]
