"""State-machine primitives for the set-state-aware GBL simulator.

GBL is a 3v3 set. Each side holds three slots (lead, safe_swap, closer) and
each slot has its *own* HP, energy, and shield count that persist across
switches. PvPoke's isolated-matchup simulator collapses a 3v3 set into a
sequence of independent 1v1 fights and therefore assumes incoming energy
is always 0 and shields are always reset. This module models the persistent
per-slot state that the rest of the engine pivots on.

Conventions:

* Energy is an integer in ``[0, ENERGY_CAP]`` (GBL caps energy at 100).
* HP is an integer in ``[0, MAX_HP]``; ``hp == 0`` implies the slot is fainted.
* Shields are an integer in ``[0, MAX_SHIELDS]`` (0, 1, or 2).
* Exactly one slot per side is ``on_field`` at any time when at least one
  slot is alive. A fully-fainted side has no slot on field (set is over).

All state objects are frozen dataclasses so transitions are explicit and
unit-tests can compare snapshots by value.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

ENERGY_CAP: int = 100
"""GBL hard cap on per-Pokémon accumulated energy."""

MAX_HP: int = 1000
"""Soft upper bound used for validation; real CP-capped HP values sit well below this."""

MAX_SHIELDS: int = 2
"""Each Pokémon brings 0–2 shields into the set."""

SLOTS_PER_SIDE: int = 3
"""3v3 GBL format — exactly three slots per side."""


@dataclass(frozen=True, slots=True)
class SlotState:
    """Per-Pokémon persistent state inside a 3v3 set.

    Attributes
    ----------
    species:
        Identifier string for the Pokémon build occupying this slot.
        Kept as a free-form string so the state module stays decoupled from
        the (not-yet-built) build/move database.
    hp:
        Current hit points. ``hp == 0`` implies the slot is fainted; a
        fainted slot must have ``alive=False`` and ``on_field=False``.
    energy:
        Accumulated energy in ``[0, ENERGY_CAP]``. Persists across switches
        — this is the field whose carry-over the v0.1 simulator must honor.
    shields:
        Remaining shields, in ``[0, MAX_SHIELDS]``.
    alive:
        Convenience flag: ``True`` iff ``hp > 0``. Tracked explicitly so a
        slot can be ``alive=True, hp=N`` without recomputing.
    on_field:
        ``True`` iff this slot is the active Pokémon for its side.
    """

    species: str
    hp: int
    energy: int = 0
    shields: int = MAX_SHIELDS
    alive: bool = True
    on_field: bool = False

    def __post_init__(self) -> None:  # pragma: no cover - pure validation
        if not (0 <= self.hp <= MAX_HP):
            raise ValueError(f"hp out of range: {self.hp}")
        if not (0 <= self.energy <= ENERGY_CAP):
            raise ValueError(f"energy out of range: {self.energy}")
        if not (0 <= self.shields <= MAX_SHIELDS):
            raise ValueError(f"shields out of range: {self.shields}")
        if self.hp == 0 and self.alive:
            raise ValueError("slot with hp == 0 cannot be alive")
        if not self.alive and self.on_field:
            raise ValueError("fainted slot cannot be on_field")

    def with_changes(self, **kwargs: object) -> "SlotState":
        """Return a new ``SlotState`` with the named fields overridden."""

        return replace(self, **kwargs)


@dataclass(frozen=True, slots=True)
class SetState:
    """One side of a 3v3 GBL set.

    Invariant: at most one slot has ``on_field=True``; if any slot is alive
    then exactly one alive slot must be on field. (A side with all slots
    fainted has no on-field slot and the set is over for that side.)
    """

    slots: tuple[SlotState, SlotState, SlotState]

    def __post_init__(self) -> None:  # pragma: no cover - pure validation
        if len(self.slots) != SLOTS_PER_SIDE:
            raise ValueError(f"expected {SLOTS_PER_SIDE} slots, got {len(self.slots)}")
        on_field = [i for i, s in enumerate(self.slots) if s.on_field]
        if len(on_field) > 1:
            raise ValueError(f"multiple slots on field: {on_field}")
        any_alive = any(s.alive for s in self.slots)
        if any_alive and not on_field:
            raise ValueError("a live side must have exactly one on-field slot")
        if on_field and not self.slots[on_field[0]].alive:
            raise ValueError("on-field slot must be alive")

    @classmethod
    def starting(
        cls,
        slots: Iterable[SlotState],
        *,
        lead_index: int = 0,
    ) -> "SetState":
        """Construct a starting state with ``lead_index`` placed on field.

        Convenience for tests and the lineup-builder: all slots default to
        full energy=0, shields=2, alive=True; only the lead is ``on_field``.
        """

        materialized = tuple(slots)
        if len(materialized) != SLOTS_PER_SIDE:
            raise ValueError(
                f"expected {SLOTS_PER_SIDE} starting slots, got {len(materialized)}"
            )
        if not (0 <= lead_index < SLOTS_PER_SIDE):
            raise ValueError(f"lead_index out of range: {lead_index}")
        adjusted = tuple(
            s.with_changes(on_field=(i == lead_index))
            for i, s in enumerate(materialized)
        )
        return cls(slots=adjusted)  # type: ignore[arg-type]

    @property
    def on_field_index(self) -> int | None:
        for i, s in enumerate(self.slots):
            if s.on_field:
                return i
        return None

    @property
    def on_field_slot(self) -> SlotState | None:
        idx = self.on_field_index
        return None if idx is None else self.slots[idx]

    def with_slot(self, index: int, new_slot: SlotState) -> "SetState":
        """Return a new ``SetState`` with ``slots[index]`` replaced."""

        if not (0 <= index < SLOTS_PER_SIDE):
            raise ValueError(f"slot index out of range: {index}")
        new_slots = list(self.slots)
        new_slots[index] = new_slot
        return SetState(slots=tuple(new_slots))  # type: ignore[arg-type]


__all__ = [
    "ENERGY_CAP",
    "MAX_HP",
    "MAX_SHIELDS",
    "SLOTS_PER_SIDE",
    "SetState",
    "SlotState",
]
