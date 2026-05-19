"""3v3 GBL set driver: chains matchups with energy/HP/shield carry-over.

The matchup resolver in :mod:`gbl_hacker.simulator.matchup` simulates a
single 1v1 fight. Real GBL is a 3v3 *set* — each side cycles through
three Pokémon, and per-slot HP / energy / shields persist across
switches. This module is the driver that walks two
:class:`~gbl_hacker.score.expected_win_rate.CandidateTeam` lineups
through a complete set and returns:

- The overall set winner ("A" / "B" / None on a turn-budget cutoff).
- The sequence of per-matchup outcomes (which slots fought, terminal
  states, turn counts).
- A switch-timer audit trail — each side's mid-set switch eligibility
  measured in 500ms ticks so a future v0.3 active-switch policy can
  read the same field without re-deriving it.

v0.2 switch policy: **faint-driven only.** When the on-field Pokémon
KOs, the next alive slot (in lineup order) enters the field at its
preserved HP / energy / shields with ``atk_stage`` / ``def_stage``
reset to 0. No active mid-matchup switching — that's a v0.3 surface.

Switch-timer semantics:
- Both sides start at ``timer_ticks=0`` (a free swap is technically
  available immediately, though we don't use it in the v0.2 policy).
- Each completed matchup increments the timer by its duration in ticks.
- A faint event resets the surviving side's timer to ``0`` to mirror
  the in-game "free swap after KO" rule on the opposing side.

These hooks are deliberately exposed so a v0.3 policy can read the
timer at decision points without re-instrumenting the driver.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

from gbl_hacker.score.expected_win_rate import CandidateTeam
from gbl_hacker.simulator.matchup import (
    MAX_TURNS,
    CombatantBuild,
    CombatantState,
    MatchupResult,
    resolve_matchup,
)
from gbl_hacker.simulator.state import MAX_SHIELDS

# Active-switch decision margin: a teammate must out-score the current
# on-field by at least this much (on the type-matchup heuristic) to
# justify burning the 45-second cooldown. Tuned so type-neutral swaps
# don't trigger but a clear typing answer does.
_SWITCH_SCORE_MARGIN: int = 3

# Low-HP active swap threshold: if the on-field is below this HP
# fraction AND a healthy teammate exists, prefer to switch out.
_LOW_HP_SWAP_FRACTION: float = 0.25

Side = Literal["A", "B"]

# GBL gives each side a 45-second active-switch lockout after using one
# of their two active swaps (the timer counts in 500ms ticks).
SWITCH_TIMER_TICKS: int = 90
"""Number of ticks (500ms each) for the GBL switch cooldown — 45 s."""


@dataclass(frozen=True, slots=True)
class SetSlot:
    """One slot inside a 3v3 set.

    Each slot keeps its own HP, energy, shields, and stat-buff stages.
    These persist across switches — switching out and back in does NOT
    reset any of them.

    ``fainted`` is True iff ``hp == 0``; carried alongside ``hp`` so
    callers can pattern-match on the flag without numerical comparisons.
    """

    build: CombatantBuild
    hp: int
    energy: int = 0
    shields: int = MAX_SHIELDS
    atk_stage: int = 0
    def_stage: int = 0
    fainted: bool = False

    def with_(self, **kwargs: object) -> "SetSlot":
        """Return a copy with the named fields replaced."""
        from dataclasses import replace

        return replace(self, **kwargs)  # type: ignore[arg-type]

    def to_combatant_state(self) -> CombatantState:
        """Project this slot into a single-matchup ``CombatantState``.

        Buff stages, HP, energy, and shields carry through. The
        ``fast_cooldown`` resets to 0 — a slot newly entering the field
        starts with a free fast move tick (the in-game behavior).
        """
        return CombatantState(
            build=self.build,
            hp=self.hp,
            energy=self.energy,
            shields=self.shields,
            fast_cooldown=0,
            atk_stage=self.atk_stage,
            def_stage=self.def_stage,
            # GBL "reset on switch": entering the field always lands in
            # the build's primary form. For dynamic-form species like
            # Aegislash this enforces Shield re-entry; for static-form
            # species (the default) this is a no-op (active_form_idx=0
            # is the only possible value).
            active_form_idx=0,
        )

    @classmethod
    def fresh(
        cls, build: CombatantBuild, *, shields: int = MAX_SHIELDS
    ) -> "SetSlot":
        """Full HP, zero energy, neutral stages — start of a fresh set."""
        return cls(
            build=build,
            hp=build.max_hp,
            energy=0,
            shields=shields,
            atk_stage=0,
            def_stage=0,
            fainted=False,
        )


@dataclass(frozen=True, slots=True)
class SetSide:
    """One side of a 3v3 set — three slots + the active-field pointer.

    Invariant: the on-field slot is alive (or the side has no live slot
    at all, in which case the set is over for this side).
    """

    slots: tuple[SetSlot, SetSlot, SetSlot]
    on_field: int = 0
    switch_timer_ticks: int = 0
    """Ticks remaining before another active switch is allowed. v0.2
    policy never schedules an active switch, so this stays at 0 except
    when the driver bumps it for instrumentation purposes."""

    @classmethod
    def starting(
        cls, team: CandidateTeam, *, shields: int = MAX_SHIELDS
    ) -> "SetSide":
        """Build the starting state from a candidate team."""
        slots = tuple(SetSlot.fresh(b, shields=shields) for b in team.slots)
        return cls(slots=slots, on_field=0, switch_timer_ticks=0)  # type: ignore[arg-type]

    @property
    def any_alive(self) -> bool:
        return any(not s.fainted for s in self.slots)

    def next_alive_index(self, *, after: int) -> int | None:
        """First alive slot strictly after ``after`` in lineup order."""
        for i in range(after + 1, len(self.slots)):
            if not self.slots[i].fainted:
                return i
        return None


@dataclass(frozen=True, slots=True)
class MatchupOutcome:
    """One on-field 1v1 fight inside a set."""

    a_slot_index: int
    b_slot_index: int
    a_terminal: SetSlot
    b_terminal: SetSlot
    winner: Side | None
    turns_elapsed: int
    charged_events: tuple


@dataclass(frozen=True, slots=True)
class SetSimulation:
    """Full 3v3 set outcome.

    Attributes
    ----------
    winner:
        ``"A"`` / ``"B"`` / ``None`` (set ran out of turn budget with
        both sides still alive).
    matchups:
        Ordered tuple of every 1v1 the driver ran. Slot indices are
        preserved so the rationale layer can render the switch order.
    total_turns:
        Sum of turns across every matchup.
    final_a, final_b:
        Terminal :class:`SetSide` snapshots — useful for rationale
        cards that want to surface remaining shields / HP.
    """

    winner: Side | None
    matchups: tuple[MatchupOutcome, ...]
    total_turns: int
    final_a: SetSide
    final_b: SetSide


def simulate_set(
    team_a: CandidateTeam,
    team_b: CandidateTeam,
    *,
    starting_shields: int = MAX_SHIELDS,
    max_total_turns: int = MAX_TURNS * 3,
    rng: random.Random | None = None,
    active_switch: bool = False,
) -> SetSimulation:
    """Walk two lineups through a complete 3v3 GBL set.

    Parameters
    ----------
    team_a, team_b:
        The two candidate lineups, lead → safe_swap → closer.
    starting_shields:
        Starting shield count for every slot (both sides). Default 2.
    max_total_turns:
        Hard cap on summed turns across all matchups. Defaults to three
        times the per-matchup cap.

    Returns
    -------
    SetSimulation
        Full audit trail — winner, every matchup, terminal side states.

    Policy notes
    ------------

    - **Faint-driven switching only.** No active mid-matchup switches in
      v0.2; the driver swaps in the next alive slot when a faint happens.
    - **Carry-over.** Each matchup hands its terminal HP / energy /
      shields / buff stages back onto the originating slot before
      considering a switch — this is the property unit-tested in the
      single-matchup ``apply_forced_switch`` resolver, lifted to the set
      level.
    - **Switch-timer instrumentation.** The driver does not schedule
      active switches, but it does maintain a ticks counter so v0.3
      policies can read it. Free swap on KO resets the surviving side's
      timer; otherwise the timer accumulates per-matchup duration.
    """

    side_a = SetSide.starting(team_a, shields=starting_shields)
    side_b = SetSide.starting(team_b, shields=starting_shields)

    matchups: list[MatchupOutcome] = []
    total_turns = 0

    while (
        side_a.any_alive
        and side_b.any_alive
        and total_turns < max_total_turns
    ):
        a_slot = side_a.slots[side_a.on_field]
        b_slot = side_b.slots[side_b.on_field]

        # Defensive: if the on-field slot is somehow fainted, advance.
        if a_slot.fainted or b_slot.fainted:
            side_a = _advance_to_next_alive(side_a)
            side_b = _advance_to_next_alive(side_b)
            if not side_a.any_alive or not side_b.any_alive:
                break
            continue

        # Active switch decision (timer-aware, type-matchup heuristic).
        # Both sides decide before the matchup resolves; a triggered
        # active swap arms the 45-second cooldown timer. Off by default
        # because the per-matchup heuristic is expensive over large
        # candidate pools.
        if active_switch:
            swap_a = _decide_active_switch(
                side_a, opponent_on_field=b_slot.build
            )
            if swap_a is not None:
                side_a = _apply_active_switch(side_a, swap_a)
                a_slot = side_a.slots[side_a.on_field]
            swap_b = _decide_active_switch(
                side_b, opponent_on_field=a_slot.build
            )
            if swap_b is not None:
                side_b = _apply_active_switch(side_b, swap_b)
                b_slot = side_b.slots[side_b.on_field]

        # Per-matchup turn budget: leave room for downstream matchups.
        remaining = max_total_turns - total_turns
        per_matchup_cap = min(MAX_TURNS, remaining)
        result: MatchupResult = resolve_matchup(
            a_slot.to_combatant_state(),
            b_slot.to_combatant_state(),
            max_turns=per_matchup_cap,
            rng=rng,
        )

        new_a_slot = _fold_result(
            a_slot,
            result,
            hp=result.a_terminal_hp,
            energy=result.a_terminal_energy,
            shields=result.a_terminal_shields,
            atk_stage=0,
            def_stage=0,
        )
        new_b_slot = _fold_result(
            b_slot,
            result,
            hp=result.b_terminal_hp,
            energy=result.b_terminal_energy,
            shields=result.b_terminal_shields,
            atk_stage=0,
            def_stage=0,
        )

        # The fold zeroes the stage on save — the next on-field appearance
        # of this slot starts neutral. (Buff stages do NOT persist
        # across switches in GBL.)

        side_a = _replace_slot(side_a, new_a_slot, index=side_a.on_field)
        side_b = _replace_slot(side_b, new_b_slot, index=side_b.on_field)

        matchups.append(
            MatchupOutcome(
                a_slot_index=side_a.on_field,
                b_slot_index=side_b.on_field,
                a_terminal=new_a_slot,
                b_terminal=new_b_slot,
                winner=result.winner,
                turns_elapsed=result.turns,
                charged_events=result.charged_events,
            )
        )
        total_turns += result.turns

        # Decrement both sides' switch cooldown by the turns spent in
        # this matchup — captures the in-game rule that the 45-second
        # cooldown ticks down during play.
        side_a = _decrement_timer(side_a, ticks=result.turns)
        side_b = _decrement_timer(side_b, ticks=result.turns)

        # Switch logic: a faint forces the losing side to bring in the
        # next slot. The surviving side's switch timer resets to 0
        # (free swap after the opponent's KO). On a double-KO both
        # sides advance.
        a_fainted = new_a_slot.fainted
        b_fainted = new_b_slot.fainted

        if a_fainted:
            side_a = _advance_to_next_alive(side_a)
            side_b = _advance_timer_on_faint(side_b)
        if b_fainted:
            side_b = _advance_to_next_alive(side_b)
            side_a = _advance_timer_on_faint(side_a)
        if not a_fainted and not b_fainted:
            # No KO yet — turn budget exhausted. Safety break so we
            # don't loop forever on a deterministic stall.
            break

    if side_a.any_alive and not side_b.any_alive:
        winner: Side | None = "A"
    elif side_b.any_alive and not side_a.any_alive:
        winner = "B"
    else:
        winner = None

    return SetSimulation(
        winner=winner,
        matchups=tuple(matchups),
        total_turns=total_turns,
        final_a=side_a,
        final_b=side_b,
    )


def _matchup_signature(b: CombatantBuild) -> tuple:
    """Hashable signature for a build's type-matchup-relevant fields.

    Only the offensive move types and defensive typing matter for the
    score function — collapse a CombatantBuild to that tuple so the
    LRU cache key is stable across distinct CombatantBuild instances
    that happen to share the same matchup profile.
    """
    return (
        b.fast.move_type.lower(),
        b.charged.move_type.lower(),
        b.charged2.move_type.lower() if b.charged2 else "",
        tuple(t.lower() for t in b.types),
    )


@lru_cache(maxsize=16384)
def _type_matchup_score_signature(
    attacker_sig: tuple, defender_sig: tuple
) -> int:
    """Cached scorer keyed on build signatures rather than full builds.

    Score components (higher = attacker advantage):

    - +3 per offensive move type that super-effective hits the defender.
    - -3 per defender's offensive move type that super-effective hits
      the attacker.
    - +1 per defender's offensive move type the attacker's defensive
      typing resists.
    """
    from gbl_hacker.gamemaster import load_default_gamemaster

    gm = load_default_gamemaster()
    chart = gm.type_chart

    a_fast, a_ch1, a_ch2, a_types = attacker_sig
    d_fast, d_ch1, d_ch2, d_types = defender_sig
    a_off = {a_fast, a_ch1, a_ch2} - {""}
    d_off = {d_fast, d_ch1, d_ch2} - {""}

    def _weak(types: tuple[str, ...]) -> set[str]:
        weak: set[str] = set()
        for t in types:
            c = chart.get(t)
            if c:
                weak.update(c.get("weaknesses", ()))
        return weak

    score = 0
    score += 3 * len(a_off & _weak(d_types))
    score -= 3 * len(d_off & _weak(a_types))

    for t in a_types:
        c = chart.get(t)
        if not c:
            continue
        resist = set(c.get("resistances", ())) | set(c.get("immunities", ()))
        score += len(resist & d_off)

    return score


def _type_matchup_score(
    attacker: CombatantBuild, defender: CombatantBuild
) -> int:
    """Cached type-matchup heuristic. See module docstring."""
    return _type_matchup_score_signature(
        _matchup_signature(attacker), _matchup_signature(defender)
    )


def _decide_active_switch(
    side: SetSide, *, opponent_on_field: CombatantBuild
) -> int | None:
    """Return the slot index to switch into, or ``None`` to stay.

    Decision rules:

    - Timer must be 0. The 45-second cooldown is enforced strictly —
      the resolver never burns it twice in a row.
    - Either:
        (a) the on-field is below ``_LOW_HP_SWAP_FRACTION`` HP AND
            an alive teammate exists at higher HP; or
        (b) a teammate's type-matchup score exceeds the current
            on-field's by at least ``_SWITCH_SCORE_MARGIN``.
    - The winning candidate (highest score, ties broken by lineup
      order) is returned.
    """

    if side.switch_timer_ticks > 0:
        return None
    current_idx = side.on_field
    current = side.slots[current_idx]
    if current.fainted:
        return None

    current_score = _type_matchup_score(current.build, opponent_on_field)
    current_hp_fraction = (
        current.hp / current.build.max_hp if current.build.max_hp else 0.0
    )

    candidates: list[tuple[int, int]] = []
    for i, slot in enumerate(side.slots):
        if i == current_idx or slot.fainted:
            continue
        s = _type_matchup_score(slot.build, opponent_on_field)
        candidates.append((i, s))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[1], reverse=True)

    best_idx, best_score = candidates[0]

    # Rule (b): type-matchup outperformance.
    if best_score >= current_score + _SWITCH_SCORE_MARGIN:
        return best_idx

    # Rule (a): low-HP triage. Swap to the best-typing teammate that's
    # also reasonably healthy.
    if current_hp_fraction < _LOW_HP_SWAP_FRACTION:
        for i, s in candidates:
            slot = side.slots[i]
            slot_hp_fraction = (
                slot.hp / slot.build.max_hp if slot.build.max_hp else 0.0
            )
            if slot_hp_fraction > current_hp_fraction + 0.2:
                return i

    return None


def _fold_result(
    slot: SetSlot,
    result: MatchupResult,
    *,
    hp: int,
    energy: int,
    shields: int,
    atk_stage: int,
    def_stage: int,
) -> SetSlot:
    """Fold a matchup's terminal values back onto the originating slot."""
    return SetSlot(
        build=slot.build,
        hp=hp,
        energy=energy,
        shields=shields,
        atk_stage=atk_stage,
        def_stage=def_stage,
        fainted=(hp == 0),
    )


def _replace_slot(side: SetSide, new_slot: SetSlot, *, index: int) -> SetSide:
    new_slots = list(side.slots)
    new_slots[index] = new_slot
    return SetSide(
        slots=tuple(new_slots),  # type: ignore[arg-type]
        on_field=side.on_field,
        switch_timer_ticks=side.switch_timer_ticks,
    )


def _apply_active_switch(side: SetSide, target_idx: int) -> SetSide:
    """Apply an operator-initiated active switch (not faint-driven).

    Sets ``switch_timer_ticks`` to the full :data:`SWITCH_TIMER_TICKS`
    cooldown so the side cannot immediately switch again until that
    many ticks have elapsed. Preserves slot HP / energy / shields and
    resets buff stages on the outgoing slot (per GBL convention —
    leaving the field clears stat buffs).
    """
    if target_idx == side.on_field:
        return side
    if not (0 <= target_idx < len(side.slots)):
        return side
    target = side.slots[target_idx]
    if target.fainted:
        return side

    outgoing_idx = side.on_field
    outgoing = side.slots[outgoing_idx]
    new_outgoing = outgoing.with_(atk_stage=0, def_stage=0)
    new_slots = list(side.slots)
    new_slots[outgoing_idx] = new_outgoing
    return SetSide(
        slots=tuple(new_slots),  # type: ignore[arg-type]
        on_field=target_idx,
        switch_timer_ticks=SWITCH_TIMER_TICKS,
    )


def _advance_to_next_alive(side: SetSide) -> SetSide:
    """Advance ``on_field`` to the next alive slot, or stay if none."""
    nxt = side.next_alive_index(after=side.on_field)
    if nxt is None:
        return side  # no live slot — set is over for this side
    return SetSide(
        slots=side.slots,
        on_field=nxt,
        switch_timer_ticks=0,
    )


def _decrement_timer(side: SetSide, *, ticks: int) -> SetSide:
    """Decrement the switch cooldown by ``ticks`` (saturating at 0)."""
    new_ticks = max(0, side.switch_timer_ticks - ticks)
    return SetSide(
        slots=side.slots,
        on_field=side.on_field,
        switch_timer_ticks=new_ticks,
    )


def _advance_timer_on_faint(side: SetSide) -> SetSide:
    """A KO on the opponent's side grants a free swap → timer = 0."""
    return SetSide(
        slots=side.slots,
        on_field=side.on_field,
        switch_timer_ticks=0,
    )


__all__ = [
    "MatchupOutcome",
    "SWITCH_TIMER_TICKS",
    "SetSide",
    "SetSimulation",
    "SetSlot",
    "Side",
    "simulate_set",
]
