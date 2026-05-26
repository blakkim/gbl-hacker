"""Asymmetric-shield-aware single-matchup resolver (Sub-AC 4.2).

A "matchup" in GBL Hacker is a fight between exactly **two on-field**
Pokémon — one per side — played to KO or to a turn-budget cutoff. This
module implements the resolver for that fight, with two properties that
PvPoke's isolated-matchup simulator is famously weak on:

1. **Per-side shield counts are independent.** The function accepts
   :attr:`CombatantState.shields` as a *per-side* field. The 3 × 3 grid
   ``(a_shields, b_shields)`` over ``{0, 1, 2}`` is fully expressible — no
   place in the API forces the two sides to share a shield count.
2. **Entry energy and entry HP are respected.** ``CombatantState`` carries
   ``hp`` and ``energy`` as inputs, so a matchup played mid-set (with a
   Pokémon that arrived mid-rotation with ``energy > 0``) is just a normal
   call to :func:`resolve_matchup` with the right starting state.

The combat model itself is intentionally a faithful-enough approximation
of GBL's turn loop for v0.1 — not a port of PvPoke's full decision tree.
That richer model is a v0.2 concern; v0.1 needs a resolver that is *good
enough to make the shield asymmetry observable in test*. Specifically:

* Each turn, both sides apply their fast move **simultaneously**: dealing
  damage and gaining energy (capped at :data:`ENERGY_CAP`). If a side
  is KO'd by fast-move damage, the matchup ends immediately — no charged
  moves resolve after the KO.
* If both sides are still alive and either side has ``energy ≥`` charged
  cost, the charged-move phase runs. Both sides that are ready fire
  exactly one charged move, in priority order (A first, then B). The
  defender of each charged move chooses to shield iff they have at least
  one shield remaining (greedy shielding — the v0.1 baseline policy).
  When shielded, the canonical GBL "1 damage gets through" rule applies.
* The loop terminates on the first turn either combatant's HP drops to
  ``0`` or below, or after :data:`MAX_TURNS` ticks (safety cutoff).
  Double-KO on the same turn yields ``winner = None``.

The resolver is deterministic: given identical input states it produces
identical output. This is what unit tests rely on to pin the asymmetric-
shield behavior down to byte-equal terminal HP.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

from gbl_hacker.simulator.state import ENERGY_CAP, MAX_SHIELDS

MAX_TURNS: int = 500
"""Safety cap on simulated turns. Real GBL matches resolve far below this."""

SHIELD_BLEED_DAMAGE: int = 1
"""GBL rule: a shielded charged move still passes 1 point of damage through."""

# GBL damage formula constants — mirror PvPoke DamageCalculator.
_DAMAGE_BONUS_MULTIPLIER: float = 1.2999999523162841796875
_STAB_MULTIPLIER: float = 1.2
_SUPER_EFFECTIVE: float = 1.6
_RESISTED: float = 0.625
_DOUBLE_RESISTED: float = 0.390625

Side = Literal["A", "B"]


@dataclass(frozen=True, slots=True)
class FastMove:
    """A fast move's per-turn contribution.

    The v0.1 model collapses GBL's per-move turn-duration into a single
    "per turn" tick. ``damage`` is the legacy absolute-value field used by
    early unit tests. New code paths that want STAB / type-effectiveness /
    stat-based damage supply ``power`` + ``move_type`` and let the
    resolver compute damage from the attacker / defender's stats.

    Resolution rule inside the matchup resolver:

    - If ``power > 0`` AND the attacker build carries ``attack > 0`` AND
      the defender build carries ``defense > 0``, the GBL formula is used
      (``power × stab × atk/def × type_eff × 0.5 × 1.3 + 1``) and
      ``damage`` is ignored.
    - Otherwise the legacy absolute ``damage`` is used as-is.
    """

    name: str
    damage: int = 0
    energy_gain: int = 0
    power: int = 0
    move_type: str = ""
    turns: int = 1
    """Turn duration in GBL ticks (500ms each). PvPoke values are 1..5.

    The v0.2 cooldown-aware resolver casts the fast move every ``turns``
    ticks; ``turns=1`` is the legacy per-turn cast that v0.1 unit tests
    assume."""

    def __post_init__(self) -> None:  # pragma: no cover - pure validation
        if self.damage < 0:
            raise ValueError(f"fast damage must be >= 0: {self.damage}")
        if self.energy_gain < 0:
            raise ValueError(
                f"fast energy_gain must be >= 0: {self.energy_gain}"
            )
        if self.power < 0:
            raise ValueError(f"fast power must be >= 0: {self.power}")
        if self.turns < 1:
            raise ValueError(f"fast turns must be >= 1: {self.turns}")


@dataclass(frozen=True, slots=True)
class ChargedMove:
    """A charged move: energy threshold + damage on hit + optional buff.

    Same dual-mode rule as :class:`FastMove`: ``power`` + ``move_type``
    activate the GBL formula; ``damage`` is the legacy absolute value used
    when the build does not carry stats.

    Buff effect fields mirror PvPoke's gamemaster:

    - ``buffs`` is ``(atk_stage_delta, def_stage_delta)`` applied on a
      successful (= unshielded) cast. Each delta in ``[-4, +4]``.
    - ``buff_chance`` is the probability the buff lands. The
      deterministic simulator applies a buff only when
      ``buff_chance >= 1.0``; sub-unit chances are dropped because the
      simulator has no RNG seam in v0.2.
    - ``buff_target`` is ``"self"`` or ``"opponent"``.

    Shielded charged moves *still* apply their buffs in the in-game
    rules — PvPoke matches that. The resolver applies the buff
    regardless of shield outcome.
    """

    name: str
    energy_cost: int
    damage: int = 0
    power: int = 0
    move_type: str = ""
    buffs: tuple[int, int] = (0, 0)
    buff_chance: float = 0.0
    buff_target: str = ""

    def __post_init__(self) -> None:  # pragma: no cover - pure validation
        if not (1 <= self.energy_cost <= ENERGY_CAP):
            raise ValueError(
                f"charged energy_cost out of range: {self.energy_cost}"
            )
        if self.damage < 0:
            raise ValueError(f"charged damage must be >= 0: {self.damage}")
        if self.power < 0:
            raise ValueError(f"charged power must be >= 0: {self.power}")


@dataclass(frozen=True, slots=True)
class CombatantBuild:
    """A GBL Pokémon's combat parameters used by the matchup resolver.

    Carries one fast move and up to two charged moves. The optional
    ``charged2`` second slot activates the bait-vs-lethal selection
    policy in :func:`resolve_matchup`; when ``None`` the resolver fires
    ``charged`` every time the attacker is energy-ready (legacy v0.1
    behavior).

    The ``attack`` / ``defense`` / ``types`` fields are optional. When
    populated, the resolver uses the GBL damage formula (STAB + type
    chart). When zero / empty, the resolver falls back to the legacy
    ``Move.damage`` absolute value, keeping unit tests that pre-compute
    damage values unchanged.
    """

    species: str
    max_hp: int
    fast: FastMove
    charged: ChargedMove
    attack: float = 0.0
    defense: float = 0.0
    types: tuple[str, ...] = ()
    charged2: ChargedMove | None = None
    form_id: int = 0
    """Upstream form discriminator. ``0`` = base form; non-zero values
    are interpreted by the render layer (typically ``1`` = shadow). The
    matchup resolver does not branch on this — shadow stat multipliers
    are already baked into ``attack`` / ``defense`` by the time a build
    reaches the resolver."""
    dex_id: int = 0
    """National Pokédex number. ``0`` when unknown (hand-built test
    fixtures). Used by team-builder logic to enforce GBL's same-dex
    uniqueness rule: a 3v3 team cannot contain two Pokémon sharing the
    same dex id, regardless of form / shadow status."""
    alt_form: "CombatantBuild | None" = None
    """Alternate form's full build for dynamic form-change species.

    Aegislash is the canonical example: the on-field Pokémon has two
    distinct stat / move profiles (Shield with 272 def / 0-power fast,
    Blade with 272 atk / normal fast) and toggles between them on
    specific in-battle events. ``None`` for the vast majority of
    species that have no runtime form change.

    See :class:`CombatantState.active_form_idx` for which form is
    currently active and :func:`CombatantState.effective_build` for
    the property the resolver actually reads."""
    form_change_to_alt_on_charged: bool = False
    """When ``True``, firing a charged move triggers a swap to
    ``alt_form`` BEFORE the cast resolves (so the cast uses alt-form
    stats / type). Aegislash's Shield → Blade transition uses this
    trigger (PvPoke's ``activate_charged`` rule)."""
    form_change_to_alt_on_shield_use: bool = False
    """When ``True``, defending against a charged move by spending a
    shield triggers a swap to ``alt_form`` AFTER the shield bleed
    damage applies. Aegislash's Blade → Shield transition uses this
    trigger (PvPoke's ``activate_shield`` rule)."""

    def __post_init__(self) -> None:  # pragma: no cover - pure validation
        if self.max_hp <= 0:
            raise ValueError(f"max_hp must be > 0: {self.max_hp}")
        if self.attack < 0:
            raise ValueError(f"attack must be >= 0: {self.attack}")
        if self.defense < 0:
            raise ValueError(f"defense must be >= 0: {self.defense}")

    @property
    def charged_moves(self) -> tuple[ChargedMove, ...]:
        """Return the build's available charged moves in order (1 or 2)."""
        if self.charged2 is None:
            return (self.charged,)
        return (self.charged, self.charged2)


@dataclass(frozen=True, slots=True)
class CombatantState:
    """One side's mid-matchup snapshot.

    Notably this carries ``shields`` per side — there is no shared
    "shield count" anywhere in the API. The 3 × 3 asymmetric-shield grid
    is just two independent values.

    ``fast_cooldown`` counts the ticks remaining before the next fast
    move resolves. ``0`` means the side is ready to cast on the current
    tick; positive values are decremented at end-of-tick. The legacy
    ``turns=1`` build produces ``fast_cooldown == 0`` every tick — i.e.
    the v0.1 per-turn cast pattern unit tests depend on.
    """

    build: CombatantBuild
    hp: int
    energy: int
    shields: int
    fast_cooldown: int = 0
    atk_stage: int = 0
    """Current attack stat-buff stage in ``[-4, +4]``; ``0`` = neutral."""
    def_stage: int = 0
    """Current defense stat-buff stage in ``[-4, +4]``; ``0`` = neutral."""
    active_form_idx: int = 0
    """Which form of a dynamic-form build is currently active.

    ``0`` = ``build`` (primary / origin form), ``1`` = ``build.alt_form``.
    For species without an alternate form (the default), this stays 0
    and ``effective_build`` returns the primary build unchanged.
    Resolver uses :func:`effective_build` rather than reading ``build``
    directly so form swaps are picked up on the next access."""

    def __post_init__(self) -> None:  # pragma: no cover - pure validation
        if not (0 <= self.hp <= self.build.max_hp):
            raise ValueError(
                f"hp {self.hp} out of [0, {self.build.max_hp}]"
            )
        if not (0 <= self.energy <= ENERGY_CAP):
            raise ValueError(f"energy out of range: {self.energy}")
        if not (0 <= self.shields <= MAX_SHIELDS):
            raise ValueError(f"shields out of range: {self.shields}")
        if self.fast_cooldown < 0:
            raise ValueError(
                f"fast_cooldown must be >= 0: {self.fast_cooldown}"
            )
        if not (-4 <= self.atk_stage <= 4):
            raise ValueError(f"atk_stage out of [-4, +4]: {self.atk_stage}")
        if not (-4 <= self.def_stage <= 4):
            raise ValueError(f"def_stage out of [-4, +4]: {self.def_stage}")

    @classmethod
    def fresh(cls, build: CombatantBuild, *, shields: int = MAX_SHIELDS) -> "CombatantState":
        """Convenience: full HP, zero energy, ``shields`` shields, neutral stages."""
        return cls(
            build=build,
            hp=build.max_hp,
            energy=0,
            shields=shields,
            fast_cooldown=0,
            atk_stage=0,
            def_stage=0,
            active_form_idx=0,
        )

    @property
    def effective_build(self) -> "CombatantBuild":
        """Return the build for the *currently active* form.

        For static-form species (the default), this is just ``build``.
        For dynamic-form species (Aegislash etc.) with
        ``active_form_idx == 1``, returns ``build.alt_form``. Falls
        back to ``build`` if ``alt_form`` is somehow missing — defense
        in depth so a misconfigured registry doesn't crash the
        resolver.
        """
        if self.active_form_idx == 1 and self.build.alt_form is not None:
            return self.build.alt_form
        return self.build


@dataclass(frozen=True, slots=True)
class ChargedEvent:
    """One charged-move resolution, recorded for downstream rationale cards."""

    turn: int
    attacker: Side
    move_name: str
    shielded: bool
    damage_applied: int


@dataclass(frozen=True, slots=True)
class MatchupResult:
    """Terminal outcome of :func:`resolve_matchup`.

    ``winner`` is ``"A"`` / ``"B"`` / ``None`` (double-KO or turn-budget
    cutoff with both sides alive). The terminal fields mirror the
    ``matchup_result`` concept in the seed ontology: switch sequence is
    omitted (this is a single matchup, not a set), but per-side terminal
    HP/energy/shields are preserved so the set-level driver can compose
    consecutive matchup results into a :data:`set_simulation` later.
    """

    winner: Side | None
    turns: int
    a_terminal_hp: int
    a_terminal_energy: int
    a_terminal_shields: int
    b_terminal_hp: int
    b_terminal_energy: int
    b_terminal_shields: int
    charged_events: tuple[ChargedEvent, ...]


# --- helpers --------------------------------------------------------------


def _type_effectiveness(
    move_type: str,
    defender_types: tuple[str, ...],
) -> float:
    """Return the combined type-effectiveness multiplier.

    Uses the packaged 18-type chart loaded on first call. Falls back to
    ``1.0`` (neutral) when the move type is empty or the type chart
    cannot be loaded — that fallback path lets unit tests that don't
    bother with types still run.
    """
    if not move_type:
        return 1.0
    try:
        from gbl_hacker.gamemaster import load_default_gamemaster
    except Exception:  # pragma: no cover - defensive
        return 1.0
    gm = load_default_gamemaster()
    eff = 1.0
    mt = move_type.lower()
    for dt in defender_types:
        traits = gm.type_chart.get(dt.lower())
        if not traits:
            continue
        if mt in traits.get("weaknesses", ()):
            eff *= _SUPER_EFFECTIVE
        elif mt in traits.get("resistances", ()):
            eff *= _RESISTED
        elif mt in traits.get("immunities", ()):
            eff *= _DOUBLE_RESISTED
    return eff


def _stage_multiplier(stage: int) -> float:
    """GBL stat-stage multiplier — mirrors :func:`gamemaster.stage_multiplier`.

    Inlined here so the resolver does not pull in the gamemaster module
    on its hot damage path. Identical formula to the public copy.
    """
    s = max(-4, min(4, stage))
    if s >= 0:
        return (4 + s) / 4.0
    return 4.0 / (4 - s)


def _move_damage(
    attacker: "CombatantBuild",
    defender: "CombatantBuild",
    *,
    move_power: int,
    move_type: str,
    legacy_damage: int,
    atk_stage: int = 0,
    def_stage: int = 0,
) -> int:
    """Compute hit damage for a move between two builds.

    Activation rule:
        Use the GBL formula when the move carries ``power > 0`` AND
        ``move_type`` is set AND the attacker carries ``attack > 0`` AND
        the defender carries ``defense > 0``. Otherwise fall back to the
        legacy absolute ``damage`` value (keeps pre-stats unit tests
        unchanged).

    Formula (mirrors PvPoke DamageCalculator):

        floor(power × stab × ((atk × atk_mult) / (def × def_mult))
              × effectiveness × 0.5 × 1.3) + 1

    where ``atk_mult`` / ``def_mult`` are the GBL stat-stage multipliers
    for the attacker's atk_stage and the defender's def_stage.
    """
    if (
        move_power > 0
        and move_type
        and attacker.attack > 0
        and defender.defense > 0
    ):
        stab = (
            _STAB_MULTIPLIER
            if move_type.lower() in {t.lower() for t in attacker.types}
            else 1.0
        )
        eff = _type_effectiveness(move_type, defender.types)
        eff_atk = attacker.attack * _stage_multiplier(atk_stage)
        eff_def = defender.defense * _stage_multiplier(def_stage)
        raw = (
            move_power
            * stab
            * (eff_atk / eff_def)
            * eff
            * 0.5
            * _DAMAGE_BONUS_MULTIPLIER
        )
        return math.floor(raw) + 1
    return legacy_damage


def _select_charged_move(
    attacker_state: CombatantState,
    defender_state: CombatantState,
) -> ChargedMove | None:
    """Pick which charged move to fire given the current state.

    Policy (deterministic, v0.2 baseline that mirrors PvPoke's greedy
    bait-or-lethal pattern):

    1. Compute the subset of charged moves the attacker can afford right
       now (``energy >= energy_cost``). Return ``None`` if empty.
    2. If the defender has zero shields:
       - Prefer a move that one-shots the defender. Among those, pick
         the cheapest (PvPoke convention: spend the least energy for
         the same KO).
       - Otherwise pick the highest damage-per-energy move so the
         turn-to-win is minimized.
    3. If the defender has shields:
       - Compute the actual unblocked damage of each move (would-be KO).
       - If any move one-shots the defender directly (rare without
         buffs but possible with type advantage), fire it.
       - Otherwise bait — fire the cheapest affordable move to drain a
         shield with minimal energy investment.

    Returns the selected :class:`ChargedMove` instance, or ``None`` when
    no move is affordable.
    """

    options = [
        m
        for m in attacker_state.effective_build.charged_moves
        if attacker_state.energy >= m.energy_cost
    ]
    if not options:
        return None
    if len(options) == 1:
        return options[0]

    # Damage calc uses the *future* (post-trigger) form when the
    # attacker has a form-change-on-charged species (Aegislash Shield
    # → Blade), since the cast resolves in alt-form stats. For static-
    # form species this collapses to the current effective build.
    if (
        attacker_state.effective_build.form_change_to_alt_on_charged
        and attacker_state.effective_build.alt_form is not None
    ):
        attacker_for_damage = attacker_state.effective_build.alt_form
    else:
        attacker_for_damage = attacker_state.effective_build

    def _hit_damage(move: ChargedMove) -> int:
        return _move_damage(
            attacker_for_damage,
            defender_state.effective_build,
            move_power=move.power,
            move_type=move.move_type,
            legacy_damage=move.damage,
            atk_stage=attacker_state.atk_stage,
            def_stage=defender_state.def_stage,
        )

    if defender_state.shields == 0:
        # No shield to bait — go for KO or best DPE.
        ko_candidates = [
            m for m in options if _hit_damage(m) >= defender_state.hp
        ]
        if ko_candidates:
            return min(ko_candidates, key=lambda m: m.energy_cost)
        return max(
            options,
            key=lambda m: _hit_damage(m) / max(1, m.energy_cost),
        )

    # Defender has a shield. Bait (cheapest move, to strip the shield)
    # ONLY when the attacker holds a move threatening enough that the
    # defender is forced to shield it — lethal or near-lethal. Stripping
    # the shield now lets that move connect later. With no such threat,
    # fire the best honest (highest-DPE) move rather than wasting tempo
    # on a bait the defender would simply call ("정직하게 던져도 이기면
    # 베이트 안 한다"). Whether a bait is *game-deciding* beyond this
    # matchup needs set-level context the resolver lacks; this is the
    # local approximation of that judgement.
    cheapest = min(options, key=lambda m: m.energy_cost)
    nuke = max(options, key=_hit_damage)
    forcing_threat = _hit_damage(nuke) >= _BAIT_THREAT_FRACTION * defender_state.hp
    if forcing_threat and cheapest.energy_cost < nuke.energy_cost:
        return cheapest
    return max(options, key=lambda m: _hit_damage(m) / max(1, m.energy_cost))


def _bounded_energy(value: int) -> int:
    """Cap energy into ``[0, ENERGY_CAP]``."""
    if value < 0:
        return 0
    if value > ENERGY_CAP:
        return ENERGY_CAP
    return value


_NON_LETHAL_SHIELD_PROBABILITY: float = 0.5
"""Probability the defender shields a *non-lethal* charged move when a
stochastic RNG is supplied. ``0.5`` is a conservative ladder baseline —
real top-rank players often farm a non-lethal first cast to bait the
shield, but they also shield-on-instinct sometimes. The defender ALWAYS
shields a lethal cast (damage >= remaining HP), regardless of RNG."""


_BAIT_THREAT_FRACTION: float = 0.85
"""Fraction of the defender's *current* HP a charged move must threaten for
the attacker to bother baiting a shield. A move doing >= 85% of current HP
(near-lethal) or more is one the defender is essentially forced to shield, so
stripping that shield with a cheaper move first is worth it. Below that, the
attacker fires its best honest move instead of baiting — needless baits get
called and tangle the position (top-player guidance)."""


def _apply_charged(
    attacker_state: CombatantState,
    defender_state: CombatantState,
    *,
    move: ChargedMove | None = None,
    rng: random.Random | None = None,
) -> tuple[CombatantState, CombatantState, bool, int]:
    """Resolve one charged move from ``attacker`` against ``defender``.

    Returns ``(new_attacker_state, new_defender_state, shielded,
    damage_applied)``. ``move`` defaults to the attacker's primary
    charged move — pass ``charged2`` explicitly to fire the secondary.

    Shield decision:
    - When ``rng is None`` (deterministic baseline) the defender always
      shields if they have a shield available — the v0.1-v0.2 default
      that unit tests pin.
    - When an ``rng`` is supplied AND the unblocked damage would not KO
      the defender, the shield is dropped with probability
      ``_NON_LETHAL_SHIELD_PROBABILITY``. Lethal threats are still
      shielded with 100% confidence.
    """
    # Form-change trigger: Shield→Blade on charged-move activation.
    # The transition fires BEFORE damage is computed so the cast uses
    # alt-form stats (Aegislash Shield's atk=97 → Blade's atk=272).
    # The flag lives on whichever build is currently active; the
    # build's own ``alt_form`` field is checked on the primary build
    # since that's where the alt was attached at materialization.
    if (
        attacker_state.build.alt_form is not None
        and attacker_state.active_form_idx == 0
        and attacker_state.effective_build.form_change_to_alt_on_charged
    ):
        attacker_state = CombatantState(
            build=attacker_state.build,
            hp=attacker_state.hp,
            energy=attacker_state.energy,
            shields=attacker_state.shields,
            fast_cooldown=attacker_state.fast_cooldown,
            atk_stage=attacker_state.atk_stage,
            def_stage=attacker_state.def_stage,
            active_form_idx=1,
        )

    selected = (
        move if move is not None else attacker_state.effective_build.charged
    )
    new_attacker = CombatantState(
        build=attacker_state.build,
        hp=attacker_state.hp,
        energy=_bounded_energy(attacker_state.energy - selected.energy_cost),
        shields=attacker_state.shields,
        fast_cooldown=attacker_state.fast_cooldown,
        atk_stage=attacker_state.atk_stage,
        def_stage=attacker_state.def_stage,
        active_form_idx=attacker_state.active_form_idx,
    )
    # Re-bind ``move`` name so the rest of the function (uses ``move.``)
    # keeps reading naturally — it was previously shadowing the parameter.
    move = selected
    raw_damage = _move_damage(
        attacker_state.effective_build,
        defender_state.effective_build,
        move_power=move.power,
        move_type=move.move_type,
        legacy_damage=move.damage,
        atk_stage=attacker_state.atk_stage,
        def_stage=defender_state.def_stage,
    )
    # Decide whether the defender shields. Stochastic: drop the shield
    # on non-lethal threats with probability _NON_LETHAL_SHIELD_PROBABILITY.
    shield_available = defender_state.shields > 0
    will_shield = shield_available
    if shield_available and rng is not None:
        is_lethal = raw_damage >= defender_state.hp
        if not is_lethal and rng.random() < _NON_LETHAL_SHIELD_PROBABILITY:
            will_shield = False
    if will_shield and shield_available:
        damage = SHIELD_BLEED_DAMAGE
        new_attacker_post = new_attacker
        # Form-change trigger: Blade→Shield on shield-use. A defender
        # in alt-form (Aegislash Blade) reverts to primary (Shield)
        # immediately after spending a shield (PvPoke's
        # ``activate_shield`` rule). Trigger flag lives on the active
        # build; alt_form attachment lives on the primary.
        new_form_idx_def = defender_state.active_form_idx
        if (
            defender_state.build.alt_form is not None
            and defender_state.active_form_idx == 1
            and defender_state.effective_build.form_change_to_alt_on_shield_use
        ):
            new_form_idx_def = 0
        new_defender = CombatantState(
            build=defender_state.build,
            hp=max(0, defender_state.hp - damage),
            energy=defender_state.energy,
            shields=defender_state.shields - 1,
            fast_cooldown=defender_state.fast_cooldown,
            atk_stage=defender_state.atk_stage,
            def_stage=defender_state.def_stage,
            active_form_idx=new_form_idx_def,
        )
        new_attacker_post, new_defender = _apply_charged_buff(
            new_attacker_post, new_defender, move=move, rng=rng
        )
        return new_attacker_post, new_defender, True, damage
    damage = raw_damage
    new_defender = CombatantState(
        build=defender_state.build,
        hp=max(0, defender_state.hp - damage),
        energy=defender_state.energy,
        shields=defender_state.shields,
        fast_cooldown=defender_state.fast_cooldown,
        atk_stage=defender_state.atk_stage,
        def_stage=defender_state.def_stage,
        active_form_idx=defender_state.active_form_idx,
    )
    new_attacker, new_defender = _apply_charged_buff(
        new_attacker, new_defender, move=move, rng=rng
    )
    return new_attacker, new_defender, False, damage


def _apply_charged_buff(
    attacker: CombatantState,
    defender: CombatantState,
    *,
    move: ChargedMove,
    rng: random.Random | None = None,
) -> tuple[CombatantState, CombatantState]:
    """Apply a charged move's buff effect.

    Activation rule:
    - ``buff_chance >= 1.0`` → always apply (deterministic).
    - ``buff_chance < 1.0`` AND ``rng is None`` → drop (legacy v0.2
      behavior; sub-unit chances had no RNG seam to sample).
    - ``buff_chance < 1.0`` AND ``rng is not None`` → sample
      ``rng.random() < buff_chance``. This is the v0.4 path that
      finally activates the long-tail moves (Ancient Power 10%,
      AeroBlast 12.5%, Crunch 30%, etc.).

    Both ``self`` and ``opponent`` targets are supported. Stages clamp
    into ``[-4, +4]``.
    """

    if move.buff_chance < 1.0:
        if rng is None:
            return attacker, defender
        if rng.random() >= move.buff_chance:
            return attacker, defender
    delta_atk, delta_def = move.buffs
    if delta_atk == 0 and delta_def == 0:
        return attacker, defender
    if move.buff_target == "self":
        new_atk = max(-4, min(4, attacker.atk_stage + delta_atk))
        new_def = max(-4, min(4, attacker.def_stage + delta_def))
        return (
            CombatantState(
                build=attacker.build,
                hp=attacker.hp,
                energy=attacker.energy,
                shields=attacker.shields,
                fast_cooldown=attacker.fast_cooldown,
                atk_stage=new_atk,
                def_stage=new_def,
                active_form_idx=attacker.active_form_idx,
            ),
            defender,
        )
    if move.buff_target == "opponent":
        new_atk = max(-4, min(4, defender.atk_stage + delta_atk))
        new_def = max(-4, min(4, defender.def_stage + delta_def))
        return (
            attacker,
            CombatantState(
                build=defender.build,
                hp=defender.hp,
                energy=defender.energy,
                shields=defender.shields,
                fast_cooldown=defender.fast_cooldown,
                atk_stage=new_atk,
                def_stage=new_def,
                active_form_idx=defender.active_form_idx,
            ),
        )
    return attacker, defender


def _apply_fast_simultaneous(
    a: CombatantState, b: CombatantState
) -> tuple[CombatantState, CombatantState]:
    """One simultaneous fast-move tick honoring per-side cooldowns.

    Both sides cast iff their ``fast_cooldown == 0``. The post-cast
    ``fast_cooldown`` is reset to ``turns - 1`` (decremented to 0 after
    ``turns - 1`` further ticks). A side waiting on cooldown decrements
    by one and deals no damage / gains no energy this tick.
    """

    a_ready = a.fast_cooldown == 0
    b_ready = b.fast_cooldown == 0

    a_eff = a.effective_build
    b_eff = b.effective_build

    a_takes = (
        _move_damage(
            b_eff,
            a_eff,
            move_power=b_eff.fast.power,
            move_type=b_eff.fast.move_type,
            legacy_damage=b_eff.fast.damage,
            atk_stage=b.atk_stage,
            def_stage=a.def_stage,
        )
        if b_ready
        else 0
    )
    b_takes = (
        _move_damage(
            a_eff,
            b_eff,
            move_power=a_eff.fast.power,
            move_type=a_eff.fast.move_type,
            legacy_damage=a_eff.fast.damage,
            atk_stage=a.atk_stage,
            def_stage=b.def_stage,
        )
        if a_ready
        else 0
    )

    a_new_energy = (
        _bounded_energy(a.energy + a.build.fast.energy_gain)
        if a_ready
        else a.energy
    )
    b_new_energy = (
        _bounded_energy(b.energy + b.build.fast.energy_gain)
        if b_ready
        else b.energy
    )

    a_cool = (a_eff.fast.turns - 1) if a_ready else (a.fast_cooldown - 1)
    b_cool = (b_eff.fast.turns - 1) if b_ready else (b.fast_cooldown - 1)

    new_a = CombatantState(
        build=a.build,
        hp=max(0, a.hp - a_takes),
        energy=a_new_energy,
        shields=a.shields,
        fast_cooldown=max(0, a_cool),
        atk_stage=a.atk_stage,
        def_stage=a.def_stage,
    )
    new_b = CombatantState(
        build=b.build,
        hp=max(0, b.hp - b_takes),
        energy=b_new_energy,
        shields=b.shields,
        fast_cooldown=max(0, b_cool),
        atk_stage=b.atk_stage,
        def_stage=b.def_stage,
    )
    return new_a, new_b


# --- public entry point ----------------------------------------------------


def resolve_matchup(
    a: CombatantState,
    b: CombatantState,
    *,
    max_turns: int = MAX_TURNS,
    rng: random.Random | None = None,
) -> MatchupResult:
    """Resolve a single 1v1 matchup to KO or turn-budget cutoff.

    Parameters
    ----------
    a, b:
        The two combatant states. Each carries its own ``shields`` count,
        so the 0/1/2 × 0/1/2 asymmetric grid is just two arguments. There
        is no place in the API that forces the two sides to share a
        shield value — that is the explicit Sub-AC 4.2 contract.
    max_turns:
        Safety cutoff. Defaults to :data:`MAX_TURNS`.

    Returns
    -------
    MatchupResult
        Terminal HP/energy/shields for both sides plus the ordered
        sequence of charged events that fired during the fight.

    Notes
    -----
    Turn ordering: fast moves resolve simultaneously each turn; charged
    moves resolve at end of turn with side ``A`` getting priority. This
    is a deliberate simplification — real GBL uses attack-stat-based
    priority — but it is **deterministic**, which is what unit tests
    need to pin a shield-asymmetric outcome.
    """
    if max_turns <= 0:
        raise ValueError(f"max_turns must be > 0: {max_turns}")

    events: list[ChargedEvent] = []
    turn = 0

    while turn < max_turns and a.hp > 0 and b.hp > 0:
        turn += 1

        # Phase 1: simultaneous fast moves.
        a, b = _apply_fast_simultaneous(a, b)
        if a.hp == 0 or b.hp == 0:
            # If both KO on the same tick, it's a double-KO.
            break

        # Phase 2: charged-move resolution with CMP (Charge Move Priority)
        # tiebreaker. When both sides are charge-ready on the same tick,
        # the side with the higher effective attack stat fires first.
        # Ties fall back to side-A priority (deterministic for tests).
        a_move = _select_charged_move(a, b)
        b_move = _select_charged_move(b, a)
        a_ready_ch = a_move is not None
        b_ready_ch = b_move is not None

        def _cmp_a_first(a_state: CombatantState, b_state: CombatantState) -> bool:
            a_atk = (
                a_state.effective_build.attack
                if a_state.effective_build.attack > 0
                else 0.0
            )
            b_atk = (
                b_state.effective_build.attack
                if b_state.effective_build.attack > 0
                else 0.0
            )
            if a_atk > b_atk:
                return True
            if b_atk > a_atk:
                return False
            # Exact tie: stochastic mode samples a 50/50 coin flip;
            # deterministic mode falls back to A-first for test stability.
            if rng is not None:
                return rng.random() < 0.5
            return True

        first_a = (
            _cmp_a_first(a, b) if (a_ready_ch and b_ready_ch) else a_ready_ch
        )

        if a_ready_ch and first_a and a_move is not None:
            a, b, shielded, dmg = _apply_charged(a, b, move=a_move, rng=rng)
            events.append(
                ChargedEvent(
                    turn=turn,
                    attacker="A",
                    move_name=a_move.name,
                    shielded=shielded,
                    damage_applied=dmg,
                )
            )
            if b.hp == 0:
                break

        if b_ready_ch and b_move is not None:
            b, a, shielded, dmg = _apply_charged(b, a, move=b_move, rng=rng)
            events.append(
                ChargedEvent(
                    turn=turn,
                    attacker="B",
                    move_name=b_move.name,
                    shielded=shielded,
                    damage_applied=dmg,
                )
            )
            if a.hp == 0:
                break

        if a_ready_ch and not first_a:
            # B got CMP priority and A still has energy → re-check after B.
            late_move = _select_charged_move(a, b)
            if late_move is not None:
                a, b, shielded, dmg = _apply_charged(a, b, move=late_move, rng=rng)
                events.append(
                    ChargedEvent(
                        turn=turn,
                        attacker="A",
                        move_name=late_move.name,
                        shielded=shielded,
                        damage_applied=dmg,
                    )
                )
                if b.hp == 0:
                    break

    if a.hp == 0 and b.hp == 0:
        winner: Side | None = None
    elif a.hp == 0:
        winner = "B"
    elif b.hp == 0:
        winner = "A"
    else:
        # Turn-budget cutoff with both still alive: no winner declared.
        winner = None

    return MatchupResult(
        winner=winner,
        turns=turn,
        a_terminal_hp=a.hp,
        a_terminal_energy=a.energy,
        a_terminal_shields=a.shields,
        b_terminal_hp=b.hp,
        b_terminal_energy=b.energy,
        b_terminal_shields=b.shields,
        charged_events=tuple(events),
    )


__all__ = [
    "MAX_TURNS",
    "SHIELD_BLEED_DAMAGE",
    "ChargedEvent",
    "ChargedMove",
    "CombatantBuild",
    "CombatantState",
    "FastMove",
    "MatchupResult",
    "Side",
    "resolve_matchup",
]
