"""Static gamemaster registry derived from PvPoke + GBL stats math.

Loads three packaged JSON dumps built once by ``scripts/build_gamemaster.py``:

- ``data/gamemaster.json``: slim ``{pokemon[], moves[]}`` from PvPoke
- ``data/cpm.json``: ``"<level>" → multiplier`` (level 1 → 51, half-level)
- ``data/type_chart.json``: ``"<type>" → {weaknesses, resistances, immunities}``

Exposes a small read-only registry that:

- Resolves a PvPoke species id / dex / Japanese display name to a
  ``GamemasterPokemon`` entry.
- Resolves a move id to a ``GamemasterMove`` entry.
- Computes 1500CP-cap-effective ``attack/defense/hp`` from the stored
  ``defaultIVs.cp1500 = [level, atk_iv, def_iv, hp_iv]`` using the CPM
  table — these are the same stats PvPoke uses for league-optimal play.
- Computes the GBL damage formula's ``effectiveness`` multiplier for a
  move type vs a defender's type tuple (using the 18-type chart).

The damage formula itself lives in :mod:`gbl_hacker.simulator.matchup`;
this module only owns the **stats / type / move lookup** seam.

Form discriminators (Taiman Party's ``form_id`` flag) are interpreted
heuristically: ``form_id != 0`` means "alternative form" (shadow / regional
variant). The mapping from dex+form → speciesId is in :func:`resolve_build`:

- ``form_id == 0`` and species_ja has no parenthesized variant → base form
  (PvPoke speciesId is the lowercase English name).
- ``form_id != 0`` and no parenthesized variant → shadow form (PvPoke
  speciesId has ``_shadow`` suffix).
- Parenthesized variant suffix in species_ja (e.g. ``ファイヤー(ガラル)``) →
  regional variant (PvPoke uses ``_galarian``, ``_alolan``, ``_hisuian``,
  ``_paldean`` suffixes); ``form_id != 0`` on top of a regional variant
  combines with ``_shadow`` (e.g. ``moltres_galarian_shadow``).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Iterable

from gbl_hacker.dex import PokedexRegistry, load_default_registry

_DATA_PACKAGE = "gbl_hacker.data"
_GAMEMASTER_FILE = "gamemaster.json"
_CPM_FILE = "cpm.json"
_TYPE_CHART_FILE = "type_chart.json"
_RANKINGS_FILE = "rankings_gl.json"
_RANKINGS_LEADS_FILE = "rankings_gl_leads.json"
_MOVE_JA_TO_PVPOKE_FILE = "move_ja_to_pvpoke.json"

# GBL damage formula constants (mirrors PvPoke DamageCalculator).
BONUS_MULTIPLIER: float = 1.2999999523162841796875
"""Pokemon GO global damage bonus — applied verbatim to every damage call."""

STAB_MULTIPLIER: float = 1.2
"""Same-type-attack bonus — applied when the move type is one of the
attacker's types."""

SUPER_EFFECTIVE: float = 1.6
RESISTED: float = 0.625
DOUBLE_RESISTED: float = 0.390625

SHADOW_ATK_MULT: float = 1.2
"""Shadow Pokémon attack multiplier (applied to the attack stat itself,
not the damage)."""

SHADOW_DEF_MULT: float = 0.8333333
"""Shadow Pokémon defense multiplier (5/6)."""

# Regional / size / state form suffix table — Japanese parenthesized
# qualifier → PvPoke speciesId suffix. Mirrors the render-layer table but
# the two are intentionally not linked; the render layer translates for
# display, this one drives speciesId resolution.
_FORM_SUFFIX_PVPOKE: dict[str, str] = {
    # Regional variants
    "ガラル": "_galarian",
    "アローラ": "_alolan",
    "ヒスイ": "_hisuian",
    "パルデア": "_paldean",
    "メガ": "_mega",
    # Aegislash (dex 681)
    "シールド": "_shield",
    "ブレード": "_blade",
    # Pumpkaboo / Gourgeist (dex 710 / 711) sizes
    "ちいさい": "_small",
    "しょうしょう": "_small",
    "ふつう": "_average",
    "おおがた": "_large",
    "とくだい": "_super",
    # Morpeko (dex 877) modes
    "まんぷく": "_full_belly",
    "はらぺこ": "_hangry",
}

# Dex-level overrides for species whose Japanese name is identical
# between base and a regional variant, but where the GBL meta meaning
# is always the variant. Without these overrides the dex_id lookup
# would land on the base entry (which PvPoke lists first) even though
# the site's bare ``species_ja`` actually means the variant.
#
# Each entry: dex_id → PvPoke speciesId of the form the JP site means
# when it emits a bare species_ja with form_id=0.
_DEX_GL_OVERRIDE: dict[int, str] = {
    # NOTE: 618 マッギョ was previously force-mapped to "stunfisk_galarian"
    # on the assumption the JP feed's bare マッギョ always means Galarian.
    # That is empirically FALSE in the recorded fixtures: the Taiman
    # pokemon_usage row for マッギョ carries ELECTRIC moves (でんきショック /
    # ほうでん — Thunder Shock / Discharge), which Galarian Stunfisk
    # (ground/steel) cannot learn — so the bare name denotes BASE Stunfisk
    # (ground/electric). The override stapled regular's electric moveset
    # onto a Galarian-typed body (a chimera). Galarian is still reachable
    # via the explicit "マッギョ(ガラル)" suffix. Keep this dict for genuine
    # name-collision cases, but verify against usage data before adding one.
}

_PARENS_RE = re.compile(r"^(.*?)\s*\(([^)]+)\)\s*$")


@dataclass(frozen=True, slots=True)
class GamemasterMove:
    """Move spec as PvPoke ships it.

    Attributes
    ----------
    move_id:
        PvPoke move id (e.g. ``WATER_GUN``).
    name:
        Display name (English).
    move_type:
        One of the 18 GO types (lowercase).
    power:
        Damage power before stats / STAB / effectiveness.
    energy:
        For charged moves: energy cost. ``0`` for fast moves.
    energy_gain:
        For fast moves: energy gained per cast. ``0`` for charged moves.
    cooldown_ms:
        Real-world ms of one cast (PvPoke uses 500ms / turn).
    turns:
        Per-turn duration (500ms each). ``None`` for charged moves (no
        notion of turn-duration there).
    buffs:
        Optional ``(atk_stage_delta, def_stage_delta)`` tuple applied
        on a successful cast. ``None`` for moves without a buff effect.
        Each delta is an integer in ``[-4, +4]``.
    buff_apply_chance:
        Probability the buff lands on a cast. The simulator's
        deterministic baseline only applies buffs with ``chance >= 1.0``;
        sub-unit chances are treated as expected-value variance and
        ignored by v0.2.
    buff_target:
        ``"self"`` or ``"opponent"`` — which combatant gets the buff
        deltas applied. Empty string for moves without a buff effect.
    """

    move_id: str
    name: str
    move_type: str
    power: int
    energy: int
    energy_gain: int
    cooldown_ms: int
    turns: int | None
    buffs: tuple[int, int] | None = None
    buff_apply_chance: float = 0.0
    buff_target: str = ""

    @property
    def is_fast(self) -> bool:
        return self.energy_gain > 0


def stage_multiplier(stage: int) -> float:
    """GBL stat-stage multiplier table.

    GBL stages range from ``-4`` to ``+4``; the multiplier formula
    matches Pokémon GO's standard buff/debuff math:

    - ``stage >= 0`` → ``(4 + stage) / 4``  (e.g. +2 → 1.5)
    - ``stage <  0`` → ``4 / (4 - stage)``  (e.g. −2 → 0.667)
    """

    s = max(-4, min(4, stage))
    if s >= 0:
        return (4 + s) / 4.0
    return 4.0 / (4 - s)


@dataclass(frozen=True, slots=True)
class GamemasterPokemon:
    """Pokémon spec slim of PvPoke's gamemaster entry.

    Attributes
    ----------
    dex:
        National Pokédex id.
    species_id:
        PvPoke unique id (``forretress``, ``moltres_galarian``,
        ``forretress_shadow``).
    species_name:
        English display name.
    base_atk, base_def, base_hp:
        Base stats from PvPoke.
    types:
        Tuple of one or two type strings.
    fast_move_ids, charged_move_ids:
        Move id lists.
    default_iv_cp1500:
        ``(level, atk_iv, def_iv, hp_iv)`` stat-product-optimal IVs
        under the 1500CP cap. ``None`` when PvPoke could not compute
        them (e.g. species ineligible for Great League).
    is_shadow:
        ``True`` iff the speciesId ends in ``_shadow``.
    """

    dex: int
    species_id: str
    species_name: str
    base_atk: int
    base_def: int
    base_hp: int
    types: tuple[str, ...]
    fast_move_ids: tuple[str, ...]
    charged_move_ids: tuple[str, ...]
    default_iv_cp1500: tuple[float, int, int, int] | None = None
    is_shadow: bool = False


@dataclass(frozen=True, slots=True)
class EffectiveStats:
    """1500CP-cap effective stats for a Pokémon at its optimal IVs.

    Attributes
    ----------
    level:
        The Pokémon's level (half-level allowed).
    attack, defense:
        Float stats used in the GBL damage formula.
    hp:
        Integer HP (PvPoke / niantic floor at the integer).
    cp:
        Computed CP value (≤ 1500 by construction).
    """

    level: float
    attack: float
    defense: float
    hp: int
    cp: int


@dataclass(frozen=True, slots=True)
class RankingEntry:
    """One PvPoke ranking entry — the recommended moveset + score.

    Attributes
    ----------
    species_id:
        PvPoke species id (matches GamemasterPokemon.species_id).
    score:
        PvPoke's overall meta score in ``[0, 100]`` — higher is better.
    rating:
        PvPoke's raw rating (~600-700 in GL).
    moveset:
        ``[fast_move_id, charged_move_1_id, charged_move_2_id]`` — the
        most accurate GBL meta-recommended moveset for this species.
    """

    species_id: str
    score: float
    rating: int
    moveset: tuple[str, ...]

    @property
    def fast_move_id(self) -> str | None:
        return self.moveset[0] if self.moveset else None

    @property
    def charged_move_ids(self) -> tuple[str, ...]:
        return self.moveset[1:] if len(self.moveset) > 1 else ()


@dataclass(frozen=True, slots=True)
class GamemasterRegistry:
    """Combined PvPoke pokemon / moves / cpm / type-chart / rankings registry."""

    pokemon_by_species_id: dict[str, GamemasterPokemon] = field(default_factory=dict)
    pokemon_by_dex_base: dict[int, GamemasterPokemon] = field(default_factory=dict)
    moves_by_id: dict[str, GamemasterMove] = field(default_factory=dict)
    cpm: dict[float, float] = field(default_factory=dict)
    type_chart: dict[str, dict[str, tuple[str, ...]]] = field(default_factory=dict)
    rankings: tuple[RankingEntry, ...] = field(default_factory=tuple)
    rankings_by_species_id: dict[str, RankingEntry] = field(default_factory=dict)
    leads_rankings: tuple[RankingEntry, ...] = field(default_factory=tuple)
    """PvPoke ``leads`` category ranking — species sorted by how strong
    they are *as a lead*. Better approximation of ladder lead
    distribution than the overall ranking."""
    move_ja_to_pvpoke: dict[str, str] = field(default_factory=dict)
    """Mapping ``Japanese move name (katakana)`` → PvPoke ``moveId``.

    Built once from PokeAPI's ``move_names.csv`` cross-referenced with
    PvPoke's ``gamemaster.moves`` set. Used to translate Taiman Party's
    ``waza1`` / ``waza2`` ladder-usage entries into PvPoke moves so the
    simulator can score the moveset operators actually run, not just
    PvPoke's recommendation."""

    # -- Pokémon lookup ----------------------------------------------------

    def get_by_species_id(self, species_id: str) -> GamemasterPokemon | None:
        return self.pokemon_by_species_id.get(species_id)

    def get_base_by_dex(self, dex_id: int) -> GamemasterPokemon | None:
        """Return the base form for a dex id (no shadow / regional)."""
        return self.pokemon_by_dex_base.get(dex_id)

    def resolve_build(
        self,
        *,
        dex_id: int | None,
        species_ja: str,
        form_id: int | None,
        dex_registry: PokedexRegistry | None = None,
    ) -> GamemasterPokemon | None:
        """Resolve a Taiman Party species reference to a gamemaster entry.

        Resolution rules (applied in this order):

        1. Strip any parenthesized variant suffix from ``species_ja``
           (e.g. ``ファイヤー(ガラル)`` → base ``ファイヤー`` + ``ガラル``).
        2. If ``dex_id`` is not given, resolve it from the base name via
           the dex registry.
        3. Pick the starting speciesId:
           - If ``dex_id`` is in :data:`_DEX_GL_OVERRIDE`, use the
             override directly (handles cases like dex 618 マッギョ where
             the JP name is identical for base and variant but the
             GBL meta meaning is always the variant).
           - Otherwise the base entry for that dex id.
        4. If a variant suffix was peeled off, swap the speciesId to the
           ``<base><suffix>`` form (e.g. ``moltres_galarian``).
        5. If ``form_id`` is non-zero, append ``_shadow``. This applies
           on top of a variant suffix too — ``サンドパン(アローラ)-1``
           resolves to ``sandslash_alolan_shadow``.

        Returns ``None`` when no PvPoke entry matches the final
        speciesId — callers should treat that as an unresolvable species
        and surface a diagnostic.
        """

        dex_reg = dex_registry or load_default_registry()

        base_name = species_ja
        variant_suffix_ja: str | None = None
        m = _PARENS_RE.match(species_ja)
        if m:
            base_name = m.group(1).strip()
            variant_suffix_ja = m.group(2).strip()

        # Step 2: resolve dex id from the base name if not provided.
        if dex_id is None:
            entry = dex_reg.lookup(species_ja=base_name)
            if entry is not None:
                dex_id = entry.dex_id

        # Step 3: pick starting speciesId — override > base entry.
        if dex_id is not None and dex_id in _DEX_GL_OVERRIDE:
            override_id = _DEX_GL_OVERRIDE[dex_id]
            base_for_variant = self.pokemon_by_species_id.get(override_id)
            if base_for_variant is None:
                base_for_variant = self.pokemon_by_dex_base.get(dex_id)
        else:
            base_for_variant = (
                self.pokemon_by_dex_base.get(dex_id) if dex_id is not None else None
            )
        if base_for_variant is None:
            return None

        species_id = base_for_variant.species_id

        # Step 4: variant suffix swap. The override entry already has
        # its own suffix in many cases (e.g. stunfisk_galarian); to keep
        # both "bare base + suffix" (e.g. forretress + _galarian) and
        # "fallback form + new suffix" (e.g. aegislash_blade →
        # aegislash + _shield) working, we strip any existing
        # underscore segment back to the stem before re-attaching.
        if variant_suffix_ja:
            suffix = _FORM_SUFFIX_PVPOKE.get(variant_suffix_ja)
            if suffix:
                stem = species_id.split("_", 1)[0] if "_" in species_id else species_id
                candidate = f"{stem}{suffix}"
                if candidate in self.pokemon_by_species_id:
                    species_id = candidate

        # Step 5: shadow suffix. Applied on top of a variant.
        if form_id is not None and form_id != 0:
            shadow_id = f"{species_id}_shadow"
            if shadow_id in self.pokemon_by_species_id:
                species_id = shadow_id

        return self.pokemon_by_species_id.get(species_id)

    # -- Moves ------------------------------------------------------------

    def get_move(self, move_id: str) -> GamemasterMove | None:
        return self.moves_by_id.get(move_id)

    # -- CPM / stats ------------------------------------------------------

    def cpm_at(self, level: float) -> float:
        """Return the CPM for ``level`` (1.0 ≤ level ≤ 51.0, half-step)."""
        try:
            return self.cpm[level]
        except KeyError as exc:
            # Fall back to half-level rounding for safety.
            rounded = round(level * 2) / 2.0
            if rounded in self.cpm:
                return self.cpm[rounded]
            raise KeyError(f"unknown level {level!r}") from exc

    def effective_stats(
        self,
        pokemon: GamemasterPokemon,
        *,
        level: float | None = None,
        atk_iv: int | None = None,
        def_iv: int | None = None,
        hp_iv: int | None = None,
    ) -> EffectiveStats:
        """Compute effective Atk / Def / HP / CP for a Pokémon.

        Defaults to the stored ``defaultIVs.cp1500`` (1500CP cap, PvPoke
        stat-product-optimal). Caller can override any of the four
        components individually (useful for experimenting with sub-optimal
        IVs or different leagues).
        """

        ivs = pokemon.default_iv_cp1500
        if ivs is not None:
            d_level, d_atk_iv, d_def_iv, d_hp_iv = ivs
        else:
            # Without PvPoke-computed IVs, fall back to "best buddy lv50,
            # all-15 IVs" — a conservative defender baseline. CP cap is
            # NOT enforced in this fallback (caller's responsibility).
            d_level, d_atk_iv, d_def_iv, d_hp_iv = 50.0, 15, 15, 15

        lv = level if level is not None else d_level
        a_iv = atk_iv if atk_iv is not None else d_atk_iv
        d_iv = def_iv if def_iv is not None else d_def_iv
        h_iv = hp_iv if hp_iv is not None else d_hp_iv

        cpm = self.cpm_at(float(lv))
        atk_mult = SHADOW_ATK_MULT if pokemon.is_shadow else 1.0
        def_mult = SHADOW_DEF_MULT if pokemon.is_shadow else 1.0

        attack = (pokemon.base_atk + a_iv) * cpm * atk_mult
        defense = (pokemon.base_def + d_iv) * cpm * def_mult
        hp_f = (pokemon.base_hp + h_iv) * cpm
        hp = max(10, math.floor(hp_f))

        cp = max(
            10,
            math.floor(
                (
                    (pokemon.base_atk + a_iv)
                    * math.sqrt(pokemon.base_def + d_iv)
                    * math.sqrt(pokemon.base_hp + h_iv)
                    * (cpm**2)
                )
                / 10.0
            ),
        )

        return EffectiveStats(level=lv, attack=attack, defense=defense, hp=hp, cp=cp)

    # -- Type effectiveness ----------------------------------------------

    def effectiveness(
        self,
        move_type: str,
        defender_types: Iterable[str],
    ) -> float:
        """Combined type-effectiveness multiplier for ``move_type`` vs
        the defender's type tuple.

        Multiplies SUPER_EFFECTIVE / RESISTED / DOUBLE_RESISTED across
        the defender's types — the same rule PvPoke's DamageCalculator
        uses.
        """

        mt = move_type.lower()
        eff = 1.0
        for dt in defender_types:
            traits = self.type_chart.get(dt.lower())
            if not traits:
                continue
            if mt in traits.get("weaknesses", ()):
                eff *= SUPER_EFFECTIVE
            elif mt in traits.get("resistances", ()):
                eff *= RESISTED
            elif mt in traits.get("immunities", ()):
                eff *= DOUBLE_RESISTED
        return eff

    def stab_multiplier(self, move_type: str, attacker_types: Iterable[str]) -> float:
        """Return 1.2 if move shares a type with attacker, else 1.0."""
        mt = move_type.lower()
        attacker = [t.lower() for t in attacker_types]
        return STAB_MULTIPLIER if mt in attacker else 1.0


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_json(filename: str) -> object:
    text = resources.files(_DATA_PACKAGE).joinpath(filename).read_text(
        encoding="utf-8"
    )
    return json.loads(text)


def _build_pokemon(entry: dict) -> GamemasterPokemon:
    base_stats = entry.get("baseStats") or {}
    types = tuple((entry.get("types") or [])[:2])
    ivs_raw = entry.get("defaultIVs_cp1500")
    ivs_t: tuple[float, int, int, int] | None
    if (
        isinstance(ivs_raw, list)
        and len(ivs_raw) == 4
        and all(isinstance(v, (int, float)) for v in ivs_raw)
    ):
        ivs_t = (
            float(ivs_raw[0]),
            int(ivs_raw[1]),
            int(ivs_raw[2]),
            int(ivs_raw[3]),
        )
    else:
        ivs_t = None
    species_id = entry["speciesId"]
    return GamemasterPokemon(
        dex=int(entry["dex"]),
        species_id=species_id,
        species_name=entry.get("speciesName", species_id),
        base_atk=int(base_stats.get("atk", 0)),
        base_def=int(base_stats.get("def", 0)),
        base_hp=int(base_stats.get("hp", 0)),
        types=tuple(t for t in types if t),
        fast_move_ids=tuple(entry.get("fastMoves") or ()),
        charged_move_ids=tuple(entry.get("chargedMoves") or ()),
        default_iv_cp1500=ivs_t,
        is_shadow=species_id.endswith("_shadow"),
    )


def _build_move(entry: dict) -> GamemasterMove:
    buffs_raw = entry.get("buffs")
    buffs: tuple[int, int] | None = None
    if isinstance(buffs_raw, list) and len(buffs_raw) == 2:
        try:
            buffs = (int(buffs_raw[0]), int(buffs_raw[1]))
        except (TypeError, ValueError):
            buffs = None
    chance_raw = entry.get("buffApplyChance")
    try:
        chance = float(chance_raw) if chance_raw is not None else 0.0
    except (TypeError, ValueError):
        chance = 0.0
    target = str(entry.get("buffTarget") or "").lower()
    return GamemasterMove(
        move_id=entry["moveId"],
        name=entry.get("name", entry["moveId"]),
        move_type=str(entry.get("type", "")).lower(),
        power=int(entry.get("power") or 0),
        energy=int(entry.get("energy") or 0),
        energy_gain=int(entry.get("energyGain") or 0),
        cooldown_ms=int(entry.get("cooldown") or 0),
        turns=int(entry["turns"]) if entry.get("turns") is not None else None,
        buffs=buffs,
        buff_apply_chance=chance,
        buff_target=target,
    )


@lru_cache(maxsize=1)
def load_default_gamemaster() -> GamemasterRegistry:
    """Build and cache the gamemaster registry from packaged JSON dumps."""

    raw_gm = _load_json(_GAMEMASTER_FILE)
    if not isinstance(raw_gm, dict):
        raise RuntimeError("gamemaster.json must be an object")
    raw_pokemon = raw_gm.get("pokemon", [])
    raw_moves = raw_gm.get("moves", [])

    pokemon_by_id: dict[str, GamemasterPokemon] = {}
    pokemon_by_dex_base: dict[int, GamemasterPokemon] = {}
    pokemon_by_dex_fallback: dict[int, GamemasterPokemon] = {}
    for entry in raw_pokemon:
        try:
            p = _build_pokemon(entry)
        except (KeyError, ValueError, TypeError):
            continue
        pokemon_by_id[p.species_id] = p
        # Preferred: the first underscore-free non-shadow entry per dex
        # — that's PvPoke's bare base form (e.g. ``forretress``).
        if not p.is_shadow and "_" not in p.species_id:
            pokemon_by_dex_base.setdefault(p.dex, p)
        # Fallback: first non-shadow entry per dex regardless of
        # underscore. Used by dexes that lack a bare base form
        # (e.g. Aegislash, Gourgeist, Morpeko) so resolve_build can
        # find a starting entry and then apply a parens-suffix swap.
        if not p.is_shadow:
            pokemon_by_dex_fallback.setdefault(p.dex, p)

    # Merge fallback into base for dexes the strict rule missed.
    for dex_id, p in pokemon_by_dex_fallback.items():
        pokemon_by_dex_base.setdefault(dex_id, p)

    moves_by_id: dict[str, GamemasterMove] = {}
    for entry in raw_moves:
        try:
            m = _build_move(entry)
        except (KeyError, ValueError, TypeError):
            continue
        moves_by_id[m.move_id] = m

    raw_cpm = _load_json(_CPM_FILE)
    if not isinstance(raw_cpm, dict):
        raise RuntimeError("cpm.json must be an object")
    cpm: dict[float, float] = {}
    for k, v in raw_cpm.items():
        try:
            level_f = float(k)
            cpm[level_f] = float(v)
        except (TypeError, ValueError):
            continue

    def _parse_rankings_list(raw: object) -> list[RankingEntry]:
        out: list[RankingEntry] = []
        if not isinstance(raw, list):
            return out
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            species_id = entry.get("speciesId")
            if not isinstance(species_id, str):
                continue
            try:
                score = float(entry.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            try:
                rating = int(entry.get("rating") or 0)
            except (TypeError, ValueError):
                rating = 0
            mv = entry.get("moveset") or []
            moveset = tuple(m for m in mv if isinstance(m, str))
            out.append(
                RankingEntry(
                    species_id=species_id,
                    score=score,
                    rating=rating,
                    moveset=moveset,
                )
            )
        return out

    raw_rankings = _load_json(_RANKINGS_FILE)
    rankings_list = _parse_rankings_list(raw_rankings)
    rankings_by_id: dict[str, RankingEntry] = {}
    for r in rankings_list:
        rankings_by_id.setdefault(r.species_id, r)

    leads_rankings_list = _parse_rankings_list(_load_json(_RANKINGS_LEADS_FILE))

    raw_move_map = _load_json(_MOVE_JA_TO_PVPOKE_FILE)
    move_ja_to_pvpoke: dict[str, str] = {}
    if isinstance(raw_move_map, dict):
        for k, v in raw_move_map.items():
            if isinstance(k, str) and isinstance(v, str):
                move_ja_to_pvpoke[k] = v

    raw_chart = _load_json(_TYPE_CHART_FILE)
    if not isinstance(raw_chart, dict):
        raise RuntimeError("type_chart.json must be an object")
    type_chart: dict[str, dict[str, tuple[str, ...]]] = {}
    for t, traits in raw_chart.items():
        if not isinstance(traits, dict):
            continue
        type_chart[t.lower()] = {
            "weaknesses": tuple(traits.get("weaknesses") or ()),
            "resistances": tuple(traits.get("resistances") or ()),
            "immunities": tuple(traits.get("immunities") or ()),
        }

    return GamemasterRegistry(
        pokemon_by_species_id=pokemon_by_id,
        pokemon_by_dex_base=pokemon_by_dex_base,
        moves_by_id=moves_by_id,
        cpm=cpm,
        type_chart=type_chart,
        rankings=tuple(rankings_list),
        rankings_by_species_id=rankings_by_id,
        leads_rankings=tuple(leads_rankings_list),
        move_ja_to_pvpoke=move_ja_to_pvpoke,
    )


__all__ = [
    "BONUS_MULTIPLIER",
    "DOUBLE_RESISTED",
    "EffectiveStats",
    "GamemasterMove",
    "GamemasterPokemon",
    "GamemasterRegistry",
    "RESISTED",
    "RankingEntry",
    "SHADOW_ATK_MULT",
    "SHADOW_DEF_MULT",
    "STAB_MULTIPLIER",
    "SUPER_EFFECTIVE",
    "load_default_gamemaster",
    "stage_multiplier",
]
