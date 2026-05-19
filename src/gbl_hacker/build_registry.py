"""Materialize Taiman Party meta species into simulator-ready builds.

Bridges the Taiman Party meta (Japanese species names + form_ids) and the
PvPoke gamemaster (English speciesId + stats + moves) so the scoring
layer can call ``expected_win_rate(...)`` with a populated build_registry.

Each ``CombatantBuild`` produced here carries:

- Effective Attack / Defense / HP at the 1500CP cap (PvPoke
  ``defaultIVs.cp1500`` + CPM table).
- The Pokémon's type tuple (drives STAB + type-effectiveness in the
  resolver's GBL damage formula).
- One fast move + one charged move picked by a default policy
  (``PvPoke fastMoves[0]`` / ``chargedMoves[0]``). Each move carries
  ``power`` + ``move_type`` so the resolver uses the GBL formula.

v0.1 simplification — form_id is read from the Taiman Party ``TeamUsage``
but the resulting build_registry keys are bare Japanese species names.
This means a species that appears in the meta as both its base form and
its shadow form collapses to the base-form stats. Shadow stat shifts
(Atk +20%, Def −16.7%) are visible in :func:`materialize_build` when the
caller passes ``form_id=1`` explicitly, but the meta-wide registry built
by :func:`build_registry_for_meta` always uses base stats. Form-aware
keying is a v0.2 concern.
"""

from __future__ import annotations

from datetime import datetime, timezone

from gbl_hacker.dex import PokedexRegistry, load_default_registry
from gbl_hacker.fetch.taiman import TAIMAN_SOURCE_CAVEAT
from gbl_hacker.gamemaster import GamemasterRegistry, load_default_gamemaster
from gbl_hacker.parse.taiman import (
    GREAT_LEAGUE_LABEL,
    MetaSnapshot,
    MoveUsage,
    PokemonUsage,
    TeamUsage,
)
from gbl_hacker.simulator import ChargedMove, CombatantBuild, FastMove


# PvPoke speciesIds that participate in a dynamic in-battle form
# change. We materialize these as ONE logical species (the entry-
# default form) with the other form as ``alt_form``. The keys are the
# primary speciesIds (what the operator selects when team-building);
# values are the alternate speciesIds and the trigger flags.
_DYNAMIC_FORM_SPECS: dict[str, tuple[str, bool, bool]] = {
    # primary_id → (alt_id, form_change_to_alt_on_charged,
    #              form_change_to_alt_on_shield_use)
    "aegislash_shield": ("aegislash_blade", True, False),
    "aegislash_blade": ("aegislash_shield", False, True),
}

# speciesIds that should NOT appear as standalone picks in the
# candidate pool because they are runtime states of another species
# (Aegislash Blade is reached by firing a charged move; it's never the
# entry form). Dropping these from the pool prevents double-counting.
_RUNTIME_ONLY_SPECIES_IDS: set[str] = {"aegislash_blade"}


def materialize_build_from_ranking(
    ranking_species_id: str,
    *,
    gm: GamemasterRegistry | None = None,
    dex: PokedexRegistry | None = None,
) -> CombatantBuild | None:
    """Build a :class:`CombatantBuild` from a PvPoke ranking entry.

    Pulls the species, effective stats, and the PvPoke-recommended
    moveset (fast + 2 charged) directly from the rankings table — far
    more accurate than ``materialize_build`` which falls back to
    ``fastMoves[0]`` / ``chargedMoves[0]`` from the raw gamemaster.

    The Pokémon's display name (``CombatantBuild.species``) is set to
    the Japanese localized name when the dex registry resolves it,
    otherwise to the English ``speciesName`` from the gamemaster. This
    keeps the render layer's species-localization path identical to the
    pool / meta paths.
    """

    gm_r = gm or load_default_gamemaster()
    dex_r = dex or load_default_registry()

    pokemon = gm_r.get_by_species_id(ranking_species_id)
    if pokemon is None:
        return None

    ranking = gm_r.rankings_by_species_id.get(ranking_species_id)
    fast_id = ranking.fast_move_id if ranking else None
    charged_ids = ranking.charged_move_ids if ranking else ()

    fast_id = fast_id or (
        pokemon.fast_move_ids[0] if pokemon.fast_move_ids else None
    )
    if not fast_id:
        return None

    if charged_ids:
        primary_id = charged_ids[0]
        secondary_id = charged_ids[1] if len(charged_ids) > 1 else None
    else:
        primary_id = (
            pokemon.charged_move_ids[0] if pokemon.charged_move_ids else None
        )
        secondary_id = None
        if primary_id and len(pokemon.charged_move_ids) >= 2:
            for mid in pokemon.charged_move_ids:
                if mid != primary_id:
                    secondary_id = mid
                    break

    if not primary_id:
        return None

    fast_move = gm_r.get_move(fast_id)
    charged_move = gm_r.get_move(primary_id)
    if fast_move is None or charged_move is None:
        return None
    charged2_move = gm_r.get_move(secondary_id) if secondary_id else None

    stats = gm_r.effective_stats(pokemon)

    # Display name: localize via dex registry when possible (so the
    # render layer's Japanese → Korean/English path works the same).
    dex_entry = dex_r.lookup(dex_id=pokemon.dex)
    display_name = dex_entry.ja if dex_entry else pokemon.species_name

    fast = FastMove(
        name=fast_move.name,
        damage=0,
        energy_gain=fast_move.energy_gain,
        power=fast_move.power,
        move_type=fast_move.move_type,
        turns=fast_move.turns or 1,
    )

    def _build_charged(m) -> ChargedMove:
        energy_cost = max(1, min(100, m.energy))
        buffs = m.buffs if m.buffs is not None else (0, 0)
        return ChargedMove(
            name=m.name,
            energy_cost=energy_cost,
            damage=0,
            power=m.power,
            move_type=m.move_type,
            buffs=buffs,
            buff_chance=m.buff_apply_chance,
            buff_target=m.buff_target,
        )

    charged = _build_charged(charged_move)
    charged2 = _build_charged(charged2_move) if charged2_move else None

    # Form id: 1 for shadow speciesIds, 0 otherwise. (Regional variants
    # also have their own speciesId so this is just shadow vs. base.)
    form_id = 1 if pokemon.is_shadow else 0

    # Dynamic-form attachment: if this speciesId has an alternate form
    # (Aegislash, etc.), recursively materialize the alt form and
    # attach it to ``alt_form``. The trigger flags govern when the
    # resolver swaps between the two.
    alt_build = None
    fc_on_charged = False
    fc_on_shield = False
    dform = _DYNAMIC_FORM_SPECS.get(ranking_species_id)
    if dform is not None:
        alt_species_id, fc_on_charged, fc_on_shield = dform
        # Recursive call — but with a guard: the alt form's own
        # ``_DYNAMIC_FORM_SPECS`` entry would otherwise produce an
        # infinite cycle (Shield→Blade→Shield→…). Build the alt
        # without further dynamic-form linkage by inlining the
        # materialization steps.
        alt_pokemon = gm_r.get_by_species_id(alt_species_id)
        alt_ranking = gm_r.rankings_by_species_id.get(alt_species_id)
        if alt_pokemon is not None:
            alt_fast_id = (
                alt_ranking.fast_move_id if alt_ranking else None
            ) or (
                alt_pokemon.fast_move_ids[0]
                if alt_pokemon.fast_move_ids
                else None
            )
            alt_charged_ids = (
                alt_ranking.charged_move_ids if alt_ranking else ()
            )
            alt_primary_id = (
                alt_charged_ids[0]
                if alt_charged_ids
                else (
                    alt_pokemon.charged_move_ids[0]
                    if alt_pokemon.charged_move_ids
                    else None
                )
            )
            alt_secondary_id = (
                alt_charged_ids[1] if len(alt_charged_ids) > 1 else None
            )
            if alt_fast_id and alt_primary_id:
                alt_fast_m = gm_r.get_move(alt_fast_id)
                alt_charged_m = gm_r.get_move(alt_primary_id)
                alt_charged2_m = (
                    gm_r.get_move(alt_secondary_id) if alt_secondary_id else None
                )
                if alt_fast_m and alt_charged_m:
                    alt_stats = gm_r.effective_stats(alt_pokemon)
                    alt_fast = FastMove(
                        name=alt_fast_m.name,
                        damage=0,
                        energy_gain=alt_fast_m.energy_gain,
                        power=alt_fast_m.power,
                        move_type=alt_fast_m.move_type,
                        turns=alt_fast_m.turns or 1,
                    )
                    alt_charged = _build_charged(alt_charged_m)
                    alt_charged2 = (
                        _build_charged(alt_charged2_m) if alt_charged2_m else None
                    )
                    # The alt build's own trigger flags — for
                    # Aegislash Blade, this is the reverse-trigger
                    # back to Shield on defensive shield use. Pull
                    # both directions from the spec so a hypothetical
                    # third dynamic-form species can configure either.
                    alt_dform = _DYNAMIC_FORM_SPECS.get(alt_species_id)
                    alt_fc_on_charged = (
                        alt_dform[1] if alt_dform else False
                    )
                    alt_fc_on_shield = (
                        alt_dform[2] if alt_dform else False
                    )
                    alt_build = CombatantBuild(
                        species=display_name,
                        max_hp=alt_stats.hp,
                        fast=alt_fast,
                        charged=alt_charged,
                        attack=alt_stats.attack,
                        defense=alt_stats.defense,
                        types=alt_pokemon.types,
                        charged2=alt_charged2,
                        form_id=1 if alt_pokemon.is_shadow else 0,
                        dex_id=alt_pokemon.dex,
                        form_change_to_alt_on_charged=alt_fc_on_charged,
                        form_change_to_alt_on_shield_use=alt_fc_on_shield,
                    )

    return CombatantBuild(
        species=display_name,
        max_hp=stats.hp,
        fast=fast,
        charged=charged,
        attack=stats.attack,
        defense=stats.defense,
        types=pokemon.types,
        charged2=charged2,
        form_id=form_id,
        dex_id=pokemon.dex,
        alt_form=alt_build,
        form_change_to_alt_on_charged=fc_on_charged,
        form_change_to_alt_on_shield_use=fc_on_shield,
    )


def build_registry_pvpoke_top(
    top_n: int = 30,
    *,
    gm: GamemasterRegistry | None = None,
    dex: PokedexRegistry | None = None,
) -> list[tuple[str, str, CombatantBuild]]:
    """Materialize the top-N PvPoke-ranked species in GL.

    Returns a list of ``(label_ja, species_id, build)`` triples in
    PvPoke ranking order so the caller can enumerate ordered 3-combos
    without re-doing the lookup. Skips entries the gamemaster cannot
    resolve.

    Unlike :func:`build_registry_for_meta`, this does NOT consult the
    Taiman Party usage feed — the source of truth is purely PvPoke's
    overall ranking. Useful when the operator wants the simulator's
    own opinion on what's strongest, including niche picks the JP meta
    underuses (the user's "마스카나 case" concern).
    """

    gm_r = gm or load_default_gamemaster()
    dex_r = dex or load_default_registry()

    result: list[tuple[str, str, CombatantBuild]] = []
    # Walk a deeper slice so dedup-filtered entries (Aegislash Blade is
    # a runtime state, not a team-pick) don't shrink the result below
    # the requested top_n.
    scan_top_n = top_n + 4 * max(1, len(_RUNTIME_ONLY_SPECIES_IDS))
    for ranking in gm_r.rankings[:scan_top_n]:
        if ranking.species_id in _RUNTIME_ONLY_SPECIES_IDS:
            continue
        build = materialize_build_from_ranking(
            ranking.species_id, gm=gm_r, dex=dex_r
        )
        if build is None:
            continue
        # Build the JP display label: ja name + shadow marker baked
        # into the species_ja so the dedup logic in the candidate
        # enumerator treats base and shadow as distinct picks.
        if build.form_id:
            label_ja = f"{build.species}#{build.form_id}"
        else:
            label_ja = build.species
        result.append((label_ja, ranking.species_id, build))
        if len(result) >= top_n:
            break
    return result


def materialize_build(
    species_ja: str,
    *,
    form_id: int = 0,
    gm: GamemasterRegistry | None = None,
    dex: PokedexRegistry | None = None,
    fast_move_id: str | None = None,
    charged_move_id: str | None = None,
    charged2_move_id: str | None = None,
) -> CombatantBuild | None:
    """Build a :class:`CombatantBuild` for a Taiman Party species reference.

    Parameters
    ----------
    species_ja:
        Upstream Japanese display name (with optional parenthesized
        regional variant suffix — e.g. ``ファイヤー(ガラル)``).
    form_id:
        Taiman Party's form discriminator. ``0`` = base form; non-zero
        means shadow when paired with no inline regional variant.
    gm, dex:
        Optional registry overrides for testing.
    fast_move_id, charged_move_id:
        Optional explicit move ids (PvPoke uppercase ids). Default
        policy: pick the first entry from PvPoke's per-species move
        list (which PvPoke roughly orders by competitive merit).

    Returns
    -------
    CombatantBuild | None
        ``None`` when the species cannot be resolved (unknown to PvPoke /
        no movesets / unknown move ids). Callers should check for ``None``
        and either skip the team or surface a diagnostic.
    """

    gm_r = gm or load_default_gamemaster()
    dex_r = dex or load_default_registry()

    pokemon = gm_r.resolve_build(
        dex_id=None,
        species_ja=species_ja,
        form_id=form_id,
        dex_registry=dex_r,
    )
    if pokemon is None:
        return None

    fast_id = fast_move_id or (
        pokemon.fast_move_ids[0] if pokemon.fast_move_ids else None
    )
    charged_id = charged_move_id or (
        pokemon.charged_move_ids[0] if pokemon.charged_move_ids else None
    )
    # Second charged move: prefer the caller's explicit override
    # (ladder-dominant pick), then fall back to PvPoke's listed pair.
    charged2_id: str | None = charged2_move_id
    if (
        charged2_id is None
        and charged_id is not None
        and len(pokemon.charged_move_ids) >= 2
    ):
        for mid in pokemon.charged_move_ids:
            if mid != charged_id:
                charged2_id = mid
                break

    if not fast_id or not charged_id:
        return None

    fast_move = gm_r.get_move(fast_id)
    charged_move = gm_r.get_move(charged_id)
    if fast_move is None or charged_move is None:
        return None
    charged2_move = gm_r.get_move(charged2_id) if charged2_id else None

    stats = gm_r.effective_stats(pokemon)

    fast = FastMove(
        name=fast_move.name,
        damage=0,
        energy_gain=fast_move.energy_gain,
        power=fast_move.power,
        move_type=fast_move.move_type,
        turns=fast_move.turns or 1,
    )

    def _build_charged(m) -> ChargedMove:
        # Charged moves clamp the energy cost into [1, ENERGY_CAP] per
        # the simulator's invariant — PvPoke uses positive integers,
        # but defend in case of a corrupt entry.
        energy_cost = max(1, min(100, m.energy))
        buffs = m.buffs if m.buffs is not None else (0, 0)
        return ChargedMove(
            name=m.name,
            energy_cost=energy_cost,
            damage=0,
            power=m.power,
            move_type=m.move_type,
            buffs=buffs,
            buff_chance=m.buff_apply_chance,
            buff_target=m.buff_target,
        )

    charged = _build_charged(charged_move)
    charged2 = _build_charged(charged2_move) if charged2_move else None

    return CombatantBuild(
        species=species_ja,
        max_hp=stats.hp,
        fast=fast,
        charged=charged,
        attack=stats.attack,
        defense=stats.defense,
        types=pokemon.types,
        charged2=charged2,
        form_id=form_id,
        dex_id=pokemon.dex,
    )


def registry_key(species_ja: str, form_id: int) -> str:
    """Form-aware registry key.

    ``form_id == 0`` keeps the bare ``species_ja`` so legacy lookups and
    tests that don't carry form information still hit the base entry.
    ``form_id != 0`` appends ``#<id>`` so shadow / regional variants do
    NOT collide with the base form in the registry.
    """
    if form_id:
        return f"{species_ja}#{form_id}"
    return species_ja


def _pick_ladder_moveset(
    usage: PokemonUsage,
    *,
    gm: GamemasterRegistry,
) -> tuple[str | None, list[str]]:
    """Pick the ladder-dominant moveset for ``usage`` if data is present.

    Translates the Japanese move names from Taiman Party's ``waza1`` /
    ``waza2`` lists into PvPoke ``moveId``s via
    ``gm.move_ja_to_pvpoke``. Returns ``(fast_id, [charged_ids…])``:

    - Fast: the single most-used fast move (``waza1[0]``) translated to
      PvPoke id. ``None`` when no move translates.
    - Charged: up to two charged moves, taken in descending usage
      order, deduplicated.

    Untranslatable moves (PokeAPI ↔ PvPoke mapping gap) are skipped —
    the caller falls back to PvPoke-recommended ids for those slots.
    """
    fast_id: str | None = None
    for mv in usage.fast_moves:
        pid = gm.move_ja_to_pvpoke.get(mv.name)
        if pid and gm.get_move(pid):
            fast_id = pid
            break

    charged_ids: list[str] = []
    for mv in usage.charged_moves:
        pid = gm.move_ja_to_pvpoke.get(mv.name)
        if pid and gm.get_move(pid) and pid not in charged_ids:
            charged_ids.append(pid)
        if len(charged_ids) == 2:
            break

    return fast_id, charged_ids


def build_registry_for_meta(
    meta: MetaSnapshot,
    *,
    gm: GamemasterRegistry | None = None,
    dex: PokedexRegistry | None = None,
    moveset_source: str = "ladder",
) -> dict[str, CombatantBuild]:
    """Materialize every species + form appearing in ``meta``.

    Iterates ``meta.pokemon_usage`` (all 50 cards) plus
    ``meta.team_usage[*].members`` paired with ``member_forms`` so a
    species that appears as both its base and shadow form gets two
    distinct registry entries keyed by :func:`registry_key`.

    Returns
    -------
    dict[str, CombatantBuild]
        Keyed by ``registry_key(species_ja, form_id)`` — i.e. the bare
        ``species_ja`` for form 0 and ``species_ja#<form_id>`` for any
        non-zero form. Compatible with the legacy lookup pattern used
        by :func:`gbl_hacker.score.expected_win_rate.materialize_opponent_team`,
        which falls back to the bare key when the form-aware one misses.
    """

    gm_r = gm or load_default_gamemaster()
    dex_r = dex or load_default_registry()

    registry: dict[str, CombatantBuild] = {}

    # Index ladder usage by species so the form-keyed lookup below can
    # find the matching PokemonUsage row when forms vary.
    by_species: dict[tuple[str, int], PokemonUsage] = {}
    for entry in meta.pokemon_usage:
        by_species[(entry.species, entry.form_id or 0)] = entry

    def _add(species_ja: str, form_id: int) -> None:
        key = registry_key(species_ja, form_id)
        if key in registry:
            return
        fast_override: str | None = None
        charged_override: str | None = None
        charged2_override: str | None = None
        if moveset_source == "ladder":
            usage = by_species.get((species_ja, form_id)) or by_species.get(
                (species_ja, 0)
            )
            if usage is not None:
                fast_override, charged_list = _pick_ladder_moveset(usage, gm=gm_r)
                if charged_list:
                    charged_override = charged_list[0]
                if len(charged_list) > 1:
                    charged2_override = charged_list[1]
        build = materialize_build(
            species_ja,
            form_id=form_id,
            gm=gm_r,
            dex=dex_r,
            fast_move_id=fast_override,
            charged_move_id=charged_override,
            charged2_move_id=charged2_override,
        )
        if build is not None:
            registry[key] = build

    for entry in meta.pokemon_usage:
        _add(entry.species, entry.form_id or 0)
    for team in meta.team_usage:
        for member, form in zip(team.members, team.member_forms):
            _add(member, form)

    return registry


PVPOKE_SYNTHETIC_CAVEAT: str = (
    "Synthetic opponent set — lineups assembled from PvPoke's GL "
    "top-N rankings (each top-ranked species used as lead, paired with "
    "two type-diverse PvPoke top-K teammates). NOT a measurement of "
    "actual Taiman Party meta usage; treat as a ladder-shape "
    "approximation that includes niche threats (rock leads, etc.) "
    "absent from the Taiman 30-team feed."
)


def _build_charged_move(m) -> ChargedMove:
    """Translate a ``GamemasterMove`` charged-spec into a ``ChargedMove``.

    Clamps energy cost into ``[1, 100]`` per the simulator's invariant,
    pulls in buff fields (PvPoke ``buffs`` / ``buffApplyChance`` /
    ``buffTarget``). Shared between :func:`materialize_build` and
    :func:`materialize_build_from_ranking` (the dynamic-form alt build
    path needs the same constructor).
    """
    energy_cost = max(1, min(100, m.energy))
    buffs = m.buffs if m.buffs is not None else (0, 0)
    return ChargedMove(
        name=m.name,
        energy_cost=energy_cost,
        damage=0,
        power=m.power,
        move_type=m.move_type,
        buffs=buffs,
        buff_chance=m.buff_apply_chance,
        buff_target=m.buff_target,
    )


def _weakness_types_of(
    types: tuple[str, ...], gm: GamemasterRegistry
) -> set[str]:
    """Return attack types that super-effective hit a Pokémon with ``types``.

    Iterates the defender's types, collecting the ``weaknesses`` set
    from the type chart for each. Doesn't bother de-multiplying when a
    type is weak on both slots (e.g. ground+water double-weakness to
    grass) — set semantics already de-dupe, which is what we want for
    teammate-selection coverage.
    """
    weak: set[str] = set()
    for t in types:
        chart = gm.type_chart.get(t.lower())
        if chart:
            weak.update(s.lower() for s in chart.get("weaknesses", ()))
    return weak


def _offensive_types_of(build: CombatantBuild) -> set[str]:
    """The set of move types this build can deal damage with."""
    types: set[str] = set()
    if build.fast.move_type:
        types.add(build.fast.move_type.lower())
    if build.charged.move_type:
        types.add(build.charged.move_type.lower())
    if build.charged2 and build.charged2.move_type:
        types.add(build.charged2.move_type.lower())
    return types


def _coverage_score(
    candidate: CombatantBuild,
    *,
    target_weaknesses: set[str],
    gm: GamemasterRegistry,
) -> int:
    """How well ``candidate`` covers ``target_weaknesses``.

    Higher is better. Score components:

    - +3 per offensive move type that super-effective-hits the
      weakness types (i.e. attack-side coverage — the candidate can
      punish whatever exploited the lead).
    - +1 if the candidate's own types include something the lead's
      weakness-attack types are themselves resisted by (defense-side
      coverage — the candidate eats the same attacks the lead can't).

    The asymmetric weight (3 vs 1) favors attack coverage because in
    GBL the safe-swap / closer's job is primarily to *answer* the
    threat, not just survive it.
    """
    score = 0
    offensive = _offensive_types_of(candidate)
    score += 3 * len(offensive & target_weaknesses)

    # Defense coverage: candidate's defensive types resist any of the
    # weakness *attack* types? (Resist list for the candidate's types
    # vs the weakness type identifiers.)
    for ct in candidate.types:
        chart = gm.type_chart.get(ct.lower())
        if not chart:
            continue
        resist = set(chart.get("resistances", ())) | set(
            chart.get("immunities", ())
        )
        if resist & target_weaknesses:
            score += 1
    return score


def _select_complementary_teammate(
    *,
    lead: CombatantBuild,
    paired_types: tuple[str, ...],
    pool: list[tuple[str, CombatantBuild]],
    exclude_dex_ids: set[int],
    gm: GamemasterRegistry,
) -> CombatantBuild | None:
    """Pick the pool entry that best covers ``paired_types`` weaknesses.

    Used for both the closer (paired_types = lead's types) and the
    safe-swap (paired_types = lead + closer's types). The candidate
    that maximizes ``_coverage_score`` wins; ties fall back to PvPoke
    rank order (the iteration order of ``pool``).

    GBL rule: same-dex Pokémon (including shadow + base) cannot share
    a team. We exclude by ``dex_id`` rather than by species name so
    Forretress + Shadow Forretress, base Quagsire + Shadow Quagsire,
    Sandslash + Alolan Sandslash, etc. are all correctly rejected.
    """

    target_weaknesses = _weakness_types_of(paired_types, gm)
    if not target_weaknesses:
        for _label, build in pool:
            if build.dex_id not in exclude_dex_ids:
                return build
        return None

    best: CombatantBuild | None = None
    best_score = -1
    for _label, build in pool:
        if build.dex_id in exclude_dex_ids:
            continue
        s = _coverage_score(build, target_weaknesses=target_weaknesses, gm=gm)
        if s > best_score:
            best_score = s
            best = build
    if best is None:
        for _label, build in pool:
            if build.dex_id not in exclude_dex_ids:
                return build
    return best


def _synthesize_team(
    lead: CombatantBuild,
    pool: list[tuple[str, CombatantBuild]],
    gm: GamemasterRegistry,
) -> list[CombatantBuild]:
    """Build a 3-Pokémon team around ``lead`` using the GBL team-building
    convention the user pinned:

    1. **Closer**: covers the lead's type weaknesses (offensive coverage).
    2. **Safe swap**: covers the union of the lead + closer's weaknesses
       — the generalist that handles whatever the other two cannot.

    Returns ``[lead, safe_swap, closer]`` in lead → safe_swap → closer
    slot order so the resulting :class:`TeamUsage` reads naturally.

    Same-dex uniqueness is enforced: regardless of base/shadow/regional
    variant, no two slots may share a dex id.
    """

    used_dex = {lead.dex_id}
    closer = _select_complementary_teammate(
        lead=lead,
        paired_types=lead.types,
        pool=pool,
        exclude_dex_ids=used_dex,
        gm=gm,
    )
    if closer is None:
        return [lead]
    used_dex.add(closer.dex_id)

    safe_swap = _select_complementary_teammate(
        lead=lead,
        paired_types=tuple(lead.types) + tuple(closer.types),
        pool=pool,
        exclude_dex_ids=used_dex,
        gm=gm,
    )
    if safe_swap is None:
        return [lead, closer]
    return [lead, safe_swap, closer]


def synthesize_pvpoke_opponent_meta(
    *,
    top_n: int = 30,
    pool_top_k: int | None = None,
    fetched_at: datetime | None = None,
    gm: GamemasterRegistry | None = None,
    dex: PokedexRegistry | None = None,
    meta_snapshot: MetaSnapshot | None = None,
    meta_lead_top_n: int = 15,
) -> tuple[MetaSnapshot, dict[str, CombatantBuild]]:
    """Build a synthetic ``MetaSnapshot`` from PvPoke's GL ranking.

    Each of the top-``top_n`` PvPoke-ranked species is used as the lead
    of one synthetic opponent team; ``_diverse_teammates`` picks two
    type-diverse companions from the top-``pool_top_k`` candidates.
    The resulting ``MetaSnapshot`` plugs directly into the scoring
    layer (``expected_win_rate`` / ``worst_case_robustness`` /
    ``meta_coverage``) so the engine can be evaluated against a
    ladder-shaped opponent distribution instead of the Taiman 30-team
    popularity feed.

    Returns ``(meta, build_registry)`` — the registry contains both
    the lead species and every selected teammate so the scoring layer
    can materialize each opponent without re-touching the gamemaster.

    Parameters
    ----------
    top_n:
        Number of synthetic opponent teams to generate (one per
        PvPoke-ranked lead). Default 30.
    pool_top_k:
        Pool depth from which teammates are picked. Default 20.
    fetched_at:
        Stamped onto the snapshot's ``fetched_at`` field. Defaults to
        UTC now.
    """

    gm_r = gm or load_default_gamemaster()
    dex_r = dex or load_default_registry()
    when = fetched_at or datetime.now(tz=timezone.utc)

    # Pool used to draw teammates from (PvPoke overall ranking — every
    # role). Must be deeper than top_n so leads near the bottom still
    # have plenty of teammate options.
    effective_pool_k = pool_top_k if pool_top_k is not None else max(
        40, top_n * 2
    )
    teammate_pool: list[tuple[str, CombatantBuild]] = []
    for _label, _species_id, build in build_registry_pvpoke_top(
        top_n=effective_pool_k, gm=gm_r, dex=dex_r
    ):
        teammate_pool.append((_label, build))

    # Lead pool: PvPoke's *leads* category ranking — proxies the
    # ladder's actual lead-frequency distribution. Falls back to the
    # overall ranking when the leads list is missing.
    #
    # **Sample-bias guard:** PvPoke's leads ranking optimizes for "is
    # this species good as a lead", which under-represents species the
    # JP meta runs constantly (e.g. Quagsire — PvPoke overall #2,
    # leads-list absent because Mud Shot's dpe is mediocre at the
    # opening exchange). When a Taiman snapshot is provided, blend its
    # top-N most-used species into the lead pool so the evaluator sees
    # the matchups the ladder actually runs — not just the textbook-
    # optimal lead distribution. Dex-uniqueness dedup avoids double-
    # counting overlap (e.g. tinkaton appears in both).
    lead_source = gm_r.leads_rankings if gm_r.leads_rankings else gm_r.rankings
    lead_pool: list[tuple[str, CombatantBuild]] = []
    seen_dex: set[int] = set()

    # Pass A: Taiman meta's top-N most-used species FIRST so the
    # synthetic lead distribution reflects what the ladder actually
    # runs (Quagsire-style species that PvPoke leads under-ranks). We
    # materialize with ladder-mode movesets so the opponent uses what
    # operators actually play.
    if meta_snapshot is not None and meta_lead_top_n > 0:
        meta_reg = build_registry_for_meta(
            meta_snapshot, gm=gm_r, dex=dex_r, moveset_source="ladder"
        )
        for entry in meta_snapshot.pokemon_usage[:meta_lead_top_n]:
            key = registry_key(entry.species, entry.form_id or 0)
            build = meta_reg.get(key)
            if build is None and (entry.form_id or 0):
                build = meta_reg.get(entry.species)
            if build is None:
                continue
            if build.dex_id and build.dex_id in seen_dex:
                continue
            seen_dex.add(build.dex_id)
            label = (
                f"{build.species}#{build.form_id}"
                if build.form_id
                else build.species
            )
            lead_pool.append((label, build))

    # Pass B: PvPoke leads ranking. Adds the textbook-strong leads that
    # the Taiman top-N doesn't already cover (e.g. Corviknight,
    # Primeape) so the opponent set has both popular and powerful
    # leads.
    for ranking in lead_source[: max(top_n * 2, 40)]:
        build = materialize_build_from_ranking(
            ranking.species_id, gm=gm_r, dex=dex_r
        )
        if build is None:
            continue
        if build.dex_id and build.dex_id in seen_dex:
            continue
        seen_dex.add(build.dex_id)
        label = (
            f"{build.species}#{build.form_id}" if build.form_id else build.species
        )
        lead_pool.append((label, build))

    if len(teammate_pool) < 3:
        raise RuntimeError(
            f"PvPoke ranking yielded {len(teammate_pool)} materializable "
            "entries — cannot synthesize a 3-Pokémon opponent set."
        )

    team_usage_rows: list[TeamUsage] = []
    pokemon_usage_rows: list[PokemonUsage] = []
    registry: dict[str, CombatantBuild] = {}
    seen_lead_species: set[str] = set()
    pokemon_seen: set[str] = set()

    leads = lead_pool[:top_n]

    for rank, (lead_label, lead_build) in enumerate(leads, start=1):
        if lead_build.species in seen_lead_species:
            continue
        seen_lead_species.add(lead_build.species)
        team_builds = _synthesize_team(lead_build, teammate_pool, gm_r)
        if len(team_builds) < 3:
            continue

        members = tuple(b.species for b in team_builds)
        member_forms = tuple(b.form_id for b in team_builds)

        # Equal usage weight per synthetic team — ladder-shape baseline.
        team_usage_rows.append(
            TeamUsage(
                members=members,  # type: ignore[arg-type]
                usage_pct=100.0 / top_n,
                rank=rank,
                usage_count=1,
                member_forms=member_forms,  # type: ignore[arg-type]
            )
        )

        for build in team_builds:
            key = registry_key(build.species, build.form_id)
            if key not in registry:
                registry[key] = build
            # Also key without the form suffix for legacy lookups.
            registry.setdefault(build.species, build)

            if build.species not in pokemon_seen:
                pokemon_seen.add(build.species)
                pokemon_usage_rows.append(
                    PokemonUsage(
                        species=build.species,
                        usage_pct=100.0 / max(1, top_n * 3),
                        rank=len(pokemon_usage_rows) + 1,
                        usage_count=1,
                        dex_id=None,
                        form_id=build.form_id,
                    )
                )

    if not team_usage_rows:
        raise RuntimeError(
            "Failed to synthesize any opponent teams from PvPoke rankings."
        )

    snapshot = MetaSnapshot(
        league=GREAT_LEAGUE_LABEL,
        rating_bracket="pvpoke_synthetic",
        fetched_at=when,
        source_url="local://pvpoke-ranking-synthetic",
        source_caveat=PVPOKE_SYNTHETIC_CAVEAT + " " + TAIMAN_SOURCE_CAVEAT,
        pokemon_usage=tuple(pokemon_usage_rows),
        team_usage=tuple(team_usage_rows),
        season=0,
        league_id=0,
    )
    return snapshot, registry


__all__ = [
    "PVPOKE_SYNTHETIC_CAVEAT",
    "build_registry_for_meta",
    "build_registry_pvpoke_top",
    "materialize_build",
    "materialize_build_from_ranking",
    "registry_key",
    "synthesize_pvpoke_opponent_meta",
]
