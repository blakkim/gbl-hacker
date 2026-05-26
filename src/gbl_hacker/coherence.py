"""Tier-1 fact layer: assert a materialized build is internally coherent.

A :class:`CombatantBuild` couples a species (its typing) with a concrete
moveset. Several construction paths can silently desync the two — most
memorably the マッギョ *chimera*: a dex-level override swapped the typing to
Galarian Stunfisk (ground/steel) while the ladder moveset picker stapled on
base Stunfisk's electric moves (Thunder Shock / Discharge), which Galarian
cannot learn. The build was then displayed under the base species' name. All
three layers — display name, typing, moveset — disagreed, and the corrupted
build leaked into opponent scoring.

That class of bug is a *deterministic fact* check, not a judgement call, so it
belongs in a code assertion rather than an after-the-fact review. Given the
gamemaster as ground truth, :func:`validate_build_coherence` asserts:

1. the build's types equal the resolved species' types, and
2. every move on the build is in that species' learnable move pool.

A build that passes both is internally consistent with PvPoke's gamemaster.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

from gbl_hacker.dex import PokedexRegistry, load_default_registry
from gbl_hacker.gamemaster import GamemasterRegistry, load_default_gamemaster
from gbl_hacker.simulator import CombatantBuild


def validate_build_coherence(
    build: CombatantBuild,
    *,
    gm: GamemasterRegistry | None = None,
    dex: PokedexRegistry | None = None,
) -> list[str]:
    """Return a list of coherence violations for ``build`` (empty == coherent).

    Resolves ``build``'s ``(species, form_id)`` back through the gamemaster
    and compares the result against the build's own typing and moveset. Each
    returned string is a human-readable violation; an empty list means the
    build is consistent with the gamemaster.

    The check is move-*name* based (the build's move objects carry names, not
    ids); PvPoke move names are unique per id, so this is exact in practice.

    A build carries its JA display name + ``form_id``, but a JA name can be
    shared across forms (ポワルン → every Castform weather form; マッギョ → base
    and Galarian). So instead of trusting a single name resolution, this
    searches every gamemaster form sharing the build's dex (and shadow status)
    and asks: does *some* real form match the build's typing **and** learn all
    its moves? If yes, the build is a coherent real Pokémon. If a typed form
    exists but cannot learn the moves — the chimera — the moves are flagged.
    If no form matches the typing at all, the typing is flagged.
    """

    gm_r = gm or load_default_gamemaster()
    dex_r = dex or load_default_registry()
    tag = f"{build.species}#{build.form_id}"

    dex_id = build.dex_id
    if not dex_id:
        pk0 = gm_r.resolve_build(
            dex_id=None,
            species_ja=build.species,
            form_id=build.form_id,
            dex_registry=dex_r,
        )
        if pk0 is None:
            return [f"{tag}: species does not resolve in the gamemaster"]
        dex_id = pk0.dex

    # Search every gamemaster form for this dex — across shadow and regional
    # variants alike. ``form_id`` is overloaded upstream (shadow for most
    # species, size for Gourgeist, the regional marker for サニーゴ(ガラル)),
    # and shadow/non-shadow share typing and move pool, so the shadow flag is
    # irrelevant to *coherence*: we only ask whether some real form explains
    # the build's typing + moveset.
    forms = [pk for pk in gm_r.pokemon_by_species_id.values() if pk.dex == dex_id]
    if not forms:
        return [f"{tag}: no gamemaster form for dex {dex_id}"]

    # 1. Typing coherence — is there any real form with this typing?
    typed = [pk for pk in forms if tuple(pk.types) == tuple(build.types)]
    if not typed:
        opts = sorted({f"{pk.species_id}{tuple(pk.types)}" for pk in forms})
        return [
            f"{tag}: typing {tuple(build.types)} matches no dex-{dex_id} form ({opts})"
        ]

    # 2. Move-pool coherence — does a same-typed form learn every move?
    build_charged = [cm.name for cm in build.charged_moves]
    closest: list[str] | None = None
    for pk in typed:
        learnable_fast = {m.name for mid in pk.fast_move_ids if (m := gm_r.get_move(mid))}
        learnable_charged = {
            m.name for mid in pk.charged_move_ids if (m := gm_r.get_move(mid))
        }
        problems: list[str] = []
        if build.fast.name not in learnable_fast:
            problems.append(
                f"{tag}: fast move {build.fast.name!r} not in {pk.species_id} "
                f"pool {sorted(learnable_fast)}"
            )
        for cm in build_charged:
            if cm not in learnable_charged:
                problems.append(
                    f"{tag}: charged move {cm!r} not in {pk.species_id} "
                    f"pool {sorted(learnable_charged)}"
                )
        if not problems:
            return []  # a real form matches both typing and full moveset
        if closest is None or len(problems) < len(closest):
            closest = problems

    return closest or []


def validate_builds(
    builds: Iterable[CombatantBuild] | Mapping[str, CombatantBuild],
    *,
    gm: GamemasterRegistry | None = None,
    dex: PokedexRegistry | None = None,
) -> dict[str, list[str]]:
    """Validate many builds; return ``{tag: [violations]}`` for offenders only.

    Accepts either an iterable of builds or a registry mapping (its values are
    validated). Builds with no violations are omitted from the result, so an
    empty dict means the whole batch is coherent.
    """

    gm_r = gm or load_default_gamemaster()
    dex_r = dex or load_default_registry()

    values = builds.values() if isinstance(builds, Mapping) else builds
    offenders: dict[str, list[str]] = {}
    for build in values:
        violations = validate_build_coherence(build, gm=gm_r, dex=dex_r)
        if violations:
            offenders[f"{build.species}#{build.form_id}"] = violations
    return offenders


__all__ = ["validate_build_coherence", "validate_builds"]
