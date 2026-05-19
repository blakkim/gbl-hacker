"""Canonical human-readable renderer for a ``MetaSnapshot``.

Sub-AC 6 contract: every meta_snapshot rendering must surface the
Taiman Party report-density caveat — not behind a ``--verbose`` flag,
not as a footnote a user can grep out, but as a structurally guaranteed
section of the output.

This module is the single source of truth for that obligation. The CLI
``gblh refresh`` and ``gblh show`` subcommands both delegate to
:func:`render_meta_snapshot`; no other code path in the engine is allowed
to format a snapshot for display. (Persistence is a different surface —
the JSON payload carries ``source_caveat`` as a required field, and
:mod:`gbl_hacker.persist.snapshot` round-trips it. Renderer guarantees
apply to *display*; persistence guarantees apply to *storage*.)

How the no-suppression invariant is enforced
--------------------------------------------

1. **No suppression parameter.** :func:`render_meta_snapshot` accepts a
   ``MetaSnapshot`` and an output stream — that's it. There is no
   ``quiet=``, ``include_caveat=``, ``no_caveat=`` or environment-
   variable override. The renderer is the only sanctioned path to
   stdout for a snapshot, and it has nothing to toggle off.

2. **Header + footer redundancy.** The caveat block is emitted at the
   top of the rendering, framed by a banner so it is hard to miss. The
   one-line footer summary also re-includes the caveat string, so a
   user grepping a single line of output still sees the warning.

3. **Defence-in-depth empty-caveat check.** The
   :class:`~gbl_hacker.parse.taiman.MetaSnapshot` dataclass already
   raises ``ValueError`` for an empty ``source_caveat`` — but if a
   hand-constructed snapshot ever slipped past that, the renderer
   raises :class:`MissingCaveatError` rather than silently producing a
   caveat-less rendering.

The module deliberately writes plain text (no colour codes, no
ANSI escapes). This keeps the output friendly to:

* re-direction into a file (``gblh show > snapshot.txt``)
* grepping the long-loop validation log
* future copy-pasting into a markdown report

ASCII-only framing characters are used for the same reason.
"""

from __future__ import annotations

import re
from typing import TextIO

from gbl_hacker.dex import Language, PokedexRegistry
from gbl_hacker.parse.taiman import MetaSnapshot, PokemonUsage, TeamUsage

# Small form-suffix translation table for the parenthesized regional
# variants Taiman Party serves inline (e.g. "ファイヤー(ガラル)"). v0.1 covers
# the variants that actually appear in current GL meta data.
_FORM_SUFFIX_TRANSLATIONS: dict[str, dict[str, str]] = {
    "ガラル": {"ko": "갈라르", "en": "Galarian"},
    "アローラ": {"ko": "알로라", "en": "Alolan"},
    "ヒスイ": {"ko": "히스이", "en": "Hisuian"},
    "パルデア": {"ko": "파르데아", "en": "Paldean"},
    "メガ": {"ko": "메가", "en": "Mega"},
}

# Suffix appended to localized species when the upstream marks a shadow
# form via ``form_id != 0`` and no parenthesized variant is present.
_SHADOW_SUFFIX: dict[str, str] = {
    "ja": "(シャドウ)",
    "ko": "(섀도우)",
    "en": "(Shadow)",
}

_PARENS_RE = re.compile(r"^(.*?)\s*\(([^)]+)\)\s*$")


CAVEAT_HEADER_LABEL: str = "DATA HONESTY — Taiman Party caveat"
"""Banner label written above the caveat block in every rendering.

Pinned as a public constant so tests can assert the label is present in
every output surface. Changing the label is a deliberate
documentation-level decision — downstream tooling that greps output for
``DATA HONESTY`` should be updated in lockstep.
"""

_BANNER_WIDTH: int = 72
"""Width of the ASCII frame around the caveat block.

72 columns matches the default terminal width pyargparse assumes; the
caveat block stays readable in narrow terminals without truncation.
"""

_HORIZONTAL_RULE: str = "=" * _BANNER_WIDTH


class MissingCaveatError(ValueError):
    """Raised when a snapshot reaches the renderer with no caveat string.

    The :class:`~gbl_hacker.parse.taiman.MetaSnapshot` dataclass already
    forbids an empty ``source_caveat`` in ``__post_init__`` — this
    exception exists for *defence-in-depth* at the rendering layer so
    that a hand-built or partially-mocked snapshot can never produce a
    caveat-less rendering.
    """


# ---------------------------------------------------------------------------
# Caveat block — the part of the rendering that satisfies AC 6
# ---------------------------------------------------------------------------


def format_caveat_block(snapshot: MetaSnapshot) -> str:
    """Render the data-honesty caveat as a framed text block.

    The block opens and closes with a horizontal rule, carries the
    pinned :data:`CAVEAT_HEADER_LABEL` banner, and prints the snapshot's
    ``source_caveat`` verbatim. The framing is intentionally heavy so
    that a casual reader cannot mistake it for a footnote.

    Parameters
    ----------
    snapshot:
        The snapshot whose caveat is being rendered.

    Returns
    -------
    str
        Multi-line text block (no trailing newline).

    Raises
    ------
    MissingCaveatError
        If ``snapshot.source_caveat`` is empty or whitespace-only.
    """

    caveat = snapshot.source_caveat or ""
    if not caveat.strip():
        raise MissingCaveatError(
            "Refusing to render a snapshot with an empty source_caveat — "
            "violates the data_honesty evaluation principle (AC 6)."
        )

    lines = [
        _HORIZONTAL_RULE,
        CAVEAT_HEADER_LABEL,
        _HORIZONTAL_RULE,
    ]
    lines.extend(_wrap_paragraph(caveat, width=_BANNER_WIDTH))
    lines.append(_HORIZONTAL_RULE)
    return "\n".join(lines)


def format_snapshot_header(snapshot: MetaSnapshot) -> str:
    """Render the snapshot's identifying metadata as a header block.

    Includes league, rating bracket, fetch timestamp, source URL, and
    counts of Pokémon / team rows. Does *not* include the caveat — that
    is the responsibility of :func:`format_caveat_block`, which is
    always concatenated above this header by
    :func:`render_meta_snapshot`.
    """

    return (
        f"league:         {snapshot.league}\n"
        f"rating_bracket: {snapshot.rating_bracket}\n"
        f"fetched_at:     {snapshot.fetched_at.isoformat()}\n"
        f"source_url:     {snapshot.source_url}\n"
        f"pokemon_rows:   {len(snapshot.pokemon_usage)}\n"
        f"team_rows:      {len(snapshot.team_usage)}"
    )


# ---------------------------------------------------------------------------
# Usage-table renderers
# ---------------------------------------------------------------------------


def _localize_species(
    species_ja: str,
    *,
    dex_id: int | None,
    form_id: int | None,
    registry: PokedexRegistry | None,
    lang: Language,
) -> str:
    """Render a species name for display in ``lang``.

    Handles three upstream quirks:

    1. Some species names already carry a parenthesized regional variant
       suffix (e.g. ``ファイヤー(ガラル)``). We strip the parenthesized part,
       localize the base name, and re-attach a translated suffix.
    2. ``form_id != 0`` with no parenthesized suffix is the upstream's
       shadow encoding. We append a language-appropriate ``(Shadow)``
       marker after localizing the base name.
    3. When the registry has no entry for a species (e.g. very newly
       added species or hand-edited fixture data), we fall back to the
       raw Japanese string so the renderer never silently drops a row.
    """

    # Step 1: peel off any parenthesized variant suffix.
    base = species_ja
    variant_suffix_ja: str | None = None
    m = _PARENS_RE.match(species_ja)
    if m:
        base = m.group(1).strip()
        variant_suffix_ja = m.group(2).strip()

    # Step 2: localize the base name.
    if registry is None:
        localized = base
    else:
        # When a variant suffix is present, the dex_id stored on
        # PokemonUsage may belong to the variant entry's master row
        # (still the same national dex number). Either way, look up by
        # ja-hrkt of the BASE name first — the variant inflates the dex
        # only via form_id, not via a new dex number.
        entry = registry.lookup(species_ja=base, dex_id=dex_id)
        if entry is not None:
            localized = entry.localize(lang)
        else:
            localized = base

    # Step 3: re-attach variant suffix (translated when possible).
    if variant_suffix_ja:
        translated = _FORM_SUFFIX_TRANSLATIONS.get(variant_suffix_ja, {}).get(lang)
        suffix_str = translated or variant_suffix_ja
        return f"{localized}({suffix_str})"

    # Step 4: shadow marker for form_id != 0 with no inline variant.
    if form_id is not None and form_id != 0:
        marker = _SHADOW_SUFFIX.get(lang, _SHADOW_SUFFIX["ja"])
        return f"{localized}{marker}"

    return localized


def _format_pokemon_table(
    rows: tuple[PokemonUsage, ...],
    *,
    limit: int,
    lang: Language,
    registry: PokedexRegistry | None,
) -> str:
    """Render the Pokémon usage rows as a fixed-width text table.

    ``limit`` caps the number of rows so the rendering stays readable
    even when the upstream returns 100+ Pokémon. ``lang`` selects the
    display language; ``registry`` is the dex localization table.
    """

    if not rows:
        return "  (no Pokémon usage rows in this snapshot)"

    visible = rows[:limit]
    truncated = len(rows) - len(visible)

    header = f"  {'rank':>5}  {'usage':>7}  {'cnt':>6}  species"
    sep = "  " + "-" * (_BANNER_WIDTH - 2)
    body_lines = [header, sep]
    for entry in visible:
        rank = "—" if entry.rank is None else str(entry.rank)
        cnt = "—" if not entry.usage_count else str(entry.usage_count)
        species_display = _localize_species(
            entry.species,
            dex_id=entry.dex_id,
            form_id=entry.form_id,
            registry=registry,
            lang=lang,
        )
        body_lines.append(
            f"  {rank:>5}  {entry.usage_pct:>6.1f}%  {cnt:>6}  {species_display}"
        )
    if truncated > 0:
        body_lines.append(f"  ... ({truncated} more not shown)")
    return "\n".join(body_lines)


def _format_team_table(
    rows: tuple[TeamUsage, ...],
    *,
    limit: int,
    lang: Language,
    registry: PokedexRegistry | None,
) -> str:
    """Render the team usage rows as a fixed-width text table."""

    if not rows:
        return "  (no team usage rows in this snapshot)"

    visible = rows[:limit]
    truncated = len(rows) - len(visible)

    header = f"  {'rank':>5}  {'usage':>7}  {'cnt':>5}  team"
    sep = "  " + "-" * (_BANNER_WIDTH - 2)
    body_lines = [header, sep]
    for entry in visible:
        rank = "—" if entry.rank is None else str(entry.rank)
        cnt = "—" if not entry.usage_count else str(entry.usage_count)
        member_strs = tuple(
            _localize_species(
                m,
                dex_id=None,
                form_id=f,
                registry=registry,
                lang=lang,
            )
            for m, f in zip(entry.members, entry.member_forms)
        )
        members = " / ".join(member_strs)
        body_lines.append(
            f"  {rank:>5}  {entry.usage_pct:>6.1f}%  {cnt:>5}  {members}"
        )
    if truncated > 0:
        body_lines.append(f"  ... ({truncated} more not shown)")
    return "\n".join(body_lines)


# ---------------------------------------------------------------------------
# The single sanctioned snapshot renderer
# ---------------------------------------------------------------------------


def render_meta_snapshot(
    snapshot: MetaSnapshot,
    *,
    stream: TextIO,
    pokemon_limit: int = 15,
    team_limit: int = 10,
    lang: Language = "ja",
    dex_registry: PokedexRegistry | None = None,
) -> None:
    """Render ``snapshot`` to ``stream`` with the caveat unconditionally surfaced.

    Output layout:

    1. ``DATA HONESTY`` framed caveat block (top of the rendering).
    2. Snapshot identifying metadata (league, bracket, fetched_at, …).
    3. Pokémon usage table (capped at ``pokemon_limit`` rows).
    4. Team usage table (capped at ``team_limit`` rows).
    5. One-line footer summary that *re-includes* the caveat string so
       a user grepping a single line of output still sees the warning.

    Parameters
    ----------
    snapshot:
        The snapshot to render.
    stream:
        Output text stream (e.g. ``sys.stdout`` or a ``io.StringIO``).
    pokemon_limit, team_limit:
        Row caps for the two tables. Defaults chosen so the full
        rendering fits in a standard 24-row terminal without scrolling.
        These are *display caps only* — the snapshot object retains all
        rows for analytical consumption.

    Raises
    ------
    MissingCaveatError
        If ``snapshot.source_caveat`` is empty. The renderer refuses to
        produce caveat-less output (AC 6, data_honesty principle).
    """

    # 1. Caveat — emitted FIRST. Function does not accept a
    #    suppression parameter, so this is structurally unavoidable.
    stream.write(format_caveat_block(snapshot))
    stream.write("\n\n")

    # 2. Identifying metadata.
    stream.write("snapshot:\n")
    stream.write(_indent(format_snapshot_header(snapshot), prefix="  "))
    stream.write("\n\n")

    # 3. Pokémon usage table.
    stream.write(f"pokemon_usage (lang={lang}):\n")
    stream.write(
        _format_pokemon_table(
            snapshot.pokemon_usage,
            limit=pokemon_limit,
            lang=lang,
            registry=dex_registry,
        )
    )
    stream.write("\n\n")

    # 4. Team usage table.
    stream.write(f"team_usage (lang={lang}):\n")
    stream.write(
        _format_team_table(
            snapshot.team_usage,
            limit=team_limit,
            lang=lang,
            registry=dex_registry,
        )
    )
    stream.write("\n\n")

    # 5. Footer summary — caveat re-stated on a single grep-friendly
    #    line. A user piping output through ``| tail -1`` still sees
    #    the warning.
    stream.write(
        f"source_caveat: {snapshot.source_caveat}\n"
    )


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------


def _wrap_paragraph(text: str, *, width: int) -> list[str]:
    """Word-wrap ``text`` to ``width`` columns without breaking words.

    A tiny, dependency-free wrapper. ``textwrap.wrap`` would do the same
    but pulls in a stdlib module the rest of the file does not need; the
    inline implementation keeps the module focused.
    """

    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        addition = len(word) + (1 if current else 0)
        if current and current_len + addition > width:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += addition
    if current:
        lines.append(" ".join(current))
    return lines


def _indent(text: str, *, prefix: str) -> str:
    """Prefix every non-empty line of ``text`` with ``prefix``."""

    return "\n".join(prefix + line if line else line for line in text.splitlines())
