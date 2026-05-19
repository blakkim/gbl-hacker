"""Human-readable renderer for the engine's top-K recommendation list.

Companion to :mod:`gbl_hacker.render.snapshot`. The snapshot renderer
formats the upstream Taiman Party meta; this one formats the *output* of
the scoring + Pareto + top-K pipeline.

Output layout:

1. ``DATA HONESTY`` caveat block — the same banner the snapshot renderer
   emits, so an operator who only sees the recommendation output still
   gets the report-density warning (AC 6 / data_honesty principle).
2. Top-K teams table with three score columns and localized member
   names.
3. One-line trailing caveat for grep-friendly tooling.

The renderer accepts a ``PokedexRegistry`` for localization. When
``None`` it falls back to the raw Japanese species names — useful for
``--lang ja`` and for tests that don't want to depend on the dex dump.
"""

from __future__ import annotations

from typing import Sequence, TextIO

from gbl_hacker.dex import Language, PokedexRegistry
from gbl_hacker.parse.taiman import MetaSnapshot
from gbl_hacker.render.snapshot import (
    CAVEAT_HEADER_LABEL,
    MissingCaveatError,
    format_caveat_block,
)
from gbl_hacker.score.pareto import ScoredTeam

# Form-suffix translation table — mirrors the one in render.snapshot. We
# duplicate the literal here rather than importing the private name so
# the two renderers can drift independently if needed.
_FORM_SUFFIX: dict[str, dict[str, str]] = {
    "ガラル": {"ko": "갈라르", "en": "Galarian"},
    "アローラ": {"ko": "알로라", "en": "Alolan"},
    "ヒスイ": {"ko": "히스이", "en": "Hisuian"},
    "パルデア": {"ko": "파르데아", "en": "Paldean"},
    "メガ": {"ko": "메가", "en": "Mega"},
}

import re

_PARENS_RE = re.compile(r"^(.*?)\s*\(([^)]+)\)\s*$")


def _localize(
    species_ja: str,
    *,
    registry: PokedexRegistry | None,
    lang: Language,
) -> str:
    """Localize one species — parens-aware, registry-aware."""

    base = species_ja
    variant: str | None = None
    m = _PARENS_RE.match(species_ja)
    if m:
        base = m.group(1).strip()
        variant = m.group(2).strip()

    if registry is None:
        localized = base
    else:
        entry = registry.lookup(species_ja=base)
        localized = entry.localize(lang) if entry else base

    if variant:
        translated = _FORM_SUFFIX.get(variant, {}).get(lang)
        suffix_str = translated or variant
        return f"{localized}({suffix_str})"
    return localized


def render_recommendation_table(
    scored_teams: Sequence[ScoredTeam],
    *,
    snapshot: MetaSnapshot,
    stream: TextIO,
    lang: Language = "ko",
    dex_registry: PokedexRegistry | None = None,
    pareto_size: int | None = None,
    all_scored: Sequence[ScoredTeam] | None = None,
    meta_top_n: int = 10,
) -> None:
    """Render the engine's top-K Pareto recommendation as a text table.

    Parameters
    ----------
    scored_teams:
        The ranked output of :func:`gbl_hacker.score.rank.rank_top_k`
        (already top-K, sorted best-first).
    snapshot:
        The originating ``MetaSnapshot`` — used to surface the
        data-honesty caveat in the header. Refusing the snapshot here
        keeps AC 6 enforced on the recommendation surface too.
    stream:
        Output text stream.
    lang:
        Display language for member species names.
    dex_registry:
        Dex localization table. ``None`` falls back to raw Japanese.
    pareto_size:
        Optional — the size of the Pareto frontier the top-K was sliced
        from. Surfaced in the header so the operator can see how much
        was discarded.

    Raises
    ------
    MissingCaveatError
        If ``snapshot.source_caveat`` is empty. Defense-in-depth — same
        rule as :func:`gbl_hacker.render.snapshot.render_meta_snapshot`.
    """

    if not snapshot.source_caveat.strip():
        raise MissingCaveatError(
            "Refusing to render recommendations with an empty source_caveat "
            "— violates the data_honesty evaluation principle (AC 6)."
        )

    stream.write(format_caveat_block(snapshot))
    stream.write("\n\n")

    stream.write("recommendation:\n")
    stream.write(f"  league:         {snapshot.league}\n")
    stream.write(f"  rating_bracket: {snapshot.rating_bracket}\n")
    stream.write(f"  fetched_at:     {snapshot.fetched_at.isoformat()}\n")
    if pareto_size is not None:
        stream.write(f"  pareto_size:    {pareto_size}\n")
    stream.write(f"  top_k:          {len(scored_teams)}\n\n")

    if not scored_teams:
        stream.write("  (no Pareto-optimal teams scored — registry coverage too thin)\n\n")
    else:
        # Localize members once so column widths can react to actual
        # content. Slot-by-slot ``CombatantBuild`` carries ``form_id`` so
        # shadow / regional variants surface in the rendered output.
        shadow_marker = {"ja": "(シャドウ)", "ko": "(섀도우)", "en": "(Shadow)"}
        rendered_members: list[str] = []
        for st in scored_teams:
            members: list[str] = []
            for build in st.team.slots:
                localized = _localize(
                    build.species, registry=dex_registry, lang=lang
                )
                # Inline parens already encode regional variants; only
                # append a shadow marker when form_id != 0 AND no inline
                # variant is already present on the localized name.
                if build.form_id and "(" not in localized:
                    localized += shadow_marker.get(lang, shadow_marker["ja"])
                members.append(localized)
            rendered_members.append(" / ".join(members))

        # Column header. Team column reads left-to-right as
        # ``lead / safe_swap / closer`` per the seed ontology.
        stream.write(
            f"  {'rank':>4}  {'ewr':>5}  {'wcr':>5}  {'cov':>5}  "
            "team (lead / safe_swap / closer)\n"
        )
        stream.write("  " + "-" * 68 + "\n")
        for i, (st, members_str) in enumerate(zip(scored_teams, rendered_members), start=1):
            sc = st.score
            stream.write(
                f"  {i:>4}  "
                f"{sc.expected_win_rate * 100:>4.1f}%  "
                f"{sc.worst_case_robustness * 100:>4.1f}%  "
                f"{sc.meta_coverage * 100:>4.1f}%  "
                f"{members_str}\n"
            )
            # Slot-by-slot moveset detail: fast / charged1 + charged2.
            # English move names from PvPoke gamemaster — we use the
            # raw English label because the move ja→ko mapping is not
            # yet wired up. Operators recognize the English names from
            # PvPoke, and the species column above is already
            # localized so the team is identifiable.
            slot_labels = ("lead", "swap", "close")
            for slot_label, build, localized in zip(
                slot_labels, st.team.slots, members_str.split(" / ")
            ):
                fast_name = build.fast.name
                charged_parts = [build.charged.name]
                if build.charged2:
                    charged_parts.append(build.charged2.name)
                charged_str = " + ".join(charged_parts)
                stream.write(
                    f"          {slot_label:<5} {localized:<22}  "
                    f"{fast_name} / {charged_str}\n"
                )
            stream.write("\n")
        stream.write("\n")
        stream.write(
            "  ewr = expected win rate  •  wcr = worst-case robustness  •  cov = meta coverage\n\n"
        )

    # Optional: surface the most-used meta teams (by upstream usage_count)
    # with their simulator scores. Lets the operator see the simulator's
    # judgement on the teams the JP meta is actually playing — including
    # shadow / regional-variant lineups that may not dominate the Pareto
    # frontier but are popular in practice.
    if all_scored:
        scored_by_team_id = {id(st.team): st for st in all_scored}
        # Map back from each meta-team's materialized CandidateTeam to
        # its ScoredTeam — but ``all_scored`` already preserves snapshot
        # team_usage order at construction, so we can index directly.
        meta_pairs = list(zip(snapshot.team_usage, all_scored))
        # Sort by upstream usage_count desc, stable by rank when tied.
        meta_pairs.sort(
            key=lambda pair: (-(pair[0].usage_count or 0), pair[0].rank or 9999)
        )
        visible = meta_pairs[: max(0, meta_top_n)]
        if visible:
            stream.write(f"meta_popular_top_{len(visible)} (sim score on the upstream's most-used teams):\n")
            stream.write(
                f"  {'rank':>4}  {'cnt':>3}  {'ewr':>5}  {'wcr':>5}  {'cov':>5}  team\n"
            )
            stream.write("  " + "-" * 68 + "\n")
            shadow_marker = {
                "ja": "(シャドウ)",
                "ko": "(섀도우)",
                "en": "(Shadow)",
            }
            for tu, st in visible:
                # Build display names from the materialized team so the
                # form_id annotation comes from the resolver, not the
                # site's raw flag (handles base/shadow disambiguation).
                names: list[str] = []
                for build in st.team.slots:
                    localized = _localize(
                        build.species, registry=dex_registry, lang=lang
                    )
                    if build.form_id and "(" not in localized:
                        localized += shadow_marker.get(
                            lang, shadow_marker["ja"]
                        )
                    names.append(localized)
                sc = st.score
                rank = tu.rank if tu.rank is not None else "—"
                stream.write(
                    f"  {rank!s:>4}  {tu.usage_count:>3}  "
                    f"{sc.expected_win_rate * 100:>4.1f}%  "
                    f"{sc.worst_case_robustness * 100:>4.1f}%  "
                    f"{sc.meta_coverage * 100:>4.1f}%  "
                    f"{' / '.join(names)}\n"
                )
            stream.write("\n")

    stream.write(
        f"source_caveat: {snapshot.source_caveat}\n"
    )


__all__ = [
    "CAVEAT_HEADER_LABEL",
    "MissingCaveatError",
    "render_recommendation_table",
]
