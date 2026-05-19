"""Independent top-tier reference team list loader (Sub-AC 5.1).

The seed's Acceptance Criterion 5 demands that the engine's recommended
teams show non-trivial overlap with at least one *independent* top-tier
reference — e.g. PvPoke's published meta team list, a known top streamer's
published lineup, or similar. That comparison is impossible without a
canonical, in-memory form of the reference list.

This module owns exactly that: it deserializes a small JSON fixture into a
``ReferenceTeamList`` whose teams expose 3-slot lineups in canonical
``(pokemon, fast_move, charge_moves)`` form — matching the seed ontology's
``pokemon_build`` shape (species, fast_move, charged_move_1,
charged_move_2). Two charged moves are first-class here even though the
v0.1 simulator's :class:`CombatantBuild` only carries one at a time; the
reference data must mirror how real GBL builds look so the downstream
overlap-check (Sub-AC 5.2+) compares like to like.

Why a separate module instead of reusing the simulator types?
    The simulator's :class:`CombatantBuild` is *internal combat state* —
    HP and per-move damage numbers, one charged move per slot to keep the
    v0.1 turn-loop simple. The reference list is *external published
    data* — it should look exactly like a PvPoke / streamer team export,
    not like a simulator input. Keeping the boundary sharp means the
    overlap-comparison code (a later Sub-AC) can construct simulator
    builds from reference entries via a registry without smearing the
    two concerns. This module owns the *external* shape only.

Why JSON instead of CSV / YAML?
    PvPoke exports team rosters as JSON; streamer lineups are most often
    informally listed but trivially JSON-able. Sticking with stdlib JSON
    keeps the loader dependency-free and the fixture diff-readable.

Canonical normalization:
    Species and move identifiers in the canonical struct are lower-cased
    with non-alphanumeric runs collapsed to single underscores. This
    matches how :func:`gbl_hacker.parse.taiman.parse_great_league_meta`
    stores the upstream's species attribute (``"medicham_shadow"``,
    ``"galarian_stunfisk"``) so the eventual overlap check can compare
    canonical identifiers directly.  The fixture preserves human-readable
    display names alongside, so rationale cards can render
    ``"Medicham (Shadow)"`` even though comparison happens on
    ``"medicham_shadow"``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

GREAT_LEAGUE_LABEL: str = "great_league"
"""Same constant as :mod:`gbl_hacker.parse.taiman` — fixed for v0.1."""

TEAM_SIZE: int = 3
"""3v3 GBL — every reference team must have exactly 3 members."""


class ReferenceLoadError(Exception):
    """Raised when a reference team-list fixture cannot be deserialized.

    Carries the offending fixture path (when available) so the operator can
    surface "what file is malformed" without having to walk the stack.
    """

    def __init__(self, message: str, *, path: Path | None = None) -> None:
        super().__init__(message)
        self.path = path


@dataclass(frozen=True, slots=True)
class ReferenceBuild:
    """One Pokémon slot of a reference team, in canonical form.

    Matches the ``pokemon_build`` ontology slice the Sub-AC pins down —
    ``(pokemon, fast_move, charge_moves)`` — with the seed's intent that
    GBL builds carry two charged moves (charge_moves is always a length-2
    tuple even if a reference publishes only one — the loader requires the
    fixture to be explicit rather than silently filling).

    All three identifiers are normalized lowercase / underscore-joined for
    diff-stable comparison against meta-snapshot species. Display names
    are preserved on :attr:`display` for human-facing rendering.

    Attributes
    ----------
    species:
        Canonical species id (``"azumarill"``, ``"medicham_shadow"``).
    fast_move:
        Canonical fast-move id (``"bubble"``, ``"counter"``).
    charge_moves:
        Canonical 2-tuple of charged-move ids in published order. The
        ordering is meaningful — many references list the more-spammed
        first / coverage move second; preserving it lets the overlap
        check distinguish (Ice Beam / Play Rough) from (Play Rough /
        Ice Beam).
    display:
        Original human-readable identifiers (species, fast, charges) as
        the upstream wrote them. Rationale cards can render these; the
        engine compares on the canonical fields above.
    """

    species: str
    fast_move: str
    charge_moves: tuple[str, str]
    display: "ReferenceBuildDisplay"

    def __post_init__(self) -> None:  # pragma: no cover - pure validation
        if not self.species:
            raise ValueError("species must be non-empty")
        if not self.fast_move:
            raise ValueError("fast_move must be non-empty")
        if len(self.charge_moves) != 2:
            raise ValueError(
                f"charge_moves must have exactly 2 entries, got {len(self.charge_moves)}"
            )
        if any(not m for m in self.charge_moves):
            raise ValueError("charge_moves entries must all be non-empty")


@dataclass(frozen=True, slots=True)
class ReferenceBuildDisplay:
    """Human-readable identifiers preserved verbatim from the fixture."""

    species: str
    fast_move: str
    charge_moves: tuple[str, str]


@dataclass(frozen=True, slots=True)
class ReferenceTeam:
    """A single published reference team (3-slot, source-tagged).

    Attributes
    ----------
    name:
        Free-form team label (``"ABR open ladder lead-Azu"``,
        ``"PvPoke #1 team"``). Used in rationale prose; never compared.
    source_label:
        Origin tag (``"pvpoke_meta"``, ``"streamer:abr"``,
        ``"reddit:silph_tournament_winner"``). The data-honesty
        principle expects this surfaced when rendering overlap reports.
    members:
        Exactly :data:`TEAM_SIZE` reference builds in slot order
        (lead / safe_swap / closer). The slot ordering mirrors how
        :class:`gbl_hacker.score.CandidateTeam` reads its slots.
    """

    name: str
    source_label: str
    members: tuple[ReferenceBuild, ReferenceBuild, ReferenceBuild]

    def __post_init__(self) -> None:  # pragma: no cover - pure validation
        if not self.name:
            raise ValueError("team name must be non-empty")
        if not self.source_label:
            raise ValueError("source_label must be non-empty")
        if len(self.members) != TEAM_SIZE:
            raise ValueError(
                f"reference team must have exactly {TEAM_SIZE} members, "
                f"got {len(self.members)}"
            )

    @property
    def species(self) -> tuple[str, str, str]:
        """Canonical species triple in slot order (handy for overlap checks)."""
        a, b, c = self.members
        return (a.species, b.species, c.species)


@dataclass(frozen=True, slots=True)
class ReferenceTeamList:
    """Deserialized container for a reference fixture file.

    A fixture file represents one *source* (PvPoke meta export at a point
    in time, or one streamer's published lineup set) and carries enough
    metadata to surface in the data-honesty caveat: when it was captured,
    where it came from, what league it covers.

    Attributes
    ----------
    source:
        Machine identifier for this fixture's origin (``"pvpoke_meta_v1"``,
        ``"streamer:abr"``). Distinct from per-team ``source_label`` —
        the list-level value names the *fixture*, the per-team value
        names a finer-grained sub-origin within it.
    source_url:
        Where the data was sourced from (the published PvPoke URL, the
        streamer's video link, etc.). Surfaced in overlap-report output.
    league:
        Always ``"great_league"`` for v0.1.
    captured_at:
        Timestamp the human recorded this fixture. Lets the engine warn
        when reference data is stale relative to the current meta.
    notes:
        Free-form caveat (``"recorded for overlap validation, not
        ground truth"``). Empty string is permitted.
    teams:
        Tuple of reference teams. Order matches the fixture order — many
        references publish teams in popularity / tier order and that
        information should not be discarded.
    """

    source: str
    source_url: str
    league: str
    captured_at: datetime
    notes: str
    teams: tuple[ReferenceTeam, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:  # pragma: no cover - pure validation
        if not self.source:
            raise ValueError("source must be non-empty")
        if self.league != GREAT_LEAGUE_LABEL:
            raise ValueError(
                f"v0.1 reference lists must target league='{GREAT_LEAGUE_LABEL}', "
                f"got {self.league!r}"
            )
        if not self.teams:
            raise ValueError("reference team list must not be empty")


# ---------------------------------------------------------------------------
# canonical normalization
# ---------------------------------------------------------------------------


_NON_ALNUM_RUN = re.compile(r"[^a-z0-9]+")


def canonical_id(raw: str) -> str:
    """Normalize a free-form identifier to lowercase / underscore-joined form.

    Examples::

        canonical_id("Medicham (Shadow)")  == "medicham_shadow"
        canonical_id("Galarian Stunfisk")  == "galarian_stunfisk"
        canonical_id("Ice Beam")           == "ice_beam"

    Leading / trailing underscores produced by punctuation at string
    edges are stripped. The empty-input case yields ``""`` so callers
    upstream can validate / raise on it themselves rather than the
    normalizer pretending to succeed.
    """

    lowered = raw.strip().lower()
    collapsed = _NON_ALNUM_RUN.sub("_", lowered)
    return collapsed.strip("_")


# ---------------------------------------------------------------------------
# public loader entry points
# ---------------------------------------------------------------------------


def load_reference_team_list(path: Path | str) -> ReferenceTeamList:
    """Load and parse a reference team-list JSON fixture from disk.

    Parameters
    ----------
    path:
        Filesystem path to the JSON fixture.

    Returns
    -------
    ReferenceTeamList
        Fully-validated, immutable reference list.

    Raises
    ------
    ReferenceLoadError
        On missing path, malformed JSON, or schema violations (wrong
        league, wrong member count, missing required fields, etc.).
        The error always carries the offending ``path``.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise ReferenceLoadError(
            f"reference fixture not found: {file_path}", path=file_path
        )

    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - filesystem edge case
        raise ReferenceLoadError(
            f"failed to read reference fixture: {exc}", path=file_path
        ) from exc

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ReferenceLoadError(
            f"reference fixture is not valid JSON: {exc.msg} (line {exc.lineno}, "
            f"col {exc.colno})",
            path=file_path,
        ) from exc

    return _deserialize(payload, path=file_path)


def load_reference_team_list_from_mapping(
    payload: Mapping[str, Any],
    *,
    path: Path | None = None,
) -> ReferenceTeamList:
    """Alternate entry point: deserialize from an already-parsed mapping.

    Useful when the caller already holds a dict (e.g. test fixtures
    embedded inline, or an in-memory PvPoke export). The validation
    path is otherwise identical to :func:`load_reference_team_list`.
    """

    return _deserialize(payload, path=path)


# ---------------------------------------------------------------------------
# deserialization
# ---------------------------------------------------------------------------


def _deserialize(payload: Any, *, path: Path | None) -> ReferenceTeamList:
    """Common validate-and-construct path for both loader entry points."""

    if not isinstance(payload, Mapping):
        raise ReferenceLoadError(
            f"top-level JSON must be an object, got {type(payload).__name__}",
            path=path,
        )

    source = _required_str(payload, "source", path=path)
    source_url = _optional_str(payload, "source_url", default="")
    league = _required_str(payload, "league", path=path)
    if league != GREAT_LEAGUE_LABEL:
        raise ReferenceLoadError(
            f"reference list must target league={GREAT_LEAGUE_LABEL!r}, got {league!r}",
            path=path,
        )
    captured_at = _required_datetime(payload, "captured_at", path=path)
    notes = _optional_str(payload, "notes", default="")

    raw_teams = payload.get("teams")
    if not isinstance(raw_teams, Sequence) or isinstance(raw_teams, (str, bytes)):
        raise ReferenceLoadError(
            "'teams' must be a JSON array", path=path
        )
    if len(raw_teams) == 0:
        raise ReferenceLoadError(
            "'teams' must contain at least one entry", path=path
        )

    teams: list[ReferenceTeam] = []
    for idx, raw_team in enumerate(raw_teams):
        teams.append(_deserialize_team(raw_team, index=idx, path=path))

    return ReferenceTeamList(
        source=source,
        source_url=source_url,
        league=league,
        captured_at=captured_at,
        notes=notes,
        teams=tuple(teams),
    )


def _deserialize_team(
    raw: Any, *, index: int, path: Path | None
) -> ReferenceTeam:
    if not isinstance(raw, Mapping):
        raise ReferenceLoadError(
            f"team[{index}] must be a JSON object, got {type(raw).__name__}",
            path=path,
        )
    name = _required_str(raw, "name", path=path, ctx=f"team[{index}]")
    source_label = _required_str(
        raw, "source_label", path=path, ctx=f"team[{index}]"
    )

    raw_members = raw.get("members")
    if not isinstance(raw_members, Sequence) or isinstance(raw_members, (str, bytes)):
        raise ReferenceLoadError(
            f"team[{index}].members must be a JSON array", path=path
        )
    if len(raw_members) != TEAM_SIZE:
        raise ReferenceLoadError(
            f"team[{index}].members must have exactly {TEAM_SIZE} entries, "
            f"got {len(raw_members)}",
            path=path,
        )

    members = tuple(
        _deserialize_build(raw_member, slot=slot, team_index=index, path=path)
        for slot, raw_member in enumerate(raw_members)
    )
    a, b, c = members
    return ReferenceTeam(
        name=name,
        source_label=source_label,
        members=(a, b, c),
    )


def _deserialize_build(
    raw: Any, *, slot: int, team_index: int, path: Path | None
) -> ReferenceBuild:
    ctx = f"team[{team_index}].members[{slot}]"
    if not isinstance(raw, Mapping):
        raise ReferenceLoadError(
            f"{ctx} must be a JSON object, got {type(raw).__name__}", path=path
        )

    species_display = _required_str(raw, "species", path=path, ctx=ctx)
    fast_display = _required_str(raw, "fast_move", path=path, ctx=ctx)

    raw_charges = raw.get("charge_moves")
    if not isinstance(raw_charges, Sequence) or isinstance(raw_charges, (str, bytes)):
        raise ReferenceLoadError(
            f"{ctx}.charge_moves must be a JSON array", path=path
        )
    if len(raw_charges) != 2:
        raise ReferenceLoadError(
            f"{ctx}.charge_moves must have exactly 2 entries, got {len(raw_charges)}",
            path=path,
        )
    charge_displays: list[str] = []
    for charge_idx, raw_charge in enumerate(raw_charges):
        if not isinstance(raw_charge, str) or not raw_charge.strip():
            raise ReferenceLoadError(
                f"{ctx}.charge_moves[{charge_idx}] must be a non-empty string",
                path=path,
            )
        charge_displays.append(raw_charge.strip())

    species_canon = canonical_id(species_display)
    fast_canon = canonical_id(fast_display)
    charges_canon = tuple(canonical_id(c) for c in charge_displays)

    if not species_canon:
        raise ReferenceLoadError(
            f"{ctx}.species normalized to empty string", path=path
        )
    if not fast_canon:
        raise ReferenceLoadError(
            f"{ctx}.fast_move normalized to empty string", path=path
        )
    if any(not c for c in charges_canon):
        raise ReferenceLoadError(
            f"{ctx}.charge_moves contains an entry that normalized to empty",
            path=path,
        )

    display = ReferenceBuildDisplay(
        species=species_display,
        fast_move=fast_display,
        charge_moves=(charge_displays[0], charge_displays[1]),
    )
    return ReferenceBuild(
        species=species_canon,
        fast_move=fast_canon,
        charge_moves=(charges_canon[0], charges_canon[1]),
        display=display,
    )


# ---------------------------------------------------------------------------
# small validation helpers
# ---------------------------------------------------------------------------


def _required_str(
    payload: Mapping[str, Any],
    key: str,
    *,
    path: Path | None,
    ctx: str | None = None,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        prefix = f"{ctx}." if ctx else ""
        raise ReferenceLoadError(
            f"missing required string field '{prefix}{key}'", path=path
        )
    return value.strip()


def _optional_str(
    payload: Mapping[str, Any], key: str, *, default: str
) -> str:
    value = payload.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        return default
    return value.strip()


def _required_datetime(
    payload: Mapping[str, Any],
    key: str,
    *,
    path: Path | None,
) -> datetime:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReferenceLoadError(
            f"missing required ISO-8601 datetime field '{key}'", path=path
        )
    raw = value.strip()
    # Accept both 'Z' suffix and explicit offsets — datetime.fromisoformat
    # in Py3.12 handles offsets but not the bare 'Z'.
    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ReferenceLoadError(
            f"'{key}' is not a valid ISO-8601 datetime: {raw!r}", path=path
        ) from exc


__all__ = [
    "GREAT_LEAGUE_LABEL",
    "TEAM_SIZE",
    "ReferenceBuild",
    "ReferenceBuildDisplay",
    "ReferenceLoadError",
    "ReferenceTeam",
    "ReferenceTeamList",
    "canonical_id",
    "load_reference_team_list",
    "load_reference_team_list_from_mapping",
]
