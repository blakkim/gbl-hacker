"""Taiman Party Great League response parser.

Consumes a ``TaimanRawSnapshot`` (one JSON payload + one HTML payload
from the two backend AJAX endpoints) and emits a normalized, typed
``MetaSnapshot`` for downstream stages.

Pokémon-level usage comes from ``getSeasonLeague.php`` JSON. Each league
entry carries a ``list`` of ``{poke_id-form_id: {poke_name, cnt, ...}}``;
we keep raw counts as ``usage_count`` and normalize to a percentage with
the league's total report volume as the denominator.

Team-level usage comes from ``BattlePvpRecommendNewVue.php`` HTML. Each
team is wrapped in ``<div class="league_party_sc flex-list">`` and
exposes:

- Rank in ``<span class="font-s12">N</span>位``
- Per-team report count in ``<span class="font-s12">N</span>件``
- Three Pokémon as both visible ``<p>`` text and a single
  ``data-party="<name>-<form>,<name>-<form>,<name>-<form>"`` attribute on
  the comment-link element

The parser prefers the ``data-party`` attribute because it is the format
the site itself uses to round-trip team identity into modal lookups, so
it's the most stable shape across upstream re-skins.

Species identifiers stay as the upstream's Japanese ``poke_name``
strings; an English / dex-id normalizer is a v0.2 concern.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

from bs4 import BeautifulSoup, Tag

from gbl_hacker.fetch.taiman import (
    GREAT_LEAGUE_ID,
    TAIMAN_SOURCE_CAVEAT,
    TaimanRawSnapshot,
)

GREAT_LEAGUE_LABEL: str = "great_league"

DEFAULT_RATING_BRACKET: str = "upper"
"""Bracket label used when the upstream payload omits a marker.

Per the seed's data-honesty principle, Taiman Party is treated as upper-
bracket-reliable. The backend XHR query schema does not currently expose
a rank-bracket parameter (the SPA's rank checkboxes appear to be
client-side filters), so v0.1 always tags snapshots ``upper``.
"""


class TaimanParseError(Exception):
    """Raised when the upstream payload cannot be parsed into a snapshot."""

    def __init__(self, message: str, *, url: str | None = None) -> None:
        super().__init__(message)
        self.url = url


@dataclass(frozen=True, slots=True)
class MoveUsage:
    """One ladder-usage entry for a fast or charged move on a species.

    Attributes
    ----------
    name:
        Move display name as the upstream renders it (Japanese — e.g.
        ``ボルトチェンジ``).
    move_id:
        Numeric move id from Taiman Party (string in the JSON, parsed
        here as ``int`` when possible; ``None`` on failure).
    move_type:
        GO move type lowercase, ``""`` when absent.
    usage_count:
        Raw count of reports using this move on this species. Larger
        means more ladder usage — drives the dominant-moveset
        selection in the build registry.
    """

    name: str
    usage_count: int
    move_type: str = ""
    move_id: int | None = None


@dataclass(frozen=True, slots=True)
class PokemonUsage:
    """Pokémon-level usage entry from the Taiman Party meta feed.

    Attributes
    ----------
    species:
        Upstream species identifier (Japanese display name). Kept as a
        free-form string so the parser does not need to own a species
        registry; downstream code may normalize via a build registry.
    usage_pct:
        Usage rate as a percentage in ``[0.0, 100.0]``, computed from
        ``usage_count / sum(league_counts) * 100``.
    rank:
        1-based usage rank within the league snapshot, or ``None`` if the
        upstream did not surface one.
    usage_count:
        Raw report count from the JSON (``cnt`` field). Preserved so
        downstream consumers can re-derive percentages with a different
        denominator if desired.
    dex_id:
        National Pokédex number, or ``None`` when the upstream omitted
        the numeric id.
    form_id:
        Upstream form discriminator (``0`` for base form, non-zero for
        shadow / alternate forms). ``None`` when absent.
    fast_moves:
        Ordered tuple of fast-move usage entries (descending by
        ``usage_count``). Empty when the upstream provided no breakdown.
    charged_moves:
        Same shape, for charged moves.
    """

    species: str
    usage_pct: float
    rank: int | None = None
    usage_count: int = 0
    dex_id: int | None = None
    form_id: int | None = None
    fast_moves: tuple[MoveUsage, ...] = ()
    charged_moves: tuple[MoveUsage, ...] = ()

    def __post_init__(self) -> None:  # pragma: no cover - pure validation
        if not self.species:
            raise ValueError("species must be non-empty")
        if not (0.0 <= self.usage_pct <= 100.0):
            raise ValueError(f"usage_pct out of range: {self.usage_pct}")
        if self.rank is not None and self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")
        if self.usage_count < 0:
            raise ValueError(f"usage_count must be >= 0, got {self.usage_count}")


@dataclass(frozen=True, slots=True)
class TeamUsage:
    """Team-level usage entry — a 3-Pokémon lineup with crowd usage share.

    Attributes
    ----------
    members:
        Three species identifiers in upstream-reported slot order. The
        site does not label slot roles, so callers should treat order as
        report order rather than lead / safe_swap / closer semantics.
    usage_pct:
        Share of reported sets this exact lineup represents, in
        ``[0.0, 100.0]``, computed from ``usage_count / total_count``.
    rank:
        1-based rank in the recommend list, or ``None`` if absent.
    usage_count:
        Raw report count for this lineup (``N`` in ``<span>N</span>件``).
    member_forms:
        Per-slot form discriminator (``0`` = base form, non-zero =
        shadow / alternate). Mirrors ``members`` slot for slot.
    """

    members: tuple[str, str, str]
    usage_pct: float
    rank: int | None = None
    usage_count: int = 0
    member_forms: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self) -> None:  # pragma: no cover - pure validation
        if len(self.members) != 3:
            raise ValueError(
                f"team must have exactly 3 members, got {len(self.members)}"
            )
        if any(not m for m in self.members):
            raise ValueError("team member species must all be non-empty")
        if not (0.0 <= self.usage_pct <= 100.0):
            raise ValueError(f"usage_pct out of range: {self.usage_pct}")
        if self.rank is not None and self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")
        if self.usage_count < 0:
            raise ValueError(f"usage_count must be >= 0, got {self.usage_count}")
        if len(self.member_forms) != 3:
            raise ValueError(
                f"member_forms must have exactly 3 entries, got {len(self.member_forms)}"
            )


@dataclass(frozen=True, slots=True)
class MetaSnapshot:
    """Normalized in-memory snapshot of a Taiman Party meta refresh.

    A snapshot is the unit of input every downstream stage consumes.
    Once constructed it is immutable; re-fetching produces a new
    snapshot rather than mutating an existing one.

    Attributes
    ----------
    league:
        Always ``"great_league"`` for v0.1.
    rating_bracket:
        Bracket label. v0.1 always emits ``"upper"`` because the backend
        XHR does not currently expose a per-bracket query parameter.
    fetched_at:
        UTC timestamp from the originating fetch (uses the recommend
        endpoint's timestamp as the canonical refresh moment).
    source_url:
        URL of the recommend endpoint actually fetched. Lets the operator
        replay the exact request.
    source_caveat:
        Data-honesty caveat string. Carried so every meta_snapshot
        rendering can surface it without consulting global config.
    pokemon_usage:
        Pokémon-level usage entries sorted by descending ``usage_count``.
    team_usage:
        Team-level usage entries in upstream rank order.
    season:
        GBL season number from the originating fetch.
    league_id:
        Taiman Party internal league id (``0`` for Great League).
    """

    league: str
    rating_bracket: str
    fetched_at: datetime
    source_url: str
    source_caveat: str
    pokemon_usage: tuple[PokemonUsage, ...] = field(default_factory=tuple)
    team_usage: tuple[TeamUsage, ...] = field(default_factory=tuple)
    season: int = 0
    league_id: int = GREAT_LEAGUE_ID

    def __post_init__(self) -> None:  # pragma: no cover - pure validation
        if self.league != GREAT_LEAGUE_LABEL:
            raise ValueError(
                f"v0.1 only supports league='{GREAT_LEAGUE_LABEL}', got {self.league!r}"
            )
        if not self.rating_bracket:
            raise ValueError("rating_bracket must be non-empty")
        if not self.source_caveat:
            raise ValueError("source_caveat must be non-empty (data_honesty principle)")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_great_league_meta(
    raw: TaimanRawSnapshot,
    *,
    rating_bracket: str = DEFAULT_RATING_BRACKET,
) -> MetaSnapshot:
    """Parse the raw fetch pair into a normalized ``MetaSnapshot``.

    Parameters
    ----------
    raw:
        The two raw responses produced by ``fetch_great_league_meta``.
    rating_bracket:
        Bracket label tag. Defaults to ``"upper"`` because the backend
        XHR does not yet expose a rank-filtered slice.

    Raises
    ------
    TaimanParseError
        If both the Pokémon-usage JSON and the team-usage HTML yield
        zero rows. An entirely empty parse is treated as upstream
        change / corruption, not a valid "no data" result.
    """

    pokemon_rows = tuple(_parse_pokemon_usage(raw))
    team_rows = tuple(_parse_team_usage(raw))

    if not pokemon_rows and not team_rows:
        raise TaimanParseError(
            "Taiman Party payload yielded zero Pokémon and zero team rows — "
            "upstream API may have changed",
            url=raw.recommend.url,
        )

    return MetaSnapshot(
        league=GREAT_LEAGUE_LABEL,
        rating_bracket=rating_bracket,
        fetched_at=raw.recommend.fetched_at,
        source_url=raw.recommend.url,
        source_caveat=raw.recommend.source_caveat or TAIMAN_SOURCE_CAVEAT,
        pokemon_usage=pokemon_rows,
        team_usage=team_rows,
        season=raw.season,
        league_id=raw.league_id,
    )


# ---------------------------------------------------------------------------
# Pokémon-usage extraction (JSON)
# ---------------------------------------------------------------------------


def _parse_pokemon_usage(raw: TaimanRawSnapshot) -> Iterable[PokemonUsage]:
    """Yield ``PokemonUsage`` rows from the season-league JSON payload."""

    try:
        payload = json.loads(raw.season_league.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TaimanParseError(
            f"Failed to decode season-league JSON: {exc}",
            url=raw.season_league.url,
        ) from exc

    if not isinstance(payload, list):
        raise TaimanParseError(
            f"Expected season-league JSON to be a list, got {type(payload).__name__}",
            url=raw.season_league.url,
        )

    league_entry = _find_league_entry(payload, league_id=raw.league_id)
    if league_entry is None:
        return

    pokemon_list = league_entry.get("list")
    if not isinstance(pokemon_list, dict):
        return

    raw_entries: list[
        tuple[str, int, int | None, int | None, tuple[MoveUsage, ...], tuple[MoveUsage, ...]]
    ] = []
    for key, value in pokemon_list.items():
        if not isinstance(value, dict):
            continue
        species = (value.get("poke_name") or "").strip()
        if not species:
            continue
        cnt = _coerce_int(value.get("cnt"))
        if cnt is None or cnt < 0:
            continue
        dex_id = _coerce_int(value.get("poke_id"))
        form_id = _coerce_int(value.get("tai_form_id"))
        if form_id is None:
            # The dict key is "<dex>-<form>"; recover form_id from it.
            parts = key.split("-")
            if len(parts) == 2:
                form_id = _coerce_int(parts[1])
        fast_uses = tuple(_parse_move_usage_list(value.get("waza1")))
        charged_uses = tuple(_parse_move_usage_list(value.get("waza2")))
        raw_entries.append((species, cnt, dex_id, form_id, fast_uses, charged_uses))

    total = sum(cnt for _, cnt, _, _, _, _ in raw_entries)
    if total <= 0:
        return

    raw_entries.sort(key=lambda row: row[1], reverse=True)

    for idx, (species, cnt, dex_id, form_id, fasts, chargeds) in enumerate(
        raw_entries, start=1
    ):
        pct = (cnt / total) * 100.0
        if pct > 100.0:
            pct = 100.0
        yield PokemonUsage(
            species=species,
            usage_pct=pct,
            rank=idx,
            usage_count=cnt,
            dex_id=dex_id,
            form_id=form_id,
            fast_moves=fasts,
            charged_moves=chargeds,
        )


def _parse_move_usage_list(raw: object) -> list[MoveUsage]:
    """Convert a ``waza1`` / ``waza2`` list into ``MoveUsage`` entries.

    Sorted by descending ``usage_count`` so callers can rely on
    ``[0]`` being the ladder-most-used move.
    """
    if not isinstance(raw, list):
        return []
    out: list[MoveUsage] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = (entry.get("waza_name") or "").strip()
        if not name:
            continue
        cnt = _coerce_int(entry.get("cnt"))
        if cnt is None or cnt < 0:
            continue
        move_type = str(entry.get("waza_type_id") or "").lower()
        move_id = _coerce_int(entry.get("waza_id"))
        out.append(
            MoveUsage(
                name=name,
                usage_count=cnt,
                move_type=move_type,
                move_id=move_id,
            )
        )
    out.sort(key=lambda m: m.usage_count, reverse=True)
    return out


def _find_league_entry(payload: list, *, league_id: int) -> dict | None:
    """Pick the league dict for ``league_id`` out of the season payload.

    The upstream JSON stores ``lr_league_id`` and ``gsl_league_id`` as
    strings; we match either, in that priority order.
    """

    for entry in payload:
        if not isinstance(entry, dict):
            continue
        for key in ("lr_league_id", "gsl_league_id"):
            value = entry.get(key)
            if value is None:
                continue
            try:
                if int(value) == league_id:
                    return entry
            except (TypeError, ValueError):
                continue
    return None


# ---------------------------------------------------------------------------
# Team-usage extraction (HTML)
# ---------------------------------------------------------------------------

_RANK_RE = re.compile(r"^\s*(\d+)\s*位\s*$")
_COUNT_RE = re.compile(r"^\s*(\d+)\s*件\s*$")


def _parse_team_usage(raw: TaimanRawSnapshot) -> Iterable[TeamUsage]:
    """Yield ``TeamUsage`` rows from the recommend HTML payload."""

    html = raw.recommend.content
    soup = BeautifulSoup(html, "html.parser")
    blocks = soup.find_all("div", class_="league_party_sc")

    raw_rows: list[tuple[int | None, int, tuple[str, str, str], tuple[int, int, int]]] = []

    for block in blocks:
        if not isinstance(block, Tag):
            continue
        rank = _extract_rank(block)
        count = _extract_count(block)
        members = _extract_members(block)
        if members is None:
            continue
        species, forms = members
        raw_rows.append((rank, count, species, forms))

    if not raw_rows:
        return

    total = sum(count for _, count, _, _ in raw_rows)
    if total <= 0:
        # Fall back to uniform weights so we still yield rows when the
        # site happens to render counts as zero (treat each team equally).
        total = len(raw_rows)
        normalize = lambda _c: 1
    else:
        normalize = lambda c: c  # noqa: E731

    for rank, count, species, forms in raw_rows:
        pct = (normalize(count) / total) * 100.0
        if pct > 100.0:
            pct = 100.0
        yield TeamUsage(
            members=species,
            usage_pct=pct,
            rank=rank,
            usage_count=count,
            member_forms=forms,
        )


def _extract_rank(block: Tag) -> int | None:
    """Pull the ``N位`` rank from the leading w12 column."""

    for span in block.find_all("span", class_="font-s12"):
        if not isinstance(span, Tag):
            continue
        parent_text = span.parent.get_text(" ", strip=True) if span.parent else ""
        match = _RANK_RE.match(parent_text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return None


def _extract_count(block: Tag) -> int:
    """Pull the ``N件`` report-count from the trailing w12 column."""

    for span in block.find_all("span", class_="font-s12"):
        if not isinstance(span, Tag):
            continue
        parent_text = span.parent.get_text(" ", strip=True) if span.parent else ""
        match = _COUNT_RE.match(parent_text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return 0


def _extract_members(
    block: Tag,
) -> tuple[tuple[str, str, str], tuple[int, int, int]] | None:
    """Extract the three member species + form ids for a team block.

    Resolution order:

    1. ``data-party="A-fa,B-fb,C-fc"`` attribute on the comment-link
       element. Most stable: the site uses this exact string to
       round-trip a team into its modal lookup.
    2. Fallback: three ``<p class="mt-5 mb-0 font-s8 text-center">``
       children in DOM order (no form ids; defaults to 0).
    """

    tag_with_party = block.find(attrs={"data-party": True})
    if isinstance(tag_with_party, Tag):
        attr = tag_with_party.get("data-party")
        if isinstance(attr, str):
            parsed = _parse_data_party(attr)
            if parsed is not None:
                return parsed

    fallback_names: list[str] = []
    for p in block.find_all("p", class_="mt-5"):
        if not isinstance(p, Tag):
            continue
        text = p.get_text(strip=True)
        if text:
            fallback_names.append(text)
        if len(fallback_names) == 3:
            break
    if len(fallback_names) == 3:
        return (
            (fallback_names[0], fallback_names[1], fallback_names[2]),
            (0, 0, 0),
        )
    return None


def _parse_data_party(
    attr: str,
) -> tuple[tuple[str, str, str], tuple[int, int, int]] | None:
    """Split ``A-fa,B-fb,C-fc`` into (names, forms).

    Returns ``None`` unless exactly three comma-separated tokens are
    present and each has the ``<name>-<form>`` shape with a non-empty
    name. Form ids that fail to parse degrade to ``0``.
    """

    parts = [p.strip() for p in attr.split(",")]
    if len(parts) != 3:
        return None

    names: list[str] = []
    forms: list[int] = []
    for part in parts:
        if not part:
            return None
        # Split on the LAST hyphen so species names containing hyphens
        # (none observed in the live data, but cheap insurance) are kept
        # intact.
        if "-" in part:
            name, _, form_str = part.rpartition("-")
            name = name.strip()
            form_id = _coerce_int(form_str)
        else:
            name = part
            form_id = 0
        if not name:
            return None
        names.append(name)
        forms.append(form_id if form_id is not None else 0)

    return (names[0], names[1], names[2]), (forms[0], forms[1], forms[2])


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _coerce_int(value: object) -> int | None:
    """Best-effort int coercion. Returns ``None`` on failure."""
    if value is None:
        return None
    if isinstance(value, bool):  # bool is a subclass of int — keep them out
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r"-?\d+", value)
        if match is None:
            return None
        try:
            return int(match.group(0))
        except ValueError:
            return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


__all__ = [
    "DEFAULT_RATING_BRACKET",
    "GREAT_LEAGUE_LABEL",
    "MetaSnapshot",
    "MoveUsage",
    "PokemonUsage",
    "TaimanParseError",
    "TeamUsage",
    "parse_great_league_meta",
]
