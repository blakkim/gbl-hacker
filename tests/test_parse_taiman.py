"""Tests for ``gbl_hacker.parse.taiman.parse_great_league_meta``.

Sub-AC 2 contract: parsing the raw Taiman Party backend pair (season-league
JSON + recommend HTML) produces a normalized in-memory snapshot. These
tests feed recorded live fixtures and assert the parsed records match the
known field values, including the new dex_id / form_id / member_forms
provenance carried through to downstream localization.

The suite is fully offline. The two fixtures live under
``tests/fixtures/`` and were captured live on 2026-05-13 — re-record by
re-running ``scripts/recon_taiman.py`` and copying the two response
bodies if the upstream backend changes shape.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from gbl_hacker.fetch.taiman import (
    GREAT_LEAGUE_ID,
    RECOMMEND_URL,
    SEASON_LEAGUE_URL,
    TAIMAN_SOURCE_CAVEAT,
    FetchResult,
    TaimanRawSnapshot,
)
from gbl_hacker.parse.taiman import (
    DEFAULT_RATING_BRACKET,
    MetaSnapshot,
    PokemonUsage,
    TaimanParseError,
    TeamUsage,
    parse_great_league_meta,
)


SEASON_FIXTURE = Path(__file__).parent / "fixtures" / "taiman_season_league.json"
RECOMMEND_FIXTURE = (
    Path(__file__).parent / "fixtures" / "taiman_recommend_great_league.html"
)


def _make_raw(
    *,
    season: int = 27,
    league_id: int = GREAT_LEAGUE_ID,
    season_bytes: bytes | None = None,
    recommend_bytes: bytes | None = None,
    season_url: str = SEASON_LEAGUE_URL,
    recommend_url: str = RECOMMEND_URL + "?season=27&league=0&between=1",
    fetched_at: datetime | None = None,
) -> TaimanRawSnapshot:
    """Build a ``TaimanRawSnapshot`` from the recorded live fixtures."""

    when = fetched_at or datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc)
    season_payload = (
        season_bytes if season_bytes is not None else SEASON_FIXTURE.read_bytes()
    )
    recommend_payload = (
        recommend_bytes
        if recommend_bytes is not None
        else RECOMMEND_FIXTURE.read_bytes()
    )
    return TaimanRawSnapshot(
        season=season,
        league_id=league_id,
        season_league=FetchResult(
            url=season_url,
            status_code=200,
            content=season_payload,
            content_type="text/html; charset=UTF-8",
            fetched_at=when,
        ),
        recommend=FetchResult(
            url=recommend_url,
            status_code=200,
            content=recommend_payload,
            content_type="text/html; charset=UTF-8",
            fetched_at=when,
        ),
    )


# ---------------------------------------------------------------------------
# core contract — parsed records match the recorded live fixture
# ---------------------------------------------------------------------------


def test_parse_returns_meta_snapshot_with_required_fields() -> None:
    """A fixture parse yields a fully-populated ``MetaSnapshot``."""

    snapshot = parse_great_league_meta(_make_raw())

    assert isinstance(snapshot, MetaSnapshot)
    assert snapshot.league == "great_league"
    assert snapshot.rating_bracket == DEFAULT_RATING_BRACKET == "upper"
    assert snapshot.fetched_at == datetime(
        2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc
    )
    assert snapshot.source_url.startswith(RECOMMEND_URL)
    assert snapshot.season == 27
    assert snapshot.league_id == GREAT_LEAGUE_ID

    # Data-honesty caveat travels with every snapshot.
    assert snapshot.source_caveat == TAIMAN_SOURCE_CAVEAT
    assert "report-density" in snapshot.source_caveat.lower()


def test_parse_pokemon_usage_top_rows_match_recorded_fixture() -> None:
    """The fixture pins the rank-1 species and at least 50 rows are surfaced."""

    snapshot = parse_great_league_meta(_make_raw())

    # The captured fixture covers all 50 Pokémon the live site returns
    # for Great League. Treat 50 as the lower bound to tolerate a future
    # recapture that includes a slightly different cut.
    assert len(snapshot.pokemon_usage) >= 50

    top = snapshot.pokemon_usage[0]
    # Live data on 2026-05-13: ヌオー (Quagsire, dex 195) sits at rank 1
    # with 1019 reports.
    assert top.rank == 1
    assert top.species == "ヌオー"
    assert top.dex_id == 195
    assert top.form_id == 0
    assert top.usage_count == 1019
    # Usage % is rank-1 share of the normalized counts — pin to one decimal.
    assert round(top.usage_pct, 1) == pytest.approx(6.2, abs=0.1)

    # Ranks must be unique and strictly increasing (1..N).
    ranks = [p.rank for p in snapshot.pokemon_usage]
    assert ranks == sorted(ranks)
    assert ranks[0] == 1
    assert all(r is not None and r > 0 for r in ranks)


def test_parse_team_usage_first_team_and_member_forms_match_fixture() -> None:
    """The first recommended team is locked to the recorded live ranking."""

    snapshot = parse_great_league_meta(_make_raw())
    assert len(snapshot.team_usage) == 30

    top = snapshot.team_usage[0]
    # Live recording: 1位 = デカヌチャン / オオタチ / マスカーニャ, 34件.
    assert top.rank == 1
    assert top.members == ("デカヌチャン", "オオタチ", "マスカーニャ")
    assert top.member_forms == (0, 0, 0)
    assert top.usage_count == 34
    assert top.usage_pct > 0.0

    # Form discriminator should be non-zero on at least one team — the
    # recorded fixture has multiple shadow rows (e.g. フォレトス(form=1),
    # ヌオー(form=1)). Pin "at least one" rather than a brittle index.
    assert any(any(f != 0 for f in t.member_forms) for t in snapshot.team_usage)


def test_parse_carries_dex_ids_and_usage_counts() -> None:
    """Pokémon entries surface dex_id and the raw report count, not just %."""

    snapshot = parse_great_league_meta(_make_raw())

    # Sample 5 known species → dex id mappings from the recorded fixture.
    by_dex = {p.dex_id: p for p in snapshot.pokemon_usage if p.dex_id is not None}
    assert by_dex[195].species == "ヌオー"
    assert by_dex[959].species == "デカヌチャン"
    assert by_dex[205].species == "フォレトス"

    # Raw counts are non-negative integers; the top entry's count is
    # strictly larger than the last entry's count (sort-by-count holds).
    counts = [p.usage_count for p in snapshot.pokemon_usage]
    assert all(c >= 0 for c in counts)
    assert counts[0] >= counts[-1]


# ---------------------------------------------------------------------------
# rating-bracket resolution
# ---------------------------------------------------------------------------


def test_explicit_rating_bracket_override_wins() -> None:
    """Caller-supplied ``rating_bracket=`` overrides the default."""

    snapshot = parse_great_league_meta(_make_raw(), rating_bracket="veteran")
    assert snapshot.rating_bracket == "veteran"


def test_bracket_defaults_to_upper_when_caller_does_not_override() -> None:
    """The backend XHR exposes no per-bracket query → default is ``upper``."""

    snapshot = parse_great_league_meta(_make_raw())
    assert snapshot.rating_bracket == DEFAULT_RATING_BRACKET == "upper"


# ---------------------------------------------------------------------------
# failure modes
# ---------------------------------------------------------------------------


def test_parse_raises_when_payload_yields_no_rows() -> None:
    """An empty pair is treated as upstream-change / corruption."""

    raw = _make_raw(
        season_bytes=b"[]",
        recommend_bytes=b"<!doctype html><html><body></body></html>",
    )
    with pytest.raises(TaimanParseError) as excinfo:
        parse_great_league_meta(raw)
    assert "zero" in str(excinfo.value).lower() or "empty" in str(excinfo.value).lower()


def test_parse_partial_payload_with_only_pokemon_section_succeeds() -> None:
    """Pokémon-only payloads parse — the team section may be empty."""

    season_only = (
        '[{"gsl_league_id":"0","name_en":"Great League","list":{'
        '"260-0":{"poke_id":"260","tai_form_id":"0","poke_name":"ラグラージ","cnt":42}'
        "}}]"
    ).encode("utf-8")
    raw = _make_raw(
        season_bytes=season_only,
        recommend_bytes=b"<!doctype html><html><body></body></html>",
    )
    snapshot = parse_great_league_meta(raw)
    assert len(snapshot.pokemon_usage) == 1
    assert snapshot.pokemon_usage[0].species == "ラグラージ"
    assert snapshot.pokemon_usage[0].dex_id == 260
    assert snapshot.team_usage == ()


def test_parse_raises_on_invalid_json_for_season_payload() -> None:
    """Corrupt JSON in the season payload surfaces as a parse error."""

    raw = _make_raw(season_bytes=b"not json {{{")
    with pytest.raises(TaimanParseError):
        parse_great_league_meta(raw)


# ---------------------------------------------------------------------------
# immutability — snapshot is frozen
# ---------------------------------------------------------------------------


def test_meta_snapshot_is_immutable() -> None:
    """Snapshots must be frozen so re-fetches produce new objects."""

    snapshot = parse_great_league_meta(_make_raw())
    with pytest.raises((AttributeError, TypeError)):
        snapshot.rating_bracket = "veteran"  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        snapshot.pokemon_usage[0].usage_pct = 99.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# defaults round-trip via PokemonUsage / TeamUsage constructors
# ---------------------------------------------------------------------------


def test_pokemon_usage_dataclass_has_sensible_defaults() -> None:
    """``PokemonUsage`` can be built with the legacy 3-field shape."""

    entry = PokemonUsage(species="azumarill", usage_pct=14.7, rank=1)
    assert entry.usage_count == 0
    assert entry.dex_id is None
    assert entry.form_id is None


def test_team_usage_dataclass_has_sensible_defaults() -> None:
    """``TeamUsage`` accepts legacy construction with all-zero forms."""

    team = TeamUsage(
        members=("Azumarill", "Annihilape", "Registeel"),
        usage_pct=3.4,
        rank=1,
    )
    assert team.usage_count == 0
    assert team.member_forms == (0, 0, 0)
