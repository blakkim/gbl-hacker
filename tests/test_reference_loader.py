"""Tests for ``gbl_hacker.reference.loader`` (Sub-AC 5.1).

The contract: a JSON fixture representing an independent top-tier reference
team list (PvPoke meta export or streamer lineup) deserializes into a
canonical ``ReferenceTeamList`` whose entries expose ``(species,
fast_move, charge_moves)`` triples in normalized lowercase/underscore form
suitable for direct overlap comparison against meta-snapshot species ids.

These tests are fully offline — the fixture
``tests/fixtures/reference_great_league_pvpoke_sample.json`` is the
recorded source of truth and defines what "well-formed reference list"
means for the engine.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gbl_hacker.reference import (
    GREAT_LEAGUE_LABEL,
    TEAM_SIZE,
    ReferenceBuild,
    ReferenceBuildDisplay,
    ReferenceLoadError,
    ReferenceTeam,
    ReferenceTeamList,
    canonical_id,
    load_reference_team_list,
    load_reference_team_list_from_mapping,
)


FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "reference_great_league_pvpoke_sample.json"
)


# ---------------------------------------------------------------------------
# Sub-AC 5.1 headline assertion — fixture deserializes to expected list
# ---------------------------------------------------------------------------


def _expected_team_list() -> ReferenceTeamList:
    """Hand-construct the expected deserialization of the fixture.

    Keeping this as code (not a second copy of the JSON) is the point —
    the test must pin the canonical-id normalization. If the fixture's
    raw ``"Medicham (Shadow)"`` does not normalize to ``"medicham_shadow"``
    the test fails before any downstream Sub-AC tries to compare against
    the meta snapshot's ``"medicham_shadow"`` key.
    """

    azumarill = ReferenceBuild(
        species="azumarill",
        fast_move="bubble",
        charge_moves=("ice_beam", "play_rough"),
        display=ReferenceBuildDisplay(
            species="Azumarill",
            fast_move="Bubble",
            charge_moves=("Ice Beam", "Play Rough"),
        ),
    )
    annihilape = ReferenceBuild(
        species="annihilape",
        fast_move="counter",
        charge_moves=("rage_fist", "ice_punch"),
        display=ReferenceBuildDisplay(
            species="Annihilape",
            fast_move="Counter",
            charge_moves=("Rage Fist", "Ice Punch"),
        ),
    )
    registeel = ReferenceBuild(
        species="registeel",
        fast_move="lock_on",
        charge_moves=("focus_blast", "zap_cannon"),
        display=ReferenceBuildDisplay(
            species="Registeel",
            fast_move="Lock-On",
            charge_moves=("Focus Blast", "Zap Cannon"),
        ),
    )
    medicham_shadow = ReferenceBuild(
        species="medicham_shadow",
        fast_move="counter",
        charge_moves=("power_up_punch", "ice_punch"),
        display=ReferenceBuildDisplay(
            species="Medicham (Shadow)",
            fast_move="Counter",
            charge_moves=("Power-Up Punch", "Ice Punch"),
        ),
    )
    lickitung_shadow = ReferenceBuild(
        species="lickitung_shadow",
        fast_move="lick",
        charge_moves=("body_slam", "power_whip"),
        display=ReferenceBuildDisplay(
            species="Lickitung (Shadow)",
            fast_move="Lick",
            charge_moves=("Body Slam", "Power Whip"),
        ),
    )
    galarian_stunfisk = ReferenceBuild(
        species="galarian_stunfisk",
        fast_move="mud_shot",
        charge_moves=("rock_slide", "earthquake"),
        display=ReferenceBuildDisplay(
            species="Galarian Stunfisk",
            fast_move="Mud Shot",
            charge_moves=("Rock Slide", "Earthquake"),
        ),
    )

    teams = (
        ReferenceTeam(
            name="Azu / Anni / Registeel core",
            source_label="pvpoke_meta",
            members=(azumarill, annihilape, registeel),
        ),
        ReferenceTeam(
            name="Shadow Medicham bully",
            source_label="pvpoke_meta",
            members=(medicham_shadow, lickitung_shadow, azumarill),
        ),
        ReferenceTeam(
            name="Galarian Stunfisk wall",
            source_label="streamer:abr",
            members=(galarian_stunfisk, azumarill, annihilape),
        ),
    )

    return ReferenceTeamList(
        source="pvpoke_meta_v1",
        source_url="https://pvpoke.com/team-builder/all/1500",
        league=GREAT_LEAGUE_LABEL,
        captured_at=datetime(2026, 5, 13, 0, 0, 0, tzinfo=timezone.utc),
        notes=(
            "Recorded fixture mirroring a PvPoke Great League meta sample. "
            "Used by the engine's overlap check; not ground-truth top-500."
        ),
        teams=teams,
    )


def test_fixture_deserializes_to_expected_team_list() -> None:
    """Sub-AC 5.1 headline: fixture round-trips to the canonical list."""

    loaded = load_reference_team_list(FIXTURE_PATH)
    assert loaded == _expected_team_list()


# ---------------------------------------------------------------------------
# field-level coverage
# ---------------------------------------------------------------------------


def test_loader_preserves_league_and_metadata() -> None:
    """League / source / capture timestamp / notes are preserved verbatim."""

    loaded = load_reference_team_list(FIXTURE_PATH)
    assert loaded.league == GREAT_LEAGUE_LABEL
    assert loaded.source == "pvpoke_meta_v1"
    assert loaded.source_url == "https://pvpoke.com/team-builder/all/1500"
    assert loaded.captured_at == datetime(
        2026, 5, 13, 0, 0, 0, tzinfo=timezone.utc
    )
    assert "PvPoke" in loaded.notes


def test_each_team_has_exactly_three_members_in_slot_order() -> None:
    """3v3 invariant — every loaded team has 3 members in fixture order."""

    loaded = load_reference_team_list(FIXTURE_PATH)
    for team in loaded.teams:
        assert len(team.members) == TEAM_SIZE
    # First team's slot order is preserved.
    first = loaded.teams[0]
    assert first.species == ("azumarill", "annihilape", "registeel")


def test_canonical_species_match_meta_snapshot_style_ids() -> None:
    """Canonical ids align with the Taiman-Party parser's species attribute style.

    The Taiman fixture stores ``data-species="medicham_shadow"`` and
    ``data-species="galarian_stunfisk"``. The reference loader must emit
    the same canonical form on the corresponding entries so the eventual
    overlap check (Sub-AC 5.2) can compare them directly.
    """

    loaded = load_reference_team_list(FIXTURE_PATH)
    all_species: set[str] = set()
    for team in loaded.teams:
        for member in team.members:
            all_species.add(member.species)
    assert "medicham_shadow" in all_species
    assert "galarian_stunfisk" in all_species
    assert "lickitung_shadow" in all_species


def test_charge_moves_are_canonical_pair_in_published_order() -> None:
    """``charge_moves`` is always a 2-tuple; ordering is preserved."""

    loaded = load_reference_team_list(FIXTURE_PATH)
    azu = loaded.teams[0].members[0]
    assert azu.species == "azumarill"
    assert azu.charge_moves == ("ice_beam", "play_rough")
    # Display tuple preserves spacing / capitalization.
    assert azu.display.charge_moves == ("Ice Beam", "Play Rough")


def test_per_team_source_label_distinguishes_pvpoke_vs_streamer() -> None:
    """Per-team ``source_label`` is preserved so multi-origin fixtures are usable."""

    loaded = load_reference_team_list(FIXTURE_PATH)
    labels = {team.source_label for team in loaded.teams}
    assert "pvpoke_meta" in labels
    assert "streamer:abr" in labels


# ---------------------------------------------------------------------------
# canonical_id behavior (pinned because downstream comparisons depend on it)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Medicham (Shadow)", "medicham_shadow"),
        ("Galarian Stunfisk", "galarian_stunfisk"),
        ("Ice Beam", "ice_beam"),
        ("Lock-On", "lock_on"),
        ("Power-Up Punch", "power_up_punch"),
        ("Mud Shot", "mud_shot"),
        ("   trailing  whitespace  ", "trailing_whitespace"),
        ("Azumarill", "azumarill"),
    ],
)
def test_canonical_id_normalizes_known_patterns(raw: str, expected: str) -> None:
    assert canonical_id(raw) == expected


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


def test_missing_path_raises_reference_load_error(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(ReferenceLoadError) as exc_info:
        load_reference_team_list(missing)
    assert exc_info.value.path == missing


def test_invalid_json_raises_reference_load_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ReferenceLoadError):
        load_reference_team_list(bad)


def test_wrong_league_is_rejected() -> None:
    payload = {
        "source": "test",
        "league": "ultra_league",
        "captured_at": "2026-05-13T00:00:00Z",
        "teams": [
            {
                "name": "x",
                "source_label": "y",
                "members": [
                    {
                        "species": "Azumarill",
                        "fast_move": "Bubble",
                        "charge_moves": ["Ice Beam", "Play Rough"],
                    }
                ]
                * 3,
            }
        ],
    }
    with pytest.raises(ReferenceLoadError, match="great_league"):
        load_reference_team_list_from_mapping(payload)


def test_wrong_member_count_is_rejected() -> None:
    payload = {
        "source": "test",
        "league": "great_league",
        "captured_at": "2026-05-13T00:00:00Z",
        "teams": [
            {
                "name": "x",
                "source_label": "y",
                "members": [
                    {
                        "species": "Azumarill",
                        "fast_move": "Bubble",
                        "charge_moves": ["Ice Beam", "Play Rough"],
                    },
                    {
                        "species": "Azumarill",
                        "fast_move": "Bubble",
                        "charge_moves": ["Ice Beam", "Play Rough"],
                    },
                ],
            }
        ],
    }
    with pytest.raises(ReferenceLoadError, match="exactly 3"):
        load_reference_team_list_from_mapping(payload)


def test_wrong_charge_move_count_is_rejected() -> None:
    payload = {
        "source": "test",
        "league": "great_league",
        "captured_at": "2026-05-13T00:00:00Z",
        "teams": [
            {
                "name": "x",
                "source_label": "y",
                "members": [
                    {
                        "species": "Azumarill",
                        "fast_move": "Bubble",
                        "charge_moves": ["Ice Beam"],
                    },
                    {
                        "species": "Azumarill",
                        "fast_move": "Bubble",
                        "charge_moves": ["Ice Beam", "Play Rough"],
                    },
                    {
                        "species": "Azumarill",
                        "fast_move": "Bubble",
                        "charge_moves": ["Ice Beam", "Play Rough"],
                    },
                ],
            }
        ],
    }
    with pytest.raises(ReferenceLoadError, match="exactly 2"):
        load_reference_team_list_from_mapping(payload)


def test_empty_teams_array_is_rejected() -> None:
    payload = {
        "source": "test",
        "league": "great_league",
        "captured_at": "2026-05-13T00:00:00Z",
        "teams": [],
    }
    with pytest.raises(ReferenceLoadError, match="at least one"):
        load_reference_team_list_from_mapping(payload)


def test_load_from_mapping_matches_load_from_disk() -> None:
    """Both entry points agree on a well-formed payload."""

    from_disk = load_reference_team_list(FIXTURE_PATH)
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    from_mapping = load_reference_team_list_from_mapping(payload)
    assert from_disk == from_mapping
