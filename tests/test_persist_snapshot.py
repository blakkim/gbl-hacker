"""Tests for ``gbl_hacker.persist.snapshot``.

Sub-AC 3 contract: a normalized ``MetaSnapshot`` can be written to a
versioned local store and read back; the central assertion is that the
written and reloaded snapshots are *equal* — value-equal, field-by-field,
including the data-honesty caveat and the immutable nested tuples.

The suite also pins the schema-version invariant (unknown versions are
rejected, missing versions are rejected) so the cache cannot silently
mis-interpret bytes from a future or hand-edited writer.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gbl_hacker.fetch.taiman import RECOMMEND_URL, TAIMAN_SOURCE_CAVEAT
from gbl_hacker.parse.taiman import MetaSnapshot, PokemonUsage, TeamUsage
from gbl_hacker.persist.snapshot import (
    SCHEMA_VERSION,
    SnapshotDecodeError,
    SnapshotPersistError,
    SnapshotSchemaError,
    StoredSnapshot,
    default_filename_for,
    latest_snapshot,
    list_snapshots,
    read_snapshot,
    snapshot_from_dict,
    snapshot_to_dict,
    write_snapshot,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _sample_snapshot(
    *,
    fetched_at: datetime | None = None,
    rating_bracket: str = "ace",
) -> MetaSnapshot:
    """Build a fully-populated snapshot for round-trip testing.

    Uses two Pokémon rows and two team rows so that the test catches both
    ``rank=None`` and ``rank=<int>`` round-trip paths, and confirms list
    ordering survives serialization.
    """

    return MetaSnapshot(
        league="great_league",
        rating_bracket=rating_bracket,
        fetched_at=fetched_at
        or datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc),
        source_url=RECOMMEND_URL,
        source_caveat=TAIMAN_SOURCE_CAVEAT,
        pokemon_usage=(
            PokemonUsage(species="azumarill", usage_pct=14.7, rank=1),
            PokemonUsage(species="registeel", usage_pct=8.2, rank=None),
        ),
        team_usage=(
            TeamUsage(
                members=("Azumarill", "Annihilape", "Registeel"),
                usage_pct=3.4,
                rank=1,
            ),
            TeamUsage(
                members=("Medicham (Shadow)", "Lickitung (Shadow)", "Azumarill"),
                usage_pct=2.9,
                rank=None,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Sub-AC 3 central assertion — written and reloaded snapshots are equal
# ---------------------------------------------------------------------------


def test_round_trip_write_then_read_equals_original(tmp_path: Path) -> None:
    """write_snapshot → read_snapshot returns a snapshot equal to the original."""

    original = _sample_snapshot()

    stored = write_snapshot(original, cache_dir=tmp_path)
    assert isinstance(stored, StoredSnapshot)
    assert stored.path.exists()
    assert stored.schema_version == SCHEMA_VERSION

    reloaded = read_snapshot(stored.path)

    # The headline Sub-AC 3 assertion — equality across all fields.
    assert reloaded == original

    # And spot-check the most safety-critical scalar fields explicitly so
    # a failure points at the offending field rather than at the whole dataclass.
    assert reloaded.league == original.league
    assert reloaded.rating_bracket == original.rating_bracket
    assert reloaded.fetched_at == original.fetched_at
    assert reloaded.source_url == original.source_url
    assert reloaded.source_caveat == original.source_caveat
    assert reloaded.pokemon_usage == original.pokemon_usage
    assert reloaded.team_usage == original.team_usage


def test_round_trip_preserves_data_honesty_caveat(tmp_path: Path) -> None:
    """The data-honesty caveat round-trips byte-for-byte.

    Per seed.yaml's ``data_honesty`` evaluation principle, the caveat
    must surface on every meta_snapshot rendering — which means it must
    not be silently dropped or rewritten by the persistence layer.
    """

    original = _sample_snapshot()
    stored = write_snapshot(original, cache_dir=tmp_path)
    reloaded = read_snapshot(stored.path)

    assert reloaded.source_caveat == original.source_caveat == TAIMAN_SOURCE_CAVEAT
    assert "report-density" in reloaded.source_caveat.lower()


def test_round_trip_via_dict_layer_is_equal() -> None:
    """``snapshot_from_dict(snapshot_to_dict(s)) == s`` for the sample snapshot.

    Validates the pure-Python boundary independently of the filesystem
    layer — useful for unit-testing engine code that wants to clone or
    log a snapshot without touching disk.
    """

    original = _sample_snapshot()
    payload = snapshot_to_dict(original)
    assert payload["schema_version"] == SCHEMA_VERSION
    reloaded = snapshot_from_dict(payload)
    assert reloaded == original


# ---------------------------------------------------------------------------
# versioned-store properties
# ---------------------------------------------------------------------------


def test_written_file_carries_schema_version(tmp_path: Path) -> None:
    """The on-disk payload exposes ``schema_version`` at the top level."""

    stored = write_snapshot(_sample_snapshot(), cache_dir=tmp_path)
    data = json.loads(stored.path.read_text(encoding="utf-8"))
    assert data["schema_version"] == SCHEMA_VERSION


def test_default_filename_is_timestamped_and_filesystem_safe(tmp_path: Path) -> None:
    """The default filename embeds the timestamp and avoids unsafe chars.

    Critical because the cache lives on the user's machine — a filename
    with ``:`` would silently fail on Windows-mounted volumes.
    """

    snapshot = _sample_snapshot()
    filename = default_filename_for(snapshot)

    assert filename.endswith(".json")
    assert "great_league" in filename
    assert "ace" in filename
    # ``:`` and ``+`` from ISO 8601 must be squashed.
    assert ":" not in filename
    assert "+" not in filename
    # And the timestamp shows up in a recognizable form.
    assert "2026-05-13" in filename

    stored = write_snapshot(snapshot, cache_dir=tmp_path)
    assert stored.path.name == filename


def test_two_snapshots_with_different_fetched_at_do_not_collide(
    tmp_path: Path,
) -> None:
    """Different fetch timestamps produce different default filenames."""

    snap_a = _sample_snapshot(
        fetched_at=datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc)
    )
    snap_b = _sample_snapshot(
        fetched_at=datetime(2026, 5, 13, 13, 0, 0, tzinfo=timezone.utc)
    )

    stored_a = write_snapshot(snap_a, cache_dir=tmp_path)
    stored_b = write_snapshot(snap_b, cache_dir=tmp_path)

    assert stored_a.path != stored_b.path
    assert read_snapshot(stored_a.path) == snap_a
    assert read_snapshot(stored_b.path) == snap_b


def test_explicit_filename_override_is_honored(tmp_path: Path) -> None:
    """A caller-supplied filename is used verbatim.

    Useful for the CLI's ``--latest`` cache slot or test fixtures that
    want a deterministic path.
    """

    stored = write_snapshot(
        _sample_snapshot(), cache_dir=tmp_path, filename="latest.json"
    )
    assert stored.path.name == "latest.json"
    assert read_snapshot(stored.path) == _sample_snapshot()


# ---------------------------------------------------------------------------
# directory helpers
# ---------------------------------------------------------------------------


def test_list_snapshots_returns_files_newest_first(tmp_path: Path) -> None:
    """``list_snapshots`` orders by mtime, newest first."""

    snap = _sample_snapshot()
    a = write_snapshot(snap, cache_dir=tmp_path, filename="a.json").path
    b = write_snapshot(snap, cache_dir=tmp_path, filename="b.json").path
    # Make ``a`` look older than ``b`` regardless of write order.
    import os
    os.utime(a, (1_700_000_000, 1_700_000_000))
    os.utime(b, (1_800_000_000, 1_800_000_000))

    files = list_snapshots(tmp_path)
    assert files == [b, a]


def test_list_snapshots_ignores_tmp_artifacts(tmp_path: Path) -> None:
    """``*.tmp`` files left by an interrupted write are not listed."""

    write_snapshot(_sample_snapshot(), cache_dir=tmp_path, filename="ok.json")
    (tmp_path / "in-flight.json.tmp").write_text("partial", encoding="utf-8")

    listed = list_snapshots(tmp_path)
    assert len(listed) == 1
    assert listed[0].name == "ok.json"


def test_list_snapshots_on_missing_dir_returns_empty(tmp_path: Path) -> None:
    """A non-existent cache dir is not a failure — it's just empty."""

    missing = tmp_path / "never-created"
    assert list_snapshots(missing) == []
    assert latest_snapshot(missing) is None


def test_latest_snapshot_returns_most_recent(tmp_path: Path) -> None:
    """``latest_snapshot`` reads back the newest cached entry."""

    older = _sample_snapshot(
        fetched_at=datetime(2026, 5, 12, 0, 0, 0, tzinfo=timezone.utc),
        rating_bracket="upper",
    )
    newer = _sample_snapshot(
        fetched_at=datetime(2026, 5, 13, 0, 0, 0, tzinfo=timezone.utc),
        rating_bracket="ace",
    )

    older_path = write_snapshot(older, cache_dir=tmp_path).path
    newer_path = write_snapshot(newer, cache_dir=tmp_path).path

    import os
    os.utime(older_path, (1_700_000_000, 1_700_000_000))
    os.utime(newer_path, (1_800_000_000, 1_800_000_000))

    got = latest_snapshot(tmp_path)
    assert got == newer


# ---------------------------------------------------------------------------
# schema / decode error paths
# ---------------------------------------------------------------------------


def test_read_snapshot_rejects_unknown_schema_version(tmp_path: Path) -> None:
    """A newer ``schema_version`` must raise rather than be re-interpreted."""

    payload = snapshot_to_dict(_sample_snapshot())
    payload["schema_version"] = SCHEMA_VERSION + 999

    bad = tmp_path / "future.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SnapshotSchemaError) as excinfo:
        read_snapshot(bad)
    assert excinfo.value.found_version == SCHEMA_VERSION + 999


def test_read_snapshot_rejects_missing_schema_version(tmp_path: Path) -> None:
    """A payload with no ``schema_version`` is rejected."""

    payload = snapshot_to_dict(_sample_snapshot())
    payload.pop("schema_version")

    bad = tmp_path / "no-version.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SnapshotSchemaError):
        read_snapshot(bad)


def test_read_snapshot_rejects_invalid_json(tmp_path: Path) -> None:
    """Corrupt JSON is a decode error, distinct from a schema mismatch."""

    bad = tmp_path / "corrupt.json"
    bad.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(SnapshotDecodeError):
        read_snapshot(bad)


def test_read_snapshot_rejects_missing_required_field(tmp_path: Path) -> None:
    """A payload missing ``source_caveat`` (data_honesty!) is rejected."""

    payload = snapshot_to_dict(_sample_snapshot())
    payload.pop("source_caveat")

    bad = tmp_path / "no-caveat.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SnapshotDecodeError):
        read_snapshot(bad)


def test_read_snapshot_propagates_io_error(tmp_path: Path) -> None:
    """A missing file path raises ``SnapshotPersistError``."""

    missing = tmp_path / "nope.json"
    with pytest.raises(SnapshotPersistError):
        read_snapshot(missing)


def test_write_snapshot_creates_cache_dir(tmp_path: Path) -> None:
    """A non-existent cache dir is created as a side effect of writing."""

    nested = tmp_path / "deep" / "cache"
    assert not nested.exists()

    stored = write_snapshot(_sample_snapshot(), cache_dir=nested)
    assert nested.is_dir()
    assert stored.path.parent == nested.resolve()


# ---------------------------------------------------------------------------
# naive-datetime tolerance — the fetch layer always writes tz-aware UTC,
# but a hand-edited file might drop the tzinfo. Treat that as UTC so the
# round-trip still succeeds rather than silently mis-interpreting the time.
# ---------------------------------------------------------------------------


def test_naive_fetched_at_in_payload_is_interpreted_as_utc(tmp_path: Path) -> None:
    """A JSON file with a naive ISO timestamp loads as UTC."""

    payload = snapshot_to_dict(_sample_snapshot())
    # Strip the trailing +00:00 offset to simulate a hand-edited file.
    payload["fetched_at"] = "2026-05-13T12:00:00"

    naive = tmp_path / "naive.json"
    naive.write_text(json.dumps(payload), encoding="utf-8")

    snapshot = read_snapshot(naive)
    assert snapshot.fetched_at == datetime(
        2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc
    )
