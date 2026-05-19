"""Local persistence for normalized Taiman Party meta snapshots.

Sub-AC 3 contract: the persistence module writes a normalized
``MetaSnapshot`` to a *versioned* local store and can read it back so that
``read_snapshot(write_snapshot(s)) == s`` holds for every well-formed
snapshot the parser can produce.

Design choices and why they exist
---------------------------------

* **JSON, not pickle.** The cache is read by humans during long-loop
  validation (the rating-change feedback loop) and must survive Python
  version bumps. JSON is the smallest format that satisfies both. Pickle
  would break the "audit your own engine" affordance the seed cares about.

* **Schema versioning is explicit.** Each file carries a top-level
  ``schema_version`` integer. Reading a file with an unknown
  ``schema_version`` raises ``SnapshotSchemaError`` — never silently
  re-interpret bytes from a future writer. ``v0.1`` writes schema_version
  ``1``; later versions add a migration step instead of editing the writer.

* **Timestamped filenames.** The default filename embeds the league,
  rating bracket, and the fetch timestamp in a filesystem-safe ISO 8601
  variant (``:`` and ``+`` replaced with ``-`` / ``_`` so Windows is
  happy). Two snapshots taken at different times never collide. A caller
  that wants a deterministic filename can pass ``filename=...``.

* **Atomic writes.** Snapshots are written to a sibling
  ``*.json.tmp`` file and then ``os.replace``-d into place so a crash
  mid-write never leaves a half-written JSON behind for the next read.

* **The data-honesty caveat round-trips.** ``source_caveat`` is
  serialized verbatim and re-validated on read, so a snapshot that's
  been on disk for weeks still surfaces the report-density warning when
  re-rendered. Dropping it on read would silently violate the
  ``data_honesty`` evaluation principle.

The module is import-cycle safe: it depends on ``parse.taiman`` (for the
dataclasses) but ``parse.taiman`` does not import this module.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from gbl_hacker.parse.taiman import (
    GREAT_LEAGUE_LABEL,
    MetaSnapshot,
    MoveUsage,
    PokemonUsage,
    TeamUsage,
)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION: Final[int] = 1
"""Current on-disk schema version written by ``write_snapshot``.

Increment when the JSON payload structure changes in a way that an older
reader cannot interpret. A reader that encounters a newer
``schema_version`` raises ``SnapshotSchemaError`` rather than guessing.
"""

DEFAULT_CACHE_SUBDIR: Final[str] = "snapshots"
"""Default sub-directory under the cache root for snapshot files."""

_FILENAME_TIMESTAMP_RE = re.compile(r"[^0-9A-Za-z._-]")
"""Characters that get squashed to ``-`` in a snapshot filename.

The default filename embeds the fetch timestamp. ISO 8601 contains ``:``
and ``+`` which are not portable across all filesystems (Windows in
particular). Anything outside ``[0-9A-Za-z._-]`` is replaced.
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SnapshotPersistError(Exception):
    """Base class for snapshot persistence failures.

    Raised when reading or writing a snapshot file fails for a reason the
    caller can act on (bad path, bad payload, version mismatch). The
    underlying cause (if any) is preserved via ``__cause__``.
    """


class SnapshotSchemaError(SnapshotPersistError):
    """Raised when an on-disk snapshot's schema cannot be loaded.

    Covers three distinct failures, all of which mean "this file is not a
    snapshot this build can interpret":

    * ``schema_version`` is missing.
    * ``schema_version`` is newer than this build supports.
    * ``schema_version`` is older AND no migration is registered.

    Carries the offending version so the caller can log it and decide
    whether to re-fetch or to upgrade the engine.
    """

    def __init__(self, message: str, *, found_version: Any | None = None) -> None:
        super().__init__(message)
        self.found_version = found_version


class SnapshotDecodeError(SnapshotPersistError):
    """Raised when the snapshot file is not valid JSON or is structurally bad.

    Distinguishes "file is corrupt" (this exception) from "file is for a
    different schema" (``SnapshotSchemaError``) — the operator's response
    differs: corruption → re-fetch; schema mismatch → upgrade.
    """


# ---------------------------------------------------------------------------
# Public dataclass — file metadata helper
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoredSnapshot:
    """Result of ``write_snapshot``: the path on disk and the schema version.

    The path is absolute so callers can log it directly. The schema version
    is included so a caller writing many snapshots can spot the moment a
    schema bump landed.
    """

    path: Path
    schema_version: int


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------


def default_filename_for(snapshot: MetaSnapshot) -> str:
    """Build a deterministic, filesystem-safe filename for ``snapshot``.

    Format: ``{league}__{rating_bracket}__{iso_fetched_at}.json``

    ``iso_fetched_at`` is the snapshot's ``fetched_at`` in UTC, rendered
    with ``isoformat()`` and squashed to a portable character set:
    ``:`` → ``-``, ``+`` → ``_``, and any other non-portable character
    replaced with ``-``. Microseconds are preserved when present so
    sub-second snapshots do not collide.
    """

    ts = _normalize_to_utc(snapshot.fetched_at)
    safe_ts = _FILENAME_TIMESTAMP_RE.sub("-", ts.isoformat())
    league = _safe_segment(snapshot.league)
    bracket = _safe_segment(snapshot.rating_bracket)
    return f"{league}__{bracket}__{safe_ts}.json"


def _safe_segment(s: str) -> str:
    """Squash a free-form segment to a filename-safe form."""
    return _FILENAME_TIMESTAMP_RE.sub("-", s)


def _normalize_to_utc(dt: datetime) -> datetime:
    """Return ``dt`` as a tz-aware UTC datetime.

    Naive datetimes are interpreted as UTC (per the fetch layer's
    convention — ``datetime.now(tz=timezone.utc)``). Timezone-aware
    datetimes are converted in-place.
    """

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def snapshot_to_dict(snapshot: MetaSnapshot) -> dict[str, Any]:
    """Convert a ``MetaSnapshot`` to a JSON-serializable dict.

    The wrapping dict carries the ``schema_version`` so a reader can
    reject unknown versions. Tuples become lists (JSON has no tuples);
    the reader converts back to tuples on the way in.
    """

    return {
        "schema_version": SCHEMA_VERSION,
        "league": snapshot.league,
        "rating_bracket": snapshot.rating_bracket,
        "fetched_at": _normalize_to_utc(snapshot.fetched_at).isoformat(),
        "source_url": snapshot.source_url,
        "source_caveat": snapshot.source_caveat,
        "season": snapshot.season,
        "league_id": snapshot.league_id,
        "pokemon_usage": [
            {
                "species": p.species,
                "usage_pct": p.usage_pct,
                "rank": p.rank,
                "usage_count": p.usage_count,
                "dex_id": p.dex_id,
                "form_id": p.form_id,
                "fast_moves": [
                    {
                        "name": m.name,
                        "usage_count": m.usage_count,
                        "move_type": m.move_type,
                        "move_id": m.move_id,
                    }
                    for m in p.fast_moves
                ],
                "charged_moves": [
                    {
                        "name": m.name,
                        "usage_count": m.usage_count,
                        "move_type": m.move_type,
                        "move_id": m.move_id,
                    }
                    for m in p.charged_moves
                ],
            }
            for p in snapshot.pokemon_usage
        ],
        "team_usage": [
            {
                "members": list(t.members),
                "usage_pct": t.usage_pct,
                "rank": t.rank,
                "usage_count": t.usage_count,
                "member_forms": list(t.member_forms),
            }
            for t in snapshot.team_usage
        ],
    }


def snapshot_from_dict(data: dict[str, Any]) -> MetaSnapshot:
    """Rebuild a ``MetaSnapshot`` from a previously-serialized dict.

    Parameters
    ----------
    data:
        The dict produced by ``snapshot_to_dict`` (or read from a JSON
        file written by ``write_snapshot``).

    Raises
    ------
    SnapshotSchemaError
        ``schema_version`` is missing or unsupported.
    SnapshotDecodeError
        A required field is missing or has the wrong type.
    """

    version = data.get("schema_version")
    if version is None:
        raise SnapshotSchemaError(
            "Snapshot file is missing 'schema_version'", found_version=None
        )
    if not isinstance(version, int):
        raise SnapshotSchemaError(
            f"Snapshot 'schema_version' must be int, got {type(version).__name__}",
            found_version=version,
        )
    if version != SCHEMA_VERSION:
        raise SnapshotSchemaError(
            (
                f"Unsupported snapshot schema_version={version!r}; "
                f"this build writes/reads version {SCHEMA_VERSION}"
            ),
            found_version=version,
        )

    try:
        league = _require_str(data, "league")
        rating_bracket = _require_str(data, "rating_bracket")
        source_url = _require_str(data, "source_url")
        source_caveat = _require_str(data, "source_caveat")
        fetched_at_raw = _require_str(data, "fetched_at")
    except KeyError as exc:
        raise SnapshotDecodeError(f"Missing required snapshot field: {exc}") from exc
    except TypeError as exc:
        raise SnapshotDecodeError(str(exc)) from exc

    if league != GREAT_LEAGUE_LABEL:
        # The MetaSnapshot dataclass would also reject this, but raising
        # here gives a more specific error class for the persistence layer.
        raise SnapshotDecodeError(
            f"Snapshot league must be {GREAT_LEAGUE_LABEL!r}, got {league!r}"
        )

    try:
        fetched_at = datetime.fromisoformat(fetched_at_raw)
    except ValueError as exc:
        raise SnapshotDecodeError(
            f"Snapshot 'fetched_at' is not a valid ISO 8601 timestamp: {fetched_at_raw!r}"
        ) from exc
    if fetched_at.tzinfo is None:
        # Files were always written with UTC tz; a missing tz means an
        # older or hand-edited file. Interpret as UTC for round-trip
        # compatibility rather than failing the read.
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)

    pokemon_usage = tuple(
        _decode_pokemon(entry, index=i)
        for i, entry in enumerate(_require_list(data, "pokemon_usage"))
    )
    team_usage = tuple(
        _decode_team(entry, index=i)
        for i, entry in enumerate(_require_list(data, "team_usage"))
    )

    season_raw = data.get("season", 0)
    league_id_raw = data.get("league_id", 0)
    if not isinstance(season_raw, int) or isinstance(season_raw, bool):
        raise SnapshotDecodeError("'season' must be int")
    if not isinstance(league_id_raw, int) or isinstance(league_id_raw, bool):
        raise SnapshotDecodeError("'league_id' must be int")

    try:
        return MetaSnapshot(
            league=league,
            rating_bracket=rating_bracket,
            fetched_at=fetched_at,
            source_url=source_url,
            source_caveat=source_caveat,
            pokemon_usage=pokemon_usage,
            team_usage=team_usage,
            season=season_raw,
            league_id=league_id_raw,
        )
    except ValueError as exc:
        # The dataclass's __post_init__ guards (e.g., empty caveat) map
        # to a decode error here — the file passed JSON parsing but failed
        # domain validation, which is still "this file is unusable".
        raise SnapshotDecodeError(f"Snapshot failed validation: {exc}") from exc


def _require_str(data: dict[str, Any], key: str) -> str:
    if key not in data:
        raise KeyError(key)
    value = data[key]
    if not isinstance(value, str):
        raise TypeError(f"Field {key!r} must be string, got {type(value).__name__}")
    return value


def _require_list(data: dict[str, Any], key: str) -> list[Any]:
    if key not in data:
        raise SnapshotDecodeError(f"Missing required snapshot field: {key!r}")
    value = data[key]
    if not isinstance(value, list):
        raise SnapshotDecodeError(
            f"Field {key!r} must be a list, got {type(value).__name__}"
        )
    return value


def _decode_pokemon(entry: Any, *, index: int) -> PokemonUsage:
    if not isinstance(entry, dict):
        raise SnapshotDecodeError(
            f"pokemon_usage[{index}] must be an object, got {type(entry).__name__}"
        )
    try:
        species = entry["species"]
        usage_pct = entry["usage_pct"]
    except KeyError as exc:
        raise SnapshotDecodeError(
            f"pokemon_usage[{index}] missing field: {exc}"
        ) from exc
    rank = entry.get("rank")
    if not isinstance(species, str):
        raise SnapshotDecodeError(
            f"pokemon_usage[{index}].species must be string"
        )
    if not isinstance(usage_pct, (int, float)) or isinstance(usage_pct, bool):
        raise SnapshotDecodeError(
            f"pokemon_usage[{index}].usage_pct must be number"
        )
    if rank is not None and not isinstance(rank, int):
        raise SnapshotDecodeError(
            f"pokemon_usage[{index}].rank must be int or null"
        )
    usage_count = entry.get("usage_count", 0)
    dex_id = entry.get("dex_id")
    form_id = entry.get("form_id")
    if not isinstance(usage_count, int) or isinstance(usage_count, bool):
        raise SnapshotDecodeError(
            f"pokemon_usage[{index}].usage_count must be int"
        )
    if dex_id is not None and (not isinstance(dex_id, int) or isinstance(dex_id, bool)):
        raise SnapshotDecodeError(
            f"pokemon_usage[{index}].dex_id must be int or null"
        )
    if form_id is not None and (not isinstance(form_id, int) or isinstance(form_id, bool)):
        raise SnapshotDecodeError(
            f"pokemon_usage[{index}].form_id must be int or null"
        )
    fast_moves = tuple(_decode_move_usage_list(entry.get("fast_moves") or []))
    charged_moves = tuple(_decode_move_usage_list(entry.get("charged_moves") or []))
    try:
        return PokemonUsage(
            species=species,
            usage_pct=float(usage_pct),
            rank=rank,
            usage_count=usage_count,
            dex_id=dex_id,
            form_id=form_id,
            fast_moves=fast_moves,
            charged_moves=charged_moves,
        )
    except ValueError as exc:
        raise SnapshotDecodeError(
            f"pokemon_usage[{index}] failed validation: {exc}"
        ) from exc


def _decode_move_usage_list(raw: Any) -> list[MoveUsage]:
    if not isinstance(raw, list):
        return []
    out: list[MoveUsage] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            continue
        usage_count = entry.get("usage_count", 0)
        if not isinstance(usage_count, int):
            continue
        move_type = entry.get("move_type") or ""
        move_id = entry.get("move_id")
        out.append(
            MoveUsage(
                name=name,
                usage_count=usage_count,
                move_type=move_type if isinstance(move_type, str) else "",
                move_id=move_id if isinstance(move_id, int) else None,
            )
        )
    return out


def _decode_team(entry: Any, *, index: int) -> TeamUsage:
    if not isinstance(entry, dict):
        raise SnapshotDecodeError(
            f"team_usage[{index}] must be an object, got {type(entry).__name__}"
        )
    try:
        members = entry["members"]
        usage_pct = entry["usage_pct"]
    except KeyError as exc:
        raise SnapshotDecodeError(f"team_usage[{index}] missing field: {exc}") from exc
    rank = entry.get("rank")
    if not isinstance(members, list) or len(members) != 3:
        raise SnapshotDecodeError(
            f"team_usage[{index}].members must be a 3-item list"
        )
    if not all(isinstance(m, str) for m in members):
        raise SnapshotDecodeError(
            f"team_usage[{index}].members entries must be strings"
        )
    if not isinstance(usage_pct, (int, float)) or isinstance(usage_pct, bool):
        raise SnapshotDecodeError(f"team_usage[{index}].usage_pct must be number")
    if rank is not None and not isinstance(rank, int):
        raise SnapshotDecodeError(f"team_usage[{index}].rank must be int or null")
    usage_count = entry.get("usage_count", 0)
    member_forms_raw = entry.get("member_forms", [0, 0, 0])
    if not isinstance(usage_count, int) or isinstance(usage_count, bool):
        raise SnapshotDecodeError(f"team_usage[{index}].usage_count must be int")
    if (
        not isinstance(member_forms_raw, list)
        or len(member_forms_raw) != 3
        or not all(
            isinstance(f, int) and not isinstance(f, bool) for f in member_forms_raw
        )
    ):
        raise SnapshotDecodeError(
            f"team_usage[{index}].member_forms must be a 3-item int list"
        )
    try:
        return TeamUsage(
            members=(members[0], members[1], members[2]),
            usage_pct=float(usage_pct),
            rank=rank,
            usage_count=usage_count,
            member_forms=(
                member_forms_raw[0],
                member_forms_raw[1],
                member_forms_raw[2],
            ),
        )
    except ValueError as exc:
        raise SnapshotDecodeError(
            f"team_usage[{index}] failed validation: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Write / read
# ---------------------------------------------------------------------------


def write_snapshot(
    snapshot: MetaSnapshot,
    *,
    cache_dir: Path,
    filename: str | None = None,
) -> StoredSnapshot:
    """Persist ``snapshot`` to the local cache as a versioned JSON file.

    Parameters
    ----------
    snapshot:
        The normalized snapshot returned by ``parse_great_league_meta``.
    cache_dir:
        Directory the file is written into. Created (with parents) if it
        does not already exist. Pass any path the caller wants — typically
        ``~/.cache/gbl-hacker/snapshots`` or a test-only ``tmp_path``.
    filename:
        Override the auto-generated filename. Useful when a caller wants
        a deterministic name (e.g. ``"latest.json"``). When ``None`` the
        default timestamped filename from ``default_filename_for`` is used.

    Returns
    -------
    StoredSnapshot
        The absolute file path and the schema version that was written.

    Raises
    ------
    SnapshotPersistError
        Any I/O error encountered while creating the directory, writing
        the temp file, or atomically replacing the destination.
    """

    if filename is None:
        filename = default_filename_for(snapshot)

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SnapshotPersistError(
            f"Could not create snapshot cache dir {cache_dir!s}: {exc}"
        ) from exc

    target = (cache_dir / filename).resolve()
    payload = snapshot_to_dict(snapshot)
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)

    # Atomic write: write to a sibling temp file, fsync, then os.replace
    # onto the destination. A crash during the write leaves the previous
    # file (or no file) in place — never a half-written JSON.
    try:
        fd, tmp_path_str = tempfile.mkstemp(
            prefix=f".{filename}.",
            suffix=".tmp",
            dir=str(cache_dir),
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(encoded)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    # Some filesystems (tmpfs in CI sandboxes) do not
                    # support fsync. The replace below is still atomic on
                    # POSIX-compatible systems, so this is non-fatal.
                    pass
            os.replace(tmp_path, target)
        except Exception:
            # Best-effort cleanup of the temp file on failure.
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
    except OSError as exc:
        raise SnapshotPersistError(
            f"Could not write snapshot to {target!s}: {exc}"
        ) from exc

    return StoredSnapshot(path=target, schema_version=SCHEMA_VERSION)


def read_snapshot(path: Path) -> MetaSnapshot:
    """Load a snapshot previously written by ``write_snapshot``.

    Parameters
    ----------
    path:
        Absolute or relative path to the JSON file.

    Returns
    -------
    MetaSnapshot
        A fully-validated, immutable snapshot equivalent to the one that
        was originally written (subject to the round-trip guarantee in
        the module docstring).

    Raises
    ------
    SnapshotPersistError
        I/O error reading the file.
    SnapshotDecodeError
        File is not valid JSON or is missing required fields.
    SnapshotSchemaError
        File's ``schema_version`` is missing or unsupported.
    """

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SnapshotPersistError(
            f"Could not read snapshot file {path!s}: {exc}"
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SnapshotDecodeError(
            f"Snapshot file {path!s} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise SnapshotDecodeError(
            f"Snapshot file {path!s} root must be a JSON object"
        )

    return snapshot_from_dict(data)


def list_snapshots(cache_dir: Path) -> list[Path]:
    """List snapshot files in ``cache_dir`` sorted by mtime (newest first).

    Returns an empty list if the directory does not exist. Files that do
    not end in ``.json`` (or that are still the ``*.tmp`` artifacts of an
    in-flight write) are skipped.
    """

    if not cache_dir.exists():
        return []
    candidates = [
        p for p in cache_dir.iterdir() if p.is_file() and p.suffix == ".json"
    ]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates


def latest_snapshot(cache_dir: Path) -> MetaSnapshot | None:
    """Return the most recent snapshot in ``cache_dir`` or ``None`` if empty.

    Convenience wrapper for the CLI "use the last fetch" code path. The
    caller can still call ``list_snapshots`` + ``read_snapshot`` directly
    when it needs the path or a non-latest entry.
    """

    files = list_snapshots(cache_dir)
    if not files:
        return None
    return read_snapshot(files[0])


__all__ = [
    "DEFAULT_CACHE_SUBDIR",
    "SCHEMA_VERSION",
    "SnapshotDecodeError",
    "SnapshotPersistError",
    "SnapshotSchemaError",
    "StoredSnapshot",
    "default_filename_for",
    "latest_snapshot",
    "list_snapshots",
    "read_snapshot",
    "snapshot_from_dict",
    "snapshot_to_dict",
    "write_snapshot",
]
