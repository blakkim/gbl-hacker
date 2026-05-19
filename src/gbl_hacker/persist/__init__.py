"""Local persistence layer for normalized meta snapshots.

Sub-modules:
    snapshot: Versioned JSON cache for ``MetaSnapshot`` instances.

The persistence layer is the boundary between in-memory engine state and
the long-loop validation log on disk. It writes timestamped, versioned
JSON files that round-trip back into immutable ``MetaSnapshot`` objects.
Keeping this layer separate from fetch/parse means a snapshot can be
re-loaded weeks later — without a live Taiman Party connection — to
reproduce a recommendation while the user reports rating-change data
back into the engine.
"""

from gbl_hacker.persist.snapshot import (
    DEFAULT_CACHE_SUBDIR,
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
