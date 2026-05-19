"""Long-loop validation rating log.

Sub-modules:
    entry: ``RatingLogEntry`` data model and JSON ser/de.
    store: Append-only JSONL store for the entry stream.

The rating-log layer is the engine's *long-loop validation* feedback
path (see ``exit_conditions.long_loop_validation`` in ``seed.yaml``).
Each entry records one real-life GBL run of a recommended team — pre-
rating, post-rating, computed delta, and free-form operator notes. The
store sub-module is the on-disk tail: a JSONL file that the operator
appends to after each set and that the engine reads back in insertion
order to compute "did the recommendations actually pay off?" stats.

Future sub-ACs add the CLI command that records new entries; this
package only exposes the data model and the local store today.
"""

from gbl_hacker.rating_log.entry import (
    SCHEMA_VERSION,
    RatingLogDecodeError,
    RatingLogEntry,
    RatingLogError,
    RatingLogSchemaError,
    RatingLogValidationError,
    entry_from_dict,
    entry_from_json,
    entry_to_dict,
    entry_to_json,
)
from gbl_hacker.rating_log.store import (
    DEFAULT_STORE_FILENAME,
    append_entries,
    append_entry,
    count_entries,
    read_entries,
)

__all__ = [
    "DEFAULT_STORE_FILENAME",
    "RatingLogDecodeError",
    "RatingLogEntry",
    "RatingLogError",
    "RatingLogSchemaError",
    "RatingLogValidationError",
    "SCHEMA_VERSION",
    "append_entries",
    "append_entry",
    "count_entries",
    "entry_from_dict",
    "entry_from_json",
    "entry_to_dict",
    "entry_to_json",
    "read_entries",
]
