"""Parsing layer — convert raw upstream responses to normalized snapshots.

Sub-modules:
    taiman: Convert a Taiman Party Great League fetch result into a typed
        ``MetaSnapshot``.

The parser layer is intentionally one-way: it consumes a ``FetchResult`` from
the fetch layer and yields immutable, validated domain objects (Pokémon
usage entries, team usage entries, the surrounding ``MetaSnapshot``). It
never performs I/O. This separation is what makes parser tests fully
offline against recorded fixture bytes.
"""

from gbl_hacker.parse.taiman import (
    DEFAULT_RATING_BRACKET,
    MetaSnapshot,
    PokemonUsage,
    TaimanParseError,
    TeamUsage,
    parse_great_league_meta,
)

__all__ = [
    "DEFAULT_RATING_BRACKET",
    "MetaSnapshot",
    "PokemonUsage",
    "TaimanParseError",
    "TeamUsage",
    "parse_great_league_meta",
]
