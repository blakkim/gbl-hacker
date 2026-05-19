"""Static pokedex localization registry.

Loads ``data/pokedex_localized.json`` (built once via
``scripts/build_dex.py`` from PokeAPI's CSV) and exposes a small,
read-only registry that resolves Taiman Party's Japanese ``poke_name``
strings into Korean and English display names.

The registry is intentionally simple:

- ``by_dex``: ``dex_id → {ja, ko, en}``
- ``by_ja``: reverse index of the Japanese (``ja-hrkt`` / katakana) form
  back to ``dex_id``, used as a fallback when upstream omits the
  numeric id.

Form discriminators (``form_id`` in Taiman Party's payload) are NOT
resolved here. Shadows, alternate forms, and regional variants share
the same dex id as the base species; the parser carries ``form_id``
separately and the render layer is responsible for applying a "(Shadow)"
or "(Galar)" style suffix when displaying.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Literal

Language = Literal["ja", "ko", "en"]

_DATA_PACKAGE = "gbl_hacker.data"
_DATA_FILE = "pokedex_localized.json"


@dataclass(frozen=True, slots=True)
class PokedexEntry:
    """Localized display names for a single Pokémon species.

    Attributes
    ----------
    dex_id:
        National Pokédex number.
    ja:
        Japanese display name (``ja-hrkt`` / katakana form — matches the
        ``poke_name`` Taiman Party serves).
    ko:
        Korean display name.
    en:
        English display name.
    """

    dex_id: int
    ja: str
    ko: str
    en: str

    def localize(self, lang: Language) -> str:
        """Return the name in the requested language, falling back to ja."""
        match lang:
            case "ko":
                return self.ko or self.ja
            case "en":
                return self.en or self.ja
            case _:
                return self.ja


@dataclass(frozen=True, slots=True)
class PokedexRegistry:
    """Immutable lookup table over ``PokedexEntry`` rows."""

    by_dex: dict[int, PokedexEntry] = field(default_factory=dict)
    by_ja: dict[str, PokedexEntry] = field(default_factory=dict)

    def lookup(
        self,
        *,
        dex_id: int | None = None,
        species_ja: str | None = None,
    ) -> PokedexEntry | None:
        """Find an entry by dex id first, then by Japanese species name.

        Either argument may be ``None``. Returns ``None`` when no row
        matches — the caller is expected to fall back to the raw
        Japanese species string for display.
        """
        if dex_id is not None:
            entry = self.by_dex.get(dex_id)
            if entry is not None:
                return entry
        if species_ja:
            return self.by_ja.get(species_ja.strip())
        return None

    def localize(
        self,
        species_ja: str,
        *,
        dex_id: int | None = None,
        lang: Language = "ja",
    ) -> str:
        """Resolve a localized display name.

        Returns the raw ``species_ja`` string unchanged when the registry
        has no matching entry. The fallback keeps render output sensible
        for newly-introduced species not yet covered by the static CSV.
        """
        entry = self.lookup(dex_id=dex_id, species_ja=species_ja)
        if entry is None:
            return species_ja
        return entry.localize(lang)


def _load_table() -> dict[str, dict[str, str]]:
    """Read the packaged JSON dump as raw dict-of-dict."""
    data_text = resources.files(_DATA_PACKAGE).joinpath(_DATA_FILE).read_text(
        encoding="utf-8"
    )
    raw = json.loads(data_text)
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"{_DATA_FILE} must contain a top-level object, got {type(raw).__name__}"
        )
    return raw


@lru_cache(maxsize=1)
def load_default_registry() -> PokedexRegistry:
    """Build (and cache) the registry from the packaged static dump.

    The build is cheap (~1k entries) but cached because the registry is
    a process-wide read-only artifact.
    """
    raw = _load_table()
    by_dex: dict[int, PokedexEntry] = {}
    by_ja: dict[str, PokedexEntry] = {}
    for dex_str, fields in raw.items():
        try:
            dex_id = int(dex_str)
        except (TypeError, ValueError):
            continue
        if not isinstance(fields, dict):
            continue
        ja = (fields.get("ja") or "").strip()
        ko = (fields.get("ko") or "").strip()
        en = (fields.get("en") or "").strip()
        if not ja and not ko and not en:
            continue
        entry = PokedexEntry(dex_id=dex_id, ja=ja, ko=ko, en=en)
        by_dex[dex_id] = entry
        if ja:
            # First-write-wins on collisions; PokeAPI's primary ja-hrkt
            # entries are unique per dex id in practice.
            by_ja.setdefault(ja, entry)
    return PokedexRegistry(by_dex=by_dex, by_ja=by_ja)


__all__ = [
    "Language",
    "PokedexEntry",
    "PokedexRegistry",
    "load_default_registry",
]
