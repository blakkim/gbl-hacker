"""Build the static dex localization table from PokeAPI's CSV dump.

Downloads ``pokemon_species_names.csv`` from the PokeAPI repo and emits
``src/gbl_hacker/data/pokedex_localized.json`` with the shape:

    {
      "1": {"ja": "フシギダネ", "ko": "이상해씨", "en": "Bulbasaur"},
      ...
    }

The runtime ``gbl_hacker.dex`` module loads this file once and exposes a
``PokedexRegistry`` that resolves Taiman Party's Japanese ``poke_name``
strings (which are katakana — matching ``ja-hrkt`` in PokeAPI's
language table) into Korean and English display names.

Run with:
    uv run python scripts/build_dex.py
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import httpx

POKEAPI_CSV = (
    "https://raw.githubusercontent.com/PokeAPI/pokeapi/master/"
    "data/v2/csv/pokemon_species_names.csv"
)

# PokeAPI ``languages.csv`` language ids (verified 2026-05-13):
#   1  = ja-hrkt (katakana — matches Taiman Party's display)
#   3  = ko
#   9  = en
LANG_IDS: dict[str, int] = {"ja": 1, "ko": 3, "en": 9}

OUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "gbl_hacker"
    / "data"
    / "pokedex_localized.json"
)


def main() -> None:
    print(f"[build_dex] downloading {POKEAPI_CSV}")
    resp = httpx.get(POKEAPI_CSV, timeout=30.0)
    resp.raise_for_status()
    text = resp.text

    reader = csv.DictReader(io.StringIO(text))
    table: dict[str, dict[str, str]] = {}
    for row in reader:
        dex = row["pokemon_species_id"]
        try:
            lang_id = int(row["local_language_id"])
        except ValueError:
            continue
        for code, code_id in LANG_IDS.items():
            if lang_id == code_id:
                table.setdefault(dex, {})[code] = row["name"]
                break

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sorted_table = {k: table[k] for k in sorted(table, key=int)}
    OUT_PATH.write_text(
        json.dumps(sorted_table, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[build_dex] wrote {len(sorted_table)} entries to {OUT_PATH}")

    # Quick sanity check on the species the live site currently surfaces.
    samples = {"195": "ヌオー", "959": "デカヌチャン", "205": "フォレトス"}
    for dex_id, expected_ja in samples.items():
        entry = sorted_table.get(dex_id, {})
        ja = entry.get("ja", "?")
        ko = entry.get("ko", "?")
        en = entry.get("en", "?")
        flag = "OK" if ja == expected_ja else "MISMATCH"
        print(f"  [{flag}] dex {dex_id}: ja={ja} ko={ko} en={en}")


if __name__ == "__main__":
    main()
