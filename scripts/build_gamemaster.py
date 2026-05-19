"""Build static PvPoke-derived game data dumps for the simulator.

Downloads PvPoke's ``gamemaster.json`` plus the two pieces it doesn't
ship in JSON form — the CPM table (in ``src/js/pokemon/Pokemon.js``) and
the type-traits chart (in ``src/js/battle/DamageCalculator.js``) — and
emits three slim files under ``src/gbl_hacker/data/``:

    gamemaster.json   — { "pokemon": [...], "moves": [...] } slim dump
    cpm.json          — { "<level_str>": <multiplier>, ... } level 1..51
    type_chart.json   — { "<type>": {weaknesses, resistances, immunities} }

The runtime ``gbl_hacker.gamemaster`` module loads these three files. They
are checked into the repo so that no live network call is needed at
runtime; re-run this script when PvPoke ships meaningful updates.

Run with:
    uv run python scripts/build_gamemaster.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx

PVPOKE_RAW = "https://raw.githubusercontent.com/pvpoke/pvpoke/master"
GAMEMASTER_URL = f"{PVPOKE_RAW}/src/data/gamemaster.json"
POKEMON_JS_URL = f"{PVPOKE_RAW}/src/js/pokemon/Pokemon.js"
DAMAGE_CALC_JS_URL = f"{PVPOKE_RAW}/src/js/battle/DamageCalculator.js"
RANKINGS_1500_URL = f"{PVPOKE_RAW}/src/data/rankings/all/overall/rankings-1500.json"
RANKINGS_LEADS_1500_URL = (
    f"{PVPOKE_RAW}/src/data/rankings/all/leads/rankings-1500.json"
)
POKEAPI_MOVE_NAMES_CSV = (
    "https://raw.githubusercontent.com/PokeAPI/pokeapi/master/"
    "data/v2/csv/move_names.csv"
)

OUT_DIR = (
    Path(__file__).resolve().parents[1] / "src" / "gbl_hacker" / "data"
)


def _http_get(url: str) -> str:
    print(f"[build_gamemaster] GET {url}")
    resp = httpx.get(url, timeout=60.0)
    resp.raise_for_status()
    return resp.text


def slim_gamemaster(raw_json_text: str) -> dict:
    """Strip ``gamemaster.json`` down to the fields the simulator needs.

    PvPoke's gamemaster is 1.6 MB. We only need:
      pokemon: dex, speciesId, speciesName, baseStats, types, fastMoves,
               chargedMoves, defaultIVs.cp1500, tags
      moves:   moveId, name, type, power, energy, energyGain, cooldown, turns
    """

    data = json.loads(raw_json_text)
    pokemon = []
    for p in data.get("pokemon", []):
        ivs = p.get("defaultIVs") or {}
        cp1500 = ivs.get("cp1500")  # [level, atk_iv, def_iv, hp_iv] or None
        pokemon.append(
            {
                "dex": p.get("dex"),
                "speciesId": p.get("speciesId"),
                "speciesName": p.get("speciesName"),
                "baseStats": p.get("baseStats"),
                "types": p.get("types"),
                "fastMoves": p.get("fastMoves") or [],
                "chargedMoves": p.get("chargedMoves") or [],
                "defaultIVs_cp1500": cp1500,
                "tags": p.get("tags") or [],
            }
        )

    moves = []
    for m in data.get("moves", []):
        moves.append(
            {
                "moveId": m.get("moveId"),
                "name": m.get("name"),
                "type": m.get("type"),
                "power": m.get("power"),
                "energy": m.get("energy"),
                "energyGain": m.get("energyGain"),
                "cooldown": m.get("cooldown"),
                "turns": m.get("turns"),
                "buffs": m.get("buffs"),
                "buffApplyChance": m.get("buffApplyChance"),
                "buffTarget": m.get("buffTarget"),
            }
        )

    return {"pokemon": pokemon, "moves": moves}


def extract_cpms(pokemon_js_text: str) -> dict[str, float]:
    """Pull the ``var cpms = [...];`` array out of PvPoke's Pokemon.js.

    Returns a mapping ``"<level>" → multiplier`` (level as string for
    JSON friendliness). The array is indexed at half-level granularity
    starting at level 1: ``cpms[0]`` = level 1, ``cpms[1]`` = 1.5, etc.
    """

    m = re.search(r"var\s+cpms\s*=\s*\[([^\]]+)\]\s*;", pokemon_js_text)
    if m is None:
        raise RuntimeError("could not locate `var cpms = [...]` in Pokemon.js")
    nums_text = m.group(1)
    values = [float(v.strip()) for v in nums_text.split(",") if v.strip()]
    table: dict[str, float] = {}
    for idx, cpm in enumerate(values):
        # cpms[2*(level-1)] = integer level
        # cpms[2*(level-1) + 1] = half level after `level`
        level = 1.0 + idx * 0.5
        # render with at most one decimal place; integers stay int-shaped
        key = f"{level:.1f}".rstrip("0").rstrip(".") if "." in f"{level:.1f}" else f"{int(level)}"
        # When the trailing-zero strip eats too much (e.g. "1." -> ""),
        # restore the integer form.
        if not key:
            key = "0"
        if key.endswith("."):
            key = key[:-1]
        table[key] = cpm
    return table


_TRAITS_RE = re.compile(
    r'case\s+"(?P<type>[a-z]+)":\s*'
    r"traits\s*=\s*\{\s*"
    r"resistances:\s*\[(?P<res>[^\]]*)\]\s*,?\s*"
    r"weaknesses:\s*\[(?P<wk>[^\]]*)\]\s*,?\s*"
    r"immunities:\s*\[(?P<im>[^\]]*)\]\s*\}"
)


def _split_quoted(items: str) -> list[str]:
    return [s.strip().strip('"').strip("'") for s in items.split(",") if s.strip()]


def extract_type_chart(damage_calc_js_text: str) -> dict[str, dict]:
    """Pull the per-type weakness/resistance/immunity traits switch.

    Returns ``{type: {"weaknesses": [...], "resistances": [...],
    "immunities": [...]}}`` for all 18 GO types.
    """

    chart: dict[str, dict] = {}
    for m in _TRAITS_RE.finditer(damage_calc_js_text):
        chart[m.group("type")] = {
            "weaknesses": _split_quoted(m.group("wk")),
            "resistances": _split_quoted(m.group("res")),
            "immunities": _split_quoted(m.group("im")),
        }
    if len(chart) != 18:
        raise RuntimeError(
            f"expected 18 type entries, parsed {len(chart)}: {sorted(chart)!r}"
        )
    return chart


def build_move_ja_to_pvpoke(
    move_names_csv: str,
    gamemaster_moves: list[dict],
) -> dict[str, str]:
    """Map Taiman Party Japanese move names → PvPoke ``moveId``.

    Strategy:

    1. Read PokeAPI's ``move_names.csv`` and bucket per-move-id by language:
       ``ja-hrkt`` (id 1) gives katakana names matching Taiman Party;
       ``en`` (id 9) gives the canonical English name.
    2. Normalize each English name to ``SCREAMING_SNAKE_CASE`` to match
       PvPoke's ``moveId`` convention (``Volt Switch`` → ``VOLT_SWITCH``).
    3. Cross-reference against the PvPoke gamemaster's move-id set so
       we never emit a mapping to a moveId PvPoke doesn't ship.

    Returns a ``{ja_name: pvpoke_move_id}`` dict.
    """
    import csv
    import io
    import re

    pvpoke_move_ids = {m["moveId"] for m in gamemaster_moves if m.get("moveId")}

    # PokeAPI language ids verified 2026-05-13:
    #   1 = ja-hrkt (katakana — Taiman Party's surface form)
    #   9 = en
    by_move: dict[str, dict[int, str]] = {}
    reader = csv.DictReader(io.StringIO(move_names_csv))
    for row in reader:
        try:
            mid = int(row["move_id"])
            lang = int(row["local_language_id"])
        except (TypeError, ValueError):
            continue
        if lang not in (1, 9):
            continue
        by_move.setdefault(mid, {})[lang] = row.get("name") or ""

    def _normalize(en_name: str) -> str:
        # Strip diacritics ("Pokémon" → "Pokemon"), collapse non-alnum
        # to underscore, uppercase. Matches PvPoke conventions for the
        # vast majority of moves.
        import unicodedata
        decomposed = unicodedata.normalize("NFKD", en_name)
        ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
        return re.sub(r"[^A-Za-z0-9]+", "_", ascii_only).strip("_").upper()

    mapping: dict[str, str] = {}
    for mid, names in by_move.items():
        ja = names.get(1) or ""
        en = names.get(9) or ""
        if not ja or not en:
            continue
        candidate = _normalize(en)
        # Try direct match first.
        if candidate in pvpoke_move_ids:
            mapping[ja] = candidate
            continue
        # A few PvPoke quirks: ``VICE_GRIP``, ``WATERFALL_CRUSH`` etc.
        # are already covered by the normalize step. Skip the move when
        # no PvPoke entry exists rather than emitting a guess.
    return mapping


def slim_rankings(raw_json_text: str, *, top_n: int = 200) -> list[dict]:
    """Strip PvPoke's rankings-1500 list down to the fields we use.

    Each entry preserved as ``{speciesId, score, moveset}`` — the
    PvPoke-recommended moveset is the most accurate GL fast+charged
    combination, far better than blindly using ``fastMoves[0]`` from
    the gamemaster's species list.
    """
    data = json.loads(raw_json_text)
    out: list[dict] = []
    for entry in data[:top_n]:
        out.append(
            {
                "speciesId": entry.get("speciesId"),
                "score": entry.get("score"),
                "rating": entry.get("rating"),
                "moveset": entry.get("moveset") or [],
            }
        )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gamemaster_text = _http_get(GAMEMASTER_URL)
    pokemon_js = _http_get(POKEMON_JS_URL)
    damage_calc_js = _http_get(DAMAGE_CALC_JS_URL)
    rankings_text = _http_get(RANKINGS_1500_URL)
    rankings_leads_text = _http_get(RANKINGS_LEADS_1500_URL)
    move_names_csv = _http_get(POKEAPI_MOVE_NAMES_CSV)

    slim = slim_gamemaster(gamemaster_text)
    cpms = extract_cpms(pokemon_js)
    type_chart = extract_type_chart(damage_calc_js)
    rankings = slim_rankings(rankings_text, top_n=200)
    rankings_leads = slim_rankings(rankings_leads_text, top_n=200)
    move_ja_to_pvpoke = build_move_ja_to_pvpoke(move_names_csv, slim["moves"])

    (OUT_DIR / "gamemaster.json").write_text(
        json.dumps(slim, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "cpm.json").write_text(
        json.dumps(cpms, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "type_chart.json").write_text(
        json.dumps(type_chart, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "rankings_gl.json").write_text(
        json.dumps(rankings, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "rankings_gl_leads.json").write_text(
        json.dumps(rankings_leads, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "move_ja_to_pvpoke.json").write_text(
        json.dumps(move_ja_to_pvpoke, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[build_gamemaster] wrote {OUT_DIR / 'gamemaster.json'}")
    print(
        f"  pokemon entries: {len(slim['pokemon'])}, "
        f"move entries: {len(slim['moves'])}"
    )
    print(f"[build_gamemaster] wrote {OUT_DIR / 'cpm.json'}  entries={len(cpms)}")
    print(
        f"[build_gamemaster] wrote {OUT_DIR / 'type_chart.json'}  "
        f"types={sorted(type_chart)}"
    )
    print(
        f"[build_gamemaster] wrote {OUT_DIR / 'rankings_gl.json'}  "
        f"top-{len(rankings)} (best: {rankings[0]['speciesId']} score={rankings[0]['score']})"
    )
    print(
        f"[build_gamemaster] wrote {OUT_DIR / 'rankings_gl_leads.json'}  "
        f"top-{len(rankings_leads)} (best lead: {rankings_leads[0]['speciesId']} score={rankings_leads[0]['score']})"
    )
    print(
        f"[build_gamemaster] wrote {OUT_DIR / 'move_ja_to_pvpoke.json'}  "
        f"entries={len(move_ja_to_pvpoke)} "
        f"(ボルトチェンジ → {move_ja_to_pvpoke.get('ボルトチェンジ', '<MISS>')})"
    )

    # Sanity samples
    for name in ("Quagsire", "Forretress", "Tinkaton"):
        p = next(
            (x for x in slim["pokemon"] if x["speciesName"] == name),
            None,
        )
        if p is not None:
            print(
                f"  [OK] {name}: dex={p['dex']} types={p['types']} "
                f"cp1500_iv={p['defaultIVs_cp1500']}"
            )
    fire = type_chart.get("fire", {})
    print(f"  [OK] fire weaknesses: {fire.get('weaknesses')}")
    print(f"  [OK] cpm @ level 40: {cpms.get('40')}")
    print(f"  [OK] cpm @ level 50: {cpms.get('50')}")


if __name__ == "__main__":
    main()
