"""Deterministic red-team critique of a candidate team — the judgment layer.

This codifies the adversarial loop the project arrived at by hand: don't argue
about a team's weaknesses rhetorically, *settle them with the simulator*.

Two stages:

1. **Offensive blind spots (the hypothesis).** Across all three slots, collect
   the move types the team can deal damage with, then find the meta species the
   team has **no super-effective answer** to (net type multiplier ≤ 1.0 for
   every one of the team's offensive types). This is a *team-wide* offensive
   gap, not a per-mon defensive weakness — the distinction that matters: a
   hand-analysis of the 메더/파이어로/쏘콘 lead's defensive weaknesses missed
   that the whole team has no super-effective leverage on Diggersby
   (normal/ground), which turned out to be its real kryptonite.

2. **Worst matchups (the verdict).** Simulate the team against every meta team
   and rank by win rate. Matchups that both lose *and* contain a blind-spot
   species are the confirmed threats — hypothesis met evidence.

The blind-spot list alone is only a hypothesis (neutral damage + bulk can still
win); the sims are what convict it.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from gbl_hacker.gamemaster import GamemasterRegistry, load_default_gamemaster
from gbl_hacker.parse.taiman import MetaSnapshot
from gbl_hacker.score.expected_win_rate import (
    CandidateTeam,
    materialize_opponent_team,
)

# PvP type-effectiveness multipliers (GO uses 1.6 / 0.625 / 0.390625).
_SE = 1.6
_NVE = 0.625
_IMMUNE = 0.390625


def net_multiplier(
    atk_type: str, defender_types: Iterable[str], gm: GamemasterRegistry
) -> float:
    """Net damage multiplier of ``atk_type`` against a defender's type combo."""
    mult = 1.0
    atk = atk_type.lower()
    for dt in defender_types:
        chart = gm.type_chart.get(dt.lower())
        if not chart:
            continue
        if atk in chart.get("weaknesses", ()):
            mult *= _SE
        elif atk in chart.get("resistances", ()):
            mult *= _NVE
        elif atk in chart.get("immunities", ()):
            mult *= _IMMUNE
    return mult


def team_offensive_types(team: CandidateTeam) -> set[str]:
    """Every move type the team can deal damage with, across all slots."""
    out: set[str] = set()
    for build in team.slots:
        if build.fast.move_type:
            out.add(build.fast.move_type.lower())
        for cm in build.charged_moves:
            if cm.move_type:
                out.add(cm.move_type.lower())
    return out


@dataclass(frozen=True)
class BlindSpot:
    """A meta species the team has no super-effective answer to."""

    species: str
    types: tuple[str, ...]
    usage_count: int
    usage_pct: float


@dataclass(frozen=True)
class Matchup:
    """The team's simulated win rate against one meta opponent team."""

    members: tuple[str, ...]
    usage_count: int
    win_rate: float
    blind_spot_members: tuple[str, ...] = ()


@dataclass(frozen=True)
class Critique:
    offensive_types: tuple[str, ...]
    blind_spots: list[BlindSpot] = field(default_factory=list)
    worst_matchups: list[Matchup] = field(default_factory=list)


def offensive_blind_spots(
    team: CandidateTeam,
    meta: MetaSnapshot,
    *,
    gm: GamemasterRegistry | None = None,
) -> list[BlindSpot]:
    """Meta species (usage-ranked) the team can't hit super-effectively."""
    gm_r = gm or load_default_gamemaster()
    off = team_offensive_types(team)
    spots: list[BlindSpot] = []
    for usage in meta.pokemon_usage:
        pk = gm_r.resolve_build(
            dex_id=None, species_ja=usage.species, form_id=usage.form_id or 0
        )
        if pk is None:
            continue
        types = tuple(t for t in pk.types if t and t != "none")
        if any(net_multiplier(ot, types, gm_r) > 1.0 for ot in off):
            continue  # team has a super-effective answer
        spots.append(
            BlindSpot(
                species=usage.species,
                types=types,
                usage_count=usage.usage_count,
                usage_pct=usage.usage_pct,
            )
        )
    spots.sort(key=lambda s: s.usage_count, reverse=True)
    return spots


def critique_team(
    team: CandidateTeam,
    meta: MetaSnapshot,
    registry: dict,
    *,
    set_win_rate_fn: Callable[[CandidateTeam, CandidateTeam], float],
    gm: GamemasterRegistry | None = None,
) -> Critique:
    """Run the full red-team pass: blind-spot hypothesis + sim verdict."""
    gm_r = gm or load_default_gamemaster()
    blind = offensive_blind_spots(team, meta, gm=gm_r)
    blind_species = {b.species for b in blind}

    matchups: list[Matchup] = []
    for tu in meta.team_usage:
        try:
            opp = materialize_opponent_team(tu, registry)
        except Exception:
            continue
        wr = set_win_rate_fn(team, opp)
        flagged = tuple(m for m in tu.members if m in blind_species)
        matchups.append(
            Matchup(
                members=tuple(tu.members),
                usage_count=tu.usage_count,
                win_rate=wr,
                blind_spot_members=flagged,
            )
        )
    matchups.sort(key=lambda m: m.win_rate)
    return Critique(
        offensive_types=tuple(sorted(team_offensive_types(team))),
        blind_spots=blind,
        worst_matchups=matchups,
    )


def format_critique(
    critique: Critique,
    *,
    team_name: str,
    localize: Callable[[str], str] = lambda s: s,
    top_n: int = 6,
) -> str:
    """Render a critique as a text report. ``localize`` maps a JA species name
    to a display name (identity by default, for tests)."""
    off = ", ".join(critique.offensive_types) or "(none)"
    lines = [
        f"RED-TEAM CRITIQUE — {team_name}",
        f"  offensive coverage types: {off}",
    ]

    lines.append("")
    if critique.blind_spots:
        lines.append(
            f"  ⚠ offensive blind spots (no super-effective answer) — "
            f"top {min(top_n, len(critique.blind_spots))} by usage:"
        )
        for b in critique.blind_spots[:top_n]:
            lines.append(
                f"     {localize(b.species):<16} {'/'.join(b.types):<16} "
                f"usage {b.usage_count}"
            )
    else:
        lines.append("  no offensive blind spots — team hits the whole meta SE.")

    lines.append("")
    lines.append(f"  worst simulated matchups (bottom {top_n}, ⚠ = has a blind-spot member):")
    for m in critique.worst_matchups[:top_n]:
        flag = " ⚠" if m.blind_spot_members else ""
        members = " / ".join(localize(x) for x in m.members)
        lines.append(f"     {m.win_rate:>5.0%}  (usage {m.usage_count:>3})  {members}{flag}")

    return "\n".join(lines)


__all__ = [
    "BlindSpot",
    "Critique",
    "Matchup",
    "critique_team",
    "format_critique",
    "net_multiplier",
    "offensive_blind_spots",
    "team_offensive_types",
]
