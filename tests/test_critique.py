"""Tests for the red-team critique judgment layer (:mod:`gbl_hacker.critique`)."""
from __future__ import annotations

from pathlib import Path

import pytest

from gbl_hacker.build_registry import (
    build_registry_for_meta,
    build_registry_pvpoke_top,
)
from gbl_hacker.critique import (
    critique_team,
    format_critique,
    net_multiplier,
    offensive_blind_spots,
    team_offensive_types,
)
from gbl_hacker.gamemaster import load_default_gamemaster
from gbl_hacker.persist.snapshot import read_snapshot
from gbl_hacker.score.expected_win_rate import CandidateTeam, set_driver_win_rate

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "snapshots"
    / "great_league__upper__2026-05-19.json"
)


@pytest.fixture(scope="module")
def snapshot():
    return read_snapshot(_FIXTURE)


@pytest.fixture(scope="module")
def meta_registry(snapshot):
    return build_registry_for_meta(snapshot)


@pytest.fixture(scope="module")
def recommended_team():
    """메더 / 파이어로 / 쏘콘 — the pinned recommendation, via the pvpoke path."""
    pvp = {b.species: b for _label, _sid, b in build_registry_pvpoke_top(30)}
    return CandidateTeam.from_slots(
        [pvp["マッギョ"], pvp["ファイアロー"], pvp["フォレトス"]]
    )


def _det_set_fn(a, b):
    return set_driver_win_rate(a, b, stochastic_samples=1, win_mode="ko")


# ----------------------------------------------------------- net_multiplier
def test_net_multiplier_known_cases():
    gm = load_default_gamemaster()
    assert net_multiplier("electric", ("ground",), gm) < 1.0  # ground immune
    assert net_multiplier("grass", ("water", "ground"), gm) == pytest.approx(1.6 * 1.6)
    assert net_multiplier("water", ("fire",), gm) == pytest.approx(1.6)
    assert net_multiplier("fire", ("water",), gm) == pytest.approx(0.625)
    assert net_multiplier("normal", ("normal",), gm) == pytest.approx(1.0)
    # Flying is SE on bug but resisted by steel → net neutral, not SE.
    assert net_multiplier("flying", ("bug", "steel"), gm) == pytest.approx(1.0)


# ------------------------------------------------------- offensive coverage
def test_team_offensive_types(recommended_team):
    assert team_offensive_types(recommended_team) == {
        "electric",
        "ground",
        "fire",
        "flying",
        "rock",
    }


def test_diggersby_is_an_offensive_blind_spot(recommended_team, snapshot):
    """The team-wide SE-gap analysis catches Diggersby (normal/ground) — the
    hole a per-mon defensive-weakness reading misses."""
    spots = offensive_blind_spots(recommended_team, snapshot)
    species = {s.species for s in spots}
    assert "ホルード" in species  # Diggersby
    # The team is not blind to *everything* — it hits a big chunk of the meta SE.
    assert len(spots) < len(snapshot.pokemon_usage)
    # Blind spots are usage-ranked descending.
    counts = [s.usage_count for s in spots]
    assert counts == sorted(counts, reverse=True)


def test_blind_spots_have_no_super_effective_answer(recommended_team, snapshot):
    gm = load_default_gamemaster()
    off = team_offensive_types(recommended_team)
    for spot in offensive_blind_spots(recommended_team, snapshot):
        assert all(net_multiplier(t, spot.types, gm) <= 1.0 for t in off)


# ------------------------------------------------ critique (hypothesis→verdict)
def test_worst_matchups_sorted_and_flagged(recommended_team, snapshot, meta_registry):
    crit = critique_team(
        recommended_team, snapshot, meta_registry, set_win_rate_fn=_det_set_fn
    )
    win_rates = [m.win_rate for m in crit.worst_matchups]
    assert win_rates == sorted(win_rates)  # ascending
    # The hypothesis (blind spots) meets the verdict (sims): at least one of the
    # very worst matchups contains a blind-spot member.
    assert any(m.blind_spot_members for m in crit.worst_matchups[:3])


def test_format_critique_report(recommended_team, snapshot, meta_registry):
    crit = critique_team(
        recommended_team, snapshot, meta_registry, set_win_rate_fn=_det_set_fn
    )
    out = format_critique(crit, team_name="메더/파이어로/쏘콘")
    assert "RED-TEAM CRITIQUE" in out
    assert "blind spot" in out.lower()
    assert "ホルード" in out  # identity localize keeps JA names
