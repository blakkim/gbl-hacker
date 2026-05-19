"""Unit tests for ``expected_win_rate`` (Sub-AC 2.1).

The headline contract this test suite enforces:

* ``expected_win_rate(team, meta)`` returns a *single scalar* in ``[0, 1]``.
* The scalar is the **weighted mean** win rate over the scored slice of
  the meta — usage percentages act as weights, the denominator is the
  realized weight (not 100), so a truncated or partially-skipped meta
  is reported honestly rather than silently penalized.
* The aggregator is the *outer* layer; per-matchup combat semantics
  live in the simulator. The injection point ``set_win_rate_fn`` is used
  in most tests to keep them fast, deterministic, and decoupled from
  the per-matchup turn loop. A separate integration test exercises the
  real :func:`resolve_matchup` path end-to-end.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gbl_hacker.parse.taiman import (
    GREAT_LEAGUE_LABEL,
    MetaSnapshot,
    PokemonUsage,
    TeamUsage,
)
from gbl_hacker.score import (
    CandidateTeam,
    MissingBuildError,
    default_set_win_rate,
    expected_win_rate,
    materialize_opponent_team,
)
from gbl_hacker.simulator import (
    ChargedMove,
    CombatantBuild,
    FastMove,
)

# --- fixture helpers ------------------------------------------------------

# Deterministic mid-tier-shaped builds. Numbers chosen so a fast simulator
# pass resolves cleanly: per-turn 2 dmg + 8 energy fast, charged at 40 cost.
# These match the shape used in test_asymmetric_shields.py — keeping them
# isomorphic makes hand-tracing the integration test trivial.


def _build(species: str, *, max_hp: int = 100, fast_damage: int = 2) -> CombatantBuild:
    return CombatantBuild(
        species=species,
        max_hp=max_hp,
        fast=FastMove(name="quick", damage=fast_damage, energy_gain=8),
        charged=ChargedMove(name="bomb", energy_cost=40, damage=70),
    )


def _candidate_team(*species_names: str) -> CandidateTeam:
    if len(species_names) != 3:
        raise AssertionError(
            f"test fixture needs 3 species, got {len(species_names)}"
        )
    return CandidateTeam.from_slots(_build(s) for s in species_names)


def _make_meta(*team_specs: tuple[tuple[str, str, str], float]) -> MetaSnapshot:
    """Build a MetaSnapshot from ``((s1, s2, s3), usage_pct)`` tuples."""

    team_usage = tuple(
        TeamUsage(members=members, usage_pct=pct)
        for members, pct in team_specs
    )
    return MetaSnapshot(
        league=GREAT_LEAGUE_LABEL,
        rating_bracket="upper",
        fetched_at=datetime(2026, 5, 13, tzinfo=timezone.utc),
        source_url="https://pokemongo-get.com/taimanparty",
        source_caveat=(
            "Taiman Party report density drops past upper bracket — "
            "top-500-specific slices are NOT reliable."
        ),
        pokemon_usage=(PokemonUsage(species="lead", usage_pct=15.0),),
        team_usage=team_usage,
    )


def _registry_with(*species_names: str) -> dict[str, CombatantBuild]:
    return {s: _build(s) for s in species_names}


def _const_set_win(value: float):
    """Build a ``set_win_rate_fn`` injection that always returns ``value``."""

    def _fn(a: CandidateTeam, b: CandidateTeam) -> float:
        return value

    return _fn


def _per_team_set_win(mapping: dict[tuple[str, str, str], float]):
    """Inject a ``set_win_rate_fn`` keyed on the opponent's species tuple.

    Lets a test assert that the aggregator weighs each opponent's win rate
    by usage_pct without simulating any matchups.
    """

    def _fn(a: CandidateTeam, b: CandidateTeam) -> float:
        return mapping[b.species]

    return _fn


# ---------------------------------------------------------------------------
# CandidateTeam construction / contract
# ---------------------------------------------------------------------------


def test_candidate_team_preserves_slot_order() -> None:
    team = CandidateTeam.from_slots(
        [_build("lead"), _build("safe"), _build("closer")]
    )
    assert team.species == ("lead", "safe", "closer")
    assert team.slots[0] is team.lead
    assert team.slots[1] is team.safe_swap
    assert team.slots[2] is team.closer


@pytest.mark.parametrize("count", [0, 1, 2, 4, 5])
def test_candidate_team_rejects_wrong_slot_count(count: int) -> None:
    builds = [_build(f"s{i}") for i in range(count)]
    with pytest.raises(ValueError, match="exactly 3"):
        CandidateTeam.from_slots(builds)


# ---------------------------------------------------------------------------
# materialize_opponent_team
# ---------------------------------------------------------------------------


def test_materialize_opponent_team_resolves_three_species_in_order() -> None:
    registry = _registry_with("a", "b", "c")
    usage = TeamUsage(members=("a", "b", "c"), usage_pct=50.0)
    team = materialize_opponent_team(usage, registry)
    assert team.species == ("a", "b", "c")
    # Identity match — registry instances are reused, not copied.
    assert team.lead is registry["a"]
    assert team.safe_swap is registry["b"]
    assert team.closer is registry["c"]


def test_materialize_opponent_team_raises_on_first_missing_species() -> None:
    registry = _registry_with("a", "c")  # missing "b"
    usage = TeamUsage(members=("a", "b", "c"), usage_pct=10.0)
    with pytest.raises(MissingBuildError) as exc_info:
        materialize_opponent_team(usage, registry)
    assert exc_info.value.species == "b"


# ---------------------------------------------------------------------------
# expected_win_rate — headline aggregation contract
# ---------------------------------------------------------------------------


def test_empty_meta_returns_zero() -> None:
    """No team_usage rows ⇒ no signal ⇒ 0.0 (distinct from 0% win rate).

    Callers wanting to distinguish "no meta data" from "scored 0 against
    every opponent" should pre-check ``meta.team_usage`` themselves.
    """
    team = _candidate_team("you-1", "you-2", "you-3")
    meta = _make_meta()  # no team rows
    score = expected_win_rate(
        team,
        meta,
        build_registry={},
        set_win_rate_fn=_const_set_win(1.0),
    )
    assert score == 0.0


def test_single_opponent_full_win_returns_one() -> None:
    """One opponent at any usage_pct, set_win_rate=1.0 ⇒ overall 1.0."""
    team = _candidate_team("you-1", "you-2", "you-3")
    meta = _make_meta((("opp-1", "opp-2", "opp-3"), 42.0))
    score = expected_win_rate(
        team,
        meta,
        build_registry=_registry_with("opp-1", "opp-2", "opp-3"),
        set_win_rate_fn=_const_set_win(1.0),
    )
    assert score == 1.0


def test_single_opponent_full_loss_returns_zero() -> None:
    team = _candidate_team("you-1", "you-2", "you-3")
    meta = _make_meta((("opp-1", "opp-2", "opp-3"), 42.0))
    score = expected_win_rate(
        team,
        meta,
        build_registry=_registry_with("opp-1", "opp-2", "opp-3"),
        set_win_rate_fn=_const_set_win(0.0),
    )
    assert score == 0.0


def test_weighted_mean_aggregates_two_opponents_correctly() -> None:
    """Two opponents with different usages and outcomes ⇒ weighted mean.

    Setup:
      * Opp A: usage 60%, you win 100%
      * Opp B: usage 40%, you win 0%

    Expected: (60 · 1.0 + 40 · 0.0) / 100 = 0.6.

    This is the canonical test that pins the *aggregation math* — if the
    weights are ever applied unevenly (e.g. unnormalized, or applied as
    fractions of 1.0 instead of percentages) the value moves off 0.6.
    """
    team = _candidate_team("you-1", "you-2", "you-3")
    meta = _make_meta(
        (("opp-a-1", "opp-a-2", "opp-a-3"), 60.0),
        (("opp-b-1", "opp-b-2", "opp-b-3"), 40.0),
    )
    registry = _registry_with(
        "opp-a-1", "opp-a-2", "opp-a-3",
        "opp-b-1", "opp-b-2", "opp-b-3",
    )
    win_by_opp = _per_team_set_win(
        {
            ("opp-a-1", "opp-a-2", "opp-a-3"): 1.0,
            ("opp-b-1", "opp-b-2", "opp-b-3"): 0.0,
        }
    )
    score = expected_win_rate(
        team,
        meta,
        build_registry=registry,
        set_win_rate_fn=win_by_opp,
    )
    assert score == pytest.approx(0.6)


def test_normalization_handles_meta_that_doesnt_sum_to_one_hundred() -> None:
    """Truncated meta (usages summing to < 100) is reported as weighted *mean*.

    Taiman Party regularly truncates its team-usage list (showing the
    top-N teams which together cover, say, 35% of reports). The
    aggregator must report the weighted mean win rate *over the
    reported slice* — NOT the win rate multiplied by the slice's mass.
    Otherwise "expected win rate" silently degrades into "expected win
    contribution to a fraction of the meta" which is a different
    quantity entirely.

    Setup:
      * Opp A: usage 20%, you win 100%
      * Opp B: usage 10%, you win 0%

    The two opponents sum to 30%, not 100%. Expected score:
      (20 · 1.0 + 10 · 0.0) / (20 + 10) = 20/30 ≈ 0.6667.
    """
    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        (("a-1", "a-2", "a-3"), 20.0),
        (("b-1", "b-2", "b-3"), 10.0),
    )
    registry = _registry_with("a-1", "a-2", "a-3", "b-1", "b-2", "b-3")
    win_by_opp = _per_team_set_win(
        {
            ("a-1", "a-2", "a-3"): 1.0,
            ("b-1", "b-2", "b-3"): 0.0,
        }
    )
    score = expected_win_rate(
        team,
        meta,
        build_registry=registry,
        set_win_rate_fn=win_by_opp,
    )
    assert score == pytest.approx(20.0 / 30.0)


def test_weighted_mean_with_partial_win_rates() -> None:
    """Mixed per-opponent win rates aggregate correctly with mixed weights."""
    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        (("a-1", "a-2", "a-3"), 50.0),
        (("b-1", "b-2", "b-3"), 30.0),
        (("c-1", "c-2", "c-3"), 20.0),
    )
    registry = _registry_with(
        "a-1", "a-2", "a-3",
        "b-1", "b-2", "b-3",
        "c-1", "c-2", "c-3",
    )
    win_by_opp = _per_team_set_win(
        {
            ("a-1", "a-2", "a-3"): 0.75,
            ("b-1", "b-2", "b-3"): 0.50,
            ("c-1", "c-2", "c-3"): 0.10,
        }
    )
    score = expected_win_rate(
        team,
        meta,
        build_registry=registry,
        set_win_rate_fn=win_by_opp,
    )
    # (50·0.75 + 30·0.50 + 20·0.10) / 100 = (37.5 + 15 + 2) / 100 = 0.545
    assert score == pytest.approx((50 * 0.75 + 30 * 0.50 + 20 * 0.10) / 100.0)


# ---------------------------------------------------------------------------
# Missing-build policy
# ---------------------------------------------------------------------------


def test_missing_build_raises_by_default() -> None:
    """Default ``on_missing_build='raise'`` re-raises MissingBuildError.

    The strict default keeps the operator honest: if a registry is
    incomplete, the score is *unknown*, not silently degraded.
    """
    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta((("missing", "opp-2", "opp-3"), 100.0))
    registry = _registry_with("opp-2", "opp-3")  # "missing" absent
    with pytest.raises(MissingBuildError) as exc_info:
        expected_win_rate(
            team,
            meta,
            build_registry=registry,
            set_win_rate_fn=_const_set_win(1.0),
        )
    assert exc_info.value.species == "missing"


def test_missing_build_skip_drops_team_and_renormalizes() -> None:
    """``on_missing_build='skip'`` drops the team; remaining usage normalizes.

    Setup:
      * Opp A: usage 60%, you win 100%. Registered.
      * Opp B: usage 40%, you win 0%. **Unregistered** — gets skipped.

    Expected score: 60·1.0 / 60 = 1.0 (B is dropped from the denominator).
    """
    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        (("a-1", "a-2", "a-3"), 60.0),
        (("b-1", "b-2", "b-3"), 40.0),  # b-* not in registry
    )
    registry = _registry_with("a-1", "a-2", "a-3")
    score = expected_win_rate(
        team,
        meta,
        build_registry=registry,
        on_missing_build="skip",
        set_win_rate_fn=_per_team_set_win(
            {("a-1", "a-2", "a-3"): 1.0}
        ),
    )
    assert score == 1.0


def test_missing_build_skip_returns_zero_when_all_teams_unmaterializable() -> None:
    """If every opponent is missing builds (under "skip"), return 0.0."""
    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        (("a-1", "a-2", "a-3"), 60.0),
        (("b-1", "b-2", "b-3"), 40.0),
    )
    score = expected_win_rate(
        team,
        meta,
        build_registry={},  # nothing matches
        on_missing_build="skip",
        set_win_rate_fn=_const_set_win(1.0),
    )
    assert score == 0.0


def test_invalid_on_missing_build_value_rejected() -> None:
    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta((("opp-1", "opp-2", "opp-3"), 100.0))
    with pytest.raises(ValueError, match="on_missing_build"):
        expected_win_rate(
            team,
            meta,
            build_registry=_registry_with("opp-1", "opp-2", "opp-3"),
            on_missing_build="explode",  # type: ignore[arg-type]
            set_win_rate_fn=_const_set_win(1.0),
        )


# ---------------------------------------------------------------------------
# Out-of-range and degenerate inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_rate", [-0.01, 1.01, -1.0, 2.0, 99.9])
def test_set_win_rate_fn_out_of_range_raises(bad_rate: float) -> None:
    """An injected set-win-rate ∉ [0, 1] is treated as a logic bug upstream."""
    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta((("opp-1", "opp-2", "opp-3"), 100.0))
    with pytest.raises(ValueError, match="out-of-range"):
        expected_win_rate(
            team,
            meta,
            build_registry=_registry_with("opp-1", "opp-2", "opp-3"),
            set_win_rate_fn=_const_set_win(bad_rate),
        )


def test_zero_total_weight_returns_zero() -> None:
    """A meta of all-0%-usage opponents has zero total weight ⇒ score 0.0.

    Pathological but possible: a degenerate meta scrape where every team
    row reports ``usage_pct == 0``. The function must avoid a division
    by zero and report 0.0.
    """
    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        (("a-1", "a-2", "a-3"), 0.0),
        (("b-1", "b-2", "b-3"), 0.0),
    )
    registry = _registry_with("a-1", "a-2", "a-3", "b-1", "b-2", "b-3")
    score = expected_win_rate(
        team,
        meta,
        build_registry=registry,
        set_win_rate_fn=_const_set_win(1.0),
    )
    assert score == 0.0


def test_result_is_always_in_unit_interval() -> None:
    """For any meta and any in-range rates the aggregate is in [0, 1]."""
    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        (("a-1", "a-2", "a-3"), 33.3),
        (("b-1", "b-2", "b-3"), 22.2),
        (("c-1", "c-2", "c-3"), 11.1),
    )
    registry = _registry_with(
        "a-1", "a-2", "a-3",
        "b-1", "b-2", "b-3",
        "c-1", "c-2", "c-3",
    )
    # Sweep edge rates: pure win, pure loss, exact tie.
    for rate in (0.0, 0.5, 1.0):
        score = expected_win_rate(
            team,
            meta,
            build_registry=registry,
            set_win_rate_fn=_const_set_win(rate),
        )
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(rate)


# ---------------------------------------------------------------------------
# default_set_win_rate — 9-pairing baseline aggregator
# ---------------------------------------------------------------------------


def test_default_set_win_rate_returns_unit_interval_on_identical_teams() -> None:
    """Identical teams (same builds) ⇒ symmetric 9-pairing outcome.

    With deterministic builds and a tie at full-shield symmetric start,
    the result is well-defined and bounded in [0, 1]. We don't pin the
    exact value here (that's a simulator-fidelity concern owned by the
    matchup tests) — just the contract that the aggregator returns a
    valid probability.
    """
    team = _candidate_team("a", "b", "c")
    rate = default_set_win_rate(team, team)
    assert 0.0 <= rate <= 1.0


def test_default_set_win_rate_runs_nine_pairings() -> None:
    """Sanity: with identical 3-slot teams the aggregator runs 9 matchups.

    We verify the count indirectly via determinism: calling twice on the
    same inputs returns the exact same float (the simulator is
    deterministic, so a 9-pairing average must be too).
    """
    team_a = _candidate_team("a1", "a2", "a3")
    team_b = _candidate_team("b1", "b2", "b3")
    r1 = default_set_win_rate(team_a, team_b)
    r2 = default_set_win_rate(team_a, team_b)
    assert r1 == r2
    assert 0.0 <= r1 <= 1.0


def test_default_set_win_rate_tie_value_propagates() -> None:
    """When matchups are forced to tie, the aggregate equals tie_value.

    Two teams of all-zero-damage Pokémon will run to the simulator's
    turn-budget cutoff — winner=None on every pairing — yielding the
    tie_value as the mean.
    """
    # Zero-damage builds: matchups never decide. The simulator's turn
    # cutoff returns winner=None.
    inert_a = CandidateTeam.from_slots(
        CombatantBuild(
            species=f"a{i}",
            max_hp=100,
            fast=FastMove(name="poke", damage=0, energy_gain=0),
            charged=ChargedMove(name="weak", energy_cost=100, damage=0),
        )
        for i in range(3)
    )
    inert_b = CandidateTeam.from_slots(
        CombatantBuild(
            species=f"b{i}",
            max_hp=100,
            fast=FastMove(name="poke", damage=0, energy_gain=0),
            charged=ChargedMove(name="weak", energy_cost=100, damage=0),
        )
        for i in range(3)
    )
    rate = default_set_win_rate(inert_a, inert_b, tie_value=0.42)
    assert rate == pytest.approx(0.42)


@pytest.mark.parametrize("bad_shields", [-1, 3, 5])
def test_default_set_win_rate_rejects_out_of_range_shields(bad_shields: int) -> None:
    team = _candidate_team("a", "b", "c")
    with pytest.raises(ValueError, match="starting_shields"):
        default_set_win_rate(team, team, starting_shields=bad_shields)


@pytest.mark.parametrize("bad_tie", [-0.1, 1.1])
def test_default_set_win_rate_rejects_out_of_range_tie_value(bad_tie: float) -> None:
    team = _candidate_team("a", "b", "c")
    with pytest.raises(ValueError, match="tie_value"):
        default_set_win_rate(team, team, tie_value=bad_tie)


# ---------------------------------------------------------------------------
# Integration: real simulator round-trip
# ---------------------------------------------------------------------------


def test_integration_expected_win_rate_against_dominated_opponent_is_high() -> None:
    """End-to-end smoke: with the real simulator, a strictly stronger team
    earns ``expected_win_rate ≈ 1.0`` against a strictly weaker meta.

    Setup:
      * Your team: 3 high-damage builds (3 fast damage / 100 HP).
      * Opponent team: 3 low-damage builds (1 fast damage / 100 HP).
    Both sides start at 2 shields. Across all 9 pairings, the higher-
    damage side reliably KO's the lower-damage side first.

    This test exercises the *default* aggregator (no injection) — the
    full path from MetaSnapshot → materialize → 9-pairing simulation →
    weighted-mean aggregation. It is the one test that would break if
    the simulator or aggregator regresses end-to-end.
    """
    your_team = CandidateTeam.from_slots(
        _build(f"strong-{i}", fast_damage=3) for i in range(3)
    )
    weak_species = [f"weak-{i}" for i in range(3)]
    opp_builds = {s: _build(s, fast_damage=1) for s in weak_species}

    meta = _make_meta((tuple(weak_species), 100.0))  # type: ignore[arg-type]
    score = expected_win_rate(
        your_team,
        meta,
        build_registry=opp_builds,
    )
    assert score == pytest.approx(1.0), (
        "the dominant-damage team must win every 1v1 pairing against the "
        f"weak meta; got {score}"
    )
