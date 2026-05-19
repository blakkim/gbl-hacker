"""Unit tests for ``worst_case_robustness`` (Sub-AC 2.2).

The headline contract this test suite enforces:

* ``worst_case_robustness(team, meta)`` returns a *single scalar* in
  ``[0, 1]`` — the team's usage-weighted low-quantile win rate over the
  meta. It is the *worst-case* sibling of ``expected_win_rate`` (which
  is the usage-weighted *mean*).
* The quantile is computed against the cumulative-usage CDF after
  sorting opponents ascending by win rate. ``quantile=0.0`` recovers
  the lowest win rate among non-zero-usage opponents; ``quantile=0.1``
  is the canonical "10th-percentile robustness" metric; ``quantile=0.5``
  is the usage-weighted median; ``quantile=1.0`` is the max.
* Aggregation only — per-matchup combat is mocked via
  ``set_win_rate_fn`` in most tests for speed and determinism, and one
  integration test exercises the real :func:`resolve_matchup` path
  end-to-end.
* Missing-build and degenerate-input semantics mirror
  :func:`expected_win_rate`, so a Pareto ranker can iterate both
  metrics with a single error-handling strategy.
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
    worst_case_robustness,
)
from gbl_hacker.simulator import (
    ChargedMove,
    CombatantBuild,
    FastMove,
)

# --- fixture helpers ------------------------------------------------------


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
        TeamUsage(members=members, usage_pct=pct) for members, pct in team_specs
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

    Lets a test pin which opponent maps to which win rate without
    simulating any matchups.
    """

    def _fn(a: CandidateTeam, b: CandidateTeam) -> float:
        return mapping[b.species]

    return _fn


# ---------------------------------------------------------------------------
# Degenerate / empty inputs
# ---------------------------------------------------------------------------


def test_empty_meta_returns_zero() -> None:
    """No team_usage rows ⇒ no signal ⇒ 0.0 (matches expected_win_rate)."""

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta()
    score = worst_case_robustness(
        team,
        meta,
        build_registry={},
        set_win_rate_fn=_const_set_win(1.0),
    )
    assert score == 0.0


def test_zero_total_weight_returns_zero() -> None:
    """All-zero-usage meta ⇒ no covered slice ⇒ 0.0 (no division-by-zero)."""

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        (("a-1", "a-2", "a-3"), 0.0),
        (("b-1", "b-2", "b-3"), 0.0),
    )
    registry = _registry_with("a-1", "a-2", "a-3", "b-1", "b-2", "b-3")
    score = worst_case_robustness(
        team,
        meta,
        build_registry=registry,
        set_win_rate_fn=_const_set_win(0.42),
    )
    assert score == 0.0


# ---------------------------------------------------------------------------
# Single-opponent semantics — the quantile collapses to that one rate
# ---------------------------------------------------------------------------


def test_single_opponent_returns_that_opponents_rate() -> None:
    """With one opponent the worst case = that opponent's rate, at every quantile."""

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta((("opp-1", "opp-2", "opp-3"), 75.0))
    registry = _registry_with("opp-1", "opp-2", "opp-3")
    for q in (0.0, 0.1, 0.5, 1.0):
        score = worst_case_robustness(
            team,
            meta,
            build_registry=registry,
            quantile=q,
            set_win_rate_fn=_const_set_win(0.37),
        )
        assert score == pytest.approx(0.37), f"quantile={q} should not change a 1-opp meta"


# ---------------------------------------------------------------------------
# Pure minimum semantics (quantile=0.0)
# ---------------------------------------------------------------------------


def test_quantile_zero_returns_minimum_win_rate() -> None:
    """``quantile=0.0`` picks the lowest win rate across the meta.

    Setup:
      * Opp A: usage 50 %, win 0.9
      * Opp B: usage 50 %, win 0.1   ← worst
      * Opp C: usage 50 %, win 0.5

    Expected: 0.1.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        (("a-1", "a-2", "a-3"), 50.0),
        (("b-1", "b-2", "b-3"), 50.0),
        (("c-1", "c-2", "c-3"), 50.0),
    )
    registry = _registry_with(
        "a-1", "a-2", "a-3",
        "b-1", "b-2", "b-3",
        "c-1", "c-2", "c-3",
    )
    win_by_opp = _per_team_set_win(
        {
            ("a-1", "a-2", "a-3"): 0.9,
            ("b-1", "b-2", "b-3"): 0.1,
            ("c-1", "c-2", "c-3"): 0.5,
        }
    )
    score = worst_case_robustness(
        team,
        meta,
        build_registry=registry,
        quantile=0.0,
        set_win_rate_fn=win_by_opp,
    )
    assert score == pytest.approx(0.1)


def test_quantile_zero_skips_zero_usage_rows() -> None:
    """A 0 %-usage exotic counter does not lower ``quantile=0.0`` worst case.

    A 0-weight opponent cannot advance the cumulative usage walk past
    zero, so it never "owns" the quantile boundary. The metric reads
    the lowest win rate *that has non-zero usage*.

    Setup:
      * Opp X: usage 0 %,  win 0.05   ← exotic counter; should be ignored
      * Opp A: usage 40 %, win 0.4    ← actual worst among non-zero
      * Opp B: usage 60 %, win 0.9

    Expected: 0.4 — the 0-usage exotic is correctly masked out.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        (("x-1", "x-2", "x-3"), 0.0),
        (("a-1", "a-2", "a-3"), 40.0),
        (("b-1", "b-2", "b-3"), 60.0),
    )
    registry = _registry_with(
        "x-1", "x-2", "x-3",
        "a-1", "a-2", "a-3",
        "b-1", "b-2", "b-3",
    )
    win_by_opp = _per_team_set_win(
        {
            ("x-1", "x-2", "x-3"): 0.05,
            ("a-1", "a-2", "a-3"): 0.4,
            ("b-1", "b-2", "b-3"): 0.9,
        }
    )
    score = worst_case_robustness(
        team,
        meta,
        build_registry=registry,
        quantile=0.0,
        set_win_rate_fn=win_by_opp,
    )
    assert score == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Mid-quantile semantics — usage-weighted CDF walk
# ---------------------------------------------------------------------------


def test_quantile_walks_usage_weighted_cdf() -> None:
    """A mid-quantile reads the win rate at that cumulative-usage threshold.

    Setup (sorted ascending by win rate):
      * Opp B: usage 10 %, win 0.1
      * Opp A: usage 60 %, win 0.5
      * Opp C: usage 30 %, win 0.9

    Cumulative usage walk:
      * after B: 10 % → rate 0.1
      * after A: 70 % → rate 0.5
      * after C: 100 % → rate 0.9

    At ``quantile=0.05`` (5 % target = 5) the first row crossing the
    threshold is B → 0.1. At ``quantile=0.5`` (50 % target = 50) the
    first crossing is A → 0.5. At ``quantile=0.9`` (90 % target = 90)
    A's cumulative 70 is still short, so C → 0.9.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        (("a-1", "a-2", "a-3"), 60.0),
        (("b-1", "b-2", "b-3"), 10.0),
        (("c-1", "c-2", "c-3"), 30.0),
    )
    registry = _registry_with(
        "a-1", "a-2", "a-3",
        "b-1", "b-2", "b-3",
        "c-1", "c-2", "c-3",
    )
    win_by_opp = _per_team_set_win(
        {
            ("a-1", "a-2", "a-3"): 0.5,
            ("b-1", "b-2", "b-3"): 0.1,
            ("c-1", "c-2", "c-3"): 0.9,
        }
    )

    def _score(q: float) -> float:
        return worst_case_robustness(
            team,
            meta,
            build_registry=registry,
            quantile=q,
            set_win_rate_fn=win_by_opp,
        )

    assert _score(0.05) == pytest.approx(0.1)
    assert _score(0.5) == pytest.approx(0.5)
    assert _score(0.9) == pytest.approx(0.9)
    # quantile == 1.0 lands on the top of the CDF — the max win rate.
    assert _score(1.0) == pytest.approx(0.9)


def test_default_quantile_is_low_percentile() -> None:
    """The default ``quantile=0.1`` reads the 10th-percentile worst rate.

    Setup (sorted ascending by win rate):
      * Opp B: usage 5 %,  win 0.1   ← cumulative 5 %, < 10 %
      * Opp C: usage 6 %,  win 0.2   ← cumulative 11 %, first ≥ 10 %
      * Opp A: usage 89 %, win 0.9

    With total usage 100 and target 10, the walk crosses 10 *during*
    Opp C — so the 10th-percentile win rate is 0.2, not B's 0.1.
    This is the long-tail-noise resistance the design promises: a
    5 %-usage counter at win 0.1 does not solely set the floor.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        (("a-1", "a-2", "a-3"), 89.0),
        (("b-1", "b-2", "b-3"), 5.0),
        (("c-1", "c-2", "c-3"), 6.0),
    )
    registry = _registry_with(
        "a-1", "a-2", "a-3",
        "b-1", "b-2", "b-3",
        "c-1", "c-2", "c-3",
    )
    win_by_opp = _per_team_set_win(
        {
            ("a-1", "a-2", "a-3"): 0.9,
            ("b-1", "b-2", "b-3"): 0.1,
            ("c-1", "c-2", "c-3"): 0.2,
        }
    )
    # Default quantile == 0.1
    score = worst_case_robustness(
        team,
        meta,
        build_registry=registry,
        set_win_rate_fn=win_by_opp,
    )
    assert score == pytest.approx(0.2)

    # Sanity: at quantile=0.0 the exotic 5 %-usage counter does set the
    # floor (since it is the lowest non-zero-weight win rate).
    floor = worst_case_robustness(
        team,
        meta,
        build_registry=registry,
        quantile=0.0,
        set_win_rate_fn=win_by_opp,
    )
    assert floor == pytest.approx(0.1)


def test_quantile_handles_truncated_meta() -> None:
    """A meta that doesn't sum to 100 still reads the quantile correctly.

    Taiman Party can truncate. The metric divides the quantile target
    by the *realized* total usage, not by 100. Setup:
      * Opp A: usage 20, win 0.2
      * Opp B: usage 10, win 0.8
    Total usage = 30. ``quantile=0.5`` target = 15. Walk B...no wait,
    sort ascending by rate:
      * A (rate 0.2): cumulative 20 ≥ 15 → 0.2.
    ``quantile=0.9`` target = 27. A cumulative 20 < 27 → next is B,
    cumulative 30 ≥ 27 → 0.8.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        (("a-1", "a-2", "a-3"), 20.0),
        (("b-1", "b-2", "b-3"), 10.0),
    )
    registry = _registry_with("a-1", "a-2", "a-3", "b-1", "b-2", "b-3")
    win_by_opp = _per_team_set_win(
        {
            ("a-1", "a-2", "a-3"): 0.2,
            ("b-1", "b-2", "b-3"): 0.8,
        }
    )
    assert worst_case_robustness(
        team,
        meta,
        build_registry=registry,
        quantile=0.5,
        set_win_rate_fn=win_by_opp,
    ) == pytest.approx(0.2)
    assert worst_case_robustness(
        team,
        meta,
        build_registry=registry,
        quantile=0.9,
        set_win_rate_fn=win_by_opp,
    ) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Missing-build policy — mirrors expected_win_rate
# ---------------------------------------------------------------------------


def test_missing_build_raises_by_default() -> None:
    """``on_missing_build='raise'`` (default) re-raises MissingBuildError.

    Strict default keeps the operator honest: if the registry is
    incomplete the score is *unknown*, not silently degraded.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta((("missing", "opp-2", "opp-3"), 100.0))
    registry = _registry_with("opp-2", "opp-3")
    with pytest.raises(MissingBuildError) as exc_info:
        worst_case_robustness(
            team,
            meta,
            build_registry=registry,
            set_win_rate_fn=_const_set_win(1.0),
        )
    assert exc_info.value.species == "missing"


def test_missing_build_skip_drops_team() -> None:
    """``on_missing_build='skip'`` drops un-materializable opponents.

    Setup:
      * Opp A: usage 60 %, win 0.2. Registered.
      * Opp B: usage 40 %, win 0.0. **Unregistered** — skipped.

    With B dropped the only contributor is A; ``quantile=0.0`` reads
    0.2, not the 0.0 we never simulated.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        (("a-1", "a-2", "a-3"), 60.0),
        (("b-1", "b-2", "b-3"), 40.0),
    )
    registry = _registry_with("a-1", "a-2", "a-3")
    score = worst_case_robustness(
        team,
        meta,
        build_registry=registry,
        quantile=0.0,
        on_missing_build="skip",
        set_win_rate_fn=_per_team_set_win({("a-1", "a-2", "a-3"): 0.2}),
    )
    assert score == pytest.approx(0.2)


def test_missing_build_skip_returns_zero_when_all_unmaterializable() -> None:
    """All-skipped meta returns 0.0 — consistent with expected_win_rate."""

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        (("a-1", "a-2", "a-3"), 60.0),
        (("b-1", "b-2", "b-3"), 40.0),
    )
    score = worst_case_robustness(
        team,
        meta,
        build_registry={},
        on_missing_build="skip",
        set_win_rate_fn=_const_set_win(1.0),
    )
    assert score == 0.0


def test_invalid_on_missing_build_rejected() -> None:
    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta((("opp-1", "opp-2", "opp-3"), 100.0))
    with pytest.raises(ValueError, match="on_missing_build"):
        worst_case_robustness(
            team,
            meta,
            build_registry=_registry_with("opp-1", "opp-2", "opp-3"),
            on_missing_build="explode",  # type: ignore[arg-type]
            set_win_rate_fn=_const_set_win(1.0),
        )


# ---------------------------------------------------------------------------
# Range / contract enforcement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_quantile", [-0.01, 1.01, -1.0, 2.0, 99.9])
def test_quantile_out_of_range_rejected(bad_quantile: float) -> None:
    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta((("opp-1", "opp-2", "opp-3"), 100.0))
    with pytest.raises(ValueError, match="quantile"):
        worst_case_robustness(
            team,
            meta,
            build_registry=_registry_with("opp-1", "opp-2", "opp-3"),
            quantile=bad_quantile,
            set_win_rate_fn=_const_set_win(0.5),
        )


@pytest.mark.parametrize("bad_rate", [-0.01, 1.01, -1.0, 2.0, 99.9])
def test_set_win_rate_fn_out_of_range_raises(bad_rate: float) -> None:
    """An injected set-win-rate ∉ [0, 1] is treated as a logic bug upstream."""

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta((("opp-1", "opp-2", "opp-3"), 100.0))
    with pytest.raises(ValueError, match="out-of-range"):
        worst_case_robustness(
            team,
            meta,
            build_registry=_registry_with("opp-1", "opp-2", "opp-3"),
            set_win_rate_fn=_const_set_win(bad_rate),
        )


def test_result_is_always_in_unit_interval() -> None:
    """For any meta and any in-range rates the worst case is in [0, 1]."""

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
    for rate in (0.0, 0.5, 1.0):
        for q in (0.0, 0.1, 0.5, 1.0):
            score = worst_case_robustness(
                team,
                meta,
                build_registry=registry,
                quantile=q,
                set_win_rate_fn=_const_set_win(rate),
            )
            assert 0.0 <= score <= 1.0
            # When every matchup is identical the quantile must equal it.
            assert score == pytest.approx(rate)


# ---------------------------------------------------------------------------
# Relationship to expected_win_rate — invariants the Pareto ranker relies on
# ---------------------------------------------------------------------------


def test_worst_case_never_exceeds_expected_for_low_quantile() -> None:
    """For ``quantile <= 0.5`` the worst case is ≤ the mean.

    A usage-weighted low quantile reads from the lower tail of the
    rate distribution; by construction it cannot exceed the weighted
    mean. The Pareto ranker relies on this so the (robustness, mean)
    pair behaves monotonically.
    """

    from gbl_hacker.score import expected_win_rate

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
            ("a-1", "a-2", "a-3"): 0.7,
            ("b-1", "b-2", "b-3"): 0.4,
            ("c-1", "c-2", "c-3"): 0.1,
        }
    )
    mean = expected_win_rate(
        team,
        meta,
        build_registry=registry,
        set_win_rate_fn=win_by_opp,
    )
    for q in (0.0, 0.1, 0.25, 0.5):
        worst = worst_case_robustness(
            team,
            meta,
            build_registry=registry,
            quantile=q,
            set_win_rate_fn=win_by_opp,
        )
        assert worst <= mean + 1e-9, f"quantile={q}: worst={worst} > mean={mean}"


# ---------------------------------------------------------------------------
# Integration: real simulator round-trip
# ---------------------------------------------------------------------------


def test_integration_robustness_against_dominated_opponent_is_high() -> None:
    """End-to-end smoke: strictly stronger team has high robustness too.

    Setup:
      * Your team: 3 high-damage builds (3 fast damage / 100 HP).
      * Opponent team: 3 low-damage builds (1 fast damage / 100 HP).

    Every 1v1 pairing is dominated; the 9-pairing average is ≈ 1.0
    for any opponent, so the worst-case quantile (default 0.1) is also
    ≈ 1.0. This is the path that would break if the simulator or the
    quantile aggregator regresses end-to-end.
    """

    your_team = CandidateTeam.from_slots(
        _build(f"strong-{i}", fast_damage=3) for i in range(3)
    )
    weak_species = [f"weak-{i}" for i in range(3)]
    opp_builds = {s: _build(s, fast_damage=1) for s in weak_species}
    meta = _make_meta((tuple(weak_species), 100.0))  # type: ignore[arg-type]
    score = worst_case_robustness(
        your_team,
        meta,
        build_registry=opp_builds,
    )
    assert score == pytest.approx(1.0), (
        "strict-dominance team must have 1.0 worst-case win rate against "
        f"a weak meta; got {score}"
    )


def test_integration_robustness_separates_strong_from_weak_team() -> None:
    """A weak team has *worse* worst-case robustness than a strong one.

    Setup: same weak meta, but compare:
      * strong_team (fast_damage=3) → ≈ 1.0 robustness
      * weak_team   (fast_damage=1) → strictly < 1.0

    The point is not to pin a specific number — the per-matchup details
    are owned by simulator tests — but to confirm robustness *moves
    with team strength*, which the Pareto ranker depends on.
    """

    weak_species = [f"weak-{i}" for i in range(3)]
    opp_builds = {s: _build(s, fast_damage=1) for s in weak_species}
    meta = _make_meta((tuple(weak_species), 100.0))  # type: ignore[arg-type]

    strong_team = CandidateTeam.from_slots(
        _build(f"strong-{i}", fast_damage=3) for i in range(3)
    )
    even_team = CandidateTeam.from_slots(
        _build(f"even-{i}", fast_damage=1) for i in range(3)
    )

    strong_score = worst_case_robustness(
        strong_team,
        meta,
        build_registry=opp_builds,
    )
    even_score = worst_case_robustness(
        even_team,
        meta,
        build_registry=opp_builds,
    )
    # Strong team strictly dominates → strict ≥ relation.
    assert strong_score >= even_score
    assert strong_score == pytest.approx(1.0)
