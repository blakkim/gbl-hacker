"""Unit tests for ``meta_coverage`` (Sub-AC 2.3).

The headline contract this test suite enforces:

* ``meta_coverage(team, meta)`` returns a *single scalar* in ``[0, 1]``
  — the usage-weighted **fraction of the meta** the team handles at or
  above a configurable win-rate threshold.
* Coverage is a **mass** statistic (count, not rate); the threshold
  partitions opponents into "handled" vs "not handled" and the numerator
  is the realized usage share of the handled set.
* The denominator is the *realized* total usage (not 100), matching the
  ``expected_win_rate`` convention — a truncated or partially-skipped
  meta is honestly reported as "X of the scored slice" rather than
  silently penalized for un-scored mass.
* The aggregator is the outer layer; per-matchup combat is mocked via
  ``set_win_rate_fn`` in most tests for speed and determinism, with one
  integration test exercising the real :func:`resolve_matchup` path.
* Missing-build and degenerate-input semantics mirror
  :func:`expected_win_rate` / :func:`worst_case_robustness`, so a Pareto
  ranker can iterate all three metrics with a single error-handling
  strategy.
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
    meta_coverage,
)
from gbl_hacker.simulator import (
    ChargedMove,
    CombatantBuild,
    FastMove,
)

# --- fixture helpers ------------------------------------------------------
# Kept structurally identical to test_expected_win_rate.py and
# test_worst_case_robustness.py so cross-checking the three score axes is
# a one-glance affair.


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
    simulating any matchups — keeps coverage-math assertions
    independent of the per-matchup engine.
    """

    def _fn(a: CandidateTeam, b: CandidateTeam) -> float:
        return mapping[b.species]

    return _fn


# ---------------------------------------------------------------------------
# Degenerate / empty inputs
# ---------------------------------------------------------------------------


def test_empty_meta_returns_zero() -> None:
    """No team_usage rows ⇒ no signal ⇒ 0.0 (matches its siblings)."""

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta()
    score = meta_coverage(
        team,
        meta,
        build_registry={},
        set_win_rate_fn=_const_set_win(1.0),
    )
    assert score == 0.0


def test_zero_total_weight_returns_zero() -> None:
    """All-zero-usage meta ⇒ no realized slice ⇒ 0.0, no division-by-zero."""

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        (("a-1", "a-2", "a-3"), 0.0),
        (("b-1", "b-2", "b-3"), 0.0),
    )
    registry = _registry_with("a-1", "a-2", "a-3", "b-1", "b-2", "b-3")
    score = meta_coverage(
        team,
        meta,
        build_registry=registry,
        set_win_rate_fn=_const_set_win(1.0),
    )
    assert score == 0.0


# ---------------------------------------------------------------------------
# Single-opponent semantics — a binary above/below threshold
# ---------------------------------------------------------------------------


def test_single_opponent_above_threshold_returns_one() -> None:
    """One opponent, rate above threshold ⇒ full coverage."""

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta((("opp-1", "opp-2", "opp-3"), 42.0))
    registry = _registry_with("opp-1", "opp-2", "opp-3")
    score = meta_coverage(
        team,
        meta,
        build_registry=registry,
        threshold=0.5,
        set_win_rate_fn=_const_set_win(0.75),
    )
    assert score == 1.0


def test_single_opponent_below_threshold_returns_zero() -> None:
    """One opponent, rate below threshold ⇒ zero coverage."""

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta((("opp-1", "opp-2", "opp-3"), 42.0))
    registry = _registry_with("opp-1", "opp-2", "opp-3")
    score = meta_coverage(
        team,
        meta,
        build_registry=registry,
        threshold=0.5,
        set_win_rate_fn=_const_set_win(0.25),
    )
    assert score == 0.0


def test_exact_tie_at_default_threshold_is_counted_as_handled() -> None:
    """``rate == threshold`` is "handled" (closed lower bound).

    At the default ``threshold = 0.5`` an exact 0.5 set-win-rate (the
    tie value) counts toward coverage. This matches the GBL operator's
    intuition that an even matchup is not a loss.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta((("opp-1", "opp-2", "opp-3"), 50.0))
    registry = _registry_with("opp-1", "opp-2", "opp-3")
    score = meta_coverage(
        team,
        meta,
        build_registry=registry,
        threshold=0.5,
        set_win_rate_fn=_const_set_win(0.5),
    )
    assert score == 1.0


def test_strict_threshold_excludes_exact_tie() -> None:
    """A caller wanting strict edge passes ``threshold = 0.501``."""

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta((("opp-1", "opp-2", "opp-3"), 50.0))
    registry = _registry_with("opp-1", "opp-2", "opp-3")
    score = meta_coverage(
        team,
        meta,
        build_registry=registry,
        threshold=0.501,
        set_win_rate_fn=_const_set_win(0.5),
    )
    assert score == 0.0


# ---------------------------------------------------------------------------
# Coverage-mass arithmetic
# ---------------------------------------------------------------------------


def test_coverage_is_usage_weighted_mass_above_threshold() -> None:
    """Two opponents partition into handled vs not; numerator = handled mass.

    Setup:
      * Opp A: usage 60 %, win 0.8 (≥ 0.5, *handled*)
      * Opp B: usage 40 %, win 0.2 (< 0.5,  not handled)

    Coverage = 60 / (60 + 40) = 0.6. This is the canonical mass test —
    it would fail if the function returned a weighted-mean *rate* (which
    would be 0.8·0.6 + 0.2·0.4 = 0.56, distinguishable from 0.6).
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        (("a-1", "a-2", "a-3"), 60.0),
        (("b-1", "b-2", "b-3"), 40.0),
    )
    registry = _registry_with("a-1", "a-2", "a-3", "b-1", "b-2", "b-3")
    win_by_opp = _per_team_set_win(
        {
            ("a-1", "a-2", "a-3"): 0.8,
            ("b-1", "b-2", "b-3"): 0.2,
        }
    )
    score = meta_coverage(
        team,
        meta,
        build_registry=registry,
        threshold=0.5,
        set_win_rate_fn=win_by_opp,
    )
    assert score == pytest.approx(0.6)


def test_coverage_is_independent_of_handled_rate_magnitude() -> None:
    """A 0.51 win counts the same as a 0.99 win — coverage is *mass*, not rate.

    Two parallel setups, same opponent weights, but different magnitudes
    of the handled rate. Both must yield identical coverage, since the
    coverage statistic does not weight by how *decisively* you beat
    each handled opponent.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        (("a-1", "a-2", "a-3"), 60.0),
        (("b-1", "b-2", "b-3"), 40.0),
    )
    registry = _registry_with("a-1", "a-2", "a-3", "b-1", "b-2", "b-3")
    barely = _per_team_set_win(
        {
            ("a-1", "a-2", "a-3"): 0.51,
            ("b-1", "b-2", "b-3"): 0.0,
        }
    )
    crushing = _per_team_set_win(
        {
            ("a-1", "a-2", "a-3"): 0.99,
            ("b-1", "b-2", "b-3"): 0.0,
        }
    )
    barely_score = meta_coverage(
        team, meta, build_registry=registry, set_win_rate_fn=barely
    )
    crushing_score = meta_coverage(
        team, meta, build_registry=registry, set_win_rate_fn=crushing
    )
    assert barely_score == pytest.approx(0.6)
    assert crushing_score == pytest.approx(0.6)
    assert barely_score == pytest.approx(crushing_score)


def test_coverage_is_monotone_decreasing_in_threshold() -> None:
    """Tightening the threshold can only shrink (or preserve) coverage.

    Setup (with rates spanning 0.3, 0.5, 0.7, 0.9):
      * Opp A: 25 %, 0.3
      * Opp B: 25 %, 0.5
      * Opp C: 25 %, 0.7
      * Opp D: 25 %, 0.9

    Coverage at successive thresholds must be non-increasing.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        (("a-1", "a-2", "a-3"), 25.0),
        (("b-1", "b-2", "b-3"), 25.0),
        (("c-1", "c-2", "c-3"), 25.0),
        (("d-1", "d-2", "d-3"), 25.0),
    )
    registry = _registry_with(
        "a-1", "a-2", "a-3",
        "b-1", "b-2", "b-3",
        "c-1", "c-2", "c-3",
        "d-1", "d-2", "d-3",
    )
    win_by_opp = _per_team_set_win(
        {
            ("a-1", "a-2", "a-3"): 0.3,
            ("b-1", "b-2", "b-3"): 0.5,
            ("c-1", "c-2", "c-3"): 0.7,
            ("d-1", "d-2", "d-3"): 0.9,
        }
    )

    def _cov(threshold: float) -> float:
        return meta_coverage(
            team,
            meta,
            build_registry=registry,
            threshold=threshold,
            set_win_rate_fn=win_by_opp,
        )

    # Threshold 0.0 — every in-range rate qualifies → 1.0.
    # Threshold 0.5 — B, C, D qualify (≥0.5)        → 0.75.
    # Threshold 0.7 — C, D qualify                  → 0.50.
    # Threshold 0.9 — D qualifies                   → 0.25.
    # Threshold 0.91 — nothing qualifies            → 0.00.
    assert _cov(0.0) == pytest.approx(1.0)
    assert _cov(0.5) == pytest.approx(0.75)
    assert _cov(0.7) == pytest.approx(0.5)
    assert _cov(0.9) == pytest.approx(0.25)
    assert _cov(0.91) == pytest.approx(0.0)

    # Monotone decreasing in threshold — sweep a fine grid as a
    # property check that no quirky threshold violates monotonicity.
    rates = [_cov(t / 100.0) for t in range(0, 101)]
    for lo, hi in zip(rates[1:], rates[:-1]):
        assert lo <= hi + 1e-12, f"coverage rose with stricter threshold: {hi} → {lo}"


def test_threshold_zero_returns_one_for_any_realized_meta() -> None:
    """``threshold = 0.0`` ⇒ every in-range rate qualifies ⇒ 1.0.

    The broadest definition of "handled" — even a 0 % win rate "meets
    the bar of 0". Useful as a degenerate-case sanity floor.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        (("a-1", "a-2", "a-3"), 10.0),
        (("b-1", "b-2", "b-3"), 90.0),
    )
    registry = _registry_with("a-1", "a-2", "a-3", "b-1", "b-2", "b-3")
    score = meta_coverage(
        team,
        meta,
        build_registry=registry,
        threshold=0.0,
        set_win_rate_fn=_const_set_win(0.0),
    )
    assert score == pytest.approx(1.0)


def test_threshold_one_only_counts_perfect_wins() -> None:
    """``threshold = 1.0`` ⇒ only perfect-win opponents qualify."""

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        (("a-1", "a-2", "a-3"), 30.0),  # perfect win
        (("b-1", "b-2", "b-3"), 70.0),  # near-perfect, doesn't qualify
    )
    registry = _registry_with("a-1", "a-2", "a-3", "b-1", "b-2", "b-3")
    win_by_opp = _per_team_set_win(
        {
            ("a-1", "a-2", "a-3"): 1.0,
            ("b-1", "b-2", "b-3"): 0.99,
        }
    )
    score = meta_coverage(
        team,
        meta,
        build_registry=registry,
        threshold=1.0,
        set_win_rate_fn=win_by_opp,
    )
    assert score == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Truncation / denominator-normalization contract
# ---------------------------------------------------------------------------


def test_truncated_meta_normalizes_against_realized_total() -> None:
    """A meta summing to < 100 is reported as fraction of *realized* slice.

    Taiman Party regularly truncates its team-usage list (top-N teams
    that together cover, say, 30 % of reports). The coverage statistic
    must divide by the realized total (30), not by 100, so the operator
    reads "this team handles X% of the *scored slice*" rather than
    silently being penalized for upstream truncation.

    Setup:
      * Opp A: usage 20, win 0.8 (handled)
      * Opp B: usage 10, win 0.2 (not handled)

    Realized total = 30. Handled mass = 20. Coverage = 20/30 ≈ 0.667.
    A bug that divided by 100 would yield 0.20 (clearly distinguishable).
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        (("a-1", "a-2", "a-3"), 20.0),
        (("b-1", "b-2", "b-3"), 10.0),
    )
    registry = _registry_with("a-1", "a-2", "a-3", "b-1", "b-2", "b-3")
    win_by_opp = _per_team_set_win(
        {
            ("a-1", "a-2", "a-3"): 0.8,
            ("b-1", "b-2", "b-3"): 0.2,
        }
    )
    score = meta_coverage(
        team,
        meta,
        build_registry=registry,
        threshold=0.5,
        set_win_rate_fn=win_by_opp,
    )
    assert score == pytest.approx(20.0 / 30.0)


# ---------------------------------------------------------------------------
# Missing-build policy — mirrors expected_win_rate / worst_case_robustness
# ---------------------------------------------------------------------------


def test_missing_build_raises_by_default() -> None:
    """Default ``on_missing_build='raise'`` re-raises MissingBuildError."""

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta((("missing", "opp-2", "opp-3"), 100.0))
    registry = _registry_with("opp-2", "opp-3")
    with pytest.raises(MissingBuildError) as exc_info:
        meta_coverage(
            team,
            meta,
            build_registry=registry,
            set_win_rate_fn=_const_set_win(1.0),
        )
    assert exc_info.value.species == "missing"


def test_missing_build_skip_drops_team_from_denominator() -> None:
    """``on_missing_build='skip'`` drops the team; remaining usage normalizes.

    Setup:
      * Opp A: usage 60 %, win 0.8 (handled). Registered.
      * Opp B: usage 40 %, win 0.0 (would-be unhandled). **Unregistered**.

    With B skipped, the realized denominator is 60, and the handled
    numerator is also 60 → coverage = 1.0. A bug that kept B in the
    denominator anyway would return 60/100 = 0.6 (distinguishable).
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        (("a-1", "a-2", "a-3"), 60.0),
        (("b-1", "b-2", "b-3"), 40.0),
    )
    registry = _registry_with("a-1", "a-2", "a-3")
    score = meta_coverage(
        team,
        meta,
        build_registry=registry,
        on_missing_build="skip",
        set_win_rate_fn=_per_team_set_win({("a-1", "a-2", "a-3"): 0.8}),
    )
    assert score == pytest.approx(1.0)


def test_missing_build_skip_returns_zero_when_all_unmaterializable() -> None:
    """All-skipped meta returns 0.0 — consistent with the other axes."""

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        (("a-1", "a-2", "a-3"), 60.0),
        (("b-1", "b-2", "b-3"), 40.0),
    )
    score = meta_coverage(
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
        meta_coverage(
            team,
            meta,
            build_registry=_registry_with("opp-1", "opp-2", "opp-3"),
            on_missing_build="explode",  # type: ignore[arg-type]
            set_win_rate_fn=_const_set_win(1.0),
        )


# ---------------------------------------------------------------------------
# Range / contract enforcement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_threshold", [-0.01, 1.01, -1.0, 2.0, 99.9])
def test_threshold_out_of_range_rejected(bad_threshold: float) -> None:
    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta((("opp-1", "opp-2", "opp-3"), 100.0))
    with pytest.raises(ValueError, match="threshold"):
        meta_coverage(
            team,
            meta,
            build_registry=_registry_with("opp-1", "opp-2", "opp-3"),
            threshold=bad_threshold,
            set_win_rate_fn=_const_set_win(0.5),
        )


@pytest.mark.parametrize("bad_rate", [-0.01, 1.01, -1.0, 2.0, 99.9])
def test_set_win_rate_fn_out_of_range_raises(bad_rate: float) -> None:
    """An injected set-win-rate ∉ [0, 1] is treated as a logic bug upstream."""

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta((("opp-1", "opp-2", "opp-3"), 100.0))
    with pytest.raises(ValueError, match="out-of-range"):
        meta_coverage(
            team,
            meta,
            build_registry=_registry_with("opp-1", "opp-2", "opp-3"),
            set_win_rate_fn=_const_set_win(bad_rate),
        )


def test_result_is_always_in_unit_interval() -> None:
    """For any meta and any in-range rates the coverage is in [0, 1]."""

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
    # Every meta-coverage call must land in [0, 1] regardless of the
    # (rate, threshold) cross product.
    for rate in (0.0, 0.25, 0.5, 0.75, 1.0):
        for threshold in (0.0, 0.25, 0.5, 0.75, 1.0):
            score = meta_coverage(
                team,
                meta,
                build_registry=registry,
                threshold=threshold,
                set_win_rate_fn=_const_set_win(rate),
            )
            assert 0.0 <= score <= 1.0
            # When the constant rate clears the threshold, every realized
            # opponent qualifies → coverage 1.0; otherwise 0.0.
            expected = 1.0 if rate >= threshold else 0.0
            assert score == pytest.approx(expected), (
                f"rate={rate}, threshold={threshold}: expected {expected}, got {score}"
            )


# ---------------------------------------------------------------------------
# Orthogonality to the other score axes — Pareto-shape invariants
# ---------------------------------------------------------------------------


def test_coverage_and_expected_win_rate_are_distinct_signals() -> None:
    """Two teams can have the same mean win rate but different coverage.

    Constructive existence proof of the Pareto-axis orthogonality:
      * Team-shape A: dominates half the meta (1.0), loses other half (0.0)
                      → mean = 0.5, coverage(thresh=0.5) = 0.5
      * Team-shape B: edges all opponents (0.5)
                      → mean = 0.5, coverage(thresh=0.5) = 1.0

    Same expected_win_rate, different meta_coverage. If meta_coverage
    were a re-derivation of mean rate, the two would be equal. The
    Pareto ranker depends on this orthogonality to span the frontier.
    """

    from gbl_hacker.score import expected_win_rate

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        (("a-1", "a-2", "a-3"), 50.0),
        (("b-1", "b-2", "b-3"), 50.0),
    )
    registry = _registry_with("a-1", "a-2", "a-3", "b-1", "b-2", "b-3")

    shape_a = _per_team_set_win(
        {
            ("a-1", "a-2", "a-3"): 1.0,
            ("b-1", "b-2", "b-3"): 0.0,
        }
    )
    shape_b = _per_team_set_win(
        {
            ("a-1", "a-2", "a-3"): 0.5,
            ("b-1", "b-2", "b-3"): 0.5,
        }
    )

    mean_a = expected_win_rate(team, meta, build_registry=registry, set_win_rate_fn=shape_a)
    mean_b = expected_win_rate(team, meta, build_registry=registry, set_win_rate_fn=shape_b)
    cov_a = meta_coverage(team, meta, build_registry=registry, set_win_rate_fn=shape_a)
    cov_b = meta_coverage(team, meta, build_registry=registry, set_win_rate_fn=shape_b)

    assert mean_a == pytest.approx(0.5)
    assert mean_b == pytest.approx(0.5)
    assert cov_a == pytest.approx(0.5)
    assert cov_b == pytest.approx(1.0)  # using the closed-bound (>=) tie rule
    assert cov_a != pytest.approx(cov_b), (
        "coverage and expected_win_rate must be orthogonal signals — "
        "two teams with equal means here have different coverages by construction"
    )


# ---------------------------------------------------------------------------
# Integration: real simulator round-trip
# ---------------------------------------------------------------------------


def test_integration_coverage_against_dominated_opponent_is_one() -> None:
    """End-to-end smoke: strictly stronger team covers 100 % of a weak meta.

    Setup:
      * Your team: 3 high-damage builds (3 fast damage / 100 HP).
      * Opponent team: 3 low-damage builds (1 fast damage / 100 HP).

    Every 1v1 pairing is dominated → 9-pairing average is 1.0 ≥ any
    threshold ≤ 1.0 → coverage is 1.0.
    """

    your_team = CandidateTeam.from_slots(
        _build(f"strong-{i}", fast_damage=3) for i in range(3)
    )
    weak_species = [f"weak-{i}" for i in range(3)]
    opp_builds = {s: _build(s, fast_damage=1) for s in weak_species}
    meta = _make_meta((tuple(weak_species), 100.0))  # type: ignore[arg-type]
    score = meta_coverage(
        your_team,
        meta,
        build_registry=opp_builds,
    )
    assert score == pytest.approx(1.0), (
        "strict-dominance team must have full coverage of a weak meta; "
        f"got {score}"
    )


def test_integration_coverage_separates_strong_from_weak_team() -> None:
    """A strong team covers more of the meta than a weak one.

    Setup: same weak meta, but compare:
      * strong_team (fast_damage=3) — should cover 100 % at default threshold
      * even_team   (fast_damage=1) — covers some lesser fraction

    The point is not the exact number (which is owned by the simulator
    tests) but the *ordering*: coverage moves monotonically with team
    strength, which the Pareto ranker depends on for tie-breaking on
    other axes.
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

    strong_score = meta_coverage(strong_team, meta, build_registry=opp_builds)
    even_score = meta_coverage(even_team, meta, build_registry=opp_builds)
    assert strong_score >= even_score
    assert strong_score == pytest.approx(1.0)
