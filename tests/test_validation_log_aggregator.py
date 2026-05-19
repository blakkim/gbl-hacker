"""Tests for ``gbl_hacker.validation_log_aggregator`` (Sub-AC 7.4).

The headline contract this suite enforces:

* :func:`aggregate_validation_log` joins
  :class:`RatingLogEntry` records to :class:`ScoredTeam` predictions by
  ``team_id``, producing one :class:`TeamValidationSummary` per team in
  the recommendation mapping.
* Per-team summaries expose count-based actual statistics (wins,
  losses, evens, mean delta), the predicted scores from the join, and
  the residuals (``actual - predicted``).
* Entries whose ``team_id`` is **not** in the recommendation mapping
  land in :attr:`ValidationLogAggregate.orphaned_entries` — surfaced,
  not dropped.
* Recommendations with **no** matched entries still appear in the
  summary list — the un-tested-team case is a feature.
* Population-level aggregates expose joined-slice and full-log views
  separately; orphans contribute to ``total_runs``/``total_delta`` but
  not to the joined-slice aggregates.
* The aggregator is pure: same inputs → byte-identical output.

Test groups:

1. Fixture builders & smoke construction.
2. Single-team join — actual stats / predicted side / residuals.
3. Multi-team join — order preservation, independent buckets.
4. Empty-entry case (un-tested team) and empty-recommendation case.
5. Orphan handling.
6. Aggregate-level statistics (joined-slice vs full-log).
7. Input validation (bad recommendations / bad entries).
8. Purity / immutability.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

import pytest

from gbl_hacker.rating_log import RatingLogEntry
from gbl_hacker.score import (
    CandidateTeam,
    Score,
    ScoredTeam,
)
from gbl_hacker.simulator import ChargedMove, CombatantBuild, FastMove
from gbl_hacker.validation_log_aggregator import (
    TeamValidationSummary,
    ValidationLogAggregate,
    aggregate_validation_log,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _build(species: str) -> CombatantBuild:
    """Build a cheap placeholder ``CombatantBuild`` for fixture use.

    The aggregator does not inspect build internals — it only consumes
    the prediction scores attached to the ``ScoredTeam``. Mirrors the
    pattern in ``test_pareto_filter.py`` for one-glance cross-check.
    """

    return CombatantBuild(
        species=species,
        max_hp=100,
        fast=FastMove(name="quick", damage=2, energy_gain=8),
        charged=ChargedMove(name="bomb", energy_cost=40, damage=70),
    )


def _team(label: str) -> CandidateTeam:
    """A 3-slot ``CandidateTeam`` whose slots are tagged for greppability."""

    return CandidateTeam.from_slots(
        [
            _build(f"{label}-lead"),
            _build(f"{label}-swap"),
            _build(f"{label}-closer"),
        ]
    )


def _scored(label: str, ev: float, worst: float, cov: float) -> ScoredTeam:
    """Convenience: ``ScoredTeam`` with the canonical three axes."""

    return ScoredTeam(
        team=_team(label),
        score=Score(
            expected_win_rate=ev,
            worst_case_robustness=worst,
            meta_coverage=cov,
        ),
    )


def _entry(
    *,
    team_id: str = "azu-anni-regi",
    pre_rating: int = 2400,
    post_rating: int = 2425,
    timestamp: datetime | None = None,
    notes: str | None = "fixture entry",
) -> RatingLogEntry:
    """Build a fully-populated ``RatingLogEntry`` for fixture use."""

    return RatingLogEntry(
        team_id=team_id,
        pre_rating=pre_rating,
        post_rating=post_rating,
        timestamp=timestamp
        or datetime(2026, 5, 13, 21, 30, 0, tzinfo=timezone.utc),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 1. Smoke construction
# ---------------------------------------------------------------------------


def test_empty_entries_and_empty_recommendations_returns_empty_aggregate() -> None:
    """Both inputs empty → empty summaries, empty orphans, zero counts."""

    aggregate = aggregate_validation_log([], {})

    assert isinstance(aggregate, ValidationLogAggregate)
    assert aggregate.summaries == ()
    assert aggregate.orphaned_entries == ()
    assert aggregate.total_runs == 0
    assert aggregate.total_delta == 0
    assert aggregate.joined_run_count == 0
    assert aggregate.joined_total_delta == 0
    assert aggregate.joined_mean_delta is None
    assert aggregate.teams_with_entries == 0
    assert aggregate.teams_without_entries == 0
    assert aggregate.orphaned_count == 0


def test_aggregate_returns_frozen_dataclass_with_tuple_fields() -> None:
    """Result is immutable: tuples, not lists; frozen, not mutable."""

    aggregate = aggregate_validation_log([_entry()], {"azu-anni-regi": _scored("a", 0.5, 0.5, 0.5)})

    assert isinstance(aggregate.summaries, tuple)
    assert isinstance(aggregate.orphaned_entries, tuple)
    assert isinstance(aggregate.summaries[0].entries, tuple)


# ---------------------------------------------------------------------------
# 2. Single-team join — actual stats / predicted side / residuals
# ---------------------------------------------------------------------------


def test_single_team_with_one_winning_entry_actual_stats() -> None:
    """One winning session: run_count=1, win_count=1, mean_delta=+25."""

    scored = _scored("a", ev=0.55, worst=0.40, cov=0.70)
    entry = _entry(pre_rating=2400, post_rating=2425, team_id="team-a")

    aggregate = aggregate_validation_log([entry], {"team-a": scored})

    assert len(aggregate.summaries) == 1
    summary = aggregate.summaries[0]

    assert summary.team_id == "team-a"
    assert summary.scored_team is scored
    assert summary.entries == (entry,)
    assert summary.run_count == 1
    assert summary.win_count == 1
    assert summary.loss_count == 0
    assert summary.even_count == 0
    assert summary.total_delta == 25
    assert summary.mean_delta == 25.0
    assert summary.actual_session_win_rate == 1.0
    assert summary.actual_non_loss_rate == 1.0


def test_single_team_predicted_side_mirrors_scored_team() -> None:
    """``predicted_*`` properties surface the joined ``Score`` axes verbatim."""

    scored = _scored("a", ev=0.55, worst=0.40, cov=0.70)
    entry = _entry(team_id="team-a")
    aggregate = aggregate_validation_log([entry], {"team-a": scored})

    summary = aggregate.summaries[0]
    assert summary.predicted_win_rate == 0.55
    assert summary.predicted_robustness == 0.40
    assert summary.predicted_meta_coverage == 0.70


def test_residual_positive_when_actual_beats_prediction() -> None:
    """1.0 actual minus 0.55 prediction → residual = +0.45."""

    scored = _scored("a", ev=0.55, worst=0.40, cov=0.70)
    entry = _entry(pre_rating=2400, post_rating=2425, team_id="team-a")
    aggregate = aggregate_validation_log([entry], {"team-a": scored})

    summary = aggregate.summaries[0]
    assert summary.win_rate_residual == pytest.approx(1.0 - 0.55)
    assert summary.robustness_residual == pytest.approx(1.0 - 0.40)


def test_residual_negative_when_actual_trails_prediction() -> None:
    """0.0 actual minus 0.65 prediction → residual = -0.65."""

    scored = _scored("a", ev=0.65, worst=0.55, cov=0.50)
    entry = _entry(pre_rating=2500, post_rating=2440, team_id="team-a")  # delta=-60
    aggregate = aggregate_validation_log([entry], {"team-a": scored})

    summary = aggregate.summaries[0]
    assert summary.win_count == 0
    assert summary.loss_count == 1
    assert summary.actual_session_win_rate == 0.0
    assert summary.win_rate_residual == pytest.approx(-0.65)
    # Non-loss rate is also 0 (no wins, no evens).
    assert summary.robustness_residual == pytest.approx(-0.55)


def test_single_team_multiple_entries_mixed_outcomes() -> None:
    """3 entries (W, L, E) → win=1, loss=1, even=1, mean=0 delta."""

    scored = _scored("a", ev=0.55, worst=0.40, cov=0.70)
    e_win = _entry(team_id="team-a", pre_rating=2400, post_rating=2430)   # +30
    e_loss = _entry(team_id="team-a", pre_rating=2400, post_rating=2370)  # -30
    e_even = _entry(team_id="team-a", pre_rating=2400, post_rating=2400)  # 0

    aggregate = aggregate_validation_log([e_win, e_loss, e_even], {"team-a": scored})
    summary = aggregate.summaries[0]

    assert summary.run_count == 3
    assert summary.win_count == 1
    assert summary.loss_count == 1
    assert summary.even_count == 1
    assert summary.total_delta == 0
    assert summary.mean_delta == 0.0
    # actual_session_win_rate = wins / runs = 1/3
    assert summary.actual_session_win_rate == pytest.approx(1 / 3)
    # actual_non_loss_rate = (wins + evens) / runs = 2/3
    assert summary.actual_non_loss_rate == pytest.approx(2 / 3)


def test_win_loss_even_counts_partition_run_count() -> None:
    """``run_count == win_count + loss_count + even_count`` always."""

    scored = _scored("a", ev=0.5, worst=0.5, cov=0.5)
    entries = [
        _entry(team_id="team-a", pre_rating=2400, post_rating=2425),
        _entry(team_id="team-a", pre_rating=2400, post_rating=2400),
        _entry(team_id="team-a", pre_rating=2400, post_rating=2300),
        _entry(team_id="team-a", pre_rating=2400, post_rating=2425),
    ]
    aggregate = aggregate_validation_log(entries, {"team-a": scored})

    s = aggregate.summaries[0]
    assert s.run_count == s.win_count + s.loss_count + s.even_count


def test_residual_uses_actual_minus_predicted_sign_convention() -> None:
    """Sign: positive = beat prediction; negative = trailed prediction."""

    scored = _scored("a", ev=0.30, worst=0.20, cov=0.10)
    # 4 wins / 4 → actual_win_rate = 1.0
    entries = [
        _entry(team_id="team-a", pre_rating=2400, post_rating=2410)
        for _ in range(4)
    ]
    aggregate = aggregate_validation_log(entries, {"team-a": scored})
    s = aggregate.summaries[0]

    # actual (1.0) - predicted (0.30) = +0.70 ⇒ positive ⇒ beat prediction.
    assert s.win_rate_residual is not None and s.win_rate_residual > 0


# ---------------------------------------------------------------------------
# 3. Multi-team join — order preservation, independent buckets
# ---------------------------------------------------------------------------


def test_summaries_follow_recommendation_mapping_iteration_order() -> None:
    """Mapping insertion order determines summary order — not entry order."""

    a = _scored("a", 0.6, 0.5, 0.5)
    b = _scored("b", 0.7, 0.6, 0.4)
    c = _scored("c", 0.5, 0.5, 0.5)
    recs: Mapping[str, ScoredTeam] = {"team-a": a, "team-b": b, "team-c": c}

    # Entries arrive in c, a, b order — but summaries still follow recs.
    entries = [
        _entry(team_id="team-c"),
        _entry(team_id="team-a"),
        _entry(team_id="team-b"),
    ]
    aggregate = aggregate_validation_log(entries, recs)

    assert [s.team_id for s in aggregate.summaries] == ["team-a", "team-b", "team-c"]


def test_multi_team_entries_route_to_correct_buckets() -> None:
    """Per-team entries route only to that team's summary."""

    a = _scored("a", 0.6, 0.5, 0.5)
    b = _scored("b", 0.7, 0.6, 0.4)
    entries = [
        _entry(team_id="team-a", pre_rating=2400, post_rating=2425),
        _entry(team_id="team-b", pre_rating=2500, post_rating=2470),
        _entry(team_id="team-a", pre_rating=2425, post_rating=2410),
    ]
    aggregate = aggregate_validation_log(entries, {"team-a": a, "team-b": b})

    sum_a = aggregate.summaries[0]
    sum_b = aggregate.summaries[1]

    assert sum_a.team_id == "team-a"
    assert sum_a.run_count == 2
    assert sum_a.total_delta == 25 + (-15)
    assert all(e.team_id == "team-a" for e in sum_a.entries)

    assert sum_b.team_id == "team-b"
    assert sum_b.run_count == 1
    assert sum_b.total_delta == -30
    assert all(e.team_id == "team-b" for e in sum_b.entries)


def test_per_team_entries_preserve_input_order() -> None:
    """A team's entries appear in the order they were yielded by the input."""

    scored = _scored("a", 0.5, 0.5, 0.5)
    e1 = _entry(team_id="team-a", pre_rating=2400, post_rating=2410,
                timestamp=datetime(2026, 5, 13, 18, 0, tzinfo=timezone.utc))
    e2 = _entry(team_id="team-a", pre_rating=2410, post_rating=2425,
                timestamp=datetime(2026, 5, 13, 19, 0, tzinfo=timezone.utc))
    e3 = _entry(team_id="team-a", pre_rating=2425, post_rating=2400,
                timestamp=datetime(2026, 5, 13, 20, 0, tzinfo=timezone.utc))

    aggregate = aggregate_validation_log([e1, e2, e3], {"team-a": scored})
    assert aggregate.summaries[0].entries == (e1, e2, e3)


# ---------------------------------------------------------------------------
# 4. Empty-entry team and empty-recommendation cases
# ---------------------------------------------------------------------------


def test_recommended_team_with_no_entries_appears_as_untested_summary() -> None:
    """Recommended-but-not-tested team gets a summary with no entries."""

    scored = _scored("a", ev=0.55, worst=0.40, cov=0.70)
    aggregate = aggregate_validation_log([], {"team-a": scored})

    assert len(aggregate.summaries) == 1
    summary = aggregate.summaries[0]
    assert summary.team_id == "team-a"
    assert summary.scored_team is scored
    assert summary.entries == ()
    assert summary.run_count == 0
    assert summary.win_count == 0
    assert summary.loss_count == 0
    assert summary.even_count == 0
    assert summary.total_delta == 0
    # No sessions → no actual stats; the None distinguishes "untested"
    # from "tested but with zero wins".
    assert summary.mean_delta is None
    assert summary.actual_session_win_rate is None
    assert summary.actual_non_loss_rate is None
    # Predicted side is still present.
    assert summary.predicted_win_rate == 0.55
    # Residuals are None because actual side is None.
    assert summary.win_rate_residual is None
    assert summary.robustness_residual is None


def test_mean_delta_zero_distinct_from_none() -> None:
    """Tested team with zero-delta session → ``mean_delta == 0.0`` (not None)."""

    scored = _scored("a", 0.5, 0.5, 0.5)
    entry = _entry(team_id="team-a", pre_rating=2400, post_rating=2400)
    aggregate = aggregate_validation_log([entry], {"team-a": scored})

    s = aggregate.summaries[0]
    assert s.mean_delta == 0.0  # NOT None
    assert s.mean_delta is not None


def test_empty_recommendations_with_entries_all_orphan() -> None:
    """No recommendations → every entry is an orphan; no summaries."""

    entries = [
        _entry(team_id="t-1"),
        _entry(team_id="t-2"),
        _entry(team_id="t-3"),
    ]
    aggregate = aggregate_validation_log(entries, {})

    assert aggregate.summaries == ()
    assert aggregate.orphaned_entries == tuple(entries)
    assert aggregate.orphaned_count == 3


# ---------------------------------------------------------------------------
# 5. Orphan handling
# ---------------------------------------------------------------------------


def test_unknown_team_id_goes_to_orphan_bucket() -> None:
    """Entry whose ``team_id`` is not in recommendations is orphaned."""

    scored = _scored("a", 0.5, 0.5, 0.5)
    matched = _entry(team_id="team-a")
    unmatched = _entry(team_id="rolled-off-team", pre_rating=2400, post_rating=2380)

    aggregate = aggregate_validation_log([matched, unmatched], {"team-a": scored})

    assert aggregate.summaries[0].entries == (matched,)
    assert aggregate.orphaned_entries == (unmatched,)
    assert aggregate.orphaned_count == 1


def test_orphan_order_preserves_input_order() -> None:
    """Orphans appear in the order they were yielded by ``entries``."""

    a = _entry(team_id="x")
    b = _entry(team_id="y")
    c = _entry(team_id="z")

    aggregate = aggregate_validation_log([c, a, b], {})
    assert aggregate.orphaned_entries == (c, a, b)


def test_orphans_excluded_from_joined_slice_aggregates() -> None:
    """Orphans contribute to total_*, not joined_*."""

    scored = _scored("a", 0.5, 0.5, 0.5)
    matched = _entry(team_id="team-a", pre_rating=2400, post_rating=2425)  # +25
    orphan = _entry(team_id="unknown", pre_rating=2400, post_rating=2300)  # -100

    aggregate = aggregate_validation_log([matched, orphan], {"team-a": scored})

    # Joined slice sees only the matched entry.
    assert aggregate.joined_run_count == 1
    assert aggregate.joined_total_delta == 25
    # Full log sees both.
    assert aggregate.total_runs == 2
    assert aggregate.total_delta == 25 + (-100)


# ---------------------------------------------------------------------------
# 6. Aggregate-level statistics
# ---------------------------------------------------------------------------


def test_joined_mean_delta_averages_only_joined_slice() -> None:
    """The joined mean ignores orphans."""

    a = _scored("a", 0.5, 0.5, 0.5)
    b = _scored("b", 0.5, 0.5, 0.5)
    entries = [
        _entry(team_id="team-a", pre_rating=2400, post_rating=2440),  # +40
        _entry(team_id="team-b", pre_rating=2400, post_rating=2380),  # -20
        _entry(team_id="team-b", pre_rating=2380, post_rating=2400),  # +20
        _entry(team_id="orphan", pre_rating=2400, post_rating=2000),  # -400
    ]
    aggregate = aggregate_validation_log(entries, {"team-a": a, "team-b": b})

    # 3 joined sessions, total delta = +40 - 20 + 20 = 40 → mean = 40/3
    assert aggregate.joined_run_count == 3
    assert aggregate.joined_total_delta == 40
    assert aggregate.joined_mean_delta == pytest.approx(40 / 3)
    # Full log includes orphan.
    assert aggregate.total_runs == 4
    assert aggregate.total_delta == 40 - 400


def test_joined_mean_delta_none_when_no_joined_sessions() -> None:
    """A recommendation set with no matching entries has ``None`` joined mean."""

    scored = _scored("a", 0.5, 0.5, 0.5)
    # Only orphans logged.
    aggregate = aggregate_validation_log(
        [_entry(team_id="other")], {"team-a": scored}
    )
    assert aggregate.joined_run_count == 0
    assert aggregate.joined_mean_delta is None
    # But the orphan still counts toward total_runs.
    assert aggregate.total_runs == 1


def test_teams_with_and_without_entries_counts() -> None:
    """Tested-vs-untested split is reflected in the aggregate."""

    a = _scored("a", 0.5, 0.5, 0.5)
    b = _scored("b", 0.5, 0.5, 0.5)
    c = _scored("c", 0.5, 0.5, 0.5)
    entries = [
        _entry(team_id="team-a"),
        _entry(team_id="team-c"),
        _entry(team_id="team-c"),
    ]
    aggregate = aggregate_validation_log(entries, {"team-a": a, "team-b": b, "team-c": c})

    assert aggregate.teams_with_entries == 2  # a, c
    assert aggregate.teams_without_entries == 1  # b


def test_total_runs_equals_joined_plus_orphan_count() -> None:
    """``total_runs == joined_run_count + orphaned_count`` invariant."""

    scored = _scored("a", 0.5, 0.5, 0.5)
    entries = [
        _entry(team_id="team-a"),
        _entry(team_id="team-a"),
        _entry(team_id="orphan-1"),
        _entry(team_id="orphan-2"),
    ]
    aggregate = aggregate_validation_log(entries, {"team-a": scored})

    assert aggregate.total_runs == aggregate.joined_run_count + aggregate.orphaned_count
    assert aggregate.total_runs == 4
    assert aggregate.joined_run_count == 2
    assert aggregate.orphaned_count == 2


# ---------------------------------------------------------------------------
# Fixture log + recommendation set (acceptance scenario)
# ---------------------------------------------------------------------------


def test_fixture_log_and_recommendation_set_produces_full_per_team_summary() -> None:
    """End-to-end fixture: 3 recommendations + 6 logged sessions + 1 orphan.

    This is the acceptance scenario for Sub-AC 7.4: a realistic
    fixture log paired with a small recommendation set, where the
    aggregator must compute every per-team and population-level
    statistic correctly. The expected values below are hand-computed
    so a regression in any branch lights up here.

    Recommendations:
        * azu-anni-regi → predicted (EV=0.60, robust=0.50, cov=0.70)
        * med-tox-trev  → predicted (EV=0.55, robust=0.45, cov=0.65)
        * carbink-mid   → predicted (EV=0.50, robust=0.55, cov=0.55)
                          [un-tested — no entries match this team]

    Log entries:
        * 4 × azu-anni-regi   → deltas (+30, +20, -10, +5)  → total +45
        * 2 × med-tox-trev    → deltas (-20, +10)            → total -10
        * 1 × stunfisk-prankster (orphan)  → delta +15
    """

    recs = {
        "azu-anni-regi": _scored("azu", ev=0.60, worst=0.50, cov=0.70),
        "med-tox-trev": _scored("med", ev=0.55, worst=0.45, cov=0.65),
        "carbink-mid": _scored("car", ev=0.50, worst=0.55, cov=0.55),
    }

    entries = [
        _entry(team_id="azu-anni-regi", pre_rating=2400, post_rating=2430),  # +30
        _entry(team_id="med-tox-trev", pre_rating=2430, post_rating=2410),   # -20
        _entry(team_id="azu-anni-regi", pre_rating=2410, post_rating=2430),  # +20
        _entry(team_id="stunfisk-prankster", pre_rating=2430, post_rating=2445),  # +15 (orphan)
        _entry(team_id="azu-anni-regi", pre_rating=2445, post_rating=2435),  # -10
        _entry(team_id="med-tox-trev", pre_rating=2435, post_rating=2445),   # +10
        _entry(team_id="azu-anni-regi", pre_rating=2445, post_rating=2450),  # +5
    ]

    aggregate = aggregate_validation_log(entries, recs)

    # --- Order ---
    assert [s.team_id for s in aggregate.summaries] == [
        "azu-anni-regi",
        "med-tox-trev",
        "carbink-mid",
    ]

    # --- azu-anni-regi summary ---
    azu = aggregate.summaries[0]
    assert azu.run_count == 4
    assert azu.win_count == 3  # +30, +20, +5
    assert azu.loss_count == 1  # -10
    assert azu.even_count == 0
    assert azu.total_delta == 30 + 20 - 10 + 5
    assert azu.mean_delta == pytest.approx(45 / 4)
    assert azu.actual_session_win_rate == pytest.approx(3 / 4)
    assert azu.actual_non_loss_rate == pytest.approx(3 / 4)
    assert azu.predicted_win_rate == 0.60
    assert azu.win_rate_residual == pytest.approx(3 / 4 - 0.60)
    assert azu.robustness_residual == pytest.approx(3 / 4 - 0.50)

    # --- med-tox-trev summary ---
    med = aggregate.summaries[1]
    assert med.run_count == 2
    assert med.win_count == 1  # +10
    assert med.loss_count == 1  # -20
    assert med.even_count == 0
    assert med.total_delta == -10
    assert med.mean_delta == pytest.approx(-5.0)
    assert med.actual_session_win_rate == pytest.approx(0.5)
    assert med.win_rate_residual == pytest.approx(0.5 - 0.55)
    assert med.robustness_residual == pytest.approx(0.5 - 0.45)

    # --- carbink-mid (un-tested) summary ---
    car = aggregate.summaries[2]
    assert car.run_count == 0
    assert car.entries == ()
    assert car.mean_delta is None
    assert car.actual_session_win_rate is None
    # Predicted side is preserved.
    assert car.predicted_win_rate == 0.50
    assert car.predicted_robustness == 0.55
    assert car.predicted_meta_coverage == 0.55
    # Residuals are None — no actual data to compare.
    assert car.win_rate_residual is None
    assert car.robustness_residual is None

    # --- Orphans ---
    assert aggregate.orphaned_count == 1
    assert aggregate.orphaned_entries[0].team_id == "stunfisk-prankster"
    assert aggregate.orphaned_entries[0].delta == 15

    # --- Population aggregates ---
    # Joined: 4 + 2 = 6 sessions; total delta = +45 + (-10) = +35.
    assert aggregate.joined_run_count == 6
    assert aggregate.joined_total_delta == 35
    assert aggregate.joined_mean_delta == pytest.approx(35 / 6)
    assert aggregate.teams_with_entries == 2
    assert aggregate.teams_without_entries == 1
    # Full log: 6 joined + 1 orphan = 7 sessions; +35 + 15 = +50.
    assert aggregate.total_runs == 7
    assert aggregate.total_delta == 50


# ---------------------------------------------------------------------------
# 7. Input validation
# ---------------------------------------------------------------------------


def test_aggregate_rejects_non_mapping_recommendations() -> None:
    """``recommendations`` must be a ``Mapping``; a list raises ``TypeError``."""

    with pytest.raises(TypeError, match="Mapping"):
        aggregate_validation_log([_entry()], [("team-a", _scored("a", 0.5, 0.5, 0.5))])  # type: ignore[arg-type]


def test_aggregate_rejects_non_string_recommendation_key() -> None:
    """Recommendation keys must be ``str``."""

    bad_recs: Mapping = {12345: _scored("a", 0.5, 0.5, 0.5)}  # type: ignore[var-annotated]
    with pytest.raises(TypeError, match="str"):
        aggregate_validation_log([_entry()], bad_recs)


def test_aggregate_rejects_non_scored_team_recommendation_value() -> None:
    """Recommendation values must be ``ScoredTeam``."""

    bad_recs: Mapping = {"team-a": "not-a-scored-team"}  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="ScoredTeam"):
        aggregate_validation_log([_entry()], bad_recs)


def test_aggregate_rejects_non_entry_in_entries() -> None:
    """Each element of ``entries`` must be ``RatingLogEntry``."""

    with pytest.raises(TypeError, match="RatingLogEntry"):
        aggregate_validation_log(
            [_entry(), {"team_id": "team-a"}],  # type: ignore[list-item]
            {"team-a": _scored("a", 0.5, 0.5, 0.5)},
        )


def test_aggregate_accepts_generator_as_entries() -> None:
    """``entries`` may be a one-shot generator (not just a list)."""

    scored = _scored("a", 0.5, 0.5, 0.5)

    def gen():
        yield _entry(team_id="team-a", pre_rating=2400, post_rating=2410)
        yield _entry(team_id="team-a", pre_rating=2410, post_rating=2425)

    aggregate = aggregate_validation_log(gen(), {"team-a": scored})
    assert aggregate.summaries[0].run_count == 2


# ---------------------------------------------------------------------------
# 8. Dataclass invariants
# ---------------------------------------------------------------------------


def test_team_validation_summary_is_frozen() -> None:
    """The summary dataclass is frozen — attribute assignment fails."""

    scored = _scored("a", 0.5, 0.5, 0.5)
    aggregate = aggregate_validation_log(
        [_entry(team_id="team-a")], {"team-a": scored}
    )
    summary = aggregate.summaries[0]
    with pytest.raises(Exception):  # FrozenInstanceError ⊂ Exception
        summary.team_id = "mutated"  # type: ignore[misc]


def test_validation_log_aggregate_is_frozen() -> None:
    """The aggregate dataclass is frozen — attribute assignment fails."""

    aggregate = aggregate_validation_log([], {})
    with pytest.raises(Exception):
        aggregate.summaries = ()  # type: ignore[misc]


def test_team_validation_summary_rejects_non_tuple_entries() -> None:
    """Direct construction with a list (not tuple) raises ``TypeError``.

    The aggregator always builds with tuples; this test guards a
    future caller that bypasses :func:`aggregate_validation_log` and
    hand-constructs a summary.
    """

    scored = _scored("a", 0.5, 0.5, 0.5)
    with pytest.raises(TypeError, match="tuple"):
        TeamValidationSummary(
            team_id="team-a",
            scored_team=scored,
            entries=[_entry()],  # type: ignore[arg-type]
        )


def test_team_validation_summary_rejects_non_entry_in_entries_tuple() -> None:
    """Construction-time entries-tuple member validation."""

    scored = _scored("a", 0.5, 0.5, 0.5)
    with pytest.raises(TypeError, match="RatingLogEntry"):
        TeamValidationSummary(
            team_id="team-a",
            scored_team=scored,
            entries=("not-an-entry",),  # type: ignore[arg-type]
        )


def test_team_validation_summary_accepts_none_scored_team() -> None:
    """``scored_team is None`` is a legal hand-construct shape."""

    entry = _entry(team_id="adhoc-team")
    summary = TeamValidationSummary(
        team_id="adhoc-team",
        scored_team=None,
        entries=(entry,),
    )
    assert summary.scored_team is None
    assert summary.predicted_win_rate is None
    assert summary.predicted_robustness is None
    assert summary.predicted_meta_coverage is None
    assert summary.win_rate_residual is None
    assert summary.robustness_residual is None
    # Actual side still computes — we have an entry.
    assert summary.run_count == 1
    assert summary.actual_session_win_rate == 1.0


# ---------------------------------------------------------------------------
# 9. Purity (same inputs → same outputs; inputs not mutated)
# ---------------------------------------------------------------------------


def test_aggregator_does_not_mutate_input_entries_list() -> None:
    """Calling the aggregator does not alter the caller's entries list."""

    scored = _scored("a", 0.5, 0.5, 0.5)
    entries = [
        _entry(team_id="team-a"),
        _entry(team_id="team-b"),  # orphan
    ]
    snapshot = list(entries)

    aggregate_validation_log(entries, {"team-a": scored})

    # Caller's list is untouched.
    assert entries == snapshot


def test_aggregator_does_not_mutate_input_recommendations_mapping() -> None:
    """Calling the aggregator does not alter the caller's recommendations."""

    scored_a = _scored("a", 0.5, 0.5, 0.5)
    scored_b = _scored("b", 0.6, 0.6, 0.6)
    recs = {"team-a": scored_a, "team-b": scored_b}
    snapshot_keys = list(recs.keys())

    aggregate_validation_log([_entry(team_id="team-a")], recs)

    assert list(recs.keys()) == snapshot_keys
    assert recs["team-a"] is scored_a
    assert recs["team-b"] is scored_b


def test_aggregator_is_deterministic_under_repeated_calls() -> None:
    """Same inputs → equal aggregates (purity / determinism)."""

    scored = _scored("a", 0.5, 0.5, 0.5)
    entries = [_entry(team_id="team-a")] * 3
    recs = {"team-a": scored}

    first = aggregate_validation_log(entries, recs)
    second = aggregate_validation_log(entries, recs)

    # Tuples / frozen dataclasses → equality works structurally.
    assert first == second
