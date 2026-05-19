"""Tests for ``gbl_hacker.reference.verify`` (Sub-AC 5.3).

The Sub-AC 5.3 contract:

    "Recommendation-vs-reference comparison CLI/module that wires the
    loader and metric to the engine output and emits a pass/fail
    verdict against a configurable threshold, with an integration test
    on a frozen snapshot of recommendations + reference asserting the
    verdict is 'pass'."

This file owns:

1. **The integration test** — the headline assertion: load both frozen
   fixtures, run :func:`verify_overlap`, assert the verdict label is
   ``"pass"`` at the default threshold. This is the AC's exit-criterion
   gate; everything else is fence.

2. **Module-unit tests** — pin the verdict contract (threshold
   semantics, axis selection, boundary case, validation errors).

3. **Recommendations-fixture loader tests** — pin the JSON deserializer
   shape so a future fixture-format drift fails loudly.

4. **CLI integration tests** — exercise the ``gblh verify-reference``
   subcommand end-to-end with the same frozen fixtures, asserting the
   exit code matches the verdict outcome.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from gbl_hacker.cli import (
    EXIT_OK,
    EXIT_USAGE,
    EXIT_VERIFY_FAIL,
    EXIT_VERIFY_LOAD,
    main,
)
from gbl_hacker.reference import (
    DEFAULT_THRESHOLD,
    OverlapReport,
    RecommendationsFixture,
    ReferenceLoadError,
    VerifyVerdict,
    compute_overlap,
    decide_verdict,
    format_verdict_summary,
    load_recommendations_fixture,
    load_recommendations_fixture_from_mapping,
    load_reference_team_list,
    verify_overlap,
)
from gbl_hacker.score import CandidateTeam


# ---------------------------------------------------------------------------
# Fixture paths — pinned to the recorded files under tests/fixtures/
# ---------------------------------------------------------------------------


FIXTURES_DIR = Path(__file__).parent / "fixtures"
REFERENCE_PATH = FIXTURES_DIR / "reference_great_league_pvpoke_sample.json"
RECOMMENDATIONS_PATH = FIXTURES_DIR / "recommendations_frozen_v1.json"


# ===========================================================================
# Sub-AC 5.3 headline integration test — verdict==pass on frozen snapshot
# ===========================================================================


def test_frozen_snapshot_verdict_is_pass_at_default_threshold() -> None:
    """Sub-AC 5.3 headline: frozen recs + reference → verdict 'pass'.

    The full end-to-end wire-up: load both fixtures from disk, run the
    headline ``verify_overlap``, assert the verdict label is exactly
    ``"pass"`` at the *configurable* default threshold. This is the
    AC's gating test — if it fails the engine is no longer producing
    output that aligns with the reference.

    Hand-computed expectation
    -------------------------
    Reference (3 teams, canonical unordered triples):
      * {azumarill, annihilape, registeel}
      * {medicham_shadow, lickitung_shadow, azumarill}
      * {galarian_stunfisk, azumarill, annihilape}

    Recommendations (3 teams, canonical):
      * {registeel, annihilape, azumarill}            ← matches ref[0]
      * {lickitung_shadow, azumarill, medicham_shadow} ← matches ref[1]
      * {swampert, gardevoir, trevenant}              ← matches none

    shared_team = 2 ; union_team = 4 → team_jaccard = 0.5
    Default threshold = 0.2 → 0.5 >= 0.2 → PASS.
    """

    recs = load_recommendations_fixture(RECOMMENDATIONS_PATH)
    ref = load_reference_team_list(REFERENCE_PATH)

    verdict = verify_overlap(recs.teams, ref)

    assert verdict.label == "pass"
    assert verdict.passed is True
    assert verdict.threshold == DEFAULT_THRESHOLD
    assert verdict.axis == "team"
    assert verdict.observed_jaccard == 0.5
    assert verdict.observed_jaccard >= verdict.threshold


def test_frozen_snapshot_verdict_carries_full_overlap_report() -> None:
    """The verdict's attached :class:`OverlapReport` matches the standalone overlap.

    Defense-in-depth: a refactor that accidentally produces a partial
    OverlapReport inside :func:`verify_overlap` would silently degrade
    downstream rendering. Pin the equality so it cannot drift.
    """

    recs = load_recommendations_fixture(RECOMMENDATIONS_PATH)
    ref = load_reference_team_list(REFERENCE_PATH)

    verdict = verify_overlap(recs.teams, ref)
    standalone = compute_overlap(recs.teams, ref)

    assert isinstance(verdict.overlap, OverlapReport)
    assert verdict.overlap == standalone
    # Hand-computed totals — pinned here so a future fixture edit must
    # also update this test, which keeps the integration honest.
    assert verdict.overlap.shared_team_count == 2
    assert verdict.overlap.union_team_count == 4
    assert verdict.overlap.shared_pokemon_count == 5
    # rec_pokemon (8) ∪ ref_pokemon (6) = 9 unique species across both.
    assert verdict.overlap.union_pokemon_count == 9


# ===========================================================================
# Threshold semantics — configurable, boundary, validation
# ===========================================================================


def test_threshold_boundary_equality_passes() -> None:
    """``observed == threshold`` is a ``"pass"`` — boundary is inclusive.

    The verdict module's contract names ``threshold`` as the *minimum*
    acceptable Jaccard. Equality at the boundary must pass; a future
    refactor that switches to strict ``>`` would silently break this.
    """

    recs = load_recommendations_fixture(RECOMMENDATIONS_PATH)
    ref = load_reference_team_list(REFERENCE_PATH)
    # The frozen snapshot's observed team_jaccard is 0.5 — pin equality.
    verdict = verify_overlap(recs.teams, ref, threshold=0.5)
    assert verdict.passed is True
    assert verdict.label == "pass"


def test_high_threshold_fails_on_frozen_snapshot() -> None:
    """A threshold above the observed Jaccard fails on the same fixtures.

    Same fixtures, threshold raised above 0.5. The verdict must flip
    to ``"fail"`` — proves the threshold is genuinely *configurable*
    and is what drives the decision, not a hard-coded constant.
    """

    recs = load_recommendations_fixture(RECOMMENDATIONS_PATH)
    ref = load_reference_team_list(REFERENCE_PATH)
    verdict = verify_overlap(recs.teams, ref, threshold=0.75)
    assert verdict.passed is False
    assert verdict.label == "fail"
    assert verdict.observed_jaccard == 0.5
    assert verdict.threshold == 0.75


def test_pokemon_axis_uses_pokemon_jaccard() -> None:
    """``axis='pokemon'`` makes the verdict consult :attr:`pokemon_jaccard`.

    Hand-computed: rec_pokemon has 8 unique species, ref_pokemon has
    6 unique species, shared is 5 → pokemon_jaccard = 5/9.
    """

    recs = load_recommendations_fixture(RECOMMENDATIONS_PATH)
    ref = load_reference_team_list(REFERENCE_PATH)
    verdict = verify_overlap(recs.teams, ref, axis="pokemon")
    assert verdict.axis == "pokemon"
    assert verdict.observed_jaccard == pytest.approx(5 / 9)
    # 5/9 ≈ 0.556 — well above the default 0.2 threshold.
    assert verdict.passed is True


@pytest.mark.parametrize("bad_threshold", [-0.01, 1.01, -1.0, 2.0])
def test_threshold_out_of_range_rejected(bad_threshold: float) -> None:
    """Threshold outside [0, 1] raises :class:`ValueError`."""

    recs = load_recommendations_fixture(RECOMMENDATIONS_PATH)
    ref = load_reference_team_list(REFERENCE_PATH)
    with pytest.raises(ValueError, match="threshold"):
        verify_overlap(recs.teams, ref, threshold=bad_threshold)


def test_unknown_axis_rejected() -> None:
    """Unknown axis raises :class:`ValueError`."""

    recs = load_recommendations_fixture(RECOMMENDATIONS_PATH)
    ref = load_reference_team_list(REFERENCE_PATH)
    with pytest.raises(ValueError, match="axis"):
        verify_overlap(recs.teams, ref, axis="bogus")  # type: ignore[arg-type]


# ===========================================================================
# decide_verdict — projection-only API
# ===========================================================================


def test_decide_verdict_does_not_recompute_overlap() -> None:
    """``decide_verdict`` projects a report; it does not re-run the math.

    A caller that already holds an :class:`OverlapReport` (e.g. sweeping
    multiple thresholds) must be able to call ``decide_verdict`` once
    per threshold without paying for the Jaccard math each time.
    """

    recs = load_recommendations_fixture(RECOMMENDATIONS_PATH)
    ref = load_reference_team_list(REFERENCE_PATH)
    overlap = compute_overlap(recs.teams, ref)

    v_low = decide_verdict(overlap, threshold=0.1)
    v_high = decide_verdict(overlap, threshold=0.9)
    # Same overlap powers both decisions.
    assert v_low.overlap is overlap
    assert v_high.overlap is overlap
    assert v_low.passed is True
    assert v_high.passed is False


# ===========================================================================
# RecommendationsFixture loader — schema, league, member-count
# ===========================================================================


def test_load_recommendations_fixture_yields_three_candidate_teams() -> None:
    """The frozen fixture deserializes to exactly 3 CandidateTeams."""

    recs = load_recommendations_fixture(RECOMMENDATIONS_PATH)
    assert isinstance(recs, RecommendationsFixture)
    assert len(recs.teams) == 3
    for team in recs.teams:
        assert isinstance(team, CandidateTeam)


def test_load_recommendations_fixture_canonicalizes_species() -> None:
    """Display-form species names normalize to canonical ids on load.

    The frozen fixture intentionally carries ``"Medicham (Shadow)"``
    so this assertion proves the loader normalizes via
    :func:`canonical_id`. If the normalization regressed, the overlap
    against the reference would silently zero out.
    """

    recs = load_recommendations_fixture(RECOMMENDATIONS_PATH)
    all_species: set[str] = set()
    for team in recs.teams:
        for species in team.species:
            all_species.add(species)
    assert "medicham_shadow" in all_species
    assert "lickitung_shadow" in all_species
    # Lowercase / underscore form pinned.
    assert all(s == s.lower() for s in all_species)


def test_load_recommendations_fixture_metadata_preserved() -> None:
    """Source / league / captured_at / notes survive the round-trip."""

    recs = load_recommendations_fixture(RECOMMENDATIONS_PATH)
    assert recs.source == "engine_output_frozen_v1"
    assert recs.league == "great_league"
    assert recs.captured_at == datetime(
        2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc
    )
    assert "frozen" in recs.notes.lower()


def test_load_recommendations_fixture_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    with pytest.raises(ReferenceLoadError) as exc_info:
        load_recommendations_fixture(missing)
    assert exc_info.value.path == missing


def test_load_recommendations_fixture_invalid_json_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ReferenceLoadError):
        load_recommendations_fixture(bad)


def test_load_recommendations_fixture_wrong_league_rejected() -> None:
    payload: dict[str, Any] = {
        "source": "test",
        "league": "ultra_league",
        "captured_at": "2026-05-13T00:00:00Z",
        "teams": [{"members": ["A", "B", "C"]}],
    }
    with pytest.raises(ReferenceLoadError, match="great_league"):
        load_recommendations_fixture_from_mapping(payload)


def test_load_recommendations_fixture_wrong_member_count_rejected() -> None:
    payload: dict[str, Any] = {
        "source": "test",
        "league": "great_league",
        "captured_at": "2026-05-13T00:00:00Z",
        "teams": [{"members": ["A", "B"]}],
    }
    with pytest.raises(ReferenceLoadError, match="exactly 3"):
        load_recommendations_fixture_from_mapping(payload)


def test_load_recommendations_fixture_empty_teams_rejected() -> None:
    payload: dict[str, Any] = {
        "source": "test",
        "league": "great_league",
        "captured_at": "2026-05-13T00:00:00Z",
        "teams": [],
    }
    with pytest.raises(ReferenceLoadError, match="at least one"):
        load_recommendations_fixture_from_mapping(payload)


def test_load_recommendations_fixture_disk_matches_mapping() -> None:
    """Both entry points agree on a well-formed payload."""

    from_disk = load_recommendations_fixture(RECOMMENDATIONS_PATH)
    payload = json.loads(RECOMMENDATIONS_PATH.read_text(encoding="utf-8"))
    from_mapping = load_recommendations_fixture_from_mapping(payload)
    # CandidateTeams compare equal by content (frozen dataclass).
    assert from_disk == from_mapping


# ===========================================================================
# CLI integration — gblh verify-reference end-to-end
# ===========================================================================


def test_cli_verify_reference_returns_exit_ok_on_frozen_snapshot() -> None:
    """``gblh verify-reference`` returns ``EXIT_OK`` on the frozen snapshot.

    Same wiring as the headline integration test, but driven through
    the CLI surface. Pins both: (a) the verdict outcome on the AC's
    "frozen snapshot of recommendations + reference" matches
    ``EXIT_OK`` and (b) the operator-facing rendering carries the
    verdict label.
    """

    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        [
            "verify-reference",
            "--recommendations",
            str(RECOMMENDATIONS_PATH),
            "--reference",
            str(REFERENCE_PATH),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    out = stdout.getvalue()
    assert exit_code == EXIT_OK, (
        f"expected EXIT_OK, got {exit_code!r}. "
        f"stdout={out!r} stderr={stderr.getvalue()!r}"
    )
    assert "verdict=PASS" in out
    assert "axis=team" in out
    assert "threshold=" in out
    assert "observed=" in out


def test_cli_verify_reference_fail_exit_when_threshold_above_observed() -> None:
    """``--threshold 0.75`` flips the verdict to fail → ``EXIT_VERIFY_FAIL``."""

    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        [
            "verify-reference",
            "--recommendations",
            str(RECOMMENDATIONS_PATH),
            "--reference",
            str(REFERENCE_PATH),
            "--threshold",
            "0.75",
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == EXIT_VERIFY_FAIL
    assert "verdict=FAIL" in stdout.getvalue()


def test_cli_verify_reference_axis_pokemon_passes_at_default_threshold() -> None:
    """``--axis pokemon`` consults the species-level Jaccard."""

    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        [
            "verify-reference",
            "--recommendations",
            str(RECOMMENDATIONS_PATH),
            "--reference",
            str(REFERENCE_PATH),
            "--axis",
            "pokemon",
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == EXIT_OK
    assert "axis=pokemon" in stdout.getvalue()


def test_cli_verify_reference_missing_recs_file_returns_load_exit_code(
    tmp_path: Path,
) -> None:
    """Missing recommendations file maps to ``EXIT_VERIFY_LOAD``."""

    missing = tmp_path / "no_such_recs.json"
    stderr = io.StringIO()

    exit_code = main(
        [
            "verify-reference",
            "--recommendations",
            str(missing),
            "--reference",
            str(REFERENCE_PATH),
        ],
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert exit_code == EXIT_VERIFY_LOAD
    assert "could not load recommendations" in stderr.getvalue().lower()


def test_cli_verify_reference_missing_ref_file_returns_load_exit_code(
    tmp_path: Path,
) -> None:
    """Missing reference file maps to ``EXIT_VERIFY_LOAD``."""

    missing = tmp_path / "no_such_ref.json"
    stderr = io.StringIO()

    exit_code = main(
        [
            "verify-reference",
            "--recommendations",
            str(RECOMMENDATIONS_PATH),
            "--reference",
            str(missing),
        ],
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert exit_code == EXIT_VERIFY_LOAD
    assert "could not load reference" in stderr.getvalue().lower()


def test_cli_verify_reference_out_of_range_threshold_returns_usage(
) -> None:
    """A threshold outside [0, 1] returns ``EXIT_USAGE``."""

    stderr = io.StringIO()
    exit_code = main(
        [
            "verify-reference",
            "--recommendations",
            str(RECOMMENDATIONS_PATH),
            "--reference",
            str(REFERENCE_PATH),
            "--threshold",
            "1.5",
        ],
        stdout=io.StringIO(),
        stderr=stderr,
    )
    assert exit_code == EXIT_USAGE
    assert "threshold" in stderr.getvalue().lower()


# ===========================================================================
# format_verdict_summary — pinned shape so CLI output cannot drift silently
# ===========================================================================


def test_format_verdict_summary_includes_required_fields() -> None:
    """The summary block surfaces verdict, threshold, observed, and sources.

    Pinning the fields keeps the data-honesty principle honest — a
    future cosmetic refactor cannot silently drop the threshold or the
    source provenance from operator-facing output.
    """

    recs = load_recommendations_fixture(RECOMMENDATIONS_PATH)
    ref = load_reference_team_list(REFERENCE_PATH)
    verdict = verify_overlap(recs.teams, ref)

    summary = format_verdict_summary(
        verdict,
        reference_source=ref.source,
        recommendation_source=recs.source,
    )

    # Header line — verdict / axis / threshold / observed all present.
    assert "verdict=PASS" in summary
    assert "axis=team" in summary
    assert "threshold=" in summary
    assert "observed=" in summary
    # Body lines — provenance and overlap counts.
    assert ref.source in summary
    assert recs.source in summary
    assert "shared team cores:" in summary
    assert "union team cores:" in summary
    assert "shared pokemon:" in summary
    assert "union pokemon:" in summary
