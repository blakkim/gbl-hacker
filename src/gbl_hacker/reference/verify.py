"""Recommendation-vs-reference verdict (Sub-AC 5.3).

The seed's Acceptance Criterion 5 asks the engine's recommendation list to
show *non-trivial overlap* with at least one independent top-tier reference
— a PvPoke meta export, a published streamer lineup, or similar. Sub-AC
5.1 (:mod:`gbl_hacker.reference.loader`) owns *how the reference arrives
in memory*; Sub-AC 5.2 (:mod:`gbl_hacker.reference.overlap`) owns *the
measurement itself* — symmetric Jaccard coefficients at team- and
Pokémon-level. This module is the **last mile**: it wires the loader and
the metric to the engine's output and emits a **pass/fail verdict**
against a configurable threshold, so the AC 5 contract can be enforced
in CI as a single shell exit code (or a single Python assertion in
tests).

Why a dedicated verdict module
------------------------------

* **One source of truth for the threshold.** The Sub-AC 5.2 overlap
  module deliberately stops at the *measurement* — it does not decide
  what counts as "non-trivial". Concentrating the threshold *here*
  means a future audit AC can move the bar without re-deriving the
  Jaccard math, and the data-honesty principle (which warns against
  hidden cutoffs) stays honest because the threshold lives in exactly
  one named location.

* **Pass/fail is a different shape from the raw report.** A
  :class:`~gbl_hacker.reference.overlap.OverlapReport` carries six
  raw sets and four derived ratios — the rendering surface for a
  rationale card. A verdict carries one boolean, the threshold, and
  the observed scalar — the rendering surface for a CI gate, a
  human-readable shell summary, or a single-line log entry. They are
  read by different audiences and should not be conflated.

* **The "engine output" boundary.** The engine produces
  ``list[CandidateTeam]`` (via :func:`gbl_hacker.score.rank_top_k`'s
  scored teams). For the integration test (Sub-AC 5.3's explicit
  requirement) we need a way to **freeze** that output into a fixture
  so a verdict can be re-run deterministically without re-running the
  full simulator. :class:`RecommendationsFixture` is that frozen
  shape, and it deserializes from a JSON sidecar mirroring how
  :class:`gbl_hacker.reference.ReferenceTeamList` does.

Threshold semantics
-------------------

The default threshold is ``0.2`` on the **team-level Jaccard** axis.
The choice is defensible rather than arbitrary:

* A team-Jaccard of ``0.2`` means at least one in five distinct cores
  is shared between recommendation and reference — well above the
  zero-overlap floor and well below a "perfect mirror" 1.0. It is
  exactly the "non-trivial but not gameable" middle ground the seed
  language pins down.

* The threshold is **configurable** per the seed's literal text in
  Sub-AC 5.3 ("emits a pass/fail verdict against a *configurable*
  threshold"). Callers tune it down for early-stage rec lists or up
  for late-stage audits, but the default is *honest enough to publish*.

* The verdict can also be computed on the **Pokémon-level Jaccard**
  axis (``axis="pokemon"``) — the finer-grained signal that catches
  partial-roster agreement when no full team matches. The team axis
  is the default because the seed's wording emphasizes "teams" — but
  exposing both axes prevents the false-negative failure mode where
  an engine that picks the right *pokémon* but a different
  *combination* gets called a verdict failure.

Boundary behavior
-----------------

* A verdict of ``"pass"`` requires the observed Jaccard to be
  ``>= threshold`` (greedy comparison). The boundary case "observed
  equals threshold" passes — the threshold names the *minimum*
  acceptable overlap, not an exclusive upper-open one.

* The verify functions are **pure**: they consume their inputs once
  and never mutate them.

* The recommendations fixture loader rejects empty team lists at
  schema time — same convention as the reference loader. A verdict
  computed on zero recommendations is a programming bug, not a
  legitimate input.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from gbl_hacker.reference.loader import (
    GREAT_LEAGUE_LABEL,
    TEAM_SIZE,
    ReferenceLoadError,
    ReferenceTeamList,
    canonical_id,
)
from gbl_hacker.reference.overlap import OverlapReport, compute_overlap
from gbl_hacker.score import CandidateTeam
from gbl_hacker.simulator import ChargedMove, CombatantBuild, FastMove

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD: float = 0.2
"""Minimum acceptable Jaccard for a ``"pass"`` verdict.

See the module docstring for the rationale behind the value. Exposed as
a module constant so a future policy change (or a doc-test) lands in one
place rather than duplicated across CLI argparse defaults and library
call sites.
"""

VerifyAxis = Literal["team", "pokemon"]
"""Which Jaccard axis the verdict is evaluated against.

``"team"`` (default) compares unordered species triples — agreement
on full team cores. ``"pokemon"`` compares the per-side species sets —
agreement on individual picks. Both are surfaced by
:class:`OverlapReport`; the verdict layer chooses which one drives the
boolean pass/fail.
"""

VerdictLabel = Literal["pass", "fail"]
"""Stable string form of the verdict — useful for stdout / log lines."""


# ---------------------------------------------------------------------------
# Recommendations fixture — frozen engine output for deterministic verdicts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecommendationsFixture:
    """Frozen snapshot of an engine recommendation list.

    The engine's runtime output flows through
    :func:`gbl_hacker.score.rank_top_k` → ``list[ScoredTeam]``. For
    Sub-AC 5.3's integration test, we need to be able to *freeze* a
    recommendation list to disk so the verdict can be re-computed
    deterministically without re-running the simulator. This dataclass
    is that frozen form.

    Only the **species triples** matter for the overlap-vs-reference
    verdict, so the on-disk schema carries species + minimal metadata
    rather than full :class:`CombatantBuild` payloads. The loader
    materializes lightweight stub builds (placeholder combat stats) for
    each species so the loaded :class:`CandidateTeam` instances satisfy
    :func:`compute_overlap`'s shape requirement.

    Attributes
    ----------
    source:
        Machine identifier for this fixture's provenance
        (``"engine_output_v1"``, ``"frozen_recs_for_ac5_test"``).
        Surfaced in CLI output.
    league:
        Always ``"great_league"`` for v0.1 — same constraint as the
        reference loader, kept consistent so a multi-league future can
        find every league check in one ``rg`` pass.
    captured_at:
        Wall-clock moment the engine produced this output.
    notes:
        Free-form caveat / context string. Empty string is permitted.
    teams:
        :class:`CandidateTeam` instances in engine-emitted rank order.
        The verdict does not consult the order — only the species
        triples — but preserving it lets a future "show top-K and
        their per-team verdicts" presentation use the same fixture.
    """

    source: str
    league: str
    captured_at: datetime
    notes: str
    teams: tuple[CandidateTeam, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:  # pragma: no cover - pure validation
        if not self.source:
            raise ValueError("source must be non-empty")
        if self.league != GREAT_LEAGUE_LABEL:
            raise ValueError(
                f"v0.1 recommendations must target league='{GREAT_LEAGUE_LABEL}', "
                f"got {self.league!r}"
            )
        if not self.teams:
            raise ValueError("recommendations fixture must not be empty")


# Stub combat stats for species-only fixtures. The overlap math consults
# only ``CombatantBuild.species``; these placeholders satisfy the
# dataclass constructors without claiming to be real GBL stats. Named so
# a future ``rg`` for "placeholder" surfaces every such stub.
_STUB_FAST = FastMove(name="placeholder", damage=1, energy_gain=1)
_STUB_CHARGED = ChargedMove(name="placeholder", energy_cost=10, damage=10)


def _stub_combatant_build(species: str) -> CombatantBuild:
    """Construct a placeholder ``CombatantBuild`` for the given species.

    The recommendations fixture serializes only species triples — the
    actual move/HP numbers do not matter to the overlap-vs-reference
    verdict. This helper inflates a species string into the
    :class:`CombatantBuild` shape that :class:`CandidateTeam` expects
    so the fixture can produce real :class:`CandidateTeam` instances
    without dragging a full build registry into the test surface.

    Parameters
    ----------
    species:
        Free-form species identifier from the fixture. The display
        form (``"Medicham (Shadow)"``) is normalized to canonical
        (``"medicham_shadow"``) before being stored on the build —
        so the resulting ``CandidateTeam.species`` triple is already
        canonical when :func:`compute_overlap` reads it.
    """

    canonical_species = canonical_id(species)
    if not canonical_species:
        raise ReferenceLoadError(
            f"species normalized to empty string: {species!r}",
        )
    return CombatantBuild(
        species=canonical_species,
        max_hp=100,
        fast=_STUB_FAST,
        charged=_STUB_CHARGED,
    )


def load_recommendations_fixture(path: Path | str) -> RecommendationsFixture:
    """Load and parse a recommendations fixture JSON file from disk.

    Schema (mirrors :func:`gbl_hacker.reference.load_reference_team_list`
    for diff-stability):

    .. code-block:: json

        {
          "source": "engine_output_frozen_v1",
          "league": "great_league",
          "captured_at": "2026-05-13T12:00:00Z",
          "notes": "Frozen engine output for the AC 5 verdict test.",
          "teams": [
            {"members": ["Azumarill", "Annihilape", "Registeel"]},
            {"members": ["Medicham (Shadow)", "Lickitung (Shadow)", "Azumarill"]}
          ]
        }

    Raises
    ------
    ReferenceLoadError
        On missing path, malformed JSON, or schema violations. The
        error always carries the offending ``path`` (when known).
    """

    file_path = Path(path)
    if not file_path.exists():
        raise ReferenceLoadError(
            f"recommendations fixture not found: {file_path}", path=file_path
        )
    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - filesystem edge case
        raise ReferenceLoadError(
            f"failed to read recommendations fixture: {exc}", path=file_path
        ) from exc
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ReferenceLoadError(
            f"recommendations fixture is not valid JSON: {exc.msg} "
            f"(line {exc.lineno}, col {exc.colno})",
            path=file_path,
        ) from exc
    return load_recommendations_fixture_from_mapping(payload, path=file_path)


def load_recommendations_fixture_from_mapping(
    payload: Mapping[str, Any],
    *,
    path: Path | None = None,
) -> RecommendationsFixture:
    """Alternate entry point: deserialize from an already-parsed mapping.

    Useful when the caller already holds a dict (e.g. tests embedding
    fixtures inline). Validation is otherwise identical to
    :func:`load_recommendations_fixture`.
    """

    if not isinstance(payload, Mapping):
        raise ReferenceLoadError(
            f"top-level JSON must be an object, got {type(payload).__name__}",
            path=path,
        )

    source = _required_str(payload, "source", path=path)
    league = _required_str(payload, "league", path=path)
    if league != GREAT_LEAGUE_LABEL:
        raise ReferenceLoadError(
            f"recommendations fixture must target league={GREAT_LEAGUE_LABEL!r}, "
            f"got {league!r}",
            path=path,
        )
    captured_at = _required_datetime(payload, "captured_at", path=path)
    notes = _optional_str(payload, "notes", default="")

    raw_teams = payload.get("teams")
    if not isinstance(raw_teams, Sequence) or isinstance(raw_teams, (str, bytes)):
        raise ReferenceLoadError("'teams' must be a JSON array", path=path)
    if len(raw_teams) == 0:
        raise ReferenceLoadError(
            "'teams' must contain at least one entry", path=path
        )

    teams: list[CandidateTeam] = []
    for idx, raw_team in enumerate(raw_teams):
        teams.append(_deserialize_recommendation_team(raw_team, index=idx, path=path))

    return RecommendationsFixture(
        source=source,
        league=league,
        captured_at=captured_at,
        notes=notes,
        teams=tuple(teams),
    )


def _deserialize_recommendation_team(
    raw: Any, *, index: int, path: Path | None
) -> CandidateTeam:
    """Decode one ``{"members": [s1, s2, s3]}`` entry to a CandidateTeam."""

    if not isinstance(raw, Mapping):
        raise ReferenceLoadError(
            f"team[{index}] must be a JSON object, got {type(raw).__name__}",
            path=path,
        )
    raw_members = raw.get("members")
    if not isinstance(raw_members, Sequence) or isinstance(raw_members, (str, bytes)):
        raise ReferenceLoadError(
            f"team[{index}].members must be a JSON array", path=path
        )
    if len(raw_members) != TEAM_SIZE:
        raise ReferenceLoadError(
            f"team[{index}].members must have exactly {TEAM_SIZE} entries, "
            f"got {len(raw_members)}",
            path=path,
        )
    builds: list[CombatantBuild] = []
    for slot_idx, raw_species in enumerate(raw_members):
        if not isinstance(raw_species, str) or not raw_species.strip():
            raise ReferenceLoadError(
                f"team[{index}].members[{slot_idx}] must be a non-empty string",
                path=path,
            )
        builds.append(_stub_combatant_build(raw_species.strip()))
    return CandidateTeam.from_slots(builds)


# ---------------------------------------------------------------------------
# Verdict dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VerifyVerdict:
    """Outcome of a recommendation-vs-reference comparison.

    The verdict is the **decision-grade** projection of an
    :class:`OverlapReport`: which axis was checked, what threshold the
    observed scalar was compared against, and the resulting pass/fail
    boolean. The full raw report is kept attached for callers that want
    to render a rationale-card-style breakdown alongside the verdict —
    rendering and decision are intentionally distinct surfaces.

    Attributes
    ----------
    passed:
        ``True`` iff the observed Jaccard on the selected ``axis`` is
        ``>= threshold``. Equality at the boundary passes — the
        threshold names the *minimum acceptable* overlap, not an
        exclusive open lower bound.
    threshold:
        The cutoff the verdict was decided against. Stored on the
        verdict (not implicit in module state) so a CI log line carries
        every input that determined the decision.
    axis:
        Which Jaccard fed the comparison — ``"team"`` (unordered
        species triples, default) or ``"pokemon"`` (individual species
        rosters).
    observed_jaccard:
        The Jaccard value on the chosen axis, copied here for easy
        rendering. Always equal to ``overlap.team_jaccard`` or
        ``overlap.pokemon_jaccard`` depending on ``axis``.
    overlap:
        The full :class:`OverlapReport` that produced the observed
        scalar. Carried by reference (frozen dataclass) so a verdict
        printer can fall through to a per-axis breakdown without
        re-running :func:`compute_overlap`.
    """

    passed: bool
    threshold: float
    axis: VerifyAxis
    observed_jaccard: float
    overlap: OverlapReport

    @property
    def label(self) -> VerdictLabel:
        """Stable string form of the verdict (``"pass"`` / ``"fail"``)."""

        return "pass" if self.passed else "fail"


# ---------------------------------------------------------------------------
# Public decision functions
# ---------------------------------------------------------------------------


def decide_verdict(
    overlap: OverlapReport,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    axis: VerifyAxis = "team",
) -> VerifyVerdict:
    """Project an :class:`OverlapReport` to a pass/fail verdict.

    This is the *decision-only* step — it does no measurement. Useful
    when a caller already has an overlap report in hand (e.g. a single
    overlap shared across multiple thresholds being audited in a
    sweep) and only wants the threshold-vs-observed comparison.

    Parameters
    ----------
    overlap:
        The :class:`OverlapReport` to evaluate.
    threshold:
        Minimum acceptable Jaccard for a ``"pass"`` verdict. Must be
        in ``[0.0, 1.0]``. Default :data:`DEFAULT_THRESHOLD`.
    axis:
        Which Jaccard to compare. ``"team"`` (default) for unordered
        species triples; ``"pokemon"`` for individual species rosters.

    Returns
    -------
    VerifyVerdict
        Frozen decision record. The boundary case
        ``observed == threshold`` passes.

    Raises
    ------
    ValueError
        If ``threshold`` is out of ``[0.0, 1.0]`` or ``axis`` is not
        one of the documented literals.
    """

    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"threshold out of range: {threshold} (must be in [0.0, 1.0])"
        )
    if axis == "team":
        observed = overlap.team_jaccard
    elif axis == "pokemon":
        observed = overlap.pokemon_jaccard
    else:
        raise ValueError(
            f"axis must be 'team' or 'pokemon', got {axis!r}"
        )
    return VerifyVerdict(
        passed=observed >= threshold,
        threshold=threshold,
        axis=axis,
        observed_jaccard=observed,
        overlap=overlap,
    )


def verify_overlap(
    recommendations: Iterable[CandidateTeam],
    reference: ReferenceTeamList,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    axis: VerifyAxis = "team",
) -> VerifyVerdict:
    """End-to-end: compute the overlap and render a pass/fail verdict.

    Headline Sub-AC 5.3 function. Composes
    :func:`gbl_hacker.reference.compute_overlap` with
    :func:`decide_verdict`, so a caller — CLI, integration test, or
    notebook — can move from "I have an engine recommendation list and
    a reference list" to a single decision boolean in one call.

    Parameters
    ----------
    recommendations:
        Iterable of :class:`CandidateTeam`. Consumed exactly once.
        Empty inputs are not an error; the resulting verdict will
        report ``observed_jaccard=0.0`` and ``passed`` reflects whether
        the threshold itself is zero.
    reference:
        Loaded :class:`ReferenceTeamList`. Construction guarantees a
        non-empty teams tuple.
    threshold, axis:
        Forwarded to :func:`decide_verdict`. See that function's
        docstring for the contract.

    Returns
    -------
    VerifyVerdict
        Frozen decision record, with the underlying overlap attached
        for downstream rendering.
    """

    overlap = compute_overlap(recommendations, reference)
    return decide_verdict(overlap, threshold=threshold, axis=axis)


# ---------------------------------------------------------------------------
# Pretty-print helper (used by the CLI; module-level so tests can pin it)
# ---------------------------------------------------------------------------


def format_verdict_summary(
    verdict: VerifyVerdict,
    *,
    reference_source: str,
    recommendation_source: str,
) -> str:
    """Render a verdict as a multi-line human-readable summary block.

    Used by the ``gblh verify-reference`` CLI subcommand to give the
    operator an at-a-glance read of the decision. The exact wording is
    pinned by a test so a future cosmetic refactor cannot silently
    drop the threshold or the observed scalar from the output.

    Parameters
    ----------
    verdict:
        The :class:`VerifyVerdict` to format.
    reference_source, recommendation_source:
        Provenance strings carried from the loaded fixtures. Surfacing
        them in the verdict block keeps the data-honesty principle
        honest — the operator can always trace a ``"pass"`` back to
        the specific source files that produced it.
    """

    header = (
        f"gblh verify-reference: verdict={verdict.label.upper()} "
        f"(axis={verdict.axis}, threshold={verdict.threshold:.4f}, "
        f"observed={verdict.observed_jaccard:.4f})"
    )
    body = (
        f"  recommendation source: {recommendation_source}\n"
        f"  reference source:      {reference_source}\n"
        f"  shared team cores:     {verdict.overlap.shared_team_count}\n"
        f"  union team cores:      {verdict.overlap.union_team_count}\n"
        f"  shared pokemon:        {verdict.overlap.shared_pokemon_count}\n"
        f"  union pokemon:         {verdict.overlap.union_pokemon_count}"
    )
    return f"{header}\n{body}"


# ---------------------------------------------------------------------------
# small validation helpers (shared shape with loader.py for diff-stability)
# ---------------------------------------------------------------------------


def _required_str(
    payload: Mapping[str, Any],
    key: str,
    *,
    path: Path | None,
    ctx: str | None = None,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        prefix = f"{ctx}." if ctx else ""
        raise ReferenceLoadError(
            f"missing required string field '{prefix}{key}'", path=path
        )
    return value.strip()


def _optional_str(
    payload: Mapping[str, Any], key: str, *, default: str
) -> str:
    value = payload.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        return default
    return value.strip()


def _required_datetime(
    payload: Mapping[str, Any],
    key: str,
    *,
    path: Path | None,
) -> datetime:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReferenceLoadError(
            f"missing required ISO-8601 datetime field '{key}'", path=path
        )
    raw = value.strip()
    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ReferenceLoadError(
            f"'{key}' is not a valid ISO-8601 datetime: {raw!r}", path=path
        ) from exc


__all__ = [
    "DEFAULT_THRESHOLD",
    "RecommendationsFixture",
    "VerdictLabel",
    "VerifyAxis",
    "VerifyVerdict",
    "decide_verdict",
    "format_verdict_summary",
    "load_recommendations_fixture",
    "load_recommendations_fixture_from_mapping",
    "verify_overlap",
]
