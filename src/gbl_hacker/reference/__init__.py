"""Independent top-tier reference team-list loader (Sub-AC 5.1+5.2+5.3).

The seed's AC 5 ("Recommendation list shows non-trivial overlap with at
least one independent top-tier reference — PvPoke meta team list, a known
top streamer's published lineup, or similar") presupposes a way to *load*
such reference lists, a way to compare the engine's recommendations
against them, AND a way to render a pass/fail verdict that a CI gate (or
a human reading a shell log) can consume. This package owns all three.

Public surface:

Sub-AC 5.1 (loader):
    * :class:`ReferenceBuild`        — canonical ``(species, fast_move,
      charge_moves)`` slot
    * :class:`ReferenceBuildDisplay` — preserved human-readable identifiers
    * :class:`ReferenceTeam`         — 3-slot reference lineup with source tag
    * :class:`ReferenceTeamList`     — deserialized fixture container
    * :class:`ReferenceLoadError`    — schema/JSON failure exception
    * :func:`canonical_id`           — identifier normalizer
    * :func:`load_reference_team_list`             — disk loader
    * :func:`load_reference_team_list_from_mapping`— in-memory loader

Sub-AC 5.2 (overlap):
    * :class:`OverlapReport`         — symmetric overlap measurement
    * :class:`TeamKey`               — unordered species-triple identity
    * :func:`compute_overlap`        — rec list × reference → OverlapReport

Sub-AC 5.3 (verdict + frozen-recommendations fixture):
    * :data:`DEFAULT_THRESHOLD`            — default 0.2 (configurable)
    * :class:`RecommendationsFixture`      — frozen engine output snapshot
    * :class:`VerifyVerdict`               — pass/fail decision record
    * :func:`decide_verdict`               — OverlapReport → VerifyVerdict
    * :func:`verify_overlap`               — recs + ref → VerifyVerdict
    * :func:`load_recommendations_fixture` — disk loader for frozen recs
    * :func:`format_verdict_summary`       — CLI-grade pretty printer
"""

from gbl_hacker.reference.loader import (
    GREAT_LEAGUE_LABEL,
    TEAM_SIZE,
    ReferenceBuild,
    ReferenceBuildDisplay,
    ReferenceLoadError,
    ReferenceTeam,
    ReferenceTeamList,
    canonical_id,
    load_reference_team_list,
    load_reference_team_list_from_mapping,
)
from gbl_hacker.reference.overlap import (
    OverlapReport,
    TeamKey,
    compute_overlap,
)
from gbl_hacker.reference.verify import (
    DEFAULT_THRESHOLD,
    RecommendationsFixture,
    VerdictLabel,
    VerifyAxis,
    VerifyVerdict,
    decide_verdict,
    format_verdict_summary,
    load_recommendations_fixture,
    load_recommendations_fixture_from_mapping,
    verify_overlap,
)

__all__ = [
    "DEFAULT_THRESHOLD",
    "GREAT_LEAGUE_LABEL",
    "TEAM_SIZE",
    "OverlapReport",
    "RecommendationsFixture",
    "ReferenceBuild",
    "ReferenceBuildDisplay",
    "ReferenceLoadError",
    "ReferenceTeam",
    "ReferenceTeamList",
    "TeamKey",
    "VerdictLabel",
    "VerifyAxis",
    "VerifyVerdict",
    "canonical_id",
    "compute_overlap",
    "decide_verdict",
    "format_verdict_summary",
    "load_recommendations_fixture",
    "load_recommendations_fixture_from_mapping",
    "load_reference_team_list",
    "load_reference_team_list_from_mapping",
    "verify_overlap",
]
