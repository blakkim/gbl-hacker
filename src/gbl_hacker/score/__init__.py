"""Scoring engine for candidate GBL teams (Sub-AC 2).

The score module turns simulator outputs into the three scalar metrics the
Pareto ranker (Sub-AC 2.4) consumes:

* ``expected_win_rate``  — weighted mean win rate against the meta (Sub-AC 2.1)
* ``worst_case_robustness`` — usage-weighted low-quantile win rate (Sub-AC 2.2)
* ``meta_coverage``         — usage-weighted fraction handled ≥ threshold (Sub-AC 2.3)

…and the Pareto-frontier filter that turns those three scalars into a
non-dominated recommendation set (Sub-AC 2.4):

* ``pareto_filter`` — non-dominated subset of scored candidate teams

…and the top-K ranker that produces the engine's final ordered output
(Sub-AC 2.5):

* ``rank_top_k`` — top-K teams from a Pareto-optimal score set, ordered
  by an equal-weight (by default) weighted-sum policy with deterministic
  lexicographic tie-breaks.

This package only owns aggregation logic. Per-matchup combat semantics live
in :mod:`gbl_hacker.simulator`; meta ingestion lives in
:mod:`gbl_hacker.parse`. Keeping the boundary sharp is what lets the v0.1
9-pairing baseline aggregator be swapped for a richer set-simulator in
later ACs without re-deriving the Pareto math.

Public surface (Sub-AC 2.1 + 2.2 + 2.3 + 2.4 + 2.5 + 3.1):
    * :class:`CandidateTeam`          — ordered 3-slot lineup
    * :class:`MissingBuildError`      — opponent species absent from registry
    * :func:`materialize_opponent_team` — TeamUsage + registry → CandidateTeam
    * :func:`default_set_win_rate`    — 9-pairing average aggregator
    * :func:`expected_win_rate`       — headline Sub-AC 2.1 function
    * :func:`worst_case_robustness`   — headline Sub-AC 2.2 function
    * :func:`meta_coverage`           — headline Sub-AC 2.3 function
    * :class:`Score`                  — 3-axis scorecard (Sub-AC 2.4)
    * :class:`ScoredTeam`             — (team, score) pairing (Sub-AC 2.4)
    * :func:`dominates`               — strict-dominance predicate (Sub-AC 2.4)
    * :func:`pareto_filter`           — headline Sub-AC 2.4 function
    * :func:`rank_top_k`               — headline Sub-AC 2.5 function
    * :class:`MetaMatchupResult`      — per-opponent record (Sub-AC 3.1)
    * :func:`select_favorable_matchups` — headline Sub-AC 3.1 function
    * :func:`select_unfavorable_matchups` — headline Sub-AC 3.2 function
    * :func:`compute_meta_coverage`   — headline Sub-AC 3.3 function
      (rationale-card sibling of :func:`meta_coverage`)
    * :class:`RationaleCard`          — assembled card data structure
      (Sub-AC 3.4)
    * :func:`build_rationale_card`    — headline Sub-AC 3.4 function
"""

from gbl_hacker.score.expected_win_rate import (
    CandidateTeam,
    MissingBuildError,
    default_set_win_rate,
    expected_win_rate,
    materialize_opponent_team,
)
from gbl_hacker.score.meta_coverage import meta_coverage
from gbl_hacker.score.pareto import (
    Score,
    ScoredTeam,
    dominates,
    pareto_filter,
)
from gbl_hacker.score.rank import rank_top_k
from gbl_hacker.score.rationale import (
    MetaMatchupResult,
    RationaleCard,
    build_rationale_card,
    compute_meta_coverage,
    select_favorable_matchups,
    select_unfavorable_matchups,
)
from gbl_hacker.score.worst_case_robustness import worst_case_robustness

__all__ = [
    "CandidateTeam",
    "MetaMatchupResult",
    "MissingBuildError",
    "RationaleCard",
    "Score",
    "ScoredTeam",
    "build_rationale_card",
    "compute_meta_coverage",
    "default_set_win_rate",
    "dominates",
    "expected_win_rate",
    "materialize_opponent_team",
    "meta_coverage",
    "pareto_filter",
    "rank_top_k",
    "select_favorable_matchups",
    "select_unfavorable_matchups",
    "worst_case_robustness",
]
