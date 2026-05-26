"""Trust signals for a recommendation: frontier fragility + pool sensitivity.

A single expected-win-rate number is only as trustworthy as the opponent pool
it was scored against. Two signals make that assumption visible instead of
implicit:

* **Opponent-pool sensitivity.** Score the same team against a *second* pool
  (PvPoke-ranking "theory-optimal" lineups vs the Taiman "what people actually
  run" meta) and report the divergence. A team that scores 93% against ranking
  but 80% against the real meta is not as strong as the headline suggests; the
  gap is the trust signal. (cf. the candidate-pool-vs-evaluation-meta split.)

* **Frontier fragility.** A 3-axis Pareto frontier that collapses to one or two
  teams usually means the opponent pool is skewed so that one lineup dominates
  every axis — not that the pick is genuinely unbeatable. We raise that as an
  explicit alarm rather than burying it.

These are deterministic post-hoc readings of the score table, so they live in a
small pure module the CLI renders and tests can exercise directly.
"""
from __future__ import annotations

from dataclasses import dataclass

# A Pareto frontier this small is treated as a skew signal, not a result.
PARETO_FRAGILE_THRESHOLD = 2
# Cross-pool EWR gap (in win-rate fraction) above which a row is flagged.
GAP_FRAGILE_THRESHOLD = 0.15


def pareto_alarm(frontier_size: int, *, opponents_label: str) -> str | None:
    """Return a fragility alarm string if the frontier is suspiciously small.

    ``None`` when ``frontier_size`` exceeds :data:`PARETO_FRAGILE_THRESHOLD`.
    """
    if frontier_size > PARETO_FRAGILE_THRESHOLD:
        return None
    return (
        f"⚠ FRAGILE FRONTIER — pareto_size={frontier_size}. The 3-axis Pareto "
        f"frontier collapsed to {frontier_size} team(s) against the "
        f"{opponents_label} opponent pool.\n"
        "  This usually means the opponent pool is skewed so one lineup "
        "dominates every axis — not that the pick is unbeatable. Treat the top "
        "pick as provisional: read the opponent-pool sensitivity below, widen "
        "--opponents-size, or cross-check with the other --opponents source."
    )


@dataclass(frozen=True)
class TrustRow:
    """One team's EWR under the primary pool vs a cross-check pool."""

    name: str
    ewr_primary: float
    ewr_cross: float

    @property
    def gap(self) -> float:
        """``ewr_primary - ewr_cross`` — positive == primary is optimistic."""
        return self.ewr_primary - self.ewr_cross

    @property
    def is_fragile(self) -> bool:
        return abs(self.gap) >= GAP_FRAGILE_THRESHOLD


def format_trust_table(
    rows: list[TrustRow],
    *,
    primary_label: str,
    cross_label: str,
) -> str:
    """Render the opponent-pool sensitivity table as a text block."""
    head = (
        "TRUST — opponent-pool sensitivity\n"
        "  Each team's EWR scored against both pools; a large gap (⚠ ≥"
        f"{GAP_FRAGILE_THRESHOLD:.0%}) means the rank is pool-dependent.\n"
        f"  ewr[{primary_label}]  ewr[{cross_label}]    gap   team\n"
        "  " + "-" * 68
    )
    lines = [head]
    for r in rows:
        flag = " ⚠" if r.is_fragile else ""
        lines.append(
            f"  {r.ewr_primary:>6.1%}{' ' * 13}{r.ewr_cross:>6.1%}{' ' * 6}"
            f"{r.gap:+6.1%}  {r.name}{flag}"
        )
    return "\n".join(lines)


__all__ = [
    "GAP_FRAGILE_THRESHOLD",
    "PARETO_FRAGILE_THRESHOLD",
    "TrustRow",
    "format_trust_table",
    "pareto_alarm",
]
