"""Tests for the recommendation trust signals (:mod:`gbl_hacker.trust`)."""
from __future__ import annotations

from gbl_hacker.trust import (
    GAP_FRAGILE_THRESHOLD,
    PARETO_FRAGILE_THRESHOLD,
    TrustRow,
    format_trust_table,
    pareto_alarm,
)


# ------------------------------------------------------------- pareto_alarm
def test_pareto_alarm_none_when_frontier_healthy():
    assert pareto_alarm(PARETO_FRAGILE_THRESHOLD + 1, opponents_label="x") is None
    assert pareto_alarm(10, opponents_label="x") is None


def test_pareto_alarm_fires_on_small_frontier():
    for size in (0, 1, PARETO_FRAGILE_THRESHOLD):
        msg = pareto_alarm(size, opponents_label="PvPoke-synthetic")
        assert msg is not None
        assert f"pareto_size={size}" in msg
        assert "PvPoke-synthetic" in msg


# ----------------------------------------------------------------- TrustRow
def test_trustrow_gap_is_primary_minus_cross():
    row = TrustRow(name="t", ewr_primary=0.93, ewr_cross=0.805)
    assert abs(row.gap - 0.125) < 1e-9


def test_trustrow_fragile_threshold():
    assert TrustRow("t", 0.93, 0.93 - GAP_FRAGILE_THRESHOLD).is_fragile
    just_under = TrustRow("t", 0.93, 0.93 - GAP_FRAGILE_THRESHOLD + 0.001)
    assert not just_under.is_fragile


def test_trustrow_fragile_is_symmetric():
    """A team much *stronger* vs the real meta than ranking is also pool-dependent."""
    assert TrustRow("t", 0.50, 0.50 + GAP_FRAGILE_THRESHOLD).is_fragile


# --------------------------------------------------------- format_trust_table
def test_format_trust_table_flags_only_fragile_rows():
    rows = [
        TrustRow("team A", 0.93, 0.805),  # gap 12.5% — not fragile
        TrustRow("team B", 0.875, 0.50),  # gap 37.5% — fragile
    ]
    out = format_trust_table(rows, primary_label="PvPoke-synthetic", cross_label="Taiman meta")
    assert "PvPoke-synthetic" in out and "Taiman meta" in out
    lines = out.splitlines()
    row_a = next(line for line in lines if "team A" in line)
    row_b = next(line for line in lines if "team B" in line)
    assert "⚠" not in row_a
    assert "⚠" in row_b
    # gap is rendered signed
    assert "+12.5%" in row_a
    assert "+37.5%" in row_b
