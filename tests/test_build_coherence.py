"""Tests for the tier-1 build-coherence fact layer (:mod:`gbl_hacker.coherence`).

The headline contract:

* Every build the engine materializes — from the Taiman meta feed and from
  the PvPoke ranking — must be *internally coherent*: its typing and every
  one of its moves must belong to a single real gamemaster form.
* The regression guard ``test_meta_registry_is_fully_coherent`` is the direct
  defense against the マッギョ *chimera* (a Galarian-typed body wearing base
  Stunfisk's electric moveset). If the ``_DEX_GL_OVERRIDE`` for dex 618 — or
  any similar typing/moveset desync — is reintroduced, this test fails.
* The validator must still *catch* a chimera when one is constructed, and must
  not false-positive on legitimate JA-name/form collisions (ポワルン → the
  Castform weather forms; マッギョ → base vs Galarian).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from gbl_hacker.build_registry import (
    build_registry_for_meta,
    build_registry_pvpoke_top,
)
from gbl_hacker.coherence import validate_build_coherence, validate_builds
from gbl_hacker.persist.snapshot import read_snapshot

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "snapshots"
    / "great_league__upper__2026-05-19.json"
)


@pytest.fixture(scope="module")
def meta_registry():
    return build_registry_for_meta(read_snapshot(_FIXTURE))


@pytest.fixture(scope="module")
def pvpoke_builds():
    return [build for _label, _sid, build in build_registry_pvpoke_top(top_n=30)]


def _build_by_species(builds, species_ja):
    for b in builds:
        if b.species == species_ja:
            return b
    raise AssertionError(f"{species_ja} not found in builds")


# --------------------------------------------------------------- regression
def test_meta_registry_is_fully_coherent(meta_registry):
    """Every Taiman-meta build resolves to one coherent real form.

    This is the chimera regression guard: with the dex-618 override in place
    the マッギョ build had Galarian typing + base-Stunfisk electric moves and
    would appear here as an offender.
    """
    offenders = validate_builds(meta_registry)
    assert offenders == {}, f"incoherent meta builds: {offenders}"


def test_pvpoke_ranking_builds_are_fully_coherent(pvpoke_builds):
    offenders = validate_builds(pvpoke_builds)
    assert offenders == {}, f"incoherent pvpoke builds: {offenders}"


# ------------------------------------------------- form-collision tolerance
def test_castform_weather_form_is_coherent(pvpoke_builds):
    """ポワルン's JA name resolves to base Castform, but the ranked build is a
    weather form. The dex-wide form search must still accept it (no false
    positive)."""
    castform = _build_by_species(pvpoke_builds, "ポワルン")
    assert validate_build_coherence(castform) == []


def test_coherent_build_returns_no_violations(meta_registry):
    stunfisk = meta_registry["マッギョ"]  # base Stunfisk, ground/electric
    assert stunfisk.types == ("ground", "electric")
    assert validate_build_coherence(stunfisk) == []


# --------------------------------------------------------- chimera detection
def test_detects_typing_chimera(meta_registry):
    """Graft Galarian typing onto base Stunfisk's electric moveset — the exact
    historical chimera. The validator must flag the electric moves as absent
    from the only ground/steel dex-618 form (Galarian)."""
    stunfisk = meta_registry["マッギョ"]
    chimera = dataclasses.replace(stunfisk, types=("ground", "steel"))
    violations = validate_build_coherence(chimera)
    assert violations, "chimera went undetected"
    joined = " ".join(violations)
    assert "stunfisk_galarian" in joined
    assert "Discharge" in joined  # an electric move Galarian cannot learn


def test_detects_unlearnable_move(meta_registry, pvpoke_builds):
    """A move no same-typed form of the dex can learn is flagged, even with
    correct typing."""
    stunfisk = meta_registry["マッギョ"]  # ground/electric
    talonflame = _build_by_species(pvpoke_builds, "ファイアロー")
    grafted = dataclasses.replace(stunfisk, fast=talonflame.fast)  # Incinerate
    violations = validate_build_coherence(grafted)
    assert any("fast move" in v and "Incinerate" in v for v in violations), violations


# --------------------------------------------------- build-pipeline guard
def test_strict_meta_registry_passes():
    """The pinned fixture must materialize cleanly under the strict policy —
    the regression guard at the pipeline boundary."""
    snap = read_snapshot(_FIXTURE)
    build_registry_for_meta(snap, coherence="raise")  # must not raise


def test_strict_pvpoke_top_passes():
    build_registry_pvpoke_top(top_n=30, coherence="raise")  # must not raise


def test_default_policy_is_silent_on_clean_fixture(recwarn):
    snap = read_snapshot(_FIXTURE)
    build_registry_for_meta(snap)  # default "warn"
    assert not [w for w in recwarn.list if "incoherent" in str(w.message)]


def test_invalid_coherence_policy_rejected():
    snap = read_snapshot(_FIXTURE)
    with pytest.raises(ValueError, match="coherence must be one of"):
        build_registry_for_meta(snap, coherence="bogus")


def test_runtime_guard_catches_reintroduced_override(monkeypatch):
    """Re-adding the dex-618 Galarian override recreates the chimera; the
    strict pipeline guard must reject it."""
    import gbl_hacker.gamemaster as gmmod

    monkeypatch.setattr(gmmod, "_DEX_GL_OVERRIDE", {618: "stunfisk_galarian"})
    snap = read_snapshot(_FIXTURE)
    with pytest.raises(ValueError, match="incoherent"):
        build_registry_for_meta(snap, coherence="raise")
