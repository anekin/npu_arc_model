"""Tests for admission control."""

from __future__ import annotations

import pytest

from scheduler.admission import AdmissionController, AdmissionResult


class TestAdmissionMemory:
    def test_admit_within_memory(self) -> None:
        ctrl = AdmissionController(memory_available_bytes=100, max_inflight_jobs=1)
        result = ctrl.admit("j1", memory_bytes=50)
        assert result.admitted is True
        assert ctrl.memory_used_bytes == 50

    def test_reject_over_memory(self) -> None:
        ctrl = AdmissionController(memory_available_bytes=100, max_inflight_jobs=1)
        result = ctrl.admit("j1", memory_bytes=101)
        assert result.admitted is False
        assert "memory" in result.reason

    def test_check_without_state_change(self) -> None:
        ctrl = AdmissionController(memory_available_bytes=100, max_inflight_jobs=1)
        result = ctrl.check("j1", memory_bytes=50)
        assert result.admitted is True
        assert ctrl.memory_used_bytes == 0


class TestAdmissionInflight:
    def test_reject_at_max_inflight(self) -> None:
        ctrl = AdmissionController(memory_available_bytes=1000, max_inflight_jobs=1)
        ctrl.admit("j1")
        result = ctrl.admit("j2")
        assert result.admitted is False
        assert "inflight" in result.reason

    def test_admit_under_max_inflight(self) -> None:
        ctrl = AdmissionController(memory_available_bytes=1000, max_inflight_jobs=2)
        ctrl.admit("j1")
        result = ctrl.admit("j2")
        assert result.admitted is True


class TestAdmissionBandwidth:
    def test_reject_bandwidth_overflow(self) -> None:
        ctrl = AdmissionController(memory_available_bytes=1000, max_inflight_jobs=2)
        ctrl.admit("j1", bandwidth_fraction=0.6)
        result = ctrl.admit("j2", bandwidth_fraction=0.5)
        assert result.admitted is False
        assert "bandwidth" in result.reason

    def test_admit_exact_bandwidth(self) -> None:
        ctrl = AdmissionController(memory_available_bytes=1000, max_inflight_jobs=2)
        ctrl.admit("j1", bandwidth_fraction=0.5)
        result = ctrl.admit("j2", bandwidth_fraction=0.5)
        assert result.admitted is True


class TestAdmissionBlocking:
    def test_lower_priority_blocking(self) -> None:
        ctrl = AdmissionController(memory_available_bytes=1000, max_inflight_jobs=2)
        result = ctrl.admit(
            "high",
            priority=5,
            exclusive_resources={"dma0"},
            lower_priority_holders={"dma0": 1},
        )
        assert result.admitted is False
        assert "lower_priority_blocking" in result.reason

    def test_equal_priority_not_blocking(self) -> None:
        ctrl = AdmissionController(memory_available_bytes=1000, max_inflight_jobs=2)
        result = ctrl.admit(
            "same",
            priority=3,
            exclusive_resources={"dma0"},
            lower_priority_holders={"dma0": 3},
        )
        assert result.admitted is True

    def test_unheld_resource_not_blocking(self) -> None:
        ctrl = AdmissionController(memory_available_bytes=1000, max_inflight_jobs=2)
        result = ctrl.admit(
            "new",
            priority=3,
            exclusive_resources={"dma1"},
            lower_priority_holders={"dma0": 1},
        )
        assert result.admitted is True


class TestAdmissionRelease:
    def test_release_restores_capacity(self) -> None:
        ctrl = AdmissionController(memory_available_bytes=100, max_inflight_jobs=1)
        ctrl.admit("j1", memory_bytes=60, bandwidth_fraction=0.4)
        assert ctrl.release("j1") is True
        assert ctrl.memory_used_bytes == 0
        assert ctrl.bandwidth_fraction_used == 0.0
        result = ctrl.admit("j2", memory_bytes=90, bandwidth_fraction=0.5)
        assert result.admitted is True

    def test_release_unknown_returns_false(self) -> None:
        ctrl = AdmissionController(memory_available_bytes=100, max_inflight_jobs=1)
        assert ctrl.release("missing") is False


class TestAdmissionInvalidConfig:
    def test_negative_memory_rejected(self) -> None:
        with pytest.raises(ValueError):
            AdmissionController(memory_available_bytes=-1, max_inflight_jobs=1)

    def test_zero_inflight_rejected(self) -> None:
        with pytest.raises(ValueError):
            AdmissionController(memory_available_bytes=100, max_inflight_jobs=0)

    def test_bad_bandwidth_fraction_rejected(self) -> None:
        with pytest.raises(ValueError):
            AdmissionController(
                memory_available_bytes=100,
                max_inflight_jobs=1,
                max_bandwidth_fraction=1.5,
            )
