"""Tests for scheduler resources and byte servers."""

from __future__ import annotations

import pytest
from contracts.errors import ConfigError
from scheduler.resources import (
    BoundedResource,
    ByteServer,
    CapacityResource,
    ResourceError,
)
from tests.oracles.scheduler import (
    equal_share_completion_times,
    single_transfer_completion,
    strict_priority_completion_times,
)


class TestCapacityResource:
    def test_zero_capacity_rejected(self) -> None:
        with pytest.raises(ResourceError):
            CapacityResource(name="zero", capacity=0)

    def test_allocate_and_release(self) -> None:
        r = CapacityResource(name="dma", capacity=2)
        assert r.allocate("a") == 1
        assert r.allocate("b") == 1
        assert r.allocate("c") == 0
        assert r.used == 2
        assert r.release("a") == 1
        assert r.allocate("c") == 1

    def test_partial_allocation(self) -> None:
        r = CapacityResource(name="partial", capacity=2, allow_partial=True)
        r.allocate("a", units=3)
        assert r.used == 2

    def test_release_all(self) -> None:
        r = CapacityResource(name="dma", capacity=4)
        r.allocate("a", 2)
        r.allocate("b", 1)
        assert r.release_all() == 3
        assert r.used == 0

    def test_negative_units_rejected(self) -> None:
        r = CapacityResource(name="dma", capacity=2)
        with pytest.raises(ResourceError):
            r.allocate("a", units=-1)


class TestBoundedResource:
    def test_zero_capacity_rejected(self) -> None:
        with pytest.raises(ResourceError):
            BoundedResource(name="fifo", capacity=0)

    def test_acquire_full_or_nothing(self) -> None:
        r = BoundedResource(name="port", capacity=2)
        assert r.acquire("a", 2) is True
        assert r.acquire("b", 1) is False
        assert r.release("a", 1) == 1
        assert r.acquire("b", 1) is True

    def test_negative_units_rejected(self) -> None:
        r = BoundedResource(name="port", capacity=2)
        with pytest.raises(ResourceError):
            r.acquire("a", -1)


class TestByteServer:
    def _bandwidth(self, bytes_per_us: float) -> float:
        """Convert bytes per microsecond to bytes per picosecond."""
        return bytes_per_us / 1_000_000.0

    def test_zero_bandwidth_rejected(self) -> None:
        with pytest.raises(ResourceError):
            ByteServer(name="mem", bandwidth_bytes_per_ps=0.0)

    def test_unknown_qos_mode_rejected(self) -> None:
        with pytest.raises(ConfigError):
            ByteServer(name="mem", bandwidth_bytes_per_ps=1.0, qos_mode="magic")

    def test_single_transfer(self) -> None:
        # 100 B at 10 B/us => 10 us = 10_000_000 ps
        server = ByteServer(name="mem", bandwidth_bytes_per_ps=self._bandwidth(10.0))
        server.add_member("a", 100)
        assert server.completion_time_ps("a") == 10_000_000

    def test_equal_share_two_members(self) -> None:
        # 100 B each at 10 B/us total => 5 B/us each => 20 us
        server = ByteServer(name="mem", bandwidth_bytes_per_ps=self._bandwidth(10.0))
        server.add_member("a", 100)
        server.add_member("b", 100)
        assert server.completion_time_ps("a") == 20_000_000
        assert server.completion_time_ps("b") == 20_000_000

    def test_strict_priority(self) -> None:
        server = ByteServer(
            name="mem",
            bandwidth_bytes_per_ps=self._bandwidth(10.0),
            qos_mode="strict_priority",
        )
        server.add_member("low", 100, priority=0)
        server.add_member("high", 100, priority=1)
        assert server.completion_time_ps("high") == 10_000_000
        assert server.completion_time_ps("low") == 20_000_000

    def test_strict_priority_tie_break_by_id(self) -> None:
        server = ByteServer(
            name="mem",
            bandwidth_bytes_per_ps=self._bandwidth(10.0),
            qos_mode="strict_priority",
        )
        server.add_member("b", 100, priority=0)
        server.add_member("a", 100, priority=0)
        # Same priority: "a" before "b" lexicographically.
        assert server.completion_time_ps("a") == 10_000_000
        assert server.completion_time_ps("b") == 20_000_000

    def test_recompute_on_remove(self) -> None:
        server = ByteServer(name="mem", bandwidth_bytes_per_ps=self._bandwidth(10.0))
        server.add_member("a", 100)
        server.add_member("b", 100)
        assert server.completion_time_ps("a") == 20_000_000
        server.remove_member("b", now_ps=0)
        # "a" now has full bandwidth; completion unchanged because it started at 0.
        assert server.completion_time_ps("a") == 10_000_000

    def test_recompute_with_nonzero_now(self) -> None:
        server = ByteServer(name="mem", bandwidth_bytes_per_ps=self._bandwidth(10.0))
        server.add_member("a", 100, now_ps=5_000_000)
        assert server.completion_time_ps("a") == 15_000_000

    def test_negative_bytes_rejected(self) -> None:
        server = ByteServer(name="mem", bandwidth_bytes_per_ps=self._bandwidth(10.0))
        with pytest.raises(ResourceError):
            server.add_member("a", -1)

    def test_unknown_member(self) -> None:
        server = ByteServer(name="mem", bandwidth_bytes_per_ps=self._bandwidth(10.0))
        with pytest.raises(ResourceError):
            server.completion_time_ps("missing")


class TestByteServerOracle:
    """Hand-audited cases verified against the independent oracle."""

    def _bw(self, bytes_per_us: float) -> float:
        return bytes_per_us / 1_000_000.0

    def test_single_100b_at_10b_us(self) -> None:
        expected = single_transfer_completion(100, 10.0)
        server = ByteServer(name="mem", bandwidth_bytes_per_ps=self._bw(10.0))
        server.add_member("a", 100)
        assert server.completion_time_ps("a") == expected

    def test_two_equal_share(self) -> None:
        expected = equal_share_completion_times([("a", 100), ("b", 100)], 10.0)
        server = ByteServer(name="mem", bandwidth_bytes_per_ps=self._bw(10.0))
        server.add_member("a", 100)
        server.add_member("b", 100)
        assert server.completion_time_ps("a") == expected["a"]
        assert server.completion_time_ps("b") == expected["b"]
        assert server.completion_time_ps("a") == 20_000_000

    def test_strict_priority_high_low(self) -> None:
        expected = strict_priority_completion_times([("low", 100, 0), ("high", 100, 1)], 10.0)
        server = ByteServer(
            name="mem",
            bandwidth_bytes_per_ps=self._bw(10.0),
            qos_mode="strict_priority",
        )
        server.add_member("low", 100, priority=0)
        server.add_member("high", 100, priority=1)
        assert server.completion_time_ps("high") == expected["high"]
        assert server.completion_time_ps("low") == expected["low"]
        assert server.completion_time_ps("high") == 10_000_000
        assert server.completion_time_ps("low") == 20_000_000
