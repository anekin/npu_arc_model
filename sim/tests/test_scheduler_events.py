"""Tests for deterministic event ordering."""

from __future__ import annotations

import pytest

from scheduler.events import Event, EventPhase, EventQueue, event_key


class TestEventOrdering:
    def test_event_key_tuple(self) -> None:
        e = Event(time_ps=100, phase=EventPhase.DISPATCH, seq=5)
        assert event_key(e) == (100, 3, 5)

    def test_release_before_dispatch_at_same_time(self) -> None:
        rel = Event(time_ps=100, phase=EventPhase.RELEASE, seq=0)
        disp = Event(time_ps=100, phase=EventPhase.DISPATCH, seq=0)
        assert rel < disp
        assert disp > rel

    def test_arrival_and_timer_equal_phase_value(self) -> None:
        arr = Event(time_ps=100, phase=EventPhase.ARRIVAL, seq=0)
        timer = Event(time_ps=100, phase=EventPhase.TIMER, seq=0)
        # ARRIVAL (1) < TIMER (2)
        assert arr < timer

    def test_insertion_sequence_tie_breaks_same_phase(self) -> None:
        first = Event(time_ps=100, phase=EventPhase.RELEASE, seq=0)
        second = Event(time_ps=100, phase=EventPhase.RELEASE, seq=1)
        assert first < second

    def test_later_time_dominates_phase(self) -> None:
        early_dispatch = Event(time_ps=50, phase=EventPhase.DISPATCH, seq=0)
        late_release = Event(time_ps=100, phase=EventPhase.RELEASE, seq=0)
        assert early_dispatch < late_release


class TestEventQueue:
    def test_push_pop_order(self) -> None:
        q = EventQueue()
        q.push(200, EventPhase.DISPATCH, "d1")
        q.push(100, EventPhase.RELEASE, "r1")
        q.push(100, EventPhase.DISPATCH, "d2")
        q.push(100, EventPhase.ARRIVAL, "a1")

        events = []
        while (ev := q.pop()) is not None:
            events.append(ev)

        assert [ev.payload for ev in events] == ["r1", "a1", "d2", "d1"]

    def test_seq_increments_on_each_push(self) -> None:
        q = EventQueue()
        e1 = q.push(10, EventPhase.RELEASE)
        e2 = q.push(10, EventPhase.RELEASE)
        assert e1.seq == 0
        assert e2.seq == 1
        assert e1 < e2

    def test_negative_time_rejected(self) -> None:
        q = EventQueue()
        with pytest.raises(ValueError):
            q.push(-1, EventPhase.RELEASE)

    def test_peek_does_not_remove(self) -> None:
        q = EventQueue()
        q.push(10, EventPhase.RELEASE)
        assert q.peek() is not None
        assert len(q) == 1

    def test_iter_is_non_destructive(self) -> None:
        q = EventQueue()
        q.push(20, EventPhase.RELEASE)
        q.push(10, EventPhase.DISPATCH)
        assert list(q)[0].phase == EventPhase.DISPATCH
        assert len(q) == 2

    def test_clear(self) -> None:
        q = EventQueue()
        q.push(10, EventPhase.RELEASE)
        q.clear()
        assert len(q) == 0
        assert q.pop() is None
