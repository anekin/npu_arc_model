"""Tests for scheduler queues."""

from __future__ import annotations

import pytest
from scheduler.queues import BoundedFIFO, MailboxLatest, QueueFullError


class TestBoundedFIFO:
    def test_zero_capacity_rejected(self) -> None:
        with pytest.raises(QueueFullError):
            BoundedFIFO(capacity=0)

    def test_fifo_order(self) -> None:
        q = BoundedFIFO(capacity=3, name="cmd")
        q.enqueue("a")
        q.enqueue("b")
        assert q.dequeue() == (0, "a")
        assert q.dequeue() == (1, "b")

    def test_capacity_limit(self) -> None:
        q = BoundedFIFO(capacity=2)
        q.enqueue("a")
        q.enqueue("b")
        assert q.is_full
        with pytest.raises(QueueFullError):
            q.enqueue("c")
        assert q.queue_full

    def test_dequeue_empty_raises(self) -> None:
        q = BoundedFIFO(capacity=2)
        with pytest.raises(QueueFullError):
            q.dequeue()

    def test_clear_resets_full_flag(self) -> None:
        q = BoundedFIFO(capacity=1)
        q.enqueue("a")
        with pytest.raises(QueueFullError):
            q.enqueue("b")
        q.clear()
        assert not q.queue_full
        q.enqueue("c")

    def test_peek_empty(self) -> None:
        q = BoundedFIFO(capacity=2)
        assert q.peek() is None


class TestMailboxLatest:
    def test_put_and_take(self) -> None:
        m = MailboxLatest(name="stream")
        m.put("s1", "first", now_ps=100)
        assert m.take("s1", now_ps=200) == (100, "first")
        assert m.take("s1", now_ps=300) is None

    def test_replace_pending(self) -> None:
        m = MailboxLatest()
        m.put("s1", "first", now_ps=100)
        replaced = m.put("s1", "second", now_ps=150)
        assert replaced is True
        age, item = m.take("s1", now_ps=250)
        # Age measures from original arrival, not replacement.
        assert age == 150
        assert item == "second"
        assert m.replacement_count == 1

    def test_per_stream_isolation(self) -> None:
        m = MailboxLatest()
        m.put("s1", "a", now_ps=0)
        m.put("s2", "b", now_ps=0)
        assert m.take("s1", now_ps=10) == (10, "a")
        assert m.take("s2", now_ps=20) == (20, "b")

    def test_peek_without_removal(self) -> None:
        m = MailboxLatest()
        m.put("s1", "x", now_ps=100)
        assert m.peek("s1", now_ps=150) == (50, "x")
        assert m.has_pending("s1")

    def test_age_unknown_stream(self) -> None:
        m = MailboxLatest()
        with pytest.raises(KeyError):
            m.age_ps("missing", now_ps=0)

    def test_clear(self) -> None:
        m = MailboxLatest()
        m.put("s1", "a", now_ps=0)
        m.put("s1", "b", now_ps=10)
        m.clear()
        assert not m.has_pending("s1")
        assert m.replacement_count == 0
