"""Scheduler queues: bounded FIFO and per-stream mailbox_latest.

``BoundedFIFO`` enforces a capacity limit and exposes a terminal
``queue_full`` flag.  ``MailboxLatest`` keeps one pending item per
stream/context, replacing an unread pending item rather than accumulating
a backlog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, TypeVar


class QueueFullError(RuntimeError):
    """Raised when an item is enqueued into a full bounded FIFO."""


T = TypeVar("T")


@dataclass
class BoundedFIFO(Generic[T]):
    """A bounded FIFO queue with terminal full detection."""

    capacity: int
    name: str = "fifo"
    _items: list[tuple[int, T]] = field(default_factory=list, repr=False)
    _sequence: int = field(default=0, repr=False)
    queue_full: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise QueueFullError(f"FIFO capacity must be positive, got {self.capacity}")

    @property
    def size(self) -> int:
        return len(self._items)

    @property
    def is_full(self) -> bool:
        return self.size >= self.capacity

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    def enqueue(self, item: T) -> int:
        """Enqueue ``item`` and return its sequence number.

        Raises ``QueueFullError`` if the FIFO is already full and sets the
        terminal ``queue_full`` flag.
        """
        if self.is_full:
            self.queue_full = True
            raise QueueFullError(
                f"FIFO {self.name!r} is full (capacity={self.capacity})"
            )
        seq = self._sequence
        self._items.append((seq, item))
        self._sequence += 1
        return seq

    def dequeue(self) -> tuple[int, T]:
        """Dequeue the oldest item and return (sequence, item)."""
        if self.is_empty:
            raise QueueFullError(f"FIFO {self.name!r} is empty")
        return self._items.pop(0)

    def peek(self) -> tuple[int, T] | None:
        """Return the oldest item without removing it."""
        if self.is_empty:
            return None
        return self._items[0]

    def clear(self) -> None:
        """Remove all items and reset the terminal flag."""
        self._items.clear()
        self.queue_full = False
        self._sequence = 0


@dataclass
class MailboxLatest(Generic[T]):
    """Per-stream/context mailbox keeping at most one pending item.

    When a new item arrives for a stream that already has a pending item,
    the pending item is replaced (not merged) and ``replacement_count`` is
    incremented.  The mailbox records the arrival time so that observation
    age can be computed.
    """

    name: str = "mailbox"
    _pending: Dict[str, tuple[int, T]] = field(default_factory=dict, repr=False)
    _arrival_ps: Dict[str, int] = field(default_factory=dict, repr=False)
    replacement_count: int = field(default=0, init=False)

    @property
    def stream_ids(self) -> list[str]:
        return sorted(self._pending.keys())

    def put(self, stream_id: str, item: T, now_ps: int) -> bool:
        """Store ``item`` for ``stream_id``.

        Returns ``True`` if an existing pending item was replaced.
        """
        replaced = stream_id in self._pending
        if replaced:
            self.replacement_count += 1
        self._pending[stream_id] = (self._arrival_ps.get(stream_id, now_ps) if replaced else now_ps, item)
        if not replaced:
            self._arrival_ps[stream_id] = now_ps
        return replaced

    def take(self, stream_id: str, now_ps: int) -> tuple[int, T] | None:
        """Return and remove the pending item, with its observation age.

        Observation age = ``now_ps - arrival_ps``.  Returns ``None`` if no
        pending item exists.
        """
        if stream_id not in self._pending:
            return None
        arrival_ps, item = self._pending.pop(stream_id)
        age_ps = now_ps - arrival_ps
        return (age_ps, item)

    def peek(self, stream_id: str, now_ps: int) -> tuple[int, T] | None:
        """Return the pending item and its age without removing it."""
        if stream_id not in self._pending:
            return None
        arrival_ps, item = self._pending[stream_id]
        return (now_ps - arrival_ps, item)

    def has_pending(self, stream_id: str) -> bool:
        return stream_id in self._pending

    def age_ps(self, stream_id: str, now_ps: int) -> int:
        """Return observation age for ``stream_id``; raises if none pending."""
        if stream_id not in self._pending:
            raise KeyError(stream_id)
        return now_ps - self._arrival_ps[stream_id]

    def clear(self) -> None:
        """Remove all pending items and reset statistics."""
        self._pending.clear()
        self._arrival_ps.clear()
        self.replacement_count = 0
