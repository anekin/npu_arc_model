"""Deterministic event abstraction for the discrete-event scheduler.

All times are integer picoseconds.  Event ordering is defined by the
tuple ``(time_ps, phase, insertion_sequence)`` so that events at the same
timestamp are processed in a deterministic, controllable order:

  RELEASE → ARRIVAL/TIMER → DISPATCH

The phase enum values encode this order directly; lower enum value =
processed earlier.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class EventPhase(IntEnum):
    """Ordered event phases.

    Values increase in the order the scheduler processes them at a given
    timestamp.
    """

    RELEASE = 0
    ARRIVAL = 1
    TIMER = 2
    DISPATCH = 3


@dataclass(frozen=True, slots=True)
class Event:
    """A single deterministic event.

    ``payload`` is an opaque callable or value; the kernel invokes it as
    ``payload(kernel)`` when the event is dispatched.
    """

    time_ps: int
    phase: EventPhase
    seq: int
    payload: Any = None

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Event):
            return NotImplemented
        return event_key(self) < event_key(other)

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Event):
            return NotImplemented
        return event_key(self) <= event_key(other)

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Event):
            return NotImplemented
        return event_key(self) > event_key(other)

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Event):
            return NotImplemented
        return event_key(self) >= event_key(other)


def event_key(event: Event) -> tuple[int, int, int]:
    """Return the canonical deterministic ordering key for ``event``."""
    return (event.time_ps, int(event.phase), event.seq)


class EventQueue:
    """Min-heap event queue with monotonic sequence counters.

    The queue guarantees that every inserted event receives a unique
    insertion sequence within the queue instance, so two events with the
    same ``(time_ps, phase)`` are still ordered deterministically.
    """

    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._seq = 0
        self._cancelled: set[int] = set()

    def push(
        self,
        time_ps: int,
        phase: EventPhase,
        payload: Any = None,
    ) -> Event:
        """Push an event and return it."""
        if time_ps < 0:
            raise ValueError(f"event time_ps must be non-negative, got {time_ps}")
        event = Event(time_ps=time_ps, phase=phase, seq=self._seq, payload=payload)
        self._seq += 1
        heapq.heappush(self._heap, event)
        return event

    def cancel(self, event: Event) -> bool:
        """Cancel a queued event.  Returns True if it was still queued."""
        if event.seq in self._cancelled:
            return False
        self._cancelled.add(event.seq)
        return True

    def pop(self) -> Event | None:
        """Pop the earliest non-cancelled event, or ``None`` if empty."""
        while self._heap:
            event = heapq.heappop(self._heap)
            if event.seq in self._cancelled:
                self._cancelled.discard(event.seq)
                continue
            return event
        return None

    def peek(self) -> Event | None:
        """Return the earliest non-cancelled event without removing it."""
        for event in sorted(self._heap):
            if event.seq not in self._cancelled:
                return event
        return None

    def __len__(self) -> int:
        return len(self._heap) - len(self._cancelled)

    def __iter__(self) -> Iterator[Event]:
        """Iterate over queued events in sorted order (non-destructive)."""
        return iter(e for e in sorted(self._heap) if e.seq not in self._cancelled)

    def clear(self) -> None:
        """Remove all queued events and cancellations."""
        self._heap.clear()
        self._cancelled.clear()
