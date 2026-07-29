"""Stable design-point identities via canonical normalized JSON SHA-256.

Every design point is identified by a deterministic digest of its normalized
configuration.  The digest is derived via SHA-256 over a canonical JSON
representation with the following guarantees:

* Sorted keys at every nesting level.
* Stable float representation (repr, not str).
* Enum values serialised as their string name.
* No timestamps, absolute paths, or iteration-order artefacts.

The same normalized input always produces the same ID.  Any change to any
design axis (engine type, array dimensions, frequency, bandwidth, etc.)
changes the ID.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

__all__ = [
    "canonical_json_bytes",
    "digest_sha256",
    "normalise_for_hashing",
]


def _normalise_value(v: Any) -> Any:
    """Recursively normalise a value for deterministic serialization.

    * float → repr (e.g. ``1.0``, ``0.85``) — stable across platforms.
    * Enum → its ``.value`` (string or primitive).
    * bool / int — pass through unchanged.
    * dict → sorted keys, recursively normalised values.
    * list / tuple → recursively normalised values.
    * None → null.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int,)):
        return v
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return v
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, bytes):
        return v.hex()
    if isinstance(v, (list, tuple)):
        return [_normalise_value(item) for item in v]
    if isinstance(v, dict):
        return {str(k): _normalise_value(val) for k, val in sorted(v.items(), key=lambda x: str(x[0]))}
    # Fallback: convert to string for unknown types
    return str(v)


def normalise_for_hashing(obj: dict[str, Any]) -> dict[str, Any]:
    """Return a recursively normalised copy of *obj* suitable for hashing.

    Keys are sorted; floats become ``repr`` strings; enums become their value.
    """
    result = _normalise_value(obj)
    if not isinstance(result, dict):
        raise ValueError(f"normalise_for_hashing requires a dict input, got {type(obj).__name__}")
    return result


def canonical_json_bytes(obj: dict[str, Any], *, indent: int | str | None = None) -> bytes:
    """Serialize *obj* to canonical JSON bytes.

    The output is deterministic — same input always yields identical bytes,
    regardless of dict-insertion order, float formatting, or enum subclass.
    """
    normalised = normalise_for_hashing(obj)
    return json.dumps(
        normalised,
        sort_keys=True,
        indent=indent,
        ensure_ascii=True,
        separators=(",", ":") if indent is None else None,
    ).encode("utf-8")


def digest_sha256(obj: dict[str, Any]) -> str:
    """Return the SHA-256 hex digest of the canonical JSON of *obj*."""
    raw = canonical_json_bytes(obj)
    return hashlib.sha256(raw).hexdigest()
