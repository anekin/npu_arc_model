"""Unified engine registry — single source of truth for all 8 engines.

Every consumer (factory, DSE, npu_sim choices, report iteration, tests)
derives its engine list from this registry. No second handwritten list.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from contracts.errors import ConfigError

# ── Canonical engine IDs and metadata ────────────────────────────────────────

_EngineFactory = Callable[[Dict[str, Any]], Any]

_CANONICAL_IDS: Tuple[str, ...] = (
    "systolic",
    "os_systolic",
    "block",
    "tensor_core",
    "wmma",
    "gmma",
    "input_stationary",
    "fsa",
)

_ENGINE_LABELS: Dict[str, str] = {
    "systolic":         "Weight-Stationary Systolic Array (TPUv1)",
    "os_systolic":      "Output-Stationary Systolic (Gemmini)",
    "block":            "Block Engine — full parallel MAC (TPUv4 VMU)",
    "tensor_core":      "Multi 16×16 Tensor Cores (A100 style)",
    "wmma":             "16×16 Warp MMA (Volta/Ampere style)",
    "gmma":             "Group MMA + TMA async DMA (Hopper H100 style)",
    "input_stationary": "Input-Stationary (Eyeriss)",
    "fsa":              "FSA — inline FlashAttention on systolic array",
}

_ENGINE_QUICK_IDS: Tuple[str, ...] = ("systolic", "block", "gmma")

# Lazy-loaded factory functions — imported on first access to avoid circular imports
_factories: Dict[str, _EngineFactory] = {}


def _init_factories() -> None:
    """Lazy-import all engine factory functions."""
    if _factories:
        return
    from engine.systolic_engine import SystolicEngine
    from engine.os_systolic_engine import OutputStationaryEngine
    from engine.block_engine import BlockEngine
    from engine.tensor_core_engine import TensorCoreEngine
    from engine.wmma_engine import WMMAEngine
    from engine.gmma_engine import GMMAEngine
    from engine.is_systolic_engine import InputStationaryEngine
    from engine.fsa_engine import FSAEngine

    _factories["systolic"] = lambda cfg: SystolicEngine(cfg)
    _factories["os_systolic"] = lambda cfg: OutputStationaryEngine(cfg)
    _factories["block"] = lambda cfg: BlockEngine(cfg)
    _factories["tensor_core"] = lambda cfg: TensorCoreEngine(cfg)
    _factories["wmma"] = lambda cfg: WMMAEngine(cfg)
    _factories["gmma"] = lambda cfg: GMMAEngine(cfg)
    _factories["input_stationary"] = lambda cfg: InputStationaryEngine(cfg)
    _factories["fsa"] = lambda cfg: FSAEngine(cfg)


# ── Public API ────────────────────────────────────────────────────────────────


def canonical_engine_ids() -> Tuple[str, ...]:
    """Return all 8 canonical engine IDs in registration order.

    Consumers MUST derive their engine lists from this function — never
    from a handwritten constant.
    """
    return _CANONICAL_IDS


def quick_engine_ids() -> Tuple[str, ...]:
    """Return the 3 priority engine IDs for quick-mode DSE."""
    return _ENGINE_QUICK_IDS


def engine_label(engine_id: str) -> str:
    """Return the human-readable description for an engine ID.

    Raises ConfigError for unknown engines.
    """
    try:
        return _ENGINE_LABELS[engine_id]
    except KeyError:
        raise ConfigError(
            f"Unknown engine type: {engine_id!r}",
            field_path="mac_engine.type",
        ) from None


def engine_choices() -> List[str]:
    """Return the CLI choices list (sorted alphabetically)."""
    return sorted(_CANONICAL_IDS)


def engine_listing() -> str:
    """Return a formatted listing for --list-engines output."""
    lines = ["Available MAC engines:"]
    for eid in _CANONICAL_IDS:
        lines.append(f"  {eid:<18} {_ENGINE_LABELS[eid]}")
    return "\n".join(lines)


def engine_full_ids() -> List[str]:
    """Return the list of all 8 canonical IDs for DSE full-mode enumeration."""
    return list(_CANONICAL_IDS)


def engine_quick_ids_list() -> List[str]:
    """Return the list of priority IDs for DSE quick-mode enumeration."""
    return list(_ENGINE_QUICK_IDS)


def create_engine_by_type(engine_type: str, config: Dict[str, Any]) -> Any:
    """Create an engine instance from the canonical engine type string.

    Uses the lazy-loaded factory registry.  Raises ``ConfigError``
    for unknown engine types (never ``ValueError``).
    """
    _init_factories()
    factory = _factories.get(engine_type)
    if factory is None:
        raise ConfigError(
            f"Unknown engine type: {engine_type!r}",
            field_path="mac_engine.type",
        )
    return factory(config)


def is_valid_engine(engine_type: str) -> bool:
    """Return True if engine_type is a canonical engine ID."""
    return engine_type in _ENGINE_LABELS


def lookup_by_prefix(prefix: str) -> str:
    """Resolve a config_label prefix to a canonical engine ID.

    DSE config_labels truncate engine names (e.g. 'syst' → 'systolic',
    'fsa ' → 'fsa').  Uses the registry to map prefixes back to canonical IDs.

    Returns the canonical ID, or raises ConfigError if no match.
    """
    _init_factories()
    prefix_clean = prefix.strip()
    # Exact match first
    if prefix_clean in _ENGINE_LABELS:
        return prefix_clean
    # Prefix match — DSE truncates to 4 chars
    matches = [eid for eid in _CANONICAL_IDS if eid.startswith(prefix_clean)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ConfigError(
            f"Cannot resolve engine prefix {prefix!r} to any canonical ID",
            field_path="config_label",
        )
    raise ConfigError(
        f"Ambiguous engine prefix {prefix!r} matches {matches}",
        field_path="config_label",
    )
