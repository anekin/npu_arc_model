"""Tests for operator registry — modeled, free/fused, and unsupported entries.

Covers:
- Modeled ops: gemm, softmax, layernorm, conv, etc.
- Free/fused ops: reshape, concat, reduce_mean, transpose (must carry fused_into)
- Unsupported ops: gather, upsample, batch_norm, unregistered
- Registry API: lookup, check, is_modeled, is_free_or_fused, is_unsupported
- Fail-closed: unregistered ops treated as unsupported
- Default op prevention: 0-cycle is only for explicitly free/fused
"""

from __future__ import annotations

import pytest

from contracts.errors import UnsupportedOperatorError
from workloads.operators import (
    DEFAULT_REGISTRY,
    OperatorDisposition,
    OperatorEntry,
    OperatorRegistry,
)


class TestModeledOps:
    """Test that all expected ops are registered as modeled."""

    MODELED_EXPECTED = [
        "gemm", "pointwise_conv", "depthwise_conv", "softmax",
        "layernorm", "rms_norm", "gelu", "relu",
        "hard_swish", "hard_sigmoid", "global_avg_pool", "max_pool",
        "add", "mul", "matmul", "conv",
    ]

    def test_all_expected_modeled_ops_registered(self):
        """Every expected modeled op should be in the default registry."""
        for op_type in self.MODELED_EXPECTED:
            assert DEFAULT_REGISTRY.is_modeled(op_type), (
                f"Expected {op_type!r} to be modeled"
            )
            disposition = DEFAULT_REGISTRY.check(op_type)
            assert disposition == OperatorDisposition.MODELED

    def test_modeled_ops_lookup_succeeds(self):
        """Lookup on modeled ops should return the entry without raising."""
        for op_type in self.MODELED_EXPECTED:
            entry = DEFAULT_REGISTRY.lookup(op_type)
            assert entry.op_type == op_type
            assert entry.disposition == OperatorDisposition.MODELED

    def test_modeled_ops_dont_have_fused_into(self):
        """Modeled ops should not have fused_into set (it's for free/fused only)."""
        for op_type in self.MODELED_EXPECTED:
            entry = DEFAULT_REGISTRY.lookup(op_type)
            assert entry.fused_into is None, (
                f"Modeled op {op_type!r} should not have fused_into"
            )

    def test_modeled_ops_frozen_set(self):
        """modeled_ops property should return a frozenset."""
        modeled = DEFAULT_REGISTRY.modeled_ops
        assert isinstance(modeled, frozenset)
        assert "gemm" in modeled
        assert "softmax" in modeled


class TestFreeFusedOps:
    """Test that free/fused ops are correctly registered with fused_into."""

    FREE_FUSED_EXPECTED = [
        "reshape", "reduce_mean", "shape", "concat", "transpose",
    ]

    def test_all_expected_free_fused_ops_registered(self):
        """Every expected free/fused op should be in the default registry."""
        for op_type in self.FREE_FUSED_EXPECTED:
            assert DEFAULT_REGISTRY.is_free_or_fused(op_type), (
                f"Expected {op_type!r} to be explicitly free_or_fused"
            )

    def test_free_fused_ops_have_fused_into(self):
        """Every free/fused op must carry a fused_into value."""
        for op_type in self.FREE_FUSED_EXPECTED:
            entry = DEFAULT_REGISTRY.lookup(op_type)
            assert entry.disposition == OperatorDisposition.EXPLICITLY_FREE_OR_FUSED
            assert entry.fused_into is not None, (
                f"Free/fused op {op_type!r} must record fused_into"
            )
            assert len(entry.fused_into) > 0, (
                f"Free/fused op {op_type!r} fused_into must not be empty"
            )

    def test_free_fused_ops_not_counted_as_modeled(self):
        """Free/fused ops should NOT be counted as modeled."""
        for op_type in self.FREE_FUSED_EXPECTED:
            assert not DEFAULT_REGISTRY.is_modeled(op_type)

    def test_free_fused_ops_lookup_succeeds(self):
        """Lookup on free/fused ops should return the entry without raising."""
        for op_type in self.FREE_FUSED_EXPECTED:
            entry = DEFAULT_REGISTRY.lookup(op_type)
            assert entry.op_type == op_type
            assert entry.disposition == OperatorDisposition.EXPLICITLY_FREE_OR_FUSED

    def test_free_fused_without_fused_into_rejected_at_construction(self):
        """Constructing a free/fused entry without fused_into must fail."""
        with pytest.raises(ValueError, match="fused_into"):
            OperatorEntry(
                op_type="some_free_op",
                disposition=OperatorDisposition.EXPLICITLY_FREE_OR_FUSED,
            )

    def test_free_fused_with_fused_into_constructs(self):
        """Free/fused entry with fused_into should construct successfully."""
        entry = OperatorEntry(
            op_type="my_fused_op",
            disposition=OperatorDisposition.EXPLICITLY_FREE_OR_FUSED,
            fused_into="preceding_gemm",
        )
        assert entry.fused_into == "preceding_gemm"


class TestUnsupportedOps:
    """Test that unsupported ops are correctly marked and fail-closed."""

    UNSUPPORTED_EXPECTED = [
        "gather", "scatter", "instance_norm", "group_norm",
        "batch_norm", "upsample",
    ]

    def test_all_expected_unsupported_ops_registered(self):
        """Every expected unsupported op should be in the default registry."""
        for op_type in self.UNSUPPORTED_EXPECTED:
            assert DEFAULT_REGISTRY.is_unsupported(op_type), (
                f"Expected {op_type!r} to be unsupported"
            )

    def test_unsupported_ops_raise_on_lookup(self):
        """Lookup on unsupported ops must raise UnsupportedOperatorError."""
        for op_type in self.UNSUPPORTED_EXPECTED:
            with pytest.raises(UnsupportedOperatorError, match=op_type):
                DEFAULT_REGISTRY.lookup(op_type)

    def test_unsupported_ops_not_modeled(self):
        """Unsupported ops should NOT be counted as modeled or free/fused."""
        for op_type in self.UNSUPPORTED_EXPECTED:
            assert not DEFAULT_REGISTRY.is_modeled(op_type)
            assert not DEFAULT_REGISTRY.is_free_or_fused(op_type)


class TestUnregisteredOpsFail:
    """Test that unregistered (unknown) operators fail-closed."""

    def test_unknown_op_is_unsupported(self):
        """An unregistered op should be treated as unsupported."""
        assert DEFAULT_REGISTRY.is_unsupported("my_custom_op_42")
        assert not DEFAULT_REGISTRY.is_modeled("my_custom_op_42")

    def test_unknown_op_raises_on_lookup(self):
        """Lookup on an unregistered op must raise UnsupportedOperatorError."""
        with pytest.raises(UnsupportedOperatorError, match="my_custom_op_42"):
            DEFAULT_REGISTRY.lookup("my_custom_op_42")

    def test_unknown_op_name_in_error(self):
        """The error should contain the op type name."""
        with pytest.raises(UnsupportedOperatorError) as exc_info:
            DEFAULT_REGISTRY.lookup("fancy_new_op")
        assert "fancy_new_op" in str(exc_info.value)

    def test_unknown_op_lookup_or_none_returns_none(self):
        """lookup_or_none should return None for unregistered ops."""
        assert DEFAULT_REGISTRY.lookup_or_none("unknown_xyz") is None

    def test_no_profile_required_default(self):
        """There is no 'profile_required' or 'unknown' default disposition."""
        # The check result for unknown is UNSUPPORTED, never MODELED or FREE_FUSED
        disposition = DEFAULT_REGISTRY.check("undefined_op_abc")
        assert disposition == OperatorDisposition.UNSUPPORTED


class TestRegistryApi:
    """Test the full OperatorRegistry API surface."""

    def test_registry_contains(self):
        """'in' operator should work."""
        assert "gemm" in DEFAULT_REGISTRY
        assert "reshape" in DEFAULT_REGISTRY
        assert "non_existent" not in DEFAULT_REGISTRY

    def test_registry_len(self):
        """len() should return total registered entries."""
        assert len(DEFAULT_REGISTRY) > 20

    def test_register_new_op(self):
        """register() should add a new modeled op."""
        reg = OperatorRegistry()
        assert not reg.is_modeled("new_op_type")
        reg.register(OperatorEntry(
            op_type="new_op_type",
            disposition=OperatorDisposition.MODELED,
            description="A test operator",
        ))
        assert reg.is_modeled("new_op_type")
        entry = reg.lookup("new_op_type")
        assert entry.description == "A test operator"

    def test_register_overrides_existing(self):
        """register() should override an existing entry."""
        reg = OperatorRegistry()
        assert reg.is_modeled("gemm")
        reg.register(OperatorEntry(
            op_type="gemm",
            disposition=OperatorDisposition.UNSUPPORTED,
        ))
        assert reg.is_unsupported("gemm")

    def test_register_invalid_op_type(self):
        """register() with empty op_type should raise."""
        reg = OperatorRegistry()
        with pytest.raises(ValueError):
            reg.register(OperatorEntry(op_type="", disposition=OperatorDisposition.MODELED))

    def test_custom_registry_independent(self):
        """Custom registries should be independent of the default."""
        reg = OperatorRegistry()
        reg.register(OperatorEntry(op_type="custom_only", disposition=OperatorDisposition.MODELED))
        assert reg.is_modeled("custom_only")
        # Default registry should not have it
        assert not DEFAULT_REGISTRY.is_modeled("custom_only")


class TestZeroCyclePrevention:
    """Verify that only explicitly free/fused ops can have 0-cycle semantics."""

    def test_arbitrary_op_not_free(self):
        """An arbitrary op not in the registry is not considered free or fused."""
        assert not DEFAULT_REGISTRY.is_free_or_fused("some_random_op")

    def test_modeled_op_not_0_cycle(self):
        """Modeled ops like gemm are NOT free/fused (they have real cost)."""
        assert DEFAULT_REGISTRY.is_modeled("gemm")
        assert not DEFAULT_REGISTRY.is_free_or_fused("gemm")

    def test_only_explicitly_free_fused_are_free(self):
        """Only ops with EXPLICITLY_FREE_OR_FUSED disposition are considered free."""
        modeled = DEFAULT_REGISTRY.modeled_ops
        free_fused = DEFAULT_REGISTRY.free_fused_ops
        # No overlap
        assert modeled.isdisjoint(free_fused), "Modeled and free/fused ops must be disjoint"
