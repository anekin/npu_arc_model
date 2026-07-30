"""Tests for dimension orthogonality — no dimension implicitly substitutes another.

Covers:
- request_batch change does NOT auto-change active_sequences or token_block
- Each dimension field is independent
- DimensionBindings to_dict() correctness
- Edge batch values from plan acceptance criteria
- Negative: invalid values rejected
"""

from __future__ import annotations

import pytest
from workloads.dimensions import (
    ACTION_HORIZON_EDGES,
    AXIS_BATCH,
    AXIS_SEQUENCES,
    AXIS_TOKEN_BLOCK,
    FLOW_STEPS_EDGES,
    IMAGE_COUNT_EDGES,
    INFLIGHT_JOBS_EDGES,
    RESIDENT_MODELS_EDGES,
    STANDARD_BATCH_EDGES,
    STRESS_BATCH_EDGES,
    TOKEN_BLOCK_EDGES,
    TOKEN_BLOCK_VLM_VLA_EXT,
    DimensionBindings,
)


class TestDimensionOrthogonality:
    """Verify that each dimension is independent and does not auto-change others."""

    def test_request_batch_independent_of_active_sequences(self):
        """Setting request_batch must not set active_sequences (or vice versa)."""
        b1 = DimensionBindings(request_batch=8)
        assert b1.request_batch == 8
        assert b1.active_sequences is None
        assert b1.token_block is None

        b2 = DimensionBindings(active_sequences=4)
        assert b2.request_batch is None
        assert b2.active_sequences == 4
        assert b2.token_block is None

    def test_request_batch_independent_of_token_block(self):
        """Setting request_batch must not set token_block."""
        b1 = DimensionBindings(request_batch=8, token_block=32)
        # Both explicitly set, but they are independent concepts
        assert b1.request_batch == 8
        assert b1.token_block == 32
        # Changing one should not change the other when constructing a new binding
        b2 = DimensionBindings(request_batch=16, token_block=32)
        assert b2.request_batch == 16
        assert b2.token_block == 32

    def test_all_dimensions_independent_when_only_one_set(self):
        """Setting just one dimension leaves all others as None."""
        test_cases = [
            ("request_batch", 8),
            ("active_sequences", 4),
            ("token_block", 64),
            ("image_count", 3),
            ("action_horizon", 50),
            ("flow_steps", 10),
            ("resident_models", 8),
            ("inflight_jobs", 16),
        ]
        for field_name, value in test_cases:
            b = DimensionBindings(**{field_name: value})
            # The set field should have the value
            assert getattr(b, field_name) == value
            # All other fields should be None
            for other_name, _ in test_cases:
                if other_name != field_name:
                    assert getattr(b, other_name) is None, f"Setting {field_name}={value} unexpectedly set {other_name}"

    def test_multiple_dimensions_set_independently(self):
        """Setting multiple dimensions at once should preserve all values independently."""
        b = DimensionBindings(
            request_batch=8,
            token_block=64,
            image_count=2,
            action_horizon=50,
        )
        assert b.request_batch == 8
        assert b.token_block == 64
        assert b.image_count == 2
        assert b.action_horizon == 50
        assert b.active_sequences is None
        assert b.flow_steps is None
        assert b.resident_models is None
        assert b.inflight_jobs is None

    def test_to_dict_only_includes_set_fields(self):
        """to_dict() should only include non-None fields."""
        b = DimensionBindings(request_batch=8, token_block=64)
        d = b.to_dict()
        assert AXIS_BATCH in d
        assert AXIS_TOKEN_BLOCK in d
        assert AXIS_SEQUENCES not in d
        assert d[AXIS_BATCH] == 8
        assert d[AXIS_TOKEN_BLOCK] == 64

    def test_to_dict_frozen_after_init(self):
        """to_dict() should be deterministic and not mutate the bindings."""
        b = DimensionBindings(request_batch=4, image_count=3)
        d1 = b.to_dict()
        d2 = b.to_dict()
        assert d1 == d2
        # The returned dict is independent
        d1["fake_key"] = 999
        assert "fake_key" not in b.to_dict()

    def test_extra_bindings_work(self):
        """Extra dimension bindings should be stored and accessible."""
        b = DimensionBindings(
            request_batch=8,
            extra={"seq_len": 1024, "vocab_size": 32000},
        )
        d = b.to_dict()
        assert d["seq_len"] == 1024
        assert d["vocab_size"] == 32000
        assert d[AXIS_BATCH] == 8

    def test_extra_shadows_canonical_rejected(self):
        """Extra bindings must not shadow canonical field names."""
        with pytest.raises(ValueError, match="shadows canonical field"):
            DimensionBindings(extra={"request_batch": 99})

    def test_extra_shadows_active_sequences_rejected(self):
        """Extra binding for 'active_sequences' must be rejected."""
        with pytest.raises(ValueError, match="shadows canonical field"):
            DimensionBindings(extra={"active_sequences": 8})


class TestEdgeBatchValues:
    """Verify that the acceptance criteria edge values are correctly defined."""

    def test_standard_batch_edges_present(self):
        """Standard batch edges should include {1, 2, 4, 8}."""
        assert STANDARD_BATCH_EDGES[AXIS_BATCH] == {1, 2, 4, 8}
        assert STANDARD_BATCH_EDGES[AXIS_SEQUENCES] == {1, 2, 4, 8}

    def test_stress_batch_edges(self):
        """Stress batch edges should include {16}."""
        assert STRESS_BATCH_EDGES[AXIS_BATCH] == {16}
        assert STRESS_BATCH_EDGES[AXIS_SEQUENCES] == {16}

    def test_token_block_edges(self):
        """Token block edges should include standard + VLM/VLA ext values."""
        assert 16 in TOKEN_BLOCK_EDGES
        assert 256 in TOKEN_BLOCK_EDGES
        assert 512 in TOKEN_BLOCK_VLM_VLA_EXT
        assert 1024 in TOKEN_BLOCK_VLM_VLA_EXT

    def test_image_count_edges(self):
        """Image count edges should be {1, 2, 3, 4}."""
        assert {1, 2, 3, 4} == IMAGE_COUNT_EDGES

    def test_action_horizon_edges(self):
        """Action horizon edges should be {8, 10, 25, 50}."""
        assert {8, 10, 25, 50} == ACTION_HORIZON_EDGES

    def test_flow_steps_edges(self):
        """Flow steps edges should be {4, 8, 10}."""
        assert {4, 8, 10} == FLOW_STEPS_EDGES

    def test_resident_models_edges(self):
        """Resident model edges should be {4, 8}."""
        assert {4, 8} == RESIDENT_MODELS_EDGES

    def test_inflight_jobs_edges(self):
        """Inflight jobs edges should be {4, 8, 16}."""
        assert {4, 8, 16} == INFLIGHT_JOBS_EDGES


class TestDimensionBindingsNegative:
    """Test that invalid dimension values are rejected."""

    def test_negative_request_batch_rejected(self):
        """Negative request_batch should be rejected."""
        with pytest.raises(ValueError, match="positive"):
            DimensionBindings(request_batch=-1)

    def test_zero_token_block_rejected(self):
        """Zero token_block should be rejected."""
        with pytest.raises(ValueError, match="positive"):
            DimensionBindings(token_block=0)

    def test_zero_image_count_rejected(self):
        """Zero image_count should be rejected."""
        with pytest.raises(ValueError, match="positive"):
            DimensionBindings(image_count=0)

    def test_frozen_after_creation(self):
        """DimensionBindings is immutable (frozen dataclass)."""
        b = DimensionBindings(request_batch=8)
        with pytest.raises(AttributeError):
            b.request_batch = 16  # type: ignore[misc]


class TestAxisConsistency:
    """Verify axis constants are consistent."""

    def test_canonical_axes_in_axis_map(self):
        """All canonical axes should appear in the to_dict mapping."""
        b = DimensionBindings(
            request_batch=1,
            active_sequences=2,
            token_block=32,
            image_count=3,
            action_horizon=50,
            flow_steps=10,
            resident_models=4,
            inflight_jobs=8,
        )
        d = b.to_dict()
        assert d[AXIS_BATCH] == 1
        assert d[AXIS_SEQUENCES] == 2
        assert d[AXIS_TOKEN_BLOCK] == 32
        assert "image_count" in d
        assert "action_horizon" in d
        assert "flow_steps" in d
        assert "resident_models" in d
        assert "inflight_jobs" in d


class TestDimensionUnbound:
    """Test unbound method for detecting missing bindings."""

    def test_all_bound_returns_empty(self):
        """When all required axes are bound, unbound returns empty set."""
        b = DimensionBindings(request_batch=8, token_block=64)
        unbound = b.unbound({AXIS_BATCH, AXIS_TOKEN_BLOCK})
        assert unbound == set()

    def test_partially_bound_returns_missing(self):
        """When some required axes are missing, unbound returns them."""
        b = DimensionBindings(request_batch=8)
        unbound = b.unbound({AXIS_BATCH, AXIS_TOKEN_BLOCK})
        assert AXIS_TOKEN_BLOCK in unbound
        assert AXIS_BATCH not in unbound

    def test_extra_bindings_count_as_bound(self):
        """Extra bindings should count as bound for unbound check."""
        b = DimensionBindings(extra={"custom_dim": 42})
        unbound = b.unbound({"custom_dim"})
        assert unbound == set()
