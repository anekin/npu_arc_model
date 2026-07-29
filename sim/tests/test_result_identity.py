"""Test stable design-point identities via contracts.identity.

Given: a normalized hardware configuration dict
When:  the digest_sha256 function is called
Then:  the same input always produces the same hex digest,
      and any change to any design axis produces a different digest.
"""

import json
from copy import deepcopy

import pytest

from contracts.identity import canonical_json_bytes, digest_sha256, normalise_for_hashing


# ── Base config fixture ───────────────────────────────────────────────────────


@pytest.fixture
def base_config():
    return {
        "version": "2",
        "mac_engine": {
            "type": "block",
            "array_height": 64,
            "array_width": 128,
            "frequency_mhz": 1000.0,
            "weight_precision_bits": 4,
            "activation_precision_bits": 8,
        },
        "memory": {
            "type": "LPDDR5-6400",
            "bandwidth_gbps": 51.2,
            "dram_efficiency": 0.85,
        },
        "sram": {
            "l1_per_core_kb": 512,
            "l2_shared_kb": 2048,
        },
    }


# ── Deterministic serialization ───────────────────────────────────────────────


class TestDeterministicSerialization:
    """Given: a config dict. When: serialised to canonical JSON bytes. Then: output is deterministic."""

    def test_same_input_same_digest(self, base_config):
        d1 = digest_sha256(base_config)
        d2 = digest_sha256(base_config)
        assert d1 == d2
        assert len(d1) == 64
        assert all(c in "0123456789abcdef" for c in d1)

    def test_dict_key_order_independent(self, base_config):
        cfg_a = dict(reversed(list(base_config.items())))
        assert digest_sha256(cfg_a) == digest_sha256(base_config)

    def test_nested_key_order_independent(self, base_config):
        cfg_a = deepcopy(base_config)
        cfg_a["mac_engine"] = dict(reversed(list(base_config["mac_engine"].items())))
        assert digest_sha256(cfg_a) == digest_sha256(base_config)

    def test_canonical_json_bytes_identical(self, base_config):
        b1 = canonical_json_bytes(base_config)
        b2 = canonical_json_bytes(base_config)
        assert b1 == b2

    def test_canonical_json_sorted_keys(self, base_config):
        result = json.loads(canonical_json_bytes(base_config).decode("utf-8"))
        keys = list(result.keys())
        assert keys == sorted(keys)

    def test_float_stable_repr(self):
        cfg = {"freq": 1000.0, "bw": 51.2}
        raw = canonical_json_bytes(cfg).decode("utf-8")
        assert "1000.0" in raw

    def test_int_not_decimal(self):
        cfg = {"h": 64, "w": 128}
        raw = canonical_json_bytes(cfg).decode("utf-8")
        assert '"64"' not in raw  # Should be 64 not "64.0"
        assert "64" in raw

    def test_bool_preserved(self):
        cfg = {"flag": True, "count": 0}
        raw = canonical_json_bytes(cfg).decode("utf-8")
        assert "true" in raw
        assert "false" not in raw

    def test_none_serialized(self):
        cfg = {"opt": None}
        raw = canonical_json_bytes(cfg).decode("utf-8")
        assert "null" in raw

    def test_list_order_preserved(self):
        cfg = {"items": [1, 2, 3]}
        raw = canonical_json_bytes(cfg).decode("utf-8")
        assert "[1,2,3]" in raw

    def test_normalise_for_hashing_returns_dict(self, base_config):
        result = normalise_for_hashing(base_config)
        assert isinstance(result, dict)
        assert "version" in result


# ── Cross-axis change detection ───────────────────────────────────────────────


class TestAxisChangeDetection:
    """Given: a base config. When: any design axis changes. Then: the ID changes."""

    def test_engine_type_change(self, base_config):
        cfg = deepcopy(base_config)
        cfg["mac_engine"]["type"] = "systolic"
        assert digest_sha256(cfg) != digest_sha256(base_config)

    def test_array_height_change(self, base_config):
        cfg = deepcopy(base_config)
        cfg["mac_engine"]["array_height"] = 128
        assert digest_sha256(cfg) != digest_sha256(base_config)

    def test_array_width_change(self, base_config):
        cfg = deepcopy(base_config)
        cfg["mac_engine"]["array_width"] = 256
        assert digest_sha256(cfg) != digest_sha256(base_config)

    def test_frequency_change(self, base_config):
        cfg = deepcopy(base_config)
        cfg["mac_engine"]["frequency_mhz"] = 800.0
        assert digest_sha256(cfg) != digest_sha256(base_config)

    def test_bandwidth_change(self, base_config):
        cfg = deepcopy(base_config)
        cfg["memory"]["bandwidth_gbps"] = 102.4
        assert digest_sha256(cfg) != digest_sha256(base_config)

    def test_dram_efficiency_change(self, base_config):
        cfg = deepcopy(base_config)
        cfg["memory"]["dram_efficiency"] = 0.95
        assert digest_sha256(cfg) != digest_sha256(base_config)

    def test_sram_change(self, base_config):
        cfg = deepcopy(base_config)
        cfg["sram"]["l1_per_core_kb"] = 1024
        assert digest_sha256(cfg) != digest_sha256(base_config)

    def test_added_field_changes_id(self, base_config):
        cfg = deepcopy(base_config)
        cfg["cores"] = 4
        assert digest_sha256(cfg) != digest_sha256(base_config)

    def test_precision_change(self, base_config):
        cfg = deepcopy(base_config)
        cfg["mac_engine"]["weight_precision_bits"] = 8
        assert digest_sha256(cfg) != digest_sha256(base_config)

    def test_no_timestamps_or_paths_in_id(self, base_config):
        digest = digest_sha256(base_config)
        assert "/" not in digest
        assert "T" not in digest

    def test_different_configs_different_digests(self, base_config):
        ids = set()
        for h in [32, 64, 128]:
            for w in [64, 128, 256]:
                cfg = deepcopy(base_config)
                cfg["mac_engine"]["array_height"] = h
                cfg["mac_engine"]["array_width"] = w
                ids.add(digest_sha256(cfg))
        assert len(ids) == 9  # All 9 combinations produce unique IDs


class TestIdentityEdgeCases:
    """Given: edge-case inputs. When: identity functions are called. Then: handle correctly."""

    def test_empty_dict(self):
        assert len(digest_sha256({})) == 64

    def test_nested_empty_dict(self):
        cfg = {"a": {"b": {}}}
        assert len(digest_sha256(cfg)) == 64

    def test_very_large_values(self):
        cfg = {"big": 2**53}
        assert len(digest_sha256(cfg)) == 64

    def test_special_float_values_repr(self):
        cfg = {"pi": 3.141592653589793}
        assert digest_sha256(cfg) == digest_sha256({"pi": 3.141592653589793})
