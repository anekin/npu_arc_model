"""Tests for the executable workload fixture catalog.

Covers:
- Discovery of all 10 named fixtures.
- Schema validity: each fixture loads into a validated WorkloadGraphV1.
- Coverage manifest over batch/token/image/horizon/flow/resident/inflight axes.
- Provenance separation: source facts carry reference URIs; no source-less market_source.
- Negative paths: missing dimension binding, source-less market_source, bad trace builder,
  duplicate workload name.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from contracts.errors import ConfigError, DimensionBindingError
from workloads.catalog import (
    CATALOG_DIR,
    build_coverage_manifest,
    discover_fixtures,
    load_all_fixtures,
    load_fixture,
)
from workloads.dimensions import (
    ACTION_HORIZON_EDGES,
    AXIS_ACTION_HORIZON,
    AXIS_BATCH,
    AXIS_FLOW_STEPS,
    AXIS_IMAGE_COUNT,
    AXIS_INFLIGHT_JOBS,
    AXIS_RESIDENT_MODELS,
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
)
from workloads.json_adapter import graph_digest
from workloads.operators import DEFAULT_REGISTRY
from workloads.validate import validate_all

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_PATH = REPO_ROOT / "sim" / "tests" / "golden" / "workload_catalog.json"

EXPECTED_FIXTURES = frozenset({
    "llm-qwen25-3b",
    "cv-yolov8n",
    "cv-vit-b16",
    "smolvla-class",
    "pi0-class",
    "openvla-baseline",
    "openvla-oft",
    "openvla-fast",
    "helix-multirate",
    "physical-ai-multijob",
})


# ── Helpers ──────────────────────────────────────────────────────────────────


def _write_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=True)


# ── Discovery and loading ────────────────────────────────────────────────────


class TestCatalogDiscovery:
    """Fixture discovery and basic loading."""

    def test_discovers_exactly_ten_fixtures(self):
        """Catalog discovers exactly the 10 named workload fixtures."""
        paths = discover_fixtures()
        names = {p.stem for p in paths}
        assert names == EXPECTED_FIXTURES
        assert len(paths) == 10

    def test_load_all_returns_ten_valid_fixtures(self):
        """All discovered fixtures load and validate successfully."""
        fixtures = load_all_fixtures()
        assert set(fixtures.keys()) == EXPECTED_FIXTURES
        for fixture in fixtures.values():
            assert fixture.version == "1"
            assert fixture.provenance.reference_uri is not None
            validate_all(fixture.graph, fixture.bindings, DEFAULT_REGISTRY)

    def test_fixture_names_match_file_stems(self):
        """Each fixture's internal name matches its YAML file stem."""
        for path in discover_fixtures():
            fixture = load_fixture(path)
            assert fixture.name == path.stem


# ── Schema and validation ────────────────────────────────────────────────────


class TestCatalogSchemaValidity:
    """Each fixture produces a schema-valid workload graph."""

    @pytest.mark.parametrize("name", sorted(EXPECTED_FIXTURES))
    def test_fixture_graph_is_dag(self, name):
        """Every fixture graph is a valid DAG."""
        fixtures = load_all_fixtures()
        fixture = fixtures[name]
        order = fixture.graph.topological_order()
        assert len(order) == len(fixture.graph.nodes)

    @pytest.mark.parametrize("name", sorted(EXPECTED_FIXTURES))
    def test_fixture_has_no_unbound_symbols(self, name):
        """All symbolic dimensions used by the graph are bound."""
        fixtures = load_all_fixtures()
        fixture = fixtures[name]
        assert fixture.graph.unbound_symbols(fixture.bindings.to_dict()) == set()

    @pytest.mark.parametrize("name", sorted(EXPECTED_FIXTURES))
    def test_fixture_footprint_digest_stable(self, name):
        """Reloading the same fixture yields the same footprint digest."""
        fixtures = load_all_fixtures()
        fixture_a = fixtures[name]
        fixture_b = load_fixture(CATALOG_DIR / f"{name}.yaml")
        assert fixture_a.footprint_digest == fixture_b.footprint_digest
        assert fixture_a.footprint_digest == graph_digest(fixture_a.graph)


# ── Coverage manifest ────────────────────────────────────────────────────────


class TestCatalogCoverage:
    """Coverage manifest spans the required dimension edges."""

    def test_coverage_manifest_includes_all_fixtures(self):
        """Manifest lists all 10 fixtures."""
        fixtures = load_all_fixtures()
        manifest = build_coverage_manifest(fixtures)
        assert set(manifest["fixture_names"]) == EXPECTED_FIXTURES
        assert manifest["fixture_count"] == 10

    def test_batch_axis_coverage(self):
        """Request-batch standard edges and stress edge are active."""
        fixtures = load_all_fixtures()
        manifest = build_coverage_manifest(fixtures)
        batch_values = set(manifest["axis_coverage"][AXIS_BATCH]["active_values"])
        assert batch_values >= STANDARD_BATCH_EDGES[AXIS_BATCH]
        assert batch_values >= STRESS_BATCH_EDGES[AXIS_BATCH]

    def test_active_sequence_edge_values_present(self):
        """Active-sequence edge values are declared in the manifest."""
        fixtures = load_all_fixtures()
        manifest = build_coverage_manifest(fixtures)
        entry = manifest["axis_coverage"][AXIS_SEQUENCES]
        assert set(entry["edge_values"]) >= STANDARD_BATCH_EDGES[AXIS_SEQUENCES]
        assert entry["active_values"]

    def test_token_block_edge_values_present(self):
        """Token-block edge values are declared in the manifest."""
        fixtures = load_all_fixtures()
        manifest = build_coverage_manifest(fixtures)
        entry = manifest["axis_coverage"][AXIS_TOKEN_BLOCK]
        assert set(entry["edge_values"]) >= TOKEN_BLOCK_EDGES
        assert set(entry["edge_values"]) & TOKEN_BLOCK_VLM_VLA_EXT
        assert entry["active_values"]

    def test_image_count_edges_covered(self):
        """Image count edges {1,2,3,4} are active."""
        fixtures = load_all_fixtures()
        manifest = build_coverage_manifest(fixtures)
        values = set(manifest["axis_coverage"][AXIS_IMAGE_COUNT]["active_values"])
        assert values == IMAGE_COUNT_EDGES

    def test_action_horizon_edges_covered(self):
        """Action horizon edges {8,10,25,50} are active."""
        fixtures = load_all_fixtures()
        manifest = build_coverage_manifest(fixtures)
        values = set(manifest["axis_coverage"][AXIS_ACTION_HORIZON]["active_values"])
        assert values == ACTION_HORIZON_EDGES

    def test_flow_steps_edges_covered(self):
        """Flow steps edges {4,8,10} are active."""
        fixtures = load_all_fixtures()
        manifest = build_coverage_manifest(fixtures)
        values = set(manifest["axis_coverage"][AXIS_FLOW_STEPS]["active_values"])
        assert values == FLOW_STEPS_EDGES

    def test_resident_models_edges_covered(self):
        """Resident model edges {4,8} are active."""
        fixtures = load_all_fixtures()
        manifest = build_coverage_manifest(fixtures)
        values = set(manifest["axis_coverage"][AXIS_RESIDENT_MODELS]["active_values"])
        assert values == RESIDENT_MODELS_EDGES

    def test_inflight_jobs_edges_covered(self):
        """Inflight jobs edges {4,8,16} are active."""
        fixtures = load_all_fixtures()
        manifest = build_coverage_manifest(fixtures)
        values = set(manifest["axis_coverage"][AXIS_INFLIGHT_JOBS]["active_values"])
        assert values == INFLIGHT_JOBS_EDGES

    def test_manifest_reports_edge_values(self):
        """Manifest exposes required edge values for every DSE axis."""
        fixtures = load_all_fixtures()
        manifest = build_coverage_manifest(fixtures)
        for axis in {
            AXIS_BATCH,
            AXIS_SEQUENCES,
            AXIS_TOKEN_BLOCK,
            AXIS_IMAGE_COUNT,
            AXIS_ACTION_HORIZON,
            AXIS_FLOW_STEPS,
            AXIS_RESIDENT_MODELS,
            AXIS_INFLIGHT_JOBS,
        }:
            entry = manifest["axis_coverage"][axis]
            assert entry["edge_values"], f"{axis} missing edge_values"
            assert entry["active_values"], f"{axis} missing active_values"


# ── Provenance ───────────────────────────────────────────────────────────────


class TestCatalogProvenance:
    """Source facts and engineering assumptions are separated and auditable."""

    def test_all_fixtures_have_reference_uri(self):
        """Every fixture provenance carries a reference URI."""
        fixtures = load_all_fixtures()
        for fixture in fixtures.values():
            assert fixture.provenance.reference_uri
            assert fixture.provenance.reference_uri.startswith(("http", "https"))

    def test_no_source_less_market_source_facts(self):
        """No source_fact with category market_source lacks a reference_uri."""
        fixtures = load_all_fixtures()
        for fixture in fixtures.values():
            for fact in fixture.source_facts:
                if fact.get("category") == "market_source":
                    assert fact.get("reference_uri"), (
                        f"{fixture.name} has source-less market_source fact {fact.get('param')}"
                    )

    @pytest.mark.parametrize("name", sorted(EXPECTED_FIXTURES))
    def test_fixture_has_engineering_assumptions(self, name):
        """Every fixture declares at least one engineering assumption."""
        fixtures = load_all_fixtures()
        fixture = fixtures[name]
        assert len(fixture.engineering_assumptions) >= 1


# ── Golden manifest ──────────────────────────────────────────────────────────


class TestCatalogGolden:
    """Deterministic golden manifest matches current catalog."""

    def test_golden_file_exists(self):
        """The golden manifest file exists."""
        assert GOLDEN_PATH.exists()

    def test_golden_matches_current_catalog(self):
        """Committed golden matches the catalog loader output."""
        fixtures = load_all_fixtures()
        current = {
            name: {
                "node_count": len(fixture.graph.nodes),
                "tensor_count": len(fixture.graph.tensors),
                "symbol_count": len(fixture.graph.symbols),
                "footprint_digest": fixture.footprint_digest,
                "provenance_summary": fixture.provenance_summary(),
            }
            for name, fixture in sorted(fixtures.items())
        }
        with GOLDEN_PATH.open("r", encoding="utf-8") as f:
            golden = json.load(f)
        assert current == golden


# ── Negative paths ───────────────────────────────────────────────────────────


class TestCatalogNegative:
    """Invalid fixtures are rejected with typed errors."""

    def test_missing_dimension_binding_rejected(self):
        """A fixture with an unbound symbolic dimension fails validation."""
        valid = load_fixture(CATALOG_DIR / "cv-vit-b16.yaml")
        data = {
            "name": "missing-dim",
            "version": "1",
            "provenance": {"source": "test", "reference_uri": "https://example.com"},
            "trace_builder": "workloads.catalog:build_vit_b16_graph",
            "dimensions": {},
            "scenario": {"type": "test"},
            "source_facts": [],
            "engineering_assumptions": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing-dim.yaml"
            _write_yaml(path, data)
            with pytest.raises(DimensionBindingError, match="unbound"):
                load_fixture(path)

    def test_source_less_market_source_rejected(self):
        """A market_source fact without reference_uri is rejected."""
        data = {
            "name": "bad-source",
            "version": "1",
            "provenance": {"source": "test", "reference_uri": "https://example.com"},
            "trace_builder": "workloads.catalog:build_vit_b16_graph",
            "dimensions": {"image_count": 1},
            "scenario": {"type": "test"},
            "source_facts": [
                {"param": "x", "value": 1, "category": "market_source"},
            ],
            "engineering_assumptions": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad-source.yaml"
            _write_yaml(path, data)
            with pytest.raises(ConfigError, match="source-less"):
                load_fixture(path)

    def test_invalid_trace_builder_rejected(self):
        """A non-existent trace builder function is rejected."""
        data = {
            "name": "bad-builder",
            "version": "1",
            "provenance": {"source": "test", "reference_uri": "https://example.com"},
            "trace_builder": "workloads.catalog:does_not_exist",
            "dimensions": {"image_count": 1},
            "scenario": {"type": "test"},
            "source_facts": [],
            "engineering_assumptions": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad-builder.yaml"
            _write_yaml(path, data)
            with pytest.raises(ConfigError, match="has no function"):
                load_fixture(path)

    def test_duplicate_workload_name_rejected(self):
        """Loading two fixtures with the same name is rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_yaml(
                tmp_path / "first.yaml",
                {
                    "name": "dup-name",
                    "version": "1",
                    "provenance": {"source": "test", "reference_uri": "https://example.com"},
                    "trace_builder": "workloads.catalog:build_vit_b16_graph",
                    "dimensions": {"image_count": 1},
                    "scenario": {"type": "test"},
                    "source_facts": [],
                    "engineering_assumptions": [],
                },
            )
            _write_yaml(
                tmp_path / "second.yaml",
                {
                    "name": "dup-name",
                    "version": "1",
                    "provenance": {"source": "test", "reference_uri": "https://example.com"},
                    "trace_builder": "workloads.catalog:build_vit_b16_graph",
                    "dimensions": {"image_count": 1},
                    "scenario": {"type": "test"},
                    "source_facts": [],
                    "engineering_assumptions": [],
                },
            )
            with pytest.raises(ConfigError, match="duplicate"):
                load_all_fixtures(config_dir=tmp_path)

    def test_unsupported_version_rejected(self):
        """A fixture with version other than '1' is rejected."""
        data = {
            "name": "bad-version",
            "version": "2",
            "provenance": {"source": "test", "reference_uri": "https://example.com"},
            "trace_builder": "workloads.catalog:build_vit_b16_graph",
            "dimensions": {"image_count": 1},
            "scenario": {"type": "test"},
            "source_facts": [],
            "engineering_assumptions": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad-version.yaml"
            _write_yaml(path, data)
            with pytest.raises(ConfigError, match="version"):
                load_fixture(path)
