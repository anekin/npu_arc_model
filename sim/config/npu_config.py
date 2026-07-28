"""NPU configuration loader.

Reads the canonical YAML spec into a plain dict so other modules can deep-copy
and override it without mutating the global configuration.
"""

from pathlib import Path

import yaml


def load_config() -> dict:
    """Load ``sim/config/npu_config.yaml`` and return it as a plain dict."""
    config_path = Path(__file__).with_suffix(".yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
