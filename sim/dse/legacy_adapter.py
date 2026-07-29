"""Legacy import-surface adapter for the DSE module.

This module preserves the pre-Todo-16 import surface:

* ``generate_configs``
* ``evaluate_config``
* ``find_pareto``

It delegates to the existing implementations in
``sim/design_space_explorer.py`` so that legacy callers (model-zoo scripts,
reports, and external notebooks) continue to work without modification.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure sim/ is importable when this module is loaded directly.
SIM_DIR = Path(__file__).resolve().parent.parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from design_space_explorer import evaluate_config, find_pareto, generate_configs

__all__ = ["generate_configs", "evaluate_config", "find_pareto"]
