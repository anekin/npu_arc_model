"""NoC analytical model — crossbar and mesh topologies with XY routing.

Pure analytical model: no dependencies on npu_sim, timeline, or
golden_executor. Follows the DMAModel pattern: config-driven constructor,
estimate_transfer / estimate_contention / estimate_total methods.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class NoCTransfer:
    """A single NoC transfer request.

    Attributes:
        src_id: Source node ID (0-indexed).
        dst_id: Destination node ID (0-indexed).
        size_bytes: Payload size in bytes.
        priority: Arbitration priority (default 0, higher = more urgent).
    """

    src_id: int
    dst_id: int
    size_bytes: int
    priority: int = 0


class NoCModel:
    """Configurable NoC analytical model: crossbar or mesh with XY routing.

    Reads an ``interconnect`` section from the configuration dict.
    Models hop latency, serialisation time, arbitration overhead,
    virtual-channel buffer depth, and port contention.

    Config keys (all under ``config["interconnect"]``):

    ====================  =======  =================================
    Key                   Default  Description
    ====================  =======  =================================
    type                  crossbar  ``crossbar`` or ``mesh``
    ports                 4         Number of endpoint ports / nodes
    bandwidth_gbps        500       Per-port bandwidth in Gbps
    hop_latency_cycles    3         Fixed latency per router hop
    flit_width_bits       256       Flit (flow-control unit) width
    vcs                   2         Virtual channels per port
    buffer_depth          4         Buffer depth in flits per VC
    arbitration           round_robin  ``round_robin`` or ``fixed_priority``
    routing               destination_tag  Routing algorithm (crossbar uses
                                    destination_tag; mesh forces XY)
    ====================  =======  =================================
    """

    def __init__(self, config: Dict[str, Any]):
        ic = config["interconnect"]
        self.topology = str(ic.get("type", "crossbar"))
        self.ports = int(ic.get("ports", 4))
        self.bandwidth_gbps = float(ic.get("bandwidth_gbps", 500.0))
        self.hop_latency_cycles = int(ic.get("hop_latency_cycles", 3))
        self.flit_width_bits = int(ic.get("flit_width_bits", 256))
        self.vcs = int(ic.get("vcs", 2))
        self.buffer_depth = int(ic.get("buffer_depth", 4))
        self.arbitration = str(ic.get("arbitration", "round_robin"))
        self.routing = str(ic.get("routing", "destination_tag"))

        # Mesh dimensions: closest-to-square 2-D grid, row-major IDs
        self._rows, self._cols = self._compute_dims(self.ports)

    # ── mesh geometry helpers ────────────────────────────────────

    @staticmethod
    def _compute_dims(ports: int):
        """Return (rows, cols) for a closest-to-square 2-D mesh.

        Prefers exact divisor pairs so every node maps to a unique
        grid position.  Falls back to ceil(sqrt(...)) with possible
        overshoot when *ports* is prime or unfactorable.
        """
        limit = int(math.ceil(math.sqrt(ports)))
        for c in range(limit, 0, -1):
            if ports % c == 0:
                r = ports // c
                # cols is the larger dimension (convention)
                return (min(r, c), max(r, c))

        # No exact factor — accept slight overshoot
        cols = limit
        rows = int(math.ceil(ports / cols))
        return (rows, cols)

    def _node_coords(self, node_id: int):
        """Convert row-major node ID to (row, col)."""
        row = node_id // self._cols
        col = node_id % self._cols
        return row, col

    def _mesh_hops(self, src_id: int, dst_id: int) -> int:
        """Manhattan distance (XY-routing hop count)."""
        src_row, src_col = self._node_coords(src_id)
        dst_row, dst_col = self._node_coords(dst_id)
        return abs(src_row - dst_row) + abs(src_col - dst_col)

    def _arbitration_cycles(self) -> int:
        """Return arbitration overhead for a single transfer.

        Aligns with ``CrossbarConfig.arbitration`` accepted values:
        - round_robin: 3 fixed cycles (default).
        - priority / fixed_priority: 1 cycle (constant grant to highest-priority port).
        - age_based: 3 cycles (oldest-wins has comparable microarchitecture cost).
        """
        arb = self.arbitration.lower()
        if arb in ("fixed_priority", "priority"):
            return 1
        # round_robin (default) or age_based
        return 3

    # ── public API ─────────────────────────────────────────────

    def estimate_transfer(self, size_bytes: int, src_id: int,
                          dst_id: int) -> int:
        """Estimate cycles for a single NoC transfer.

        Components:
        * Serialisation: ``ceil(size_bytes / bytes_per_flit)`` flits.
        * Hop latency: 1 hop (crossbar) or XY Manhattan distance (mesh).
        * Arbitration overhead (crossbar only).
        * Buffer-depth contention penalty.

        Returns an integer number of cycles.
        """
        if size_bytes <= 0:
            return 0

        bytes_per_flit = max(1, self.flit_width_bits // 8)
        num_flits = int(math.ceil(size_bytes / bytes_per_flit))
        serial_cycles = num_flits

        if self.topology == "crossbar":
            hop_cycles = self.hop_latency_cycles
            arb_cycles = self._arbitration_cycles()
            # Buffer depth adds a fixed pipeline stall per traversal
            buf_penalty = self.buffer_depth
            return hop_cycles + serial_cycles + arb_cycles + buf_penalty

        # mesh topology
        hop_count = self._mesh_hops(src_id, dst_id)
        hop_cycles = hop_count * self.hop_latency_cycles
        # Buffer-depth penalty scales with hop count (each hop may stall)
        buf_penalty = self.buffer_depth * hop_count
        return int(hop_cycles + serial_cycles + buf_penalty)

    def estimate_contention(self, num_active_ports: int,
                            total_ports: int) -> float:
        """Return a contention factor in [0.0, 1.0].

        Formula: ``max(0, active - 1) / total_ports``.
        A single active port experiences no contention.
        """
        if total_ports <= 0:
            return 0.0
        return max(0.0, float(num_active_ports - 1)) / float(total_ports)

    def estimate_total(self, transfers: List[NoCTransfer]) -> int:
        """Aggregate cycles for a batch of transfers.

        Uses ``max(individual cycles)`` as the ideal-parallel base,
        then applies a contention penalty via :meth:`estimate_contention`.
        """
        if not transfers:
            return 0
        individual = [
            self.estimate_transfer(t.size_bytes, t.src_id, t.dst_id)
            for t in transfers
        ]
        base = max(individual)
        contention = self.estimate_contention(len(transfers), self.ports)
        return int(base * (1.0 + contention))
