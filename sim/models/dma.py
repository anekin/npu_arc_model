"""DMA bandwidth model — multi-channel descriptor-based DMA engine"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# Priority lookup for fixed_priority arbitration: higher value = higher priority.
_REQUEST_TYPE_PRIORITY: Dict[str, int] = {
    "weight_load": 2,
    "kv_access": 1,
    "output_store": 0,
}


@dataclass
class DMARequest:
    """A single DMA transfer request queued on a channel.

    Attributes:
        request_type: Logical category (weight_load, kv_access, output_store).
        size_bytes: Total payload bytes to transfer.
        direction: 'load' (DRAM→SRAM) or 'store' (SRAM→DRAM).
        block_count: Number of contiguous/scatter-gather blocks (default 1).
        priority: Arbitration priority (default 0, higher = more urgent).
    """
    request_type: str
    size_bytes: int
    direction: str
    block_count: int = 1
    priority: int = 0


class DMAModel:
    """DMA engine with configurable channels, arbitration, and burst transfers.

    Models: LPDDR5 ↔ L2 SRAM data movement.
    Key insight: DMA loads next layer's weights while MXU computes current layer.

    Arbitration modes:
        round_robin (default) — FIFO order on each channel.
        fixed_priority — sort requests by type priority:
            weight_load (2) > kv_access (1) > output_store (0).
    """

    def __init__(self, config: Dict[str, Any]):
        dma = config["dma"]
        self.burst_size = int(dma["burst_size_bytes"])          # 256
        self.descriptor_overhead = int(dma["descriptor_overhead_cycles"])  # 5
        self.max_pending = int(dma.get("max_pending_descriptors", 16))

        # DW_axi_dmac-spec configurable parameters (v0.4)
        self.num_channels = int(dma.get("num_channels", 2))
        self.fifo_depth = int(dma.get("per_channel_fifo_depth", 64))
        self.max_burst_length = int(dma.get("max_burst_length", 8))
        self.multi_block_mode = str(dma.get("multi_block_mode", "linked_list"))
        self.ll_prefetch_en = bool(dma.get("ll_prefetch_en", True))

        # Arbitration policy (v0.5)
        self.arbitration = str(dma.get("arbitration", "round_robin"))

        # Per-channel request queues for FIFO backpressure modelling
        self.channel_queues: List[List[DMARequest]] = [
            [] for _ in range(self.num_channels)
        ]

        mem = config["memory"]
        self.bw_bytes_per_cycle = float(mem["bandwidth_bytes_per_cycle"])  # 51.2

    def estimate_transfer(self, size_bytes: int, direction: str = "load") -> int:
        """Estimate cycles for a single DMA transfer.

        direction: 'load' (DRAM→SRAM) or 'store' (SRAM→DRAM)

        Returns total cycles including descriptor overhead.
        """
        if size_bytes <= 0:
            return 0

        # Number of bursts
        num_bursts = math.ceil(size_bytes / self.burst_size)

        # Transfer time: bytes / bandwidth
        transfer_cycles = size_bytes / self.bw_bytes_per_cycle

        # Burst overhead: one cycle per burst for address handshake
        burst_overhead = num_bursts

        total = (self.descriptor_overhead + transfer_cycles + burst_overhead)
        return int(math.ceil(total))

    def estimate_weight_load(self, K: int, N: int, weight_bits: int = 4) -> int:
        """Estimate cycles to load weight matrix (K×N) from DRAM to SRAM.

        Internally enqueues a DMARequest, estimates via channel queues,
        then dequeues so callers see an unchanged queue state.
        """
        size_bytes = math.ceil(K * N * weight_bits / 8)
        request = DMARequest(
            request_type="weight_load",
            size_bytes=size_bytes,
            direction="load",
            block_count=1,
        )
        self.enqueue(request)
        cycles = self.estimate_total_cycles()
        # Dequeue: remove the request we just added (FIFO order)
        ch = self.allocate_channel("weight_load")
        self.channel_queues[ch].pop()
        return cycles

    def allocate_channel(self, request_type: str) -> int:
        """Map a DMA request type to a channel index.

        request_type: 'weight_load', 'kv_access', 'output_store'
        Returns channel index in [0, num_channels).
        """
        mapping = {"weight_load": 0, "kv_access": 1, "output_store": 2}
        base = mapping.get(request_type, 0)
        return base % self.num_channels

    def enqueue(self, request: DMARequest) -> int:
        """Enqueue a DMA request onto the appropriate channel queue.

        In round_robin mode, appends to the channel FIFO.
        In fixed_priority mode, inserts in priority order (highest first).

        Returns the channel index assigned via allocate_channel().
        """
        ch = self.allocate_channel(request.request_type)
        if self.arbitration == "fixed_priority":
            # Insert at correct priority position (highest first)
            queue = self.channel_queues[ch]
            req_prio = _REQUEST_TYPE_PRIORITY.get(request.request_type, 0)
            insert_at = 0
            for i, existing in enumerate(queue):
                existing_prio = _REQUEST_TYPE_PRIORITY.get(existing.request_type, 0)
                if req_prio > existing_prio:
                    insert_at = i
                    break
                insert_at = i + 1
            queue.insert(insert_at, request)
        else:
            self.channel_queues[ch].append(request)
        return ch

    def estimate_channel_cycles(self, channel_idx: int) -> int:
        """Estimate total cycles for all queued requests on one channel.

        Order follows the configured arbitration policy set during enqueue:
        round_robin → FIFO, fixed_priority → sorted by request-type priority.

        Components:
        - Per-transfer estimate_transfer() cycles for each request.
        - Multi-block overhead: block_count * descriptor_overhead.
        - Linked-list pointer fetch: block_count * 2 (halved if ll_prefetch_en).
        - FIFO backpressure: stall cycles when queued bytes exceed
          fifo_capacity = per_channel_fifo_depth * burst_size.
        """
        queue = self.channel_queues[channel_idx]
        if not queue:
            return 0

        total_cycles = 0
        total_queued_bytes = 0

        for req in queue:
            # Base transfer cycles
            transfer = self.estimate_transfer(req.size_bytes, req.direction)
            total_cycles += transfer

            # Multi-block descriptor overhead
            total_cycles += req.block_count * self.descriptor_overhead

            # Linked-list pointer fetch overhead
            if self.multi_block_mode == "linked_list":
                ll_cycles = req.block_count * 2
                if self.ll_prefetch_en:
                    ll_cycles = max(1, ll_cycles // 2)
                total_cycles += ll_cycles

            total_queued_bytes += req.size_bytes

        # FIFO backpressure: stall when queued data exceeds FIFO capacity
        fifo_capacity = self.fifo_depth * self.burst_size
        if total_queued_bytes > fifo_capacity:
            excess_bytes = total_queued_bytes - fifo_capacity
            stall_cycles = int(math.ceil(excess_bytes / self.burst_size))
            total_cycles += stall_cycles

        return total_cycles

    def estimate_total_cycles(self) -> int:
        """Estimate total DMA cycles across all channels.

        Since DMA channels operate in parallel, the total time is bounded
        by the busiest channel (conservative exposed stall).
        """
        if self.num_channels == 0:
            return 0
        return max(
            self.estimate_channel_cycles(ch) for ch in range(self.num_channels)
        )

    def estimate_effective(self, transfer_cycles: int,
                           compute_cycles: int) -> Tuple[int, int]:
        """Calculate effective (non-overlapped) DMA cycles.

        Returns (effective_cycles, hidden_cycles).
        effective = DMA cycles that block (couldn't overlap with compute)
        hidden = DMA cycles hidden behind compute
        """
        hidden = min(transfer_cycles, compute_cycles)
        effective = max(0, transfer_cycles - compute_cycles)
        return effective, hidden
