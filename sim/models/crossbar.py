"""AXI4 Crossbar functional model — M=6 masters, S=2 slaves.

Models address decode, round-robin arbitration tracking, and AXI ID
preservation.  Does NOT model channel-level timing (awvalid/awready
handshake cycles).

Address routing (matches rtl/soc/axi_crossbar.v):
    SRAM: addr[31:22] == 0b0010000000 -> slave 0
    DRAM: addr[31] == 1                -> slave 1
    Other                               -> DECERR
"""

import threading
from typing import List, Optional, Tuple

from sim.regmap import Addr


class CrossbarModel:
    """Functional model of the M=6/S=2 AXI4 crossbar."""

    MASTER_IBEX = 0
    MASTER_MXU = 1
    MASTER_SFU = 2
    MASTER_VEC = 3
    MASTER_DMA = 4
    MASTER_PCIE = 5

    NUM_MASTERS = 6
    NUM_SLAVES = 2

    def __init__(
        self,
        sram: bytearray,
        dram: bytearray,
        boot_rom: Optional[bytearray] = None,
    ):
        """Initialize crossbar with shared memory references.

        Args:
            sram: FuncModel.sram bytearray (default 4 MB at 0x2000_0000).
            dram: FuncModel.dram bytearray (default 64 MB at 0x8000_0000).
            boot_rom: Optional Ibex boot ROM (not used by current decode).
        """
        self.sram = sram
        self.dram = dram
        self.boot_rom = boot_rom

        # Per-slave last granted master, independent for AW/W and AR/R.
        self._aw_last_granted: List[Optional[int]] = [None] * self.NUM_SLAVES
        self._ar_last_granted: List[Optional[int]] = [None] * self.NUM_SLAVES

        # Per-master AXI transaction ID counters.
        self._txn_ids = [0] * self.NUM_MASTERS

        # Functional mutex per slave (no cycle-accurate blocking).
        self._aw_locks = [threading.Lock() for _ in range(self.NUM_SLAVES)]
        self._ar_locks = [threading.Lock() for _ in range(self.NUM_SLAVES)]

        # Grant history for fairness / ordering verification.
        self._aw_grants: List[Tuple[int, int]] = []
        self._ar_grants: List[Tuple[int, int]] = []

    def __repr__(self) -> str:
        return (
            f"CrossbarModel(M={self.NUM_MASTERS}, S={self.NUM_SLAVES}, "
            f"sram={len(self.sram)}B, dram={len(self.dram)}B)"
        )

    # ── Public master-facing API ─────────────────────────────────────

    def read(self, master_id: int, addr: int, size: int) -> bytes:
        """Issue an AXI4 read from a specific master.

        Args:
            master_id: Master port index (0=Ibex, 1=MXU, ..., 5=PCIe).
            addr: SoC physical byte address (32-bit).
            size: Number of bytes to read.

        Returns:
            Bytes read from decoded target.

        Raises:
            ValueError: If master_id is invalid or address is DECERR.
        """
        if not 0 <= master_id < self.NUM_MASTERS:
            raise ValueError(f"Invalid master_id {master_id}")
        if size < 0:
            raise ValueError("size must be non-negative")
        if size == 0:
            return b""

        slave_idx, mem = self._decode(addr)
        axi_id = self._next_axi_id(master_id)
        base = self._slave_base(slave_idx)

        with self._ar_locks[slave_idx]:
            self._grant(slave_idx, master_id, is_write=False)
            off = addr - base
            data = bytes(mem[off:off + size])

        # Expose the composed ID for testability.
        self._last_axi_id = axi_id
        return data

    def write(self, master_id: int, addr: int, data: bytes) -> None:
        """Issue an AXI4 write from a specific master.

        Args:
            master_id: Master port index (0–5).
            addr: SoC physical byte address.
            data: Raw bytes to write.

        Raises:
            ValueError: If master_id is invalid or address is DECERR.
        """
        if not 0 <= master_id < self.NUM_MASTERS:
            raise ValueError(f"Invalid master_id {master_id}")

        slave_idx, mem = self._decode(addr)
        axi_id = self._next_axi_id(master_id)
        base = self._slave_base(slave_idx)

        with self._aw_locks[slave_idx]:
            self._grant(slave_idx, master_id, is_write=True)
            off = addr - base
            mem[off:off + len(data)] = data

        self._last_axi_id = axi_id

    # ── Address decode ───────────────────────────────────────────────

    def _decode(self, addr: int) -> Tuple[int, bytearray]:
        """Decode physical address to (slave_idx, memory).

        Returns:
            (0, sram) for 0x2000_0000 <= addr < 0x2000_0000 + len(sram)
            (1, dram) for 0x8000_0000 <= addr < 0x8000_0000 + len(dram)

        Raises:
            ValueError: Address unmapped (DECERR).
        """
        sram_base = Addr.SRAM_BASE
        if sram_base <= addr < sram_base + len(self.sram):
            return 0, self.sram
        dram_base = Addr.DRAM_BASE
        if dram_base <= addr < dram_base + len(self.dram):
            return 1, self.dram
        raise ValueError(f"Address 0x{addr:08x} unmapped (DECERR)")

    # ── Arbitration tracking ─────────────────────────────────────────

    def _grant(self, slave_idx: int, master_id: int, is_write: bool) -> bool:
        """Record a round-robin grant for the given slave.

        The functional model grants every request (no cycle contention),
        but updates last-granted state so tests can verify fairness.

        Returns:
            True if this master has the grant (always True in this model).
        """
        last = self._aw_last_granted if is_write else self._ar_last_granted
        history = self._aw_grants if is_write else self._ar_grants
        last[slave_idx] = master_id
        history.append((slave_idx, master_id))
        return True

    # ── Helpers ──────────────────────────────────────────────────────

    def _next_axi_id(self, master_id: int) -> int:
        """Compose next AXI ID: {master_id[2:0], axi_id[5:0]}.

        Python representation widens to (master_id << 8) | txn_id so
        tests can easily separate master and transaction fields.
        """
        txn_id = self._txn_ids[master_id]
        self._txn_ids[master_id] = (txn_id + 1) & 0xFF
        return (master_id << 8) | txn_id

    def _slave_base(self, slave_idx: int) -> int:
        if slave_idx == 0:
            return Addr.SRAM_BASE
        if slave_idx == 1:
            return Addr.DRAM_BASE
        raise ValueError(f"Invalid slave_idx {slave_idx}")


# ── APB Decoder ─────────────────────────────────────────────────────

class APBDecoder:
    """APB address decoder — 1 master -> 7 slaves, 4 KB windows.

    Matches rtl/soc/apb_decoder.v psel/paddr decode logic.

    Slave mapping:
        slave0 = MXU       0x4000_0000–0x4000_0FFF
        slave1 = SFU       0x4000_1000–0x4000_1FFF
        slave2 = VECTOR    0x4000_2000–0x4000_2FFF
        slave3 = DMA       0x4000_3000–0x4000_3FFF
        slave4 = PCIe      0x4000_4000–0x4000_4FFF
        slave5 = DOORBELL  0x4000_5000–0x4000_5FFF
        slave6 = INTC      0x4000_6000–0x4000_6FFF
    """

    SLAVES = {
        0: ("MXU", Addr.MXU_BASE),
        1: ("SFU", Addr.SFU_BASE),
        2: ("VECTOR", Addr.VECTOR_BASE),
        3: ("DMA", Addr.DMA_BASE),
        4: ("PCIe", Addr.PCIE_BASE),
        5: ("DOORBELL", Addr.DOORBELL),
        6: ("INTC", Addr.INTC_BASE),
    }

    def __init__(self):
        self._slave_map = {
            idx: (base, 0x1000) for idx, (_, base) in self.SLAVES.items()
        }

    def decode(self, paddr: int) -> int:
        """Decode APB address to slave index (0–6).

        Args:
            paddr: 32-bit APB address.

        Returns:
            Slave index 0–6.

        Raises:
            ValueError: If paddr is out of the MMIO range.
        """
        if not Addr.MXU_BASE <= paddr < Addr.INTC_BASE + 0x1000:
            raise ValueError(f"APB address 0x{paddr:08x} out of MMIO range")
        return (paddr >> 12) & 0xF

    def get_slave_name(self, slave_idx: int) -> str:
        """Return human-readable slave name for debug."""
        return self.SLAVES.get(slave_idx, ("UNKNOWN", 0))[0]

    @property
    def slave_map(self) -> dict:
        """Return {idx: (base, size)} for all 7 slaves."""
        return self._slave_map.copy()
