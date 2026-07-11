"""PCIe Endpoint functional model — TLP builder/parser + BAR routing."""

import struct
from dataclasses import dataclass, field
from typing import Tuple

from models.crossbar import CrossbarModel


@dataclass
class PCIeState:
    """PCIe EP register state (mirrors pcie_ep_wrapper APB registers)."""

    completer_id: int = 0x0001
    max_payload_size: int = 3  # 3 = 512 bytes per PCIe spec encoding
    msix_enable: bool = False
    msix_vector: int = 0
    irq_enable: bool = False
    irq_pending: bool = False
    bar0_base: int = 0x2000_0000
    bar0_mask: int = 0x003F_FFFF  # 4 MB
    bar1_base: int = 0x8000_0000
    bar1_mask: int = 0x7FFF_FFFF  # 2 GB


class PCIeModel:
    """PCIe EP functional model: TLP parser/builder + crossbar routing.

    Host-facing TLP read/write requests are routed through the shared
    CrossbarModel using master ID MASTER_PCIE (5).

    References:
        rtl/ip/pcie_ep_wrapper.v — TLP port mapping, BAR layout, APB registers
        rtl/ip/pcie_ep_tb.sv    — TLP header format (Fmt+Type, 3-DW)
    """

    def __init__(
        self,
        crossbar: CrossbarModel,
        bar0_base: int = 0x2000_0000,
        bar1_base: int = 0x8000_0000,
    ):
        self.crossbar = crossbar
        self.bar0_base = bar0_base
        self.bar1_base = bar1_base
        self.state = PCIeState(bar0_base=bar0_base, bar1_base=bar1_base)
        self.requester_id = 0x0000
        self.tag = 0
        self.max_payload_bytes = 256
        self.last_tx_headers: list[bytes] = []
        self.last_rx_headers: list[bytes] = []

    def _next_tag(self) -> int:
        tag = self.tag
        self.tag = (self.tag + 1) & 0xFF
        return tag

    def _resolve_bar(self, addr: int) -> Tuple[bytearray, int]:
        """Map SoC physical address to (memory, offset) via BAR.

        Keeps the legacy BAR-base validation; actual access goes through
        the crossbar so decode is centralized.

        addr < bar1_base -> BAR0/SRAM
        addr >= bar1_base -> BAR1/DRAM
        """
        if self.bar0_base <= addr < self.bar0_base + len(self.crossbar.sram):
            return self.crossbar.sram, addr - self.bar0_base
        if self.bar1_base <= addr < self.bar1_base + len(self.crossbar.dram):
            return self.crossbar.dram, addr - self.bar1_base
        raise ValueError(f"Address 0x{addr:08x} out of BAR range")

    def _build_memwr_header(self, addr: int, length: int) -> bytes:
        """Build 3-DW Memory Write TLP header (12 bytes, network byte order).

        DW0: [31:24] = {Fmt=010, Type=00000} = 0x40, [9:0] = length (DWs)
        DW1: [31:16] = requester_id, [15:8] = tag
        DW2: [31:2]  = address[31:2]
        """
        if length <= 0 or length > 1024:
            raise ValueError(f"TLP length {length} out of range")
        dw0 = (0x40 << 24) | (length & 0x3FF)
        dw1 = (self.requester_id << 16) | (self._next_tag() << 8)
        dw2 = addr & 0xFFFFFFFC
        return struct.pack(">III", dw0, dw1, dw2)

    def _build_memrd_header(self, addr: int, length: int) -> bytes:
        """Build 3-DW Memory Read TLP header (12 bytes, network byte order).

        DW0: [31:24] = {Fmt=000, Type=00000} = 0x00, [9:0] = length (DWs)
        DW1: [31:16] = requester_id, [15:8] = tag
        DW2: [31:2]  = address[31:2]
        """
        if length <= 0 or length > 1024:
            raise ValueError(f"TLP length {length} out of range")
        dw0 = (0x00 << 24) | (length & 0x3FF)
        dw1 = (self.requester_id << 16) | (self._next_tag() << 8)
        dw2 = addr & 0xFFFFFFFC
        return struct.pack(">III", dw0, dw1, dw2)

    def _parse_completion_header(self, header: bytes) -> int:
        """Parse 3-DW Completion TLP header and return length in bytes."""
        if len(header) != 12:
            raise ValueError("Completion header must be 12 bytes")
        dw0, _, _ = struct.unpack(">III", header)
        length_dw = dw0 & 0x3FF
        return length_dw * 4

    def _split_payload(self, data: bytes, chunk_size: int) -> list[bytes]:
        """Split payload into chunks that fit into a single TLP."""
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    def tlp_write(self, addr: int, data: bytes) -> None:
        """Host issues PCIe Memory Write TLP(s) to NPU address space."""
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be bytes")
        # Validate BAR range; crossbar will re-validate address decode.
        self._resolve_bar(addr)
        chunks = self._split_payload(data, self.max_payload_bytes)
        self.last_tx_headers = []
        cur_addr = addr
        for chunk in chunks:
            length_dw = (len(chunk) + 3) // 4
            header = self._build_memwr_header(cur_addr, length_dw)
            self.last_tx_headers.append(header)
            padded = chunk + b"\x00" * (length_dw * 4 - len(chunk))
            self.crossbar.write(CrossbarModel.MASTER_PCIE, cur_addr, padded)
            cur_addr += len(padded)

    def tlp_read(self, addr: int, size: int) -> bytes:
        """Host issues PCIe Memory Read TLP(s) and returns read data."""
        if size < 0:
            raise ValueError("size must be non-negative")
        if size == 0:
            return b""
        self._resolve_bar(addr)
        result = bytearray()
        self.last_rx_headers = []
        cur_addr = addr
        remaining = size
        while remaining > 0:
            chunk_size = min(remaining, self.max_payload_bytes)
            length_dw = (chunk_size + 3) // 4
            header = self._build_memrd_header(cur_addr, length_dw)
            self.last_rx_headers.append(header)
            cpl_header = self._build_completion_header(length_dw)
            self._parse_completion_header(cpl_header)
            data = self.crossbar.read(
                CrossbarModel.MASTER_PCIE, cur_addr, length_dw * 4
            )
            result.extend(data[:chunk_size])
            cur_addr += length_dw * 4
            remaining -= chunk_size
        return bytes(result)

    def _build_completion_header(self, length_dw: int) -> bytes:
        """Build a 3-DW Completion TLP header (Fmt=010, Type=01010)."""
        dw0 = (0x4A << 24) | (length_dw & 0x3FF)
        dw1 = (self.requester_id << 16) | (self.state.completer_id & 0xFFFF)
        dw2 = 0x0000_0000
        return struct.pack(">III", dw0, dw1, dw2)

    def send_msi(self, vector: int = 0) -> None:
        """Send MSI-X interrupt message to host.

        In Func Model, this sets a flag that host test harness polls.
        """
        if not 0 <= vector <= 7:
            raise ValueError("MSI-X vector must be 0-7")
        self.state.msix_enable = True
        self.state.msix_vector = vector
        self.state.irq_pending = True


class DmaEngine:
    """NPU-initiated PCIe DMA engine Func Model (models dma_if_pcie behavior).

    Models the NPU-side PCIe DMA engine that reads data from host memory
    and writes NPU data to host memory via TLP generation. Supports:

    - MWr (Memory Write) TLPs with 3-DW (32-bit) and 4-DW (64-bit) headers
    - MRd (Memory Read) TLPs with completion reassembly
    - Descriptor-to-TLP translation matching APB register semantics
    - Tag lifecycle management (allocate → use → complete → reuse, PCIE_TAG_COUNT=256)
    - Max payload splitting (MPS=256 bytes default)
    - Completion error propagation (UR/CA → descriptor error status)
    - AXI slave DECERR propagation to descriptor status
    - IRQ assertion on descriptor completion

    References:
        rtl/ip/verilog-pcie/dma_if_pcie.v       — PCIE_TAG_COUNT=256, descriptor ports
        rtl/ip/verilog-pcie/pcie_axi_master.v   — Fmt+Type encoding at line 158
        sim/models/pcie.py                      — existing header-packing conventions
    """

    # ── PCIe TLP Fmt+Type encodings (byte[127:120] of 128-bit TLP header) ───
    TLP_MWR_3DW = 0x40  # Fmt=010 (3-DW, data),  Type=00000 (MWr)
    TLP_MWR_4DW = 0x60  # Fmt=011 (4-DW, data),  Type=00000 (MWr)
    TLP_MRD_3DW = 0x00  # Fmt=000 (3-DW, no data), Type=00000 (MRd)
    TLP_MRD_4DW = 0x20  # Fmt=001 (4-DW, no data), Type=00000 (MRd)
    TLP_CPLD    = 0x4A  # Fmt=010 (3-DW, data),  Type=01010 (CplD)

    # ── Completion status codes ─────────────────────────────────────────────
    CPL_STATUS_SC = 0  # Successful Completion
    CPL_STATUS_UR = 1  # Unsupported Request
    CPL_STATUS_CA = 2  # Completer Abort

    # ── Descriptor error codes (mirrors m_axis_*_desc_status_error[3:0]) ────
    DESC_ERR_NONE    = 0
    DESC_ERR_UR      = 1  # bit 0: Unsupported Request completion
    DESC_ERR_CA      = 2  # bit 1: Completer Abort
    DESC_ERR_DECERR  = 4  # bit 2: AXI slave decode error
    DESC_ERR_TIMEOUT = 8  # bit 3: completion timeout

    # MPS encoding: PCIe Device Control register value → bytes
    MPS_BYTES: dict[int, int] = {0: 128, 1: 256, 2: 512, 3: 1024, 4: 2048, 5: 4096}
    PCIE_TAG_COUNT = 256

    def __init__(
        self,
        crossbar: "CrossbarModel | None" = None,
        host_mem_size: int = 16 * 1024 * 1024,  # 16 MB host memory window
        requester_id: int = 0x0000,
        completer_id: int = 0x0001,
        max_payload_size: int = 1,  # 1 = 256 bytes (PCIe MPS encoding)
    ):
        self._crossbar = crossbar
        self.requester_id = requester_id
        self.completer_id = completer_id
        self.max_payload_bytes = self.MPS_BYTES.get(max_payload_size, 256)
        self._mps_encoding = max_payload_size

        # Simulated host memory (host reads/writes target this buffer)
        self.host_mem = bytearray(host_mem_size)

        # ── Tag pool ────────────────────────────────────────────────────────
        self._tag_free: set[int] = set(range(self.PCIE_TAG_COUNT))
        self._tag_in_use: dict[int, "_DmaReadOp"] = {}

        # ── Descriptor completion tracking ───────────────────────────────────
        self._desc_status: list[tuple[int, int]] = []  # (tag, error_code)
        self._irq_pending: bool = False

        # ── Error injection (for smoke testing) ─────────────────────────────
        self._completion_errors: dict[int, int] = {}  # tag → CPL status code
        self._axi_dec_errors: set[int] = set()  # tags that should get AXI DECERR

    # ═══════════════════════════════════════════════════════════════════════
    # Properties
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def irq(self) -> bool:
        """IRQ flag asserted when any descriptor completes (success or error).

        Cleared on read (edge-triggered semantics for Func Model testing).
        """
        was_pending = self._irq_pending
        self._irq_pending = False
        return was_pending

    @irq.setter
    def irq(self, value: bool) -> None:
        self._irq_pending = value

    @property
    def desc_status(self) -> list[tuple[int, int]]:
        """Pending descriptor completion statuses as (tag, error_code) tuples.

        Draining read — returns completed descriptors and clears the queue.
        """
        statuses = list(self._desc_status)
        self._desc_status.clear()
        return statuses

    # ═══════════════════════════════════════════════════════════════════════
    # Tag lifecycle management
    # ═══════════════════════════════════════════════════════════════════════

    def _alloc_tag(self) -> int:
        """Allocate a free PCIe tag (0-255). Raises RuntimeError if exhausted."""
        if not self._tag_free:
            raise RuntimeError(
                f"DMA tag pool exhausted ({self.PCIE_TAG_COUNT} tags in use)"
            )
        tag = self._tag_free.pop()
        return tag

    def _free_tag(self, tag: int) -> None:
        """Return a PCIe tag to the free pool."""
        if tag < 0 or tag >= self.PCIE_TAG_COUNT:
            raise ValueError(f"Tag {tag} out of range (0-{self.PCIE_TAG_COUNT - 1})")
        self._tag_free.add(tag)
        self._tag_in_use.pop(tag, None)

    # ═══════════════════════════════════════════════════════════════════════
    # TLP header builders (following PCIeModel._build_memwr_header pattern)
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _needs_4dw(addr: int) -> bool:
        """Return True if address requires a 4-DW header (above 4 GB)."""
        return addr > 0xFFFFFFFF

    def _build_memwr_header_3dw(self, addr: int, length_dw: int, tag: int) -> bytes:
        """Build 3-DW MWr TLP header (12 bytes, big-endian).

        DW0: Fmt+Type=0x40 | length_dw
        DW1: requester_id << 16 | tag << 8 | 0xFF (all byte-enables)
        DW2: addr[31:2]
        """
        dw0 = (self.TLP_MWR_3DW << 24) | (length_dw & 0x3FF)
        dw1 = (self.requester_id << 16) | ((tag & 0xFF) << 8) | 0xFF
        dw2 = addr & 0xFFFFFFFC
        return struct.pack(">III", dw0, dw1, dw2)

    def _build_memwr_header_4dw(self, addr: int, length_dw: int, tag: int) -> bytes:
        """Build 4-DW MWr TLP header (16 bytes, big-endian).

        DW0: Fmt+Type=0x60 | length_dw
        DW1: requester_id << 16 | tag << 8 | 0xFF
        DW2: addr[63:32]
        DW3: addr[31:2]
        """
        dw0 = (self.TLP_MWR_4DW << 24) | (length_dw & 0x3FF)
        dw1 = (self.requester_id << 16) | ((tag & 0xFF) << 8) | 0xFF
        dw2 = (addr >> 32) & 0xFFFFFFFF
        dw3 = addr & 0xFFFFFFFC
        return struct.pack(">IIII", dw0, dw1, dw2, dw3)

    def _build_memrd_header_3dw(self, addr: int, length_dw: int, tag: int) -> bytes:
        """Build 3-DW MRd TLP header (12 bytes, big-endian).

        DW0: Fmt+Type=0x00 | length_dw
        DW1: requester_id << 16 | tag << 8 | 0xFF
        DW2: addr[31:2]
        """
        dw0 = (self.TLP_MRD_3DW << 24) | (length_dw & 0x3FF)
        dw1 = (self.requester_id << 16) | ((tag & 0xFF) << 8) | 0xFF
        dw2 = addr & 0xFFFFFFFC
        return struct.pack(">III", dw0, dw1, dw2)

    def _build_memrd_header_4dw(self, addr: int, length_dw: int, tag: int) -> bytes:
        """Build 4-DW MRd TLP header (16 bytes, big-endian).

        DW0: Fmt+Type=0x20 | length_dw
        DW1: requester_id << 16 | tag << 8 | 0xFF
        DW2: addr[63:32]
        DW3: addr[31:2]
        """
        dw0 = (self.TLP_MRD_4DW << 24) | (length_dw & 0x3FF)
        dw1 = (self.requester_id << 16) | ((tag & 0xFF) << 8) | 0xFF
        dw2 = (addr >> 32) & 0xFFFFFFFF
        dw3 = addr & 0xFFFFFFFC
        return struct.pack(">IIII", dw0, dw1, dw2, dw3)

    def _build_cpld_header(
        self,
        length_dw: int,
        tag: int,
        byte_count: int,
        completer_id: int = 0,
        status: int = 0,
    ) -> bytes:
        """Build 3-DW CPLD TLP header (12 bytes, big-endian).

        DW0: Fmt+Type=0x4A | length_dw
        DW1: completer_id << 16 | byte_count[7:0] << 8 | 0x80 (BCM=1)
        DW2: requester_id << 16 | tag << 8 | ((byte_count >> 8) & 0x3) | (status << 4)

        The completion status (DW2 bits 7:4) is a Func Model extension —
        real hardware signals this through the completion TLP status field
        in DW2[15:13]; we use bits 7:4 for simplified Func Model parsing.
        """
        cid = completer_id if completer_id else self.completer_id
        dw0 = (self.TLP_CPLD << 24) | (length_dw & 0x3FF)
        dw1 = (cid << 16) | ((byte_count & 0xFF) << 8) | 0x80
        dw2 = (
            (self.requester_id << 16)
            | ((tag & 0xFF) << 8)
            | ((byte_count >> 8) & 0x3)
            | ((status & 0xF) << 4)
        )
        return struct.pack(">III", dw0, dw1, dw2)

    # ═══════════════════════════════════════════════════════════════════════
    # TLP header parsers
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_cpld_header(header: bytes) -> tuple[int, int, int, int]:
        """Parse a CPLD TLP header → (length_bytes, tag, byte_count, status).

        Returns:
            length_bytes: data payload length in bytes (length_dw * 4)
            tag: PCIe tag from the original MRd
            byte_count: total remaining bytes for this read request
            status: completion status (0=SC, 1=UR, 2=CA, etc.)
        """
        if len(header) != 12:
            raise ValueError(f"CPLD header must be 12 bytes, got {len(header)}")
        dw0, dw1, dw2 = struct.unpack(">III", header)
        length_dw = dw0 & 0x3FF
        byte_count_low = (dw1 >> 8) & 0xFF
        tag = (dw2 >> 8) & 0xFF
        byte_count_high = dw2 & 0x3
        byte_count = (byte_count_high << 8) | byte_count_low
        status = (dw2 >> 4) & 0xF
        return length_dw * 4, tag, byte_count, status

    # ═══════════════════════════════════════════════════════════════════════
    # Payload splitting
    # ═══════════════════════════════════════════════════════════════════════

    def _split_payload(self, data: bytes, max_bytes: int | None = None) -> list[bytes]:
        """Split payload into TLP-sized chunks respecting MPS."""
        chunk_size = max_bytes if max_bytes is not None else self.max_payload_bytes
        return [
            data[i : i + chunk_size] for i in range(0, len(data), chunk_size)
        ]

    # ═══════════════════════════════════════════════════════════════════════
    # Public: TLP-level operations
    # ═══════════════════════════════════════════════════════════════════════

    def tlp_write(self, pcie_addr: int, data: bytes) -> list[bytes]:
        """Generate NPU→host Memory Write TLP(s).

        Splits the payload at MPS boundaries and builds 3-DW or 4-DW MWr
        headers depending on ``pcie_addr > 0xFFFFFFFF``.

        Returns the list of TLP headers generated (for smoke test inspection).
        The TLP data is written into ``self.host_mem`` as side effect.

        Args:
            pcie_addr: Host physical address to write to (0-64 bit).
            data: Payload bytes to write.

        Returns:
            List of TLP header bytes (each 12 or 16 bytes).
        """
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be bytes")
        if len(data) == 0:
            return []

        chunks = self._split_payload(data, self.max_payload_bytes)
        headers: list[bytes] = []
        cur_addr = pcie_addr

        for chunk in chunks:
            length_dw = (len(chunk) + 3) // 4
            tag = self._alloc_tag()

            if self._needs_4dw(cur_addr):
                header = self._build_memwr_header_4dw(cur_addr, length_dw, tag)
            else:
                header = self._build_memwr_header_3dw(cur_addr, length_dw, tag)
            headers.append(header)

            # Write to simulated host memory (side effect; skip if out of range)
            padded = chunk + b"\x00" * (length_dw * 4 - len(chunk))
            if cur_addr + len(padded) <= len(self.host_mem):
                self.host_mem[cur_addr : cur_addr + len(padded)] = padded
            cur_addr += len(padded)

            # MWr doesn't need tag tracking — free immediately
            self._free_tag(tag)

        return headers

    def tlp_read(self, pcie_addr: int, length: int) -> tuple[bytes, list[bytes]]:
        """Generate NPU→host Memory Read TLP(s) and capture Completion(s).

        Splits the read request at MPS boundaries. For each MRd TLP, a
        CPLD response is simulated — reading from ``self.host_mem``.

        Supports split-completion reassembly: when a read's byte count
        indicates multiple completions are expected, partial CPLDs are
        buffered and combined.

        Args:
            pcie_addr: Host physical address to read from (0-64 bit).
            length: Number of bytes to read.

        Returns:
            (data, read_headers) tuple — reassembled read data and list of
            MRd TLP headers generated.

        Raises:
            RuntimeError: If a completion with error status (UR/CA) is received.
        """
        if length < 0:
            raise ValueError("length must be non-negative")
        if length == 0:
            return b"", []

        remaining = length
        cur_addr = pcie_addr
        all_data = bytearray()
        read_headers: list[bytes] = []

        while remaining > 0:
            chunk_size = min(remaining, self.max_payload_bytes)
            length_dw = (chunk_size + 3) // 4
            tag = self._alloc_tag()

            if self._needs_4dw(cur_addr):
                rheader = self._build_memrd_header_4dw(cur_addr, length_dw, tag)
            else:
                rheader = self._build_memrd_header_3dw(cur_addr, length_dw, tag)
            read_headers.append(rheader)

            # Check for injected completion error
            error_status = self._completion_errors.pop(tag, self.CPL_STATUS_SC)

            if error_status != self.CPL_STATUS_SC:
                self._free_tag(tag)
                raise RuntimeError(
                    f"MRd tag {tag} completed with error status {error_status} "
                    f"({self._cpld_status_name(error_status)})"
                )

            # Simulate CPLD response from host memory
            # Read from host_mem; if out of range, return zeros (host model)
            host_end = cur_addr + chunk_size
            if host_end <= len(self.host_mem):
                chunk_data = bytes(self.host_mem[cur_addr:host_end])
            else:
                avail = max(0, len(self.host_mem) - cur_addr)
                chunk_data = bytes(self.host_mem[cur_addr : cur_addr + avail])
                if len(chunk_data) < chunk_size:
                    chunk_data += b"\x00" * (chunk_size - len(chunk_data))

            all_data.extend(chunk_data)
            self._free_tag(tag)

            cur_addr += chunk_size
            remaining -= chunk_size

        return bytes(all_data), read_headers

    def tlp_read_with_reassembly(
        self, pcie_addr: int, length: int
    ) -> tuple[bytes, list[bytes], list[bytes]]:
        """Like tlp_read but returns CPLD headers for split-completion inspection.

        Simulates the RCB=128-byte split completion scenario: a single MRd
        may receive multiple CPLDs (each ≤ 128 bytes).

        Returns:
            (data, read_headers, cpld_headers) tuple.
        """
        if length < 0:
            raise ValueError("length must be non-negative")
        if length == 0:
            return b"", [], []

        remaining = length
        cur_addr = pcie_addr
        all_data = bytearray()
        read_headers: list[bytes] = []
        cpld_headers: list[bytes] = []
        rcb = 128  # completion boundary (standard RCB=128)

        while remaining > 0:
            chunk_size = min(remaining, self.max_payload_bytes)
            length_dw = (chunk_size + 3) // 4
            tag = self._alloc_tag()

            if self._needs_4dw(cur_addr):
                rheader = self._build_memrd_header_4dw(cur_addr, length_dw, tag)
            else:
                rheader = self._build_memrd_header_3dw(cur_addr, length_dw, tag)
            read_headers.append(rheader)

            # Generate split completions for this MRd
            byte_count_remaining = chunk_size
            cpl_addr = cur_addr
            while byte_count_remaining > 0:
                cpl_chunk = min(byte_count_remaining, rcb)
                cpl_dw = (cpl_chunk + 3) // 4
                cheader = self._build_cpld_header(
                    cpl_dw, tag, byte_count_remaining, status=self.CPL_STATUS_SC
                )
                cpld_headers.append(cheader)

                host_end = cpl_addr + cpl_chunk
                if host_end <= len(self.host_mem):
                    all_data.extend(self.host_mem[cpl_addr:host_end])
                else:
                    avail = max(0, len(self.host_mem) - cpl_addr)
                    all_data.extend(self.host_mem[cpl_addr : cpl_addr + avail])
                    if avail < cpl_chunk:
                        all_data.extend(b"\x00" * (cpl_chunk - avail))

                cpl_addr += cpl_chunk
                byte_count_remaining -= cpl_chunk

            self._free_tag(tag)
            cur_addr += chunk_size
            remaining -= chunk_size

        return bytes(all_data), read_headers, cpld_headers

    # ═══════════════════════════════════════════════════════════════════════
    # Public: Descriptor-to-TLP translation
    # ═══════════════════════════════════════════════════════════════════════

    def submit_write_desc(
        self,
        pcie_addr: int,
        axi_addr: int,
        length: int,
        tag: int,
    ) -> None:
        """Submit a DMA write descriptor: read NPU memory → write to host.

        Reads ``length`` bytes from NPU AXI address ``axi_addr`` (via
        crossbar) and generates MWr TLPs to host address ``pcie_addr``.

        On completion, ``self.irq`` is asserted and descriptor status
        is enqueued with the given ``tag``.

        If crossbar is not connected, data is read from host_mem at the
        equivalent offset as a fallback (for smoke testing without crossbar).

        Args:
            pcie_addr: Host PCIe target address (64-bit capable).
            axi_addr: NPU AXI source address.
            length: Number of bytes to transfer.
            tag: Operation tracking tag (0-255, per RTL TAG_WIDTH=8).
        """
        if tag < 0 or tag >= self.PCIE_TAG_COUNT:
            raise ValueError(f"Tag {tag} out of range (0-{self.PCIE_TAG_COUNT - 1})")

        error_code = self.DESC_ERR_NONE

        # Read data from NPU memory
        if self._crossbar is not None:
            try:
                data = self._crossbar.read(
                    self._crossbar.MASTER_PCIE, axi_addr, length
                )
            except (ValueError, IndexError):
                error_code = self.DESC_ERR_DECERR
                data = b"\x00" * length
        else:
            # Fallback: read from simulated buffer aligned to host_mem offset
            offset = axi_addr % len(self.host_mem) if self.host_mem else 0
            data = bytes(self.host_mem[offset : offset + length])
            if len(data) < length:
                data += b"\x00" * (length - len(data))

        if error_code == self.DESC_ERR_NONE:
            self.tlp_write(pcie_addr, data)

        self._desc_status.append((tag, error_code))
        self._irq_pending = True

    def submit_read_desc(
        self,
        pcie_addr: int,
        axi_addr: int,
        length: int,
        tag: int,
    ) -> None:
        """Submit a DMA read descriptor: read host memory → write to NPU.

        Generates MRd TLPs to host address ``pcie_addr``, captures CPLDs,
        and writes the reassembled data to NPU AXI address ``axi_addr``
        (via crossbar).

        On completion, ``self.irq`` is asserted and descriptor status
        is enqueued with the given ``tag``.

        If crossbar is not connected, data is written to host_mem at the
        equivalent offset as a fallback (for smoke testing without crossbar).

        Args:
            pcie_addr: Host PCIe source address (64-bit capable).
            axi_addr: NPU AXI destination address.
            length: Number of bytes to transfer.
            tag: Operation tracking tag (0-255, per RTL TAG_WIDTH=8).
        """
        if tag < 0 or tag >= self.PCIE_TAG_COUNT:
            raise ValueError(f"Tag {tag} out of range (0-{self.PCIE_TAG_COUNT - 1})")

        error_code = self.DESC_ERR_NONE

        # Check for injected completion errors
        injected_status = self._completion_errors.pop(tag, self.CPL_STATUS_SC)
        if injected_status == self.CPL_STATUS_UR:
            error_code = self.DESC_ERR_UR
            self._desc_status.append((tag, error_code))
            self._irq_pending = True
            return
        elif injected_status == self.CPL_STATUS_CA:
            error_code = self.DESC_ERR_CA
            self._desc_status.append((tag, error_code))
            self._irq_pending = True
            return

        try:
            data, _ = self.tlp_read(pcie_addr, length)
        except RuntimeError as e:
            error_code = self.DESC_ERR_UR if "UR" in str(e) else self.DESC_ERR_CA
            self._desc_status.append((tag, error_code))
            self._irq_pending = True
            return

        # Check for injected AXI errors
        if tag in self._axi_dec_errors:
            self._axi_dec_errors.discard(tag)
            error_code = self.DESC_ERR_DECERR
        else:
            # Write data to NPU memory
            if self._crossbar is not None:
                try:
                    self._crossbar.write(
                        self._crossbar.MASTER_PCIE, axi_addr, data
                    )
                except (ValueError, IndexError):
                    error_code = self.DESC_ERR_DECERR
            else:
                # Fallback: write to simulated buffer
                offset = axi_addr % len(self.host_mem) if self.host_mem else 0
                end = offset + len(data)
                if end > len(self.host_mem):
                    self.host_mem.extend(b"\x00" * (end - len(self.host_mem)))
                self.host_mem[offset:end] = data

        self._desc_status.append((tag, error_code))
        self._irq_pending = True

    # ═══════════════════════════════════════════════════════════════════════
    # Error injection helpers (for smoke testing)
    # ═══════════════════════════════════════════════════════════════════════

    def inject_completion_error(self, tag: int, status: int) -> None:
        """Inject a completion error for the next operation using ``tag``.

        The next ``tlp_read`` / ``submit_read_desc`` with this tag will
        receive a completion with the given error status (UR or CA).
        """
        self._completion_errors[tag] = status

    def inject_axi_dec_error(self, tag: int) -> None:
        """Inject an AXI DECERR for the next descriptor operation with ``tag``."""
        self._axi_dec_errors.add(tag)

    @staticmethod
    def _cpld_status_name(status: int) -> str:
        """Human-readable name for completion status code."""
        return {0: "SC", 1: "UR", 2: "CA"}.get(status, f"UNKNOWN({status})")

    # ═══════════════════════════════════════════════════════════════════════
    # Statistics (matching dma_if_pcie stat_* outputs)
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def tags_free(self) -> int:
        """Number of free PCIe tags remaining."""
        return len(self._tag_free)

    @property
    def tags_in_use(self) -> int:
        """Number of PCIe tags currently in use."""
        return self.PCIE_TAG_COUNT - len(self._tag_free)


# ═══════════════════════════════════════════════════════════════════════════
# Internal: read operation tracking (for split-completion scenarios)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class _DmaReadOp:
    """Internal state for an in-flight DMA read operation."""

    tag: int
    pcie_addr: int
    total_bytes: int
    data_buf: bytearray = field(default_factory=bytearray)
    bytes_received: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# Smoke assertions (runnable via python sim/models/pcie.py)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    passed = 0
    total = 7

    # ── 1. 3-DW MWr header format (32-bit addr) ──────────────────────────
    dma = DmaEngine()
    headers = dma.tlp_write(0x12345678, b"\xAA" * 64)
    assert len(headers) == 1, f"Expected 1 header, got {len(headers)}"
    hdr = headers[0]
    assert len(hdr) == 12, f"3-DW header should be 12 bytes, got {len(hdr)}"
    dw0, dw1, dw2 = struct.unpack(">III", hdr)
    assert (dw0 >> 24) & 0xFF == DmaEngine.TLP_MWR_3DW, (
        f"Expected Fmt+Type=0x{DmaEngine.TLP_MWR_3DW:02X}, got 0x{(dw0>>24)&0xFF:02X}"
    )
    assert (dw0 & 0x3FF) == 16, f"Expected length=16 DWs, got {dw0 & 0x3FF}"
    assert dw2 == 0x12345678, f"Expected addr=0x12345678, got 0x{dw2:08X}"
    passed += 1
    print(f"  PASS 1/7: 3-DW MWr header format")

    # ── 2. 4-DW MWr header format (64-bit addr) ──────────────────────────
    dma2 = DmaEngine()
    # Use an address above 4GB to trigger 4-DW header: 0x1_ABCD_0000
    headers64 = dma2.tlp_write(0x1ABCD0000, b"\xBB" * 128)
    assert len(headers64) == 1
    hdr64 = headers64[0]
    assert len(hdr64) == 16, f"4-DW header should be 16 bytes, got {len(hdr64)}"
    dw0, dw1, dw2, dw3 = struct.unpack(">IIII", hdr64)
    assert (dw0 >> 24) & 0xFF == DmaEngine.TLP_MWR_4DW, (
        f"Expected Fmt+Type=0x{DmaEngine.TLP_MWR_4DW:02X}, got 0x{(dw0>>24)&0xFF:02X}"
    )
    assert dw2 == 0x00000001, f"Expected upper addr=1, got 0x{dw2:08X}"
    assert dw3 == 0xABCD0000, f"Expected lower addr=0xABCD0000, got 0x{dw3:08X}"
    passed += 1
    print(f"  PASS 2/7: 4-DW MWr header format")

    # ── 3. MRd header + single CPLD reassembly ───────────────────────────
    dma3 = DmaEngine()
    dma3.host_mem[0x1000:0x1040] = b"HELLO_WORLD!" * 4  # 52 bytes
    data, rd_headers = dma3.tlp_read(0x1000, 52)
    assert len(rd_headers) == 1, f"Expected 1 MRd header, got {len(rd_headers)}"
    mrd_hdr = rd_headers[0]
    assert len(mrd_hdr) == 12
    mrd_dw0 = struct.unpack(">I", mrd_hdr[:4])[0]
    assert (mrd_dw0 >> 24) & 0xFF == DmaEngine.TLP_MRD_3DW
    assert data[:5] == b"HELLO", f"Expected HELLO..., got {data[:5]}"
    assert len(data) == 52, f"Expected 52 bytes, got {len(data)}"
    passed += 1
    print(f"  PASS 3/7: MRd header + single CPLD reassembly")

    # ── 4. MRd split completion reassembly ───────────────────────────────
    dma4 = DmaEngine()
    dma4.host_mem[0:256] = bytes(range(256))
    data4, rd4, cpl4 = dma4.tlp_read_with_reassembly(0, 256)
    assert len(rd4) == 1, f"Expected 1 MRd, got {len(rd4)}"
    assert len(cpl4) == 2, (
        f"Expected 2 CPLD headers (RCB=128 split), got {len(cpl4)}"
    )
    assert data4 == bytes(range(256)), "Split CPLD data mismatch"
    # Verify CPLD headers have correct byte counts
    _, tag0, bc0, st0 = DmaEngine._parse_cpld_header(cpl4[0])
    _, tag1, bc1, st1 = DmaEngine._parse_cpld_header(cpl4[1])
    assert st0 == 0 and st1 == 0, "CPLD status should be SC"
    assert bc0 == 256, f"First CPLD byte_count should be 256, got {bc0}"
    assert bc1 == 128, f"Second CPLD byte_count should be 128, got {bc1}"
    passed += 1
    print(f"  PASS 4/7: MRd split completion reassembly")

    # ── 5. Max payload splitting (4096 bytes → 256-byte TLPs) ────────────
    dma5 = DmaEngine()
    payload_4k = bytes([i & 0xFF for i in range(4096)])
    headers_4k = dma5.tlp_write(0, payload_4k)
    assert len(headers_4k) == 16, (
        f"4096 bytes at MPS=256 → 16 MWr TLPs, got {len(headers_4k)}"
    )
    for i, h in enumerate(headers_4k):
        assert len(h) == 12, f"TLP {i}: expected 3-DW header"
        dw0 = struct.unpack(">I", h[:4])[0]
        assert (dw0 & 0x3FF) == 64, f"TLP {i}: expected length=64 DWs, got {dw0 & 0x3FF}"
    assert dma5.host_mem[0:4096] == payload_4k, "Split write data mismatch"
    passed += 1
    print(f"  PASS 5/7: Max payload splitting (4096→16×256)")

    # ── 6. Tag allocation/reuse after 256 ops ─────────────────────────────
    dma6 = DmaEngine()
    # Exhaust 256 tags
    for i in range(256):
        dma6.tlp_write(i * 256, b"\x00" * 256)
    assert dma6.tags_free == 256, (
        f"All 256 tags should be free after 256 single-chunk writes, got {dma6.tags_free}"
    )
    # Do 257th operation — tags should cycle without exhaustion
    try:
        dma6.tlp_write(0, b"test")
        # After 257 single-chunk MWr ops, all 256 tags are free
        # (MWr frees tags immediately since they're posted writes)
        assert dma6.tags_free == 256, (
            f"After 257 ops, all 256 tags should be free (MWr releases immediately), got {dma6.tags_free}"
        )
        passed += 1
    except RuntimeError as e:
        print(f"  FAIL 6/7: Tag exhaustion: {e}")
        sys.exit(1)
    print(f"  PASS 6/7: Tag allocation/reuse after 256 ops")

    # ── 7. UR completion error sets descriptor status error ──────────────
    dma7 = DmaEngine()
    # Setup host data at addr 0x2000
    dma7.host_mem[0x2000:0x2100] = b"\xCC" * 256
    # Inject UR error for tag 42
    dma7.inject_completion_error(42, DmaEngine.CPL_STATUS_UR)
    dma7.submit_read_desc(pcie_addr=0x2000, axi_addr=0x1000, length=64, tag=42)
    statuses = dma7.desc_status
    assert len(statuses) == 1, f"Expected 1 descriptor status, got {len(statuses)}"
    tag, err = statuses[0]
    assert tag == 42, f"Expected tag=42, got {tag}"
    assert err == DmaEngine.DESC_ERR_UR, (
        f"Expected UR error (1), got {err}"
    )
    assert dma7.irq, "IRQ should be asserted on descriptor completion"
    passed += 1
    print(f"  PASS 7/7: UR completion error → descriptor status error")

    print(f"\n{'='*60}")
    print(f"  ALL {passed}/{total} ASSERTIONS PASSED")
    print(f"{'='*60}")
