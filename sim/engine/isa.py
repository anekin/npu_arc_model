"""NPU 指令集架构（ISA）定义 + 解码器 + 编码器

CISC 风格，32-bit 定长指令字。一条指令 = 一个完整算子。
参考: NPU硬件详细架构设计v0.1 §3.7
"""

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, Tuple


class OpCode(IntEnum):
    """NPU 指令操作码（5 bits）"""
    MMUL     = 0x00   # 矩阵乘法
    SOFTMAX  = 0x01   # Softmax 激活
    LAYERNORM= 0x02   # Layer Normalization
    GELU     = 0x03   # GELU 激活
    RELU     = 0x04   # ReLU 激活
    ROPE     = 0x05   # RoPE 位置编码
    SILU     = 0x06   # SiLU 激活
    MAXPOOL  = 0x07   # Max Pooling 2x2
    AVGPOOL  = 0x08   # Avg Pooling 2x2
    DMA_LD   = 0x09   # DRAM → SRAM 加载
    DMA_ST   = 0x0A   # SRAM → DRAM 存储
    KV_LOAD  = 0x0B   # 加载指定 token 的 KV 到 SRAM
    KV_STORE = 0x0C   # 存储当前计算的 KV 到 DRAM
    BARRIER  = 0x0D   # 流水线同步栅栏
    NOP      = 0x0E   # 空操作
    # ── Vector unit instructions (v2) ────────────────────────────
    VADD     = 0x0F   # 逐元素加
    VMUL     = 0x10   # 逐元素乘
    VRED_MAX = 0x11   # 树规约求最大值
    VRED_SUM = 0x12   # 树规约求和
    VCONV    = 0x13   # INT32 → BF16 类型转换
    VRESID   = 0x14   # 残差连接: da = sa + sb
    # ── DMA descriptor instructions (v2) ─────────────────────────
    DMA_LDD  = 0x15   # DMA 加载（描述符链模式）
    DMA_STD  = 0x16   # DMA 存储（描述符链模式）
    RMSNORM  = 0x17   # RMS Normalization (two-pass FPU)
    VCONV_F16_I32 = 0x18   # FP16 → INT32 类型转换
    # ── Future extension ────────────────────────────────────────────

    @classmethod
    def from_mnemonic(cls, mnemonic: str) -> "OpCode":
        mn_map = {
            "mmul": cls.MMUL, "softmax": cls.SOFTMAX,
            "layernorm": cls.LAYERNORM, "gelu": cls.GELU,
            "relu": cls.RELU, "rope": cls.ROPE, "silu": cls.SILU,
            "maxpool": cls.MAXPOOL, "avgpool": cls.AVGPOOL,
            "dma_ld": cls.DMA_LD, "dma_st": cls.DMA_ST,
            "kv_load": cls.KV_LOAD, "kv_store": cls.KV_STORE,
            "barrier": cls.BARRIER, "nop": cls.NOP,
            # v2: Vector
            "vadd": cls.VADD, "vmul": cls.VMUL,
            "vred_max": cls.VRED_MAX, "vred_sum": cls.VRED_SUM,
            "vconv": cls.VCONV, "vresid": cls.VRESID,
            "vconv_f16_i32": cls.VCONV_F16_I32,
            # v2: DMA descriptor
            "dma_ldd": cls.DMA_LDD, "dma_std": cls.DMA_STD,
            # v3: RMSNorm
            "rmsnorm": cls.RMSNORM,
        }
        return mn_map[mnemonic.lower()]


# ── Instruction data structure ──────────────────────────────────────

@dataclass
class NPUInstruction:
    """Decoded NPU instruction."""
    opcode: OpCode
    operands: Dict[str, int] = field(default_factory=dict)
    comment: str = ""
    raw: int = 0  # 32-bit encoded form

    @property
    def mnemonic(self) -> str:
        return self.opcode.name.lower()

    def __repr__(self):
        ops = ", ".join(f"{k}={v}" for k, v in self.operands.items())
        comment = f"  # {self.comment}" if self.comment else ""
        return f"{self.mnemonic:10s} {ops}{comment}"


# ── Encoder / Decoder ───────────────────────────────────────────────

class NPUEncoder:
    """Encode NPUInstruction → 32-bit binary."""

    # Bit allocation (32-bit total):
    # [31:27] opcode (5)
    # [26:23] dest_reg / flags (4)
    # [22:8]  operand A (15)
    # [7:0]   operand B / immediate (8)
    # Extended operands go in subsequent words (for MMUL with 4 operands)

    @staticmethod
    def encode(instr: NPUInstruction) -> List[int]:
        """Encode instruction to 32-bit words."""
        words = []
        op = instr.opcode.value
        ops = instr.operands

        if op in (OpCode.MMUL,):
            # MMUL wa, ia, oa, N → 2 words: op+wa+ia, oa+N
            wa = ops.get("wa", 0)
            ia = ops.get("ia", 0)
            oa = ops.get("oa", 0)
            n  = ops.get("N", 0)
            w0 = (op << 27) | ((wa & 0x3FF) << 17) | ((ia & 0x3FF) << 7) | (n & 0x7F)
            w1 = (oa & 0xFFFF) << 16  # 2nd word carries oa
            words = [w0, w1]
        elif op in (OpCode.DMA_LD, OpCode.DMA_ST,):
            # DMA_LD dram, sram, size
            dram = ops.get("dram", 0) & 0xFFFF
            sram = ops.get("sram", 0) & 0xFFF
            size = ops.get("size", 0) & 0xFFF
            w0 = (op << 27) | ((dram >> 4) & 0xFFF) << 15 | ((sram >> 4) & 0xFF) << 7 | (size & 0x7F)
            words = [w0]
        elif op in (OpCode.MAXPOOL, OpCode.AVGPOOL,):
            # MAXPOOL/AVGPOOL sa, da, H, W
            sa = ops.get("sa", 0)
            da = ops.get("da", 0)
            h  = ops.get("H", 0) & 0xF
            w  = ops.get("W", 0) & 0xF
            w0 = (op << 27) | ((sa & 0xFFF) << 15) | ((da & 0xFFF) << 3) | ((h & 0x3) << 1) | (w & 0x1)
            words = [w0]
        elif op in (OpCode.ROPE,):
            # ROPE sa, da, len, theta
            sa = ops.get("sa", 0)
            da = ops.get("da", 0)
            length = ops.get("len", 0) & 0xFFF
            w0 = (op << 27) | ((sa & 0xFFF) << 15) | ((da & 0xFFF) << 3) | (length & 0x7)
            words = [w0]
        elif op in (OpCode.KV_LOAD, OpCode.KV_STORE,):
            # KV_LOAD/KV_STORE token_id
            tid = ops.get("token_id", 0) & 0xFFF
            w0 = (op << 27) | (tid & 0xFFF)
            words = [w0]
        elif op == OpCode.RMSNORM:
            # RMSNORM sa, da, elements (two-pass, FPU datapath)
            sa = ops.get("sa", 0)
            da = ops.get("da", 0)
            elements = ops.get("elements", 0) & 0xFFF
            w0 = (op << 27) | ((sa & 0xFFF) << 15) | ((da & 0xFFF) << 3) | (elements & 0x7)
            words = [w0]
        elif op in (OpCode.BARRIER, OpCode.NOP,):
            words = [op << 27]
        elif op in (OpCode.VADD, OpCode.VMUL, OpCode.VRED_MAX,
                    OpCode.VRED_SUM, OpCode.VCONV, OpCode.VRESID,
                    OpCode.VCONV_F16_I32,
                    OpCode.DMA_LDD, OpCode.DMA_STD,):
            # Generic Vector/DMA: op + sa + da + len
            sa = ops.get("sa", 0)
            da = ops.get("da", 0)
            length = ops.get("len", 0) & 0xFFF
            w0 = (op << 27) | ((sa & 0xFFF) << 15) | ((da & 0xFFF) << 3) | (length & 0x7)
            words = [w0]
        else:
            # Generic: SOFTMAX, LAYERNORM, GELU, RELU, SILU
            # Format: op + sa + da + len
            sa = ops.get("sa", 0)
            da = ops.get("da", 0)
            length = ops.get("len", 0) & 0xFFF
            w0 = (op << 27) | ((sa & 0xFFF) << 15) | ((da & 0xFFF) << 3) | (length & 0x7)
            words = [w0]

        instr.raw = words[0] if words else 0
        return words


class NPUDecoder:
    """Decode 32-bit binary → NPUInstruction."""

    @staticmethod
    def decode(words: List[int]) -> NPUInstruction:
        """Decode one or more 32-bit words to an instruction."""
        w0 = words[0]
        op = OpCode((w0 >> 27) & 0x1F)
        operands: Dict[str, int] = {}

        if op == OpCode.MMUL:
            wa = (w0 >> 17) & 0x3FF
            ia = (w0 >> 7) & 0x3FF
            n  = w0 & 0x7F
            oa = (words[1] >> 16) & 0xFFFF if len(words) > 1 else 0
            operands = {"wa": wa, "ia": ia, "oa": oa, "N": n}
        elif op in (OpCode.DMA_LD, OpCode.DMA_ST):
            dram = ((w0 >> 15) & 0xFFF) << 4
            sram = ((w0 >> 7) & 0xFF) << 4
            size = w0 & 0x7F
            operands = {"dram": dram, "sram": sram, "size": size}
        elif op in (OpCode.KV_LOAD, OpCode.KV_STORE):
            operands = {"token_id": w0 & 0xFFF}
        elif op in (OpCode.MAXPOOL, OpCode.AVGPOOL):
            operands = {
                "sa": (w0 >> 15) & 0xFFF,
                "da": (w0 >> 3) & 0xFFF,
                "H": ((w0 >> 1) & 0x3) + 1,
                "W": (w0 & 0x1) + 1,
            }
        elif op == OpCode.ROPE:
            operands = {
                "sa": (w0 >> 15) & 0xFFF,
                "da": (w0 >> 3) & 0xFFF,
                "len": (w0 & 0x7),
            }
        elif op in (OpCode.BARRIER, OpCode.NOP):
            pass
        elif op == OpCode.RMSNORM:
            operands = {
                "sa": (w0 >> 15) & 0xFFF,
                "da": (w0 >> 3) & 0xFFF,
                "elements": w0 & 0x7,
            }
        elif op in (OpCode.VADD, OpCode.VMUL, OpCode.VRED_MAX,
                    OpCode.VRED_SUM, OpCode.VCONV, OpCode.VRESID,
                    OpCode.VCONV_F16_I32,
                    OpCode.DMA_LDD, OpCode.DMA_STD,):
            # Generic Vector/DMA
            operands = {
                "sa": (w0 >> 15) & 0xFFF,
                "da": (w0 >> 3) & 0xFFF,
                "len": w0 & 0x7,
            }
        else:
            # Generic SFU: SOFTMAX, LAYERNORM, GELU, RELU, SILU
            operands = {
                "sa": (w0 >> 15) & 0xFFF,
                "da": (w0 >> 3) & 0xFFF,
                "len": w0 & 0x7,
            }

        return NPUInstruction(opcode=op, operands=operands)


# ── Parser: text → NPUInstruction ──────────────────────────────────

def parse_instruction(line: str) -> NPUInstruction:
    """Parse a text instruction line into NPUInstruction.

    Examples:
        "mmul wa=0, ia=1, oa=2, N=2560"
        "softmax sa=0, da=1, len=2560"
        "dma_ld dram=0x1000, sram=0x200, size=2500"
        "nop"
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return NPUInstruction(OpCode.NOP)

    # Split mnemonic and operands
    parts = line.split(None, 1)
    mnemonic = parts[0].lower()
    opcode = OpCode.from_mnemonic(mnemonic)

    operands: Dict[str, int] = {}
    if len(parts) > 1:
        for token in parts[1].split(","):
            token = token.strip()
            if "=" in token:
                key, val = token.split("=", 1)
                key = key.strip()
                val = val.strip()
                # Handle hex
                if val.startswith("0x") or val.startswith("0X"):
                    operands[key] = int(val, 16)
                else:
                    operands[key] = int(val)

    return NPUInstruction(opcode=opcode, operands=operands)


def parse_isa_program(text: str) -> List[NPUInstruction]:
    """Parse a multi-line ISA program into instruction list."""
    instrs = []
    for line in text.strip().splitlines():
        instr = parse_instruction(line)
        if instr.opcode != OpCode.NOP or instr.operands:
            instrs.append(instr)
    return instrs
