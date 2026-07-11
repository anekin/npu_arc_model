#!/usr/bin/env python3
"""软件开销评估 — 七引擎 × 三种场景对比"""
import math, sys, yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from engine.mac_engine import create_engine
from engine.compiler import NPUCompiler
from engine.isa import OpCode
from npu_sim import generate_qwen3b_trace
from models.sw_overhead import SWOverheadModel

def count_isa_per_type(program):
    from collections import Counter
    cnt = Counter()
    for instr in program:
        cnt[instr.opcode.name] += 1
    return cnt

def estimate_riscv_from_isa(op_counts):
    """从 ISA 指令流估算 RISC-V 执行开销"""
    riscv_per_op = {
        'MMUL': 4, 'DMA_LD': 6, 'SOFTMAX': 3, 'LAYERNORM': 3,
        'ROPE': 3, 'GELU': 3, 'KV_LOAD': 4, 'KV_STORE': 4,
        'BARRIER': 5, 'NOP': 1,
    }
    total = sum(cnt * riscv_per_op.get(op, 4) for op, cnt in op_counts.items())
    return int(total * 1.2 * 5)  # CPI=1.2, cycle_ratio=5

def main():
    trace = generate_qwen3b_trace(prompt_len=1)
    compiler = NPUCompiler(num_cores=1)
    program = compiler.compile_decode(trace)
    op_counts = count_isa_per_type(program)
    total_isa = len(program)
    
    # 三种场景
    sw = SWOverheadModel()
    total_tiles = sum(math.ceil(K/128) * math.ceil(N/128) for _, K, N, _, _ in trace)
    
    scenarios = {
        "A. DMA chain (理想)": sw.estimate(28, 0, total_isa),
        "B. 无 DMA chain (保守)": sw.estimate(28, total_tiles, total_isa, has_dma_chain=False),
        "C. +Host PCIe (实际)": sw.estimate(28, 0, total_isa, kv_evict_per_layer=False),
    }
    # Scenario C adds host-side overhead manually
    host_pcie_us = 100  # μs: PCIe doorbell + interrupt + cmd buffer
    
    config = yaml.safe_load(open(Path(__file__).parent / 'config' / 'npu_config.yaml'))
    
    engines = ['systolic', 'block', 'gmma', 'os_systolic', 'tensor_core', 'input_stationary', 'wmma']
    
    print("=" * 100)
    print("  软件开销评估 — 七引擎 × 四种场景")
    print("=" * 100)
    print(f"  ISA 指令: {total_isa} 条/层, Tile 总数: {total_tiles:,}")
    print()
    
    # 表头
    print(f"{'Engine':20s} {'HW tok/s':>7s} {'HW μs':>8s}  {'SW(A)%':>6s} {'SW(B)%':>6s} {'SW(C)%':>6s}  A tok/s  B tok/s  C tok/s")
    print("-" * 100)
    
    for eng_name in engines:
        cfg = yaml.safe_load(yaml.dump(config))
        cfg['mac_engine'] = dict(cfg.get('mxu', {}))
        cfg['mac_engine']['type'] = eng_name
        engine = create_engine(cfg)
        
        # HW cycles
        mxu_cycles = 0
        for M, K, N, _, _ in trace:
            r = engine.estimate(M, K, N)
            mxu_cycles += r.total_cycles
        hw_us = mxu_cycles / 1000
        hw_tok = 1e6 / hw_us if hw_us > 0 else 0
        
        # SW overhead (scenario A — DMA chain)
        sw_a = sw.estimate(28, 0, total_isa)
        sw_a_us = sw_a.total_cycles / 1000
        
        # Scenario B — no DMA chain
        sw_b = sw.estimate(28, total_tiles, total_isa, has_dma_chain=False)
        sw_b_us = sw_b.total_cycles / 1000
        
        # Scenario C — + host PCIe
        sw_c_us = sw_a_us + host_pcie_us
        
        pct_a = sw_a_us / hw_us * 100
        pct_b = sw_b_us / hw_us * 100
        pct_c = sw_c_us / hw_us * 100
        
        tok_a = 1e6 / (hw_us + sw_a_us)
        tok_b = 1e6 / (hw_us + sw_b_us)
        tok_c = 1e6 / (hw_us + sw_c_us)
        
        print(f"{eng_name:20s} {hw_tok:7.0f} {hw_us:8.0f}  {pct_a:5.1f}% {pct_b:5.1f}% {pct_c:5.1f}%  {tok_a:7.0f} {tok_b:7.0f} {tok_c:7.0f}")
    
    print()
    print("  场景说明:")
    print("    A = DMA descriptor chain (硬件自遍历) + 无 host 开销")
    print("    B = 无 DMA chain (每个 tile RISC-V 写 descriptor)")
    print("    C = DMA chain + 100μs host PCIe (实际部署)")
    print()
    print("  关键发现:")
    print(f"    1. DMA descriptor chain 至关重要 — 无 chain 时 per-tile 开销累计")
    print(f"    2. Host PCIe 开销 (100μs) 对小 batch 影响显著")
    print(f"    3. 快引擎 (block/gmma) 受 SW 开销影响最大 (HW 时间短, SW 占比高)")
    print(f"    4. 慢引擎 (systolic/wmma) SW 占比可忽略")

if __name__ == '__main__':
    main()
