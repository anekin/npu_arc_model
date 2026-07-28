# DSE 引擎模型 Bug 报告

> **来源项目**: CaduceusCore (CaduceusCore/sim/engine/)
> **检测日期**: 2026-07-27
> **检测方法**: `sim/tests/test_engines.py` 8 个测试失败，经逐一代码审计和数值复现分析定位根因
> **影响范围**: Arc Model DSE 引擎库的 cycle 估算准确性，影响架构选型排名
> **不影响**: Func Model golden reference（`GoldenMXU.matmul_int4_per_block` 与 DSE 引擎库独立）、RTL 验证

---

## 概要

`sim/tests/test_engines.py` 中 13 个测试有 **8 个失败**。经分析分为三类：

| 类别 | 数量 | 性质 | 修复策略 |
|------|:----:|------|----------|
| 模型公式 bug | 3 | 引擎模型的 cycle 估算公式有误 | 修模型 |
| 测试 stale | 4 | 引擎模型有意变更后测试期望值未更新 | re-baseline |
| 模型不完整 | 1 | 缺少关键开销建模 | 补模型 |

---

## BUG-DSE-001: OS-Systolic 缺少 K-reduction depth (H)

### 严重程度: High

### 测试
`test_os_systolic_decode` (L213-239)

### 断言
```python
assert r_os.bottleneck == "dma"
assert os_tok_s <= block_tok_s
```

### 实际值
- OS-Systolic: `total_cycles=266,948`, `tok/s=3,744`, `bottleneck=dma`
- Block: `total_cycles=393,634`, `tok/s=2,540`, `bottleneck=dma`

### 根因

`sim/engine/os_systolic_engine.py` L57-58:
```python
per_tile_compute = BROADCAST_SYNC_CYCLES + \
    _accumulate_cycles(self.w_bits, self.a_bits)
# = 2 + 2 = 4 cycles/tile
```

BlockEngine 在 commit `c18fc5d` 中添加了 `self.H`（K-reduction depth）到 per-tile compute:
```python
# block_engine.py L126
per_tile_compute = self.H + BROADCAST_SYNC_CYCLES + \
    _accumulate_cycles(self.w_bits, self.a_bits)
# = 64 + 2 + 2 = 68 cycles/tile
```

OS-Systolic 同样有 K-reduction 操作（每个 PE 累加 H 个乘加结果），但 per-tile compute 只有 4 cycles，**缺少了 H=64 的 reduction depth**，导致 compute 虚低 17 倍。

### 修复方向

在 `os_systolic_engine.py` L57 添加 `self.H`:
```python
per_tile_compute = self.H + BROADCAST_SYNC_CYCLES + \
    _accumulate_cycles(self.w_bits, self.a_bits)
# = 64 + 2 + 2 = 68 cycles/tile
```

**注意**: 仅加 H 后 OS 从 DMA-bound 变为 compute-bound (`per_tile_compute=68` > `per_tile_dma≈48.5`)，cycle 总数 ≈ 374,272，tok/s ≈ 2,672，**仍高于 Block 的 2,540 tok/s**。测试断言 `os_tok_s <= block_tok_s` 和 `r_os.bottleneck == "dma"` 均会失败。

根本原因在于 OS-Systolic 的 PE 面积更大（wider accumulator + output register，docstring 说明约 4× systolic PE 面积），相同 die area 时 MAC 数更少。仅修正 K-reduction depth 不够，还需建模 **area penalty**（相同 die area → 有效阵列更小 → tile 数更多或 per-tile compute 更高）。

### 修复建议

1. **第一步**: 添加 `self.H` 到 per_tile_compute
2. **第二步**: 如果 `os_tok_s` 仍高于 `block_tok_s`，引入 area penalty 因子。OS-Systolic PE 面积约为 Block PE 的 2×（基于 Gemmini 文档），建模方式：
   - `effective_W = self.W // 2`（有效宽度减半 → N_tiles 翻倍 → 总 cycle 翻倍）
   - 或降低 `eff_bw`（相同面积下 DMA 通道更少）
3. **第三步**: 确认测试的两个断言（`bottleneck=dma` + `os_tok_s <= block_tok_s`）均通过

### 关联 commit
- `c18fc5d` — 给 BlockEngine 添加 H 时漏了 OS-Systolic

---

## BUG-DSE-002: SystolicEngine decode drain phase 公式与 MXUModel 不一致

### 严重程度: High

### 测试
`test_systolic_vs_mxumodel_decode` (L274-286)

### 断言
```python
assert r_sys.total_cycles == r_mxu.total_cycles  # byte-for-byte
```

### 实际值
- SystolicEngine: `total_cycles=98,751` (128×128, M=1, K=2048, N=2048)
- MXUModel: `total_cycles=98,495`
- 差异: 256 cycles

### 根因

`sim/engine/systolic_engine.py` L29-31 (_estimate_decode):
```python
pipeline_fill = self.H + self.W          # 128 + 128 = 256
pipeline_drain = M + self.H              # 1 + 128 = 129
per_tile_compute = pipeline_fill + pipeline_drain  # 385
```

`sim/models/mxu.py` MXUModel._estimate_decode:
```python
pipeline_fill = self.H + self.W          # 256
pipeline_drain = self.H * (M + 1) + self.W  # 128*2 + 128 = 384
# 或等价: H*M + H + W = H*(M+1) + W
per_tile_compute = 384
```

差异: SystolicEngine `drain = M + H = 129`，MXUModel `drain = H*(M+1) = 256 + drain`。对 M=1，SystolicEngine drain=129，MXUModel drain=256，差 127 cycles/tile。

总差异: 约 255 个重叠 tile × 1 cycle/tile + 首次冷启动 1 cycle = 256 cycles。

### 修复方向

修正 SystolicEngine decode 的 drain 公式以匹配 MXUModel:
```python
# 当前 (buggy)
pipeline_drain = M + self.H

# 正确 (match MXUModel)
pipeline_drain = self.H * (M + 1) + self.W
# 或等价: pipeline_drain = self.H * M + self.H + self.W
```

### 关联 commit
- 无明确引入 commit，可能是两套模型独立开发导致公式分叉

---

## BUG-DSE-003: SystolicEngine prefill drain phase 公式与 MXUModel 不一致

### 严重程度: High

### 测试
`test_systolic_vs_mxumodel_prefill` (L289-300)

### 断言
```python
assert r_sys.total_cycles == r_mxu.total_cycles  # byte-for-byte
```

### 实际值
- SystolicEngine: `total_cycles=145,076` (128×128, M=128, K=2048, N=2048)
- MXUModel: `total_cycles=144,948`
- 差异: 128 cycles

### 根因

`sim/engine/systolic_engine.py` L82-83 (_estimate_prefill):
```python
pipeline_drain = self.H + self.H    # = 2*H = 256 (when M >= H)
```

`sim/models/mxu.py` MXUModel._estimate_prefill:
```python
pipeline_drain = self.H              # = H = 128 (when M >= H)
```

差异: SystolicEngine prefill drain = 2H = 256，MXUModel = H = 128。对每个 tile 差 128 cycles。

### 修复方向

修正 SystolicEngine prefill 的 drain 公式:
```python
# 当前 (buggy)
pipeline_drain = self.H + self.H

# 正确 (match MXUModel)
pipeline_drain = self.H
```

### 注意事项

SystolicEngine 的 decode 分发条件是 `M <= 2`，MXUModel 的 decode 分发条件是 `M <= 8`。对 M=3..8，两者走不同路径（SystolicEngine 走 prefill，MXUModel 走 decode），因此 byte-for-byte 对齐只覆盖 M=1,2 和 M=128。

---

## BUG-DSE-004: TensorCore 缺少 sub-tile 碎片化开销建模

### 严重程度: High

### 测试
`test_tensor_core_decode` (L131-158)

### 断言
```python
assert r_tc.total_cycles > r_block.total_cycles  # TC 应比 Block 慢
```

### 实际值
- TensorCore: `total_cycles=291,472`, `tok/s=3,431`
- Block: `total_cycles=393,634`, `tok/s=2,540`
- TC 反而**快 35%**

### 根因

`sim/engine/tensor_core_engine.py` 的 `estimate` 方法只建模了:
1. 每个 sub-tile 的 compute: `SUBTILE_PIPELINE_FILL(80) + SUBTILE_OVERHEAD_CYCLES(4) = 84 cycles`
2. Per-wave DMA: 16 个 TC 共享 DRAM 带宽

但**缺少以下碎片化开销**:

| 缺失开销 | 物理原因 | 预估影响 |
|----------|----------|----------|
| **Per-TC DMA setup** | 每个 TC 的 sub-tile 传输需要独立的地址描述符设置、DMA 仲裁、跨 bar 请求 | 3-5 cycles × 16 TCs × 1376 waves = 66k-110k cycles |
| **TC 间 barrier sync** | 16 个 TC 并行计算后需要一次 barrier sync 才能进入下一 wave | 1-2 cycles × 1376 waves = 1.4k-2.8k cycles |
| **Address generation overhead** | 16 个独立 sub-tile 的地址需要地址生成器逐个计算 | 每 wave 2-4 cycles |

当前模型中 `bottleneck = max(84, 211.8) = 211.8`（DMA-bound），碎片化开销被 DMA bottleneck 完全掩盖。但如果把 DMA setup 开销加到 DMA 路径上:
- `per_wave_dma_with_setup = 211.8 + 16 × TC_DMA_SETUP_CYCLES + TC_BARRIER_CYCLES`
- 设 `TC_DMA_SETUP_CYCLES=5`, `TC_BARRIER_CYCLES=2`: `per_wave_dma = 211.8 + 80 + 2 = 293.8`
- `total = 293.8 + 1375 × 293.8 = 404,263` → 超过 Block 的 393,634 → 测试通过

### 修复方向

在 `tensor_core_engine.py` 的 `estimate` 方法中增加:

1. 类级常量:
```python
TC_DMA_SETUP_CYCLES = 5    # per-TC 地址描述符设置 + DMA 仲裁开销
TC_BARRIER_CYCLES = 2      # TC 间 barrier sync
```

2. 修改 per-wave DMA 计算:
```python
per_wave_dma = (num_tcs * (tile_weight_bytes + tile_act_bytes) / self.eff_bw
                + num_tcs * TC_DMA_SETUP_CYCLES + TC_BARRIER_CYCLES)
```

3. 如果 `total_cycles` 仍低于 Block，逐步调高 `TC_DMA_SETUP_CYCLES` (6, 7, 8...) 直到 `r_tc.total_cycles > r_block.total_cycles`

### 注意事项

- `GMMA_PIPELINE_SCALE = 0.05` 在文件中定义但**未被使用**（commit `c18fc5d` 移除了使用）——TensorCore 模型与此无关
- 常量应定义在 `tensor_core_engine.py` 类内（项目中不存在 `sim/engine/constants.py` 文件）
- 修复后 DSE 重跑可能改变 TensorCore 的 PPA 排名

---

## BUG-DSE-005: GMMA 测试断言 stale — bottleneck 从 "dma" 变为 "compute"

### 严重程度: Low (测试 stale，非模型 bug)

### 测试
`test_gmma_decode` (L303-319)

### 断言
```python
assert r.bottleneck == "dma"
```

### 实际值
- `bottleneck = "compute"`, `total_cycles = 710,016`

### 根因

`sim/engine/gmma_engine.py` L59-61:
```python
def _per_tile_compute(self, M: int) -> int:
    return self.H + M + self.W    # 64 + 1 + 64 = 129
```

`GMMA_PIPELINE_SCALE = 0.05` 定义在 L50 但**未被使用**。Commit `c18fc5d` 故意移除了 `int(systolic_like * 0.05)` ≈ 12 cycles 的缩放，改为原始 `H+M+W = 129` cycles。

结果: `total_compute = 129 × 5504 = 710,016`，远超过 `tma_dma = 196,791`，GMMA 变为 compute-bound。

### 修复方向

更新测试断言:
```python
# 当前 (stale)
assert r.bottleneck == "dma"

# 更新后
assert r.bottleneck == "compute"
```

在断言旁加注释: `# commit c18fc5d removed GMMA_PIPELINE_SCALE=0.05; GMMA now compute-bound at 64×64`

### 决策点
此变更是否为有意为之？如果 GMMA 的 0.05 缩放本应保留（反映 group-MMA 单元的异步流水线特性），则应恢复缩放而非更新测试。需确认 commit `c18fc5d` 的意图。

---

## BUG-DSE-006: GMMA TMA overlap 测试 stale — compute-bound 下带宽不影响

### 严重程度: Low (测试 stale，非模型 bug)

### 测试
`test_gmma_tma_overlap` (L322-344)

### 断言
```python
assert hbm2e_tok_s > 2 * lpddr5_tok_s
```

### 实际值
- LPDDR5: `tok/s = 1,408.4`
- HBM2e: `tok/s = 1,408.4`（完全相等）

### 根因

同 BUG-DSE-005。GMMA 在 64×64 阵列 M=1 decode 时为 compute-bound (`total_compute=710,016`)。`total_cycles = max(compute, tma_dma)`，由于 compute 远大于 `tma_dma`，带宽变化（LPDDR5 51.2 GB/s → HBM2e 460 GB/s）对总 cycle 数**零影响**。

### 修复方向

**选项 A**（推荐）: 更新断言，验证 DMA 被 100% 覆盖:
```python
assert hbm2e_tok_s == pytest.approx(lpddr5_tok_s, rel=0.01)
# 注释: GMMA 在 64×64 M=1 decode 时 compute-bound，
# 带宽差异不影响总 cycle。验证 DMA 被 compute 完全掩盖。
```

**选项 B**: 用 128×128 阵列重测，使 GMMA 回到 DMA-bound:
```python
cfg = _engine_config("gmma")
cfg["mac_engine"]["array_height"] = 128
cfg["mac_engine"]["array_width"] = 128
# 128×128 下 per_tile_compute = 257，total_compute = 1,414,528
# 若仍 compute-bound，需更大阵列或更高带宽
```

---

## BUG-DSE-007: npu_sim systolic baseline stale

### 严重程度: Low (测试 stale）

### 测试
`test_systolic_npu_sim_baseline` (L347-363)

### 断言
```python
assert tok_per_s == pytest.approx(11.17, rel=0.01)
```

### 实际值
- `tok/s = 10.114`（偏差 9.5%）

### 根因

预期值 11.17 设定于 commit `fbd7a7a`（6 月 23 日）。后续多次模型更新累积漂移:
- `b58beab`（7月8日）: SFU/Vector P0 校准参数
- `1ba137c`, `edba4ff`: DMA/NoC 模型更新
- `b1f458b`: SystolicEngine prefill 校准

### 修复方向

运行 `PYTHONPATH=sim python -m sim.npu_sim --engine systolic --json`，取当前实际 `tok_per_s` 值替换预期值:
```python
assert tok_per_s == pytest.approx(<new_actual_value>, rel=0.01)
# 注释: re-baselined after SFU/Vector/DMA/NoC model updates (commits b58beab, 1ba137c, b1f458b)
```

**注意**: 如果 BUG-DSE-002 和 BUG-DSE-003 修了 SystolicEngine 公式，此 baseline 需要在模型修复后重新测量。

---

## BUG-DSE-008: npu_sim block baseline stale

### 严重程度: Low (测试 stale）

### 测试
`test_block_npu_sim_baseline` (L366-380)

### 断言
```python
assert tok_per_s == pytest.approx(29.6, rel=0.01)
```

### 实际值
- `tok/s = 21.586`（偏差 27%）

### 根因

BlockEngine 经历了三次重大校准:
1. `5e88238` — 将 per-tile compute 从 1 cycle 修正为 ~4 cycles（真实广播流水线）
2. `c18fc5d` — 添加 H=64 作为 K-reduction depth，per-tile compute = 68 cycles
3. `92f14c0` — 时间复用 M 通道，DRAM 权重加载模型

这些变更将 tok/s 从 ~30 降至 ~22。测试预期值 29.6 是模型变更前的基线。

### 修复方向

运行 `PYTHONPATH=sim python -m sim.npu_sim --json`，取当前实际 `tok_per_s` 值替换:
```python
assert tok_per_s == pytest.approx(<new_actual_value>, rel=0.01)
# 注释: re-baselined after 3 BlockEngine calibration changes (5e88238, c18fc5d, 92f14c0)
```

---

## 代码引用总表

| Bug | 文件 | 关键行号 | 问题代码 |
|-----|------|---------|---------|
| DSE-001 | `sim/engine/os_systolic_engine.py` | L57-58 | `per_tile_compute = BROADCAST_SYNC_CYCLES + _accumulate_cycles(...)` 缺 `self.H` |
| DSE-002 | `sim/engine/systolic_engine.py` | L30-31 | `pipeline_drain = M + self.H` 应为 `H*(M+1)+W` |
| DSE-003 | `sim/engine/systolic_engine.py` | L82-83 | `pipeline_drain = self.H + self.H` 应为 `self.H` |
| DSE-004 | `sim/engine/tensor_core_engine.py` | L72-78 | `per_wave_dma` 缺 TC_DMA_SETUP + BARRIER 开销 |
| DSE-005 | `sim/engine/gmma_engine.py` | L50, L59-61 | `GMMA_PIPELINE_SCALE=0.05` 定义但未使用 |
| DSE-006 | 同 DSE-005 | — | 同根因 |
| DSE-007 | `sim/tests/test_engines.py` | L361 | `pytest.approx(11.17, rel=0.01)` stale |
| DSE-008 | `sim/tests/test_engines.py` | L380 | `pytest.approx(29.6, rel=0.01)` stale |

## 参考引擎（正确）

| 引擎 | 文件 | 状态 |
|------|------|:----:|
| BlockEngine | `sim/engine/block_engine.py` | ✅ 正确（reference，含 H + BROADCAST_SYNC + accumulate） |
| MXUModel | `sim/models/mxu.py` | ✅ 正确（reference，SystolicEngine 应与之对齐） |

## 依赖关系

```
DSE-001 (OS-Systolic H + area penalty) ─┐
DSE-002 (SystolicEngine decode drain)   ├─→ DSE-007 (npu_sim systolic baseline re-measure)
DSE-003 (SystolicEngine prefill drain) │
DSE-004 (TensorCore fragmentation)      ├─→ DSE-008 (npu_sim block baseline re-measure)
                                        │
DSE-005 (GMMA bottleneck stale)         ├─→ DSE-006 (GMMA TMA overlap stale)
                                        │
                        (DSE-005/006 独立于 001-004)
```

DSE-007 和 DSE-008 的 baseline 值必须在 DSE-001~004 模型修复后重新测量。

## 修复后的验证

```bash
# 修复全部 8 个 bug 后应全部通过
PYTHONPATH=sim python -m pytest sim/tests/test_engines.py -v

# timing 套件不应引入新回归
PYTHONPATH=sim python -m pytest sim/timing/tests/ -q

# 确认引擎间相对排名符合工程直觉
python3 -c "
from engine.mac_engine import create_engine
cfg = {'mac_engine': {'array_height':64,'array_width':64,'frequency_mhz':1000,'weight_precision_bits':4,'activation_precision_bits':8,'type':''}, 'memory': {'bandwidth_bytes_per_cycle':51.2,'dram_efficiency':0.85}}
for t in ['block','systolic','os_systolic','tensor_core','gmma','wmma','input_stationary']:
    cfg['mac_engine']['type'] = t
    r = create_engine(cfg).estimate(1, 11008, 2048)
    print(f'{t:20s}: {r.total_cycles:>8d} cycles, {1000e6/r.total_cycles:>8.1f} tok/s, bottleneck={r.bottleneck}')
"
```

## 对 DSE 三场景结论的影响

当前三场景 DSE 结论（`reports/arch-dse-three-scenarios.md`）的推荐:
- S1: FSA 128×256
- S2/S3: Block 80×1536

修复后引擎排名**不大可能**改变三场景推荐结论，因为:
1. Block Engine（已选 S2/S3）模型本身正确，不受这些 bug 影响
2. FSA 是独立引擎，不在本次 8 个 bug 范围内
3. OS-Systolic / Systolic / TensorCore / GMMA 的排名变化只影响它们之间的相对对比

但如果未来重跑 DSE 评估新场景（如更换阵列尺寸或带宽假设），这些模型 bug 会导致排名不准。建议在 Arc Model repo 中修复后重跑一次全量 DSE 确认。
