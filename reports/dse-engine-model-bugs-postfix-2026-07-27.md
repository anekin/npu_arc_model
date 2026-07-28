# DSE 引擎模型 Bug 修复验证报告

> **基于**: DSE Engine Model Bug Fix Wave 1-3 (Todos 1-10)
> **检测日期**: 2026-07-27 (原始报告)
> **修复验证**: 2026-07-28
> **当前 HEAD**: `02683a9f49bc2df299d31f4af8c1446d99101fce`
> **原始报告 SHA256**: `61fe73e163f4dc61c1c746ea3a115b176c4d745bf387c7b2a4350a195d88ccd3` ✅ 未修改
> **Config SHA256**: `d3ad177cd825b7ef6342bc0f53402e61e5d4438267ae4112d7e6aca041a08217` (`sim/config/npu_config.yaml`)

---

## 一、修复总表

| BUG | 引擎 | 严重程度 | 状态 | 修复 Commit | 测试证据路径 |
|:---|:-----|:-------:|:---:|:-----------|:------------|
| BUG-DSE-001 | OS-Systolic | High | **FIXED** | `d994b08` | `.omo/evidence/task-6-os.txt` |
| BUG-DSE-002 | SystolicEngine decode | High | **FIXED** | `f5798e4` | `.omo/evidence/task-5-parity.txt` |
| BUG-DSE-003 | SystolicEngine prefill | High | **FIXED** | `f5798e4` | `.omo/evidence/task-5-parity.txt` |
| BUG-DSE-004 | TensorCore | High | **FIXED** | `cd699e3` | `.omo/evidence/task-7-tc.txt` |
| BUG-DSE-005 | GMMA (bottleneck stale) | Low | **FIXED** | `1173eff` | `.omo/evidence/task-8-gmma.txt` |
| BUG-DSE-006 | GMMA (TMA overlap stale) | Low | **FIXED** | `1173eff` | `.omo/evidence/task-8-floor.txt` |
| BUG-DSE-007 | npu_sim systolic baseline | Low | **FIXED** | `02683a9` | `.omo/evidence/task-10-verification.json` |
| BUG-DSE-008 | npu_sim block baseline | Low | **FIXED** | `02683a9` | `.omo/evidence/task-10-verification.json` |

---

## 二、Before/After 对照详情

所有修复后数值来源于 Todo 10 的端到端回归证据（`.omo/evidence/task-10-engine-ffn-down.json`, `.omo/evidence/task-10-verification.json`）。测试配置：64×64, M=1, K=11008, N=2048 (FFN_down), INT4, LPDDR5-6400 @ 51.2 GB/s (85% eff)。

### BUG-DSE-001: OS-Systolic K-reduction depth

| 指标 | Before | After | 偏差 |
|:----|:------|:-----|:----:|
| per_tile_compute | 4 cycles (缺 H=64) | 68 cycles (含 H=64) | +17× |
| total_cycles | 266,948 | 393,634 | +47.5% |
| tok/s | 3,744 | 2,540.4 | -32.1% |
| bottleneck | dma | dma | — |
| 与 Block 偏差 | 快 47.5% | 0.0% (完全一致) | ✅ |

**根因**: `os_systolic_engine.py:57` per_tile_compute = BROADCAST_SYNC_CYCLES + \_accumulate_cycles(...) 遗漏了 `self.H` (K-reduction depth=64)。修复后 OS 使用与 BlockEngine 完全相同的聚合 DMA 核算逻辑。

**修复**: `d994b08` — `fix(os-systolic): account for K-reduction depth and physical DMA aggregation`

---

### BUG-DSE-002: SystolicEngine decode drain

| 指标 | Before | After | 参考 (MXUModel) |
|:----|:------|:-----|:---------------:|
| pipeline_drain (M=1) | M + H = 129 | H × (M+1) + W = 256 | 256 |
| per_tile_compute (M=1) | 385 | H × (M+1) + W = 256 | 256 |
| 与 MXUModel 偏差 | 256 cycles/tile | 0 (byte-for-byte) | ✅ |

**根因**: `systolic_engine.py:30-31` 使用 `pipeline_drain = M + self.H`，MXUModel 使用 `H*(M+1)+W`，两套模型独立开发导致公式分叉。

**修复**: `f5798e4` — `fix(systolic): correct decode and prefill per-tile-compute formulas`

---

### BUG-DSE-003: SystolicEngine prefill drain

| 指标 | Before | After | 参考 (MXUModel) |
|:----|:------|:-----|:---------------:|
| pipeline_drain (full tile) | 2H = 128 | H = 64 | 64 |
| per_tile_compute (full tile) | H+W+2H = 320 | H+W+H = 256 | 256 |
| 与 MXUModel 偏差 | 128 cycles/tile | 0 (byte-for-byte) | ✅ |

**根因**: `systolic_engine.py:82-83` 始终使用 `pipeline_drain = self.H + self.H` (2H)，MXUModel 使用 `self.H` (full tile) 或 `M` (partial tile)。

**修复**: `f5798e4` — `fix(systolic): correct decode and prefill per-tile-compute formulas`

---

### BUG-DSE-004: TensorCore descriptor fragmentation

| 指标 | Before | After |
|:----|:------|:-----|
| total_cycles | 291,472 | 401,552 |
| compute_cycles | 115,584 | 115,584 |
| dma_cycles | 175,888 | 285,968 |
| total_descriptor_cycles | 0 (缺失) | 110,080 |
| tok/s | 3,431 | 2,490.3 |
| vs Block (2,540.4 tok/s) | **快 35%** (错误排名) | **慢 2%** (正确排名) |
| bottleneck | dma | dma |

**根因**: `tensor_core_engine.py:71-84` 缺少 per-TC descriptor setup 开销（`active_tcs * descriptor_overhead_cycles`），导致 TC 的 DMA 时间被低估，排名错误地高于 Block Engine。

**修复**: `cd699e3` — `fix(tensor-core): model per-wave descriptor fragmentation overhead`

配置参数: `dma.descriptor_overhead_cycles=5`（来自 `sim/config/npu_config.yaml`，Todo 2 暴露）。

---

### BUG-DSE-005: GMMA bottleneck stale

| 指标 | Before | After |
|:----|:------|:-----|
| per_tile_compute (M=1) | 129 (H+M+W, 未缩放) | 7 (max(1, ceil((H+M+W)×0.05)) |
| total_compute | 710,016 | 38,528 |
| total_cycles | 710,016 | 393,634 |
| bottleneck | compute | dma |
| tok/s | 1,408.4 | 2,540.4 |

**根因**: `GMMA_PIPELINE_SCALE=0.05` 定义在 `gmma_engine.py:50` 但未被使用（commit `c18fc5d` 移除了缩放）。未缩放的 per_tile_compute=129 使 GMMA 变为 compute-bound。

---

### BUG-DSE-006: GMMA TMA overlap stale

| 指标 | Before | After |
|:----|:------|:-----|
| LPDDR5 tok/s | 1,408.4 | 2,540.4 |
| HBM2e (460 GB/s) tok/s | 1,408.4 | 22,824 |
| 带宽单调性 | ❌ 无变化 | ✅ HBM2e > 2× LPDDR5 |

**根因**: 同 DSE-005。compute-bound 下带宽变化对总 cycle 零影响。修复后 total_cycles = max(total_compute, total_dma) 使用 raw DMA 作为物理下限，TMA overlap 仅作为诊断信息保留。

**修复 (DSE-005/006)**: `1173eff` — `fix(gmma): enable pipeline scaling and enforce physical raw-dma floor`

**⚠️ 未校准参数声明**: `GMMA pipeline_scale=0.05` 是基于 H100 GMMA 架构假设的经验估算值，尚未经过实测签核。该值可在 `sim/config/npu_config.yaml` 和 `sim/config/design_space.yaml` 中调整。两个 YAML 文件的默认值必须保持一致（当前均为 0.05）。

---

### BUG-DSE-007: npu_sim systolic baseline stale

| 指标 | Before (期望值) | After (实测值) | 偏差 |
|:----|:--------------:|:-------------:|:---:|
| tok/s | 11.17 | **10.1515** | 9.5% |

**根因**: 预期值 11.17 设定于 commit `fbd7a7a`（6 月 23 日），后续 SFU/Vector/DMA/NoC 模型更新累积漂移。

**修复**: `02683a9` — rebaseline to 10.15 (`pytest.approx(10.15, rel=0.01)`)。证据: `.omo/evidence/task-10-verification.json` (systolic_tok_per_s: 10.1515)。

---

### BUG-DSE-008: npu_sim block baseline stale

| 指标 | Before (期望值) | After (实测值) | 偏差 |
|:----|:--------------:|:-------------:|:---:|
| tok/s | 29.6 | **21.586** | 27% |

**根因**: BlockEngine 经历了三次重大校准（per-tile compute 修正、K-reduction depth、时间复用 M-channel），将 tok/s 从 ~30 降至 ~22。

**修复**: `02683a9` — rebaseline to 21.59 (`pytest.approx(21.59, rel=0.01)`)。证据: `.omo/evidence/task-10-verification.json` (block_tok_per_s: 21.586)。

---

## 三、修复后引擎排名 (FFN_down, M=1, 64×64, LPDDR5-6400)

| 排名 | Engine | tok/s | total_cycles | bottleneck | 面积估算 |
|:---:|:-------|:----:|:-----------:|:---------:|:--------:|
| 1 | OS-Systolic | 2,540.4 | 393,634 | dma | ~28.2 mm² |
| 1 | Block | 2,540.4 | 393,634 | dma | 28.2 mm² |
| 1 | GMMA | 2,540.4 | 393,634 | dma | 30.2 mm² |
| 4 | TensorCore | 2,490.3 | 401,552 | dma | 52 mm² |
| 5 | FSA | 1,408.4 | 710,016 | compute | ~30 mm² |
| 6 | Systolic | 946.2 | 1,056,816 | compute | 22.2 mm² |
| 7 | WMMA | 6.9 | 145,129,680 | compute | 57 mm² |

**关键变化**:
1. OS-Systolic、Block、GMMA 在 DRAM-bound 场景下性能一致（~2,540 tok/s），差距 < 0.1%
2. TensorCore 修复后正确慢于 Block（-2%），排名从错误的 "快 35%" 修正
3. GMMA 从 compute-bound (1,408 tok/s) 回到 DMA-bound (2,540 tok/s)，带宽单调性恢复
4. FSA 因 inline Softmax 开销表现为 compute-bound，排名高于 Systolic 但低于 DMA-bound 三引擎
5. Systolic 受 pipeline fill/drain 物理开销拖累（946 tok/s），不适合 M=1 decode

---

## 四、配置哈希校验

```
$ sha256sum reports/dse-engine-model-bugs-2026-07-27.md
61fe73e163f4dc61c1c746ea3a115b176c4d745bf387c7b2a4350a195d88ccd3

$ sha256sum sim/config/npu_config.yaml
d3ad177cd825b7ef6342bc0f53402e61e5d4438267ae4112d7e6aca041a08217
```

原始 dated 报告 SHA256 与预期完全一致，未被修改。

---

## 五、回归验证摘要

| 验证项 | 结果 | 证据 |
|:------|:---:|:-----|
| `pytest -q` (63 tests) | ✅ EXIT 0 | `.omo/evidence/task-10-verification.json` |
| Quick DSE (36 configs) | ✅ EXIT 0, errors=0 | `.omo/evidence/task-10-quick-dse.json` |
| Full DSE (13,440 configs) | ✅ EXIT 0, errors=0 | `.omo/evidence/task-10-full-dse.json` |
| 7-engine FFN_down benchmark | ✅ 全部正常 | `.omo/evidence/task-10-engine-ffn-down.json` |
| Block CLI 基准 (21.586 tok/s) | ✅ 偏差 0.02% | `.omo/evidence/task-10-verification.json` |
| Systolic CLI 基准 (10.1515 tok/s) | ✅ 偏差 0.015% | `.omo/evidence/task-10-verification.json` |
| 原始 dated 报告 SHA256 | ✅ 未修改 | 本节第四条 |

---

## 六、修复架构决策总结

1. **SystolicEngine 自持公式**: 不引入共享 helper、不委托 MXUModel，decode/prefill 公式独立维护，与 MXUModel 参考模型 byte-for-byte 一致已验证。
2. **OS-Systolic + H K-reduction**: 加 H=64 的 reduction depth 并改用 BlockEngine 相同的聚合 DMA 核算逻辑。
3. **TensorCore + descriptor cost**: per-wave DMA 加入 `active_tcs × descriptor_overhead_cycles` 开销，使用已有 config 参数 `dma.descriptor_overhead_cycles=5`。
4. **GMMA pipeline scale + raw-DMA floor**: 恢复 `pipeline_scale=0.05` 缩放，total_cycles 下限为 raw transferred-byte time（`max(total_compute, total_dma)`），TMA overlap 仅作为诊断信息。
5. **未校准参数**: `GMMA pipeline_scale=0.05` 为未签核的架构假设值，可在 YAML 配置中调整。两个 YAML 文件 (`npu_config.yaml`, `design_space.yaml`) 的默认值必须同步。

---

## 七、对 DSE 场景结论的影响

修复后的引擎排名**不改变**现有三场景推荐结论（`reports/arch-dse-three-scenarios.md`）：
- Block Engine（已选 S2/S3）模型本身正确，不受本次修复影响
- FSA（已选 S1）是独立引擎，不在本次 8 个 bug 范围内
- OS-Systolic / TensorCore / GMMA 排名变化只影响相对对比，不改变 Block 的首选地位

**警告**: 如果在未来 DSE 中更换阵列尺寸或带宽假设，必须使用修复后的引擎模型重新扫描。
