# Arc Model Bug Tracker

> Tracker revision: 1.4
> Last updated: 2026-07-15
> Scope: Arc Model framework、NPU Engine、DSE、场景配置、报告与可复现性

本文档是与代码一起版本化的 **Signoff 缺陷台账**。GitHub Issue 用于讨论、
分派和通知；本台账用于记录会影响 Framework、Engine 或 Scenario Signoff 的
缺陷及其关闭证据。两者都存在时，必须互相链接，状态以本台账随代码合入的
版本为准。

## 1. Bug 与模型缺口的边界

以下情况必须登记为 Bug：

- 违反物理上限、单位、单调性或确定性等已声明不变量；
- 实现不符合 Engine contract、场景需求或报告字段语义；
- 搜索遗漏、排序错误、false PASS/false FAIL；
- 同一输入无法复现，或受支持平台的正式入口无法运行；
- 已发布结果因代码错误而需要重跑。

尚未校准的经验常数、论文外推误差和暂不支持的工作负载，如果模型已明确降级
成熟度并给出 warning，应记录在 Engine manifest 的 `known_gaps`，不自动视为
Bug。一旦其行为超出声明范围、没有正确降级，或导致错误推荐，则转为 Bug。

## 2. 编号、严重级别与状态

### 2.1 编号

- 格式：`ARC-BUG-NNN`，三位十进制流水号；
- ID 永不复用，关闭或误报也保留记录；
- 新 Bug 使用当前最大编号加一；
- GitHub Issue 标题以 `[ARC-BUG-NNN]` 开头。

### 2.2 严重级别

| 等级 | 定义 | Signoff 影响 |
|---|---|---|
| P0 | 物理上限、单位、false PASS、不可复现推荐 | 阻止全部相关 Signoff，不允许 waiver |
| P1 | 主要周期路径、排序、搜索遗漏、正式入口不可用 | 阻止受影响层级，原则上修复后方可继续 |
| P2 | 已正确隔离的非主路径误差、非关键报告问题 | 可在有 owner、保护措施和期限时 waiver |
| P3 | 文档、展示和低风险可用性问题 | 不阻止 Signoff，但必须排期 |

严重级别遵循 `docs/arc-model-test-signoff-plan.md` 第 13 节。P0/P1 不能通过降低
Engine maturity 来掩盖；maturity 只表达证据成熟度，不替代缺陷管理。

### 2.3 状态流转

```text
NEW -> TRIAGED -> IN_PROGRESS -> FIXED -> VERIFIED -> CLOSED
  \        \             \          \-> REOPENED
   \        \-> DEFERRED  \-> WAIVED
    \-> DUPLICATE / NOT_A_BUG
```

| 状态 | 进入条件 |
|---|---|
| NEW | 已复现或有可信失败证据，尚未完成影响分析 |
| TRIAGED | 严重级别、组件、owner、影响范围已确认 |
| IN_PROGRESS | 修复正在开发 |
| FIXED | 修复完成，但回归或独立复核尚未完成 |
| VERIFIED | 原复现失败、针对性回归及相关测试均通过 |
| CLOSED | 修复和测试已合入目标分支，受影响报告已处置 |
| REOPENED | 修复后再次复现或影响范围扩大 |
| DEFERRED | 暂缓处理；必须写明理由、owner 和复审日期 |
| WAIVED | 仅 P2/P3；必须记录保护措施、owner 和到期日 |
| DUPLICATE / NOT_A_BUG | 保留证据并链接主 Bug 或模型契约 |

## 3. 每条 Bug 的必填信息

每条记录至少包含：ID、标题、严重级别、状态、发现日期、组件、影响版本、
发现测试、复现方法、期望/实际行为、根因、影响范围、修复、回归测试、owner、
修复提交、GitHub Issue、验证日期和 Signoff 处置。缺少根因或回归测试的修复
不能进入 `VERIFIED`；没有合入提交的记录不能进入 `CLOSED`。

## 4. 当前摘要

| ID | 严重级别 | 状态 | 组件 | 标题 | Signoff 处置 |
|---|---|---|---|---|---|
| ARC-BUG-001 | P0 | CLOSED | GMMA/DMA | TMA overlap 突破物理 DMA ceiling | 已在 `2897dff` 合入并关闭 |
| ARC-BUG-002 | P1 | CLOSED | GMMA scheduler | Weight-cache pair 可能产生性能倒退 | 已在 `2897dff` 合入并关闭 |
| ARC-BUG-003 | P0 | CLOSED | EngineResult/metrics | Engine `ops` 单位不一致 | 已在 `2897dff` 合入；历史指标按范围处置 |
| ARC-BUG-004 | P1 | CLOSED | Windows CLI | GBK stdout 导致 DSE 输出前崩溃 | 已在 `2897dff` 合入并关闭 |
| ARC-BUG-005 | P1 | CLOSED | WC PPA/reporting | WC 成本缺失且 ON/OFF 被合并 | 已在 `2ddcccf` 合入并关闭 |
| ARC-BUG-006 | P1 | CLOSED | Systolic WC scheduler | WC pair 缺少单调 fallback | 已在 `2ddcccf` 合入并关闭 |

当前没有 `NEW`、`TRIAGED`、`IN_PROGRESS` 或 `REOPENED` 的 P0/P1；六项修复及
回归测试已由提交 `2897dff` 和 `2ddcccf` 合入目标功能分支，状态均为 `CLOSED`。

## 5. 缺陷记录

### ARC-BUG-001 — GMMA TMA overlap 突破物理 DMA ceiling

| 字段 | 内容 |
|---|---|
| Severity / Status | P0 / CLOSED |
| Detected / Verified | 2026-07-15 / 2026-07-15 |
| Component / Owner | `sim/engine/gmma_engine.py` / Arc Model maintainer |
| Affected versions | v3.5 及以前；first-bad version 待历史二分 |
| Detected by | Engine admission physical microbench、bandwidth property test |
| Reproduction | 对 DMA-bound GMMA shape 比较传输字节、物理带宽和 `total_cycles` |
| Expected | Kernel 周期不得小于完整数据传输的物理周期 |
| Actual | 旧模型以 `total_dma × (1-TMA_OVERLAP)` 作为 roofline，可能虚构带宽 |
| Root cause | 把计算/DMA 并发收益错误地施加到物理 DMA 完成时间，而非调度重叠 |
| Impact | 可能高估 GMMA TPS、改变 Engine 排序；所有旧 GMMA DSE 结果不作 Signoff 依据 |
| Fix | 使用 `max(total_compute, total_dma)`；overlap 不再缩短物理传输下限 |
| Regression | `test_every_engine_microbench_is_finite_and_physically_bounded`；全量 118 tests |
| Evidence | `sim/tests/test_engine_admission.py`、`sim/results/engine_admission_audit_v36.json` |
| Commit / GitHub Issue | `2897dffa0004d1dfa9626f4bcd1bea8e49ff3530` / not created |
| Signoff disposition | 已关闭；涉及 GMMA 的历史报告必须重跑或标记 superseded |

### ARC-BUG-002 — GMMA weight-cache pair 可能产生性能倒退

| 字段 | 内容 |
|---|---|
| Severity / Status | P1 / CLOSED |
| Detected / Verified | 2026-07-15 / 2026-07-15 |
| Component / Owner | `sim/engine/gmma_engine.py` / Arc Model maintainer |
| Affected versions | v3.5 及以前；first-bad version 待历史二分 |
| Detected by | pair scheduler non-regression property test |
| Reproduction | 比较 `estimate_weight_cache_pair(M,K,N)` 与两次独立 `estimate` |
| Expected | 可选 pair 优化不得比合法的独立调度更慢 |
| Actual | 某些 shape 下合并 Gate/Up 的估算周期大于两个独立 GEMM |
| Root cause | scheduler 只计算合并路径，没有保留独立 GEMM 作为合法 fallback |
| Impact | 低估 GMMA FFN 性能，并破坏“启用优化不应倒退”的单调性 |
| Fix | pair 较慢时返回两个独立 GEMM，并记录 `scheduler_fallback` 原因 |
| Regression | `test_every_engine_bandwidth_preload_pair_and_ppa_properties`；全量 118 tests |
| Evidence | `sim/tests/test_engine_admission.py`、Engine admission audit |
| Commit / GitHub Issue | `2897dffa0004d1dfa9626f4bcd1bea8e49ff3530` / not created |
| Signoff disposition | 已关闭；GMMA 仍保持 M1，不因本 Bug 修复自动升级 maturity |

### ARC-BUG-003 — EngineResult.ops 单位不一致

| 字段 | 内容 |
|---|---|
| Severity / Status | P0 / CLOSED |
| Detected / Verified | 2026-07-15 / 2026-07-15 |
| Component / Owner | Systolic、Tensor Core、WMMA、GMMA、Input Stationary / Arc Model maintainer |
| Affected versions | v3.5 及以前；first-bad version 待历史二分 |
| Detected by | 10-Engine 统一 microbench contract test |
| Reproduction | 对同一 GEMM 检查 `result.ops == M×K×N×ops_per_mac` |
| Expected | 所有 Engine 的 `ops` 均表示 operation count |
| Actual | 五个 Engine 返回 MAC count，Block/OS/FSA 返回 operation count |
| Root cause | EngineResult 没有统一的单位契约，各 Engine 独立实现时采用了不同定义 |
| Impact | Engine 间 TOPS/ops/利用率证据不可直接比较；周期和 TPS 主路径不因该字段修正而改变 |
| Fix | 五个 Engine 统一乘 `ops_per_mac`，并同步修正 ideal-cycle 分母以保持利用率物理含义 |
| Regression | `test_every_engine_microbench_is_finite_and_physically_bounded` 中的统一 ops 断言 |
| Evidence | `sim/tests/test_engine_admission.py`、全量 118 tests |
| Commit / GitHub Issue | `2897dffa0004d1dfa9626f4bcd1bea8e49ff3530` / not created |
| Signoff disposition | 已关闭；引用旧 `ops`/TOPS 字段的报告需重算，纯周期/TPS 结果无需仅因本项重跑 |

### ARC-BUG-004 — Windows GBK stdout 导致 CLI 崩溃

| 字段 | 内容 |
|---|---|
| Severity / Status | P1 / CLOSED |
| Detected / Verified | 2026-07-15 / 2026-07-15 |
| Component / Owner | `sim/design_space_explorer.py` CLI / Arc Model maintainer |
| Affected versions | v3.5 及以前的 Windows GBK/redirected stdout 路径 |
| Detected by | Windows 场景 DSE 集成运行 |
| Reproduction | 在 GBK stdout 下运行正式 DSE，使报告输出 Unicode 单位或状态字符 |
| Expected | CLI 完成搜索并写出 UTF-8 JSON |
| Actual | `UnicodeEncodeError`，搜索结果输出前退出 |
| Root cause | CLI 假设 stdout 支持全部 Unicode，没有固定正式输出编码 |
| Impact | Windows 正式入口不可用；计算结果本身不受影响 |
| Fix | `main()` 启动时将 stdout reconfigure 为 UTF-8，错误策略为 replacement |
| Regression | Windows 下完成场景 A 与 Agent 全量 DSE；全量 118 tests |
| Evidence | `sim/results/scenario_a_lpddr5_v36_engine_evidence.json`、Agent DSE JSON |
| Commit / GitHub Issue | `2897dffa0004d1dfa9626f4bcd1bea8e49ff3530` / not created |
| Signoff disposition | 已关闭；后续补跨平台 subprocess CLI gate |

### ARC-BUG-005 — Weight Cache hardware cost omitted and variants collapsed

| Field | Content |
|---|---|
| Severity / Status | P1 / CLOSED |
| Detected / Verified | 2026-07-15 / 2026-07-15 |
| Component / Owner | `sim/engine/ppa_model.py`, DSE reporting / Arc Model maintainer |
| Affected versions | v3.6 and earlier |
| Detected by | Review of WC ON/OFF architecture implementation cost |
| Reproduction | Evaluate the same WC-capable engine/configuration with `weight_cache=false/true` and inspect PPA plus `engine_comparison` |
| Expected | WC ON has explicit nonzero area/power cost and is reported separately from WC OFF |
| Actual | WC ON gained FFN performance at identical area/power; engine-level reporting collapsed both variants |
| Root cause | PPA ignored `optimizations.weight_cache`; reporting grouped only by engine name |
| Impact | Underestimated WC hardware cost and hid implementation alternatives in DSE reports |
| Fix | Configurable per-engine WC PE-area proxy, shared area/power accounting, structured hardware identity, and `engine_variant_comparison` |
| Regression | `test_weight_cache_hardware_variant_has_nonzero_ppa_cost`; `test_weight_cache_variants_are_not_collapsed` |
| Evidence | Updated LPDDR5 scenario A and Agent DSE artifacts; 125-test full regression |
| Commit / GitHub Issue | `2ddcccf` / not created |
| Signoff disposition | Closed in target branch; WC PPA remains architecture-stage until independent calibration |

### ARC-BUG-006 — Systolic WC pair could regress selected FFN shapes

| Field | Content |
|---|---|
| Severity / Status | P1 / CLOSED |
| Detected / Verified | 2026-07-15 / 2026-07-15 |
| Component / Owner | `sim/engine/systolic_engine.py` / Arc Model maintainer |
| Affected versions | v3.6 and v3.7 pre-fix working tree |
| Detected by | WC ON/OFF Agent report review |
| Reproduction | Compare WC pair with two independent GEMMs for `(M,K,N)=(1,2048,11008)` on the selected large Systolic array |
| Expected | Optional WC scheduling never takes more cycles than the legal two-GEMM fallback |
| Actual | WC ON produced 7.60 TPS while WC OFF produced 7.61 TPS at the selected Agent point |
| Root cause | Systolic dual-weight schedule lacked the fallback already present in Block/GMMA |
| Impact | Small decode regression and misleading WC comparison for some shapes |
| Fix | Return two independent GEMMs when the combined WC schedule is not faster |
| Regression | `test_weight_cache_pair_is_monotonic_for_decode_and_agent_ffn_shapes` |
| Evidence | Updated scenario A/Agent WC variant DSE artifacts; 125-test full regression |
| Commit / GitHub Issue | `2ddcccf` / not created |
| Signoff disposition | Closed in target branch; monotonic fallback is a permanent regression gate |

## 6. 新 Bug 登记模板

复制以下内容到本文件末尾，同时创建 GitHub Issue：

```markdown
### ARC-BUG-NNN — Short title

| 字段 | 内容 |
|---|---|
| Severity / Status | P? / NEW |
| Detected / Verified | YYYY-MM-DD / — |
| Component / Owner | path-or-component / owner |
| Affected versions | version or commit range |
| Detected by | test, report or reviewer |
| Reproduction | minimal deterministic steps |
| Expected | declared contract |
| Actual | observed behavior |
| Root cause | pending until triaged |
| Impact | metrics, engines, scenarios and reports |
| Fix | pending |
| Regression | required test name |
| Evidence | logs/results/report links |
| Commit / GitHub Issue | pending / URL |
| Signoff disposition | blocking scope or waiver |
```

## 7. 每次 Signoff 的 Bug 检查

1. 搜索所有非 `CLOSED` Bug，并按 P0→P3 审核；
2. 确认相关 P0/P1 已 `VERIFIED`，且目标分支发布前已 `CLOSED`；
3. 检查 P2/P3 waiver 的 owner、保护措施和到期日；
4. 将修复关联到至少一个稳定回归用例；
5. 重跑受影响的 Engine/Scenario，并处置历史报告；
6. 在 Signoff 报告中附本台账 revision、未关闭列表和 waiver 列表。

## 8. Revision history

| Revision | Date | Change |
|---|---|---|
| 1.0 | 2026-07-15 | 建立正式流程并登记 ARC-BUG-001～004 |
| 1.1 | 2026-07-15 | `2897dff` 合入修复，ARC-BUG-001～004 更新为 CLOSED |
| 1.2 | 2026-07-15 | 登记并验证 ARC-BUG-005；WC ON/OFF 独立建模与报告 |
| 1.3 | 2026-07-15 | 登记并验证 ARC-BUG-006；Systolic WC 调度增加单调 fallback |
| 1.4 | 2026-07-15 | `2ddcccf` 合入 WC 修复，ARC-BUG-005/006 更新为 CLOSED |
