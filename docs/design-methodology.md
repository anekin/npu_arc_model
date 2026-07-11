# CaduceusCore 设计方法论

## 核心原则

**Func Model 是 Golden Reference。RTL 按照 Func Model 的接口和行为来实现，而不是反过来。**

```
Func Model → 定义 Spec（硬件应该做什么）
    ↓
RTL → 按 Func Model 接口实现
    ↓
验证 → RTL 输出 == Func Model Golden Reference
```

## 铁律

1. **Func Model 定义设计意图**：如果 Func Model 的行为与 RTL 不一致，优先修改 RTL。
2. **RTL 是实现细节**：RTL 的架构选择（如累加器数量、FSM 状态）服务于 Func Model 定义的 Spec。
3. **验证是单向的**：RTL vs Func Model 对比中，Func Model 是答案，RTL 是被测试对象。

## 示例

| 场景 | 错误做法 | 正确做法 |
|------|----------|----------|
| RTL 不支持 batch 权重复用 | 修改 Func Model 加 M-tiling 去匹配 RTL | Func Model 保持权重共享 Spec，RTL 加多累加器支持 |
| RTL cycle 数多于 Func Model | 调高 Func Model 的 cycle 估算 | 把 RTL cycle 差异记录为实现差距，排入优化计划 |
| Func Model 的 DDR 模型与 RTL AXI trace 不符 | 改 Func Model 去拟合 trace | 用 trace 校准模型参数，但不改变 Spec 行为 |
