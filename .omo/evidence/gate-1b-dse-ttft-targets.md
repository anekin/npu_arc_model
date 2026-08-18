# Gate 1b DSE TTFT 目标值

## 目标配置

Block 64×64 @ 1 GHz, LPDDR5-64b, INT4, qwen2.5-3b（36 layers）

## M = 128

- wc=True: **ttft_ms = 3211.00** → Func Model 3911.05 ms → 比值 **1.218** ✅
- wc=False: **ttft_ms = 6457.39** → Func Model 3911.05 ms → 比值 **0.606**（参考）
- 证据：`sim/../.omo/evidence/task-7-arc-prefill-ttft-dse-m128.json`
- SHA-256: `b33d2e3175210421bac5b9462580e465ca8aae3e2e619b7e180f8cf2b7af6ddd`

## M = 2000

- wc=True: **ttft_ms = 50171.91** → Func Model 63924.19 ms → 比值 **1.274** ✅
- wc=False: **ttft_ms = 100896.77** → Func Model 63924.19 ms → 比值 **0.634**（参考）
- 证据：`sim/../.omo/evidence/task-8-arc-prefill-ttft-dse-m2000.json`
- SHA-256: `a5540350d4443a3688eed226b58496b0bdfec3c2d86f8c54673a6269543aa858`

## 判定规则

Gate 1b PASS  iff  `0.5 <= Func_Model_TTFT / Arc_TTFT <= 2.0`

主目标取 **wc=True**；wc=False 仅作参考。

## 实施说明

- KV prefill 写回简化为 0 cycles（与参考实现一致）。
- Arc Model 已按 qwen2.5-3b spec.layers=36 计算；历史 `_NUM_LAYERS=28` 已修正。
- 与参考实现（2649.49 ms @ M=128 28层）的差异来源：层数 36 vs 28，以及引擎公式修正。
- 文档同步：`docs/arc_vs_func.md` 已更新 prefill/TTFT 实现状态与证据引用。
