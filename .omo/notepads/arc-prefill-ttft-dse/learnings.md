## [2026-08-18] T0 blocked: qoder routing
- 全局 ~/.config/opencode/oh-my-openagent.json 已配置 kimi/deepseek 路由（可用）。
- 项目级 .opencode/oh-my-openagent.json 原先覆盖为 qoder，导致子 agent 走 qoder 触发 400。
- 已将项目级配置全部改为 deepseek/kimi；但 opencode 启动时缓存配置，需用户重启当前会话才能生效。
- T0 已标记 -[~]，重启后可恢复执行。

## [2026-08-18] T0 completed
- 手动完成 lift-decision-grade-fail 遗留提交（子 agent 因 qoder 400 反复失败）。
- 提交分组：calibrate(tensor_core/dram/freq)、feat(calibration)、docs(trust/readme/arch)、chore(omo)、docs(arc requirement)、evidence(t0)。
- 验证：除 pre-existing 的 test_scenario_cli_produces_v2_schema 挂起外，其余 pytest 全绿。
- 已 push origin main。

