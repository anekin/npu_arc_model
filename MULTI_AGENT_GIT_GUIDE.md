# 多 Agent 协同 Git 工作流指南（npu_arc_model）

> **部署拓扑（重要）**：不同 Agent 在**不同电脑**上工作，每个工作目录同时只有一个 Agent。
> 因此每个 Agent 各自有独立的 clone、独立的 working tree、独立的 index、独立的
> `.git/index.lock`——**不存在同机锁竞争**。本指南据此给出规范。
>
> **背景事故**：一次把 `origin/feat/scenario-driven-dse` 合并进本地 `main` 时，被中断的
> `merge` 残留了 `index.lock`，重试时撞锁；且该仓库存在 `.git` 引用写入不持久的内部隐患，
> 导致操作连锁失败、工作树一度丢了 116 个被跟踪文件。
> **根因**：本地 `merge` 中途被中断 + 仓库自身的 refs 写入问题，**与跨机协作无关**。
> 跨机协作真正要防的是**集成期冲突**——两台机器各自改了同一批文件，开 PR 时才撞（见 §5）。

---

## 0. 铁律（Core Principle）

1. **一个 Agent = 一台机器 = 一个独立 clone（天然满足，通常无需 worktree）。**
   你们的拓扑下各自 clone 已天然隔离；直接用 feature-branch + PR（见 §3）即可。
   worktree 只在「同一台机器跑多个 Agent」时才作为隔离手段需要。
2. **`main` / 共享分支只允许通过 PR（Merge Request）合入。**
   任何 Agent 不得在 `main` 上直接 `commit` / `push`。
3. **任何破坏性操作前，先备份 `.git`。**

---

## 1. worktree 解决什么、什么时候才需要

- worktree 的价值是**隔离同机并发**：每个 worktree 有独立的 working tree + index +
  `index.lock`，同一台机器上跑多个 Agent 时不会互相锁死或污染。
- 但**你们是跨机部署**（每台机器一个 Agent、各自 clone），不存在同机锁竞争——
  这种情况**不需要 worktree**，直接在主 clone 上用 feature-branch + PR（§3）就是标准且最简单的方式。
- 跨机协作真正要防的是**集成期冲突**：两台机器各自改了同一批文件，开 PR 时才会撞。
  这靠 §5 的「文件所有权」+ §4 的「PR 串行集成」解决，而不是靠 worktree。
- 补充：worktree 能缩小「工作树级」失误的爆炸半径，但**无法**防住本次那种
  `.git` 对象库/引用级损坏（`.git` 在所有 worktree 间共享）——这类只能靠 §6 的
  `.git` 备份。所以无论用不用 worktree，备份铁律不变。

> 若将来出现「同一台机器并行多个 Agent」的场景，再启用 worktree：
> `git worktree add ../npu_am_<agent> -b feat/<agent>-<topic> origin/main`，
> 其余流程不变。

---

## 2. 每个 Agent 在自己的 feature 分支上开发

```bash
# 在各自机器上的 clone 里操作（已经在自己的目录，无需 worktree）
git fetch origin
git checkout -b feat/<agent>-<topic> origin/main   # 基于最新 main 起分支

# —— 开发 ——

# 提交：小步、语义化、按 feature 拆；只 add 具体文件，不要 git add -A 一把梭
git add <specific files>
git commit -m "feat(<scope>): <what and why>"

# 推送并提 PR
git push -u origin feat/<agent>-<topic>
```

- 永远在自己的 `feat/*` / `fix/*` 分支上工作；**绝不**在 `main` 上直接提交。
- 提交粒度要小，便于 rebase 与 review。
- 开 PR 前先 `git rebase origin/main`，把自己的分支对齐到最新 main，冲突在自己机器上先解掉。

---

## 3. Agent 日常开发流程（标准 feature-branch + PR）

```bash
# 开始会话：同步最新 main（只在自己的 feature 分支上操作，绝不直接改 main）
git fetch origin
git rebase origin/main          # 把自己的 feature 分支 rebase 到最新 main 之上

# —— 开发 ——

# 提交：小步、语义化、按 feature 拆；只 add 具体文件，不要 git add -A 一把梭
git add <specific files>
git commit -m "feat(<scope>): <what and why>"

# 推送并提 PR
git push -u origin feat/<agent>-<topic>
```

- 永远在自己的 `feat/*` / `fix/*` 分支上工作；**绝不**在 `main` 上直接提交。
- 提交粒度要小，便于 rebase 与 review。
- 本 Agent 的工作区里出现冲突，只在**自己这个 feature 分支 / clone** 内解决，冲突范围天然很小。

---

## 4. 功能合入 main（集成）

- 通过 **PR / Merge Request** 合入，**禁止**直接 push `main`。
- 推荐 **squash-merge**，保持 `main` 历史线性、可读。
- 合入前必须：`git rebase origin/main` 解决自己分支的冲突。
- **集成由一个人 / 一个 Agent 串行执行**，避免两个 Agent 同时合入互相覆盖。
- 大范围目录重组（如本次 `docs/` → `docs/cv`、`docs/llm`、`docs/embodied`）**必须放在
  独立分支 + PR** 里做，不要和引擎改动混在同一分支，否则合并时必然爆炸。

---

## 5. 冲突前置规避：文件所有权（File Ownership）

跨机 Agent 各自改文件、开 PR 时才可能撞冲突。明确**文件所有权**，集成时按权威方裁定：

| 路径 | 权威方 | 集成策略 |
|---|---|---|
| `sim/engine/**`、`sim/config/**`、`sim/dse/**`、`sim/design_space_explorer.py`、`sim/arc_model.py`、`sim/dse_scenario.py`、`sim/model_specs.py`、`sim/models/**` | **远端仓库（upstream / repo）** | 本地对这些文件的改动在集成时**让位于 repo 版本**（本次结论：本地改动可放弃） |
| `docs/**`、`reports/**`、`scripts/**` | 本地 Agent | 保留本地版本 |
| `README.md` | 传入方（repo） | 以远端为准 |

- 跨所有权的改动，提前在 PR 描述里说明，由负责人裁定。
- 若两个 Agent 必须碰同一批文件，先拆目录 / 拆模块，做不到就串行排队。

---

## 6. 破坏性操作安全准则

- **任何破坏性操作前先备份 `.git`**：
  ```bash
  cp -r .git .git.bak.$(date +%s)     # 约 150MB，秒级完成
  ```
- **禁止**对同一工作树并发跑 git（同一台机器上若有多个 Agent 或后台任务，必须先串行化）。
- `git reset --hard` / `git checkout -f` / `git clean -f` 会**抹掉未提交工作**，
  仅在确认要丢弃时使用，且先用上面的方式备份。
- 不要用 `git merge` 去「拉平」大量重组过的分支；优先 `rebase` 或走 PR。
- **绝对不要**手工删除 `.git/refs`、`index` 等内部文件，除非在下面的恢复流程里且已备份。

> **本仓库特例（refs 写入不持久，务必注意）**：实测 `git update-ref` / `git branch` /
> `git fetch` 会返回 exit 0，但 `.git/refs/` 下不落文件、`origin/main` 取回后回退到旧值，
> 只有「手工写 loose-ref 文件」（如 `printf <sha> > .git/refs/heads/<branch>`）能持久化。
> 任何建分支 / 推分支 / 更新 remote-tracking 的操作，**都要用 `git rev-parse <ref>` 复核引用
> 是否真的写入**，不要迷信命令返回码。建议优先排查 `.git` 所在文件系统的权限 / 符号链接问题。

---

## 7. 损坏恢复手册（出现异常时照做）

**症状**：`.git/index.lock` 报错、`not a git repository`、工作树大量文件被标记为
`deleted`、merge 卡死。

1. **确认没有 git 进程在跑**：
   ```bash
   ps aux | grep -i git        # Linux/macOS
   # Windows：任务管理器检查，或 PowerShell: Get-Process git
   ```
2. **stale lock**：若确无进程，`rm -f .git/index.lock`。
3. **merge 卡死 / 中途损坏**：
   ```bash
   git merge --abort                        # 回到 HEAD
   git checkout -f HEAD -- .                # 把工作树恢复成 HEAD 状态
   git status --short                       # 应只剩 untracked
   ```
4. **refs / index 丢失 / not a git repository**：
   从最近的 `.git.bak.*` 恢复：`cp -r .git.bak.<ts> .git`，再 `git fsck` 校验；
   必要时按 `ORIG_HEAD` 用「手工写 loose-ref 文件」重建分支指针
   （见 §6 本仓库特例）。
5. **清理孤儿 reflog**（本次出现过的坑）：若无实际 stash，删除 `.git/logs/refs/stash`
   可消除 `fsck` 的 `invalid reflog entry` 报错。
6. **永远保留 `.git.bak.*`**，确认仓库完全健康后再删。

---

## 8. 本仓库约定速查

- **远端**：`origin`（`git@github.com:anekin/npu_arc_model.git`）
- **分支命名**：`feat/<agent>-<topic>`、`fix/<agent>-<bug>`
- **引擎改动**（`sim/engine`、`sim/config`、`sim/dse` 等）→ 走 PR，本地改动在集成时让位于 repo
- **文档 / 报告 / 脚本**（`docs`、`reports`、`scripts`）→ 本地 Agent 直接提交即可
- **每个 Agent 启动时**：先 `git fetch`，再在**自己的 feature 分支**上 `git rebase origin/main`
- **集成串行化**：合入 `main` 一个人做，避免并发 push / merge
- **跨机天然隔离**：每台机器各自 clone，无需 worktree；冲突只发生在 PR 集成期，靠文件所有权裁定

---

## 9. 本次事故复盘（作为反面教材）

| 步骤 | 实际发生 | 按本指南应怎样 |
|---|---|---|
| 同步远端 | 直接在共享 `main` 上 `merge feat/...` | 在自己的 feature 分支上 rebase，再走 PR |
| 中断处理 | 上一次 merge 残留 `index.lock`，重试时撞锁 | 先 `rm -f .git/index.lock`（确认无进程后） |
| 目录重组 | 文档大迁徙与引擎改动混在同一分支 | 拆成独立 PR，分别合入 |
| 破坏性操作 | 未备份 `.git` 就操作 | 先 `cp -r .git .git.bak.<ts>` |
| 工作树丢失 | 116 个文件被标记为 deleted | `git checkout -f HEAD -- .` 恢复（已执行） |
| 引用写入 | `git branch`/`update-ref` 返回成功却不落文件 | 改用手写 loose-ref + `git rev-parse` 复核（§6） |
| 冲突解决 | 直接在 main 上解 100+ 冲突 | 各自分支内小范围解决，PR 合入 |

> 一句话：**分分支、串行集成、先备份后破坏、按文件所有权裁定冲突。** 跨机 Agent 各自
> clone 已天然隔离，守住这四条就不会再踩到这次的坑。
