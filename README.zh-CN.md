[English](README.md) | 简体中文

# Godot 游戏代码审查 Agent

一个面向 Godot 4 GDScript 游戏项目的 AI 驱动代码评审与质量保障流水线。围绕 [Harvester 太空采矿游戏](https://github.com/gdquest-demos/godot-2d-space-game)的实际运行构建并测试，可迁移到其他 Godot 项目。

## 概览

Harvester-Agent 使用多阶段 LangGraph 流水线分析 GDScript 代码，组合了：

- **静态分析**：Godot CLI 校验（语法、import、资源）
- **LLM 代码评审**：AI 驱动的评审，带跨文件上下文注入
- **自动化验证**：propose-verify（提出—验证）模式，降低误报
- **交叉关联**：跨流水线与代码发现，检测根因

## 它抓到的一个真实 Bug

对 Harvester 游戏做了一次全项目扫描（62 个文件，一遍跑完），流水线只报了一个警告——而那是一个一直悄悄藏在音频代码里的真 bug：

`LoopingAudioStreamPlayer2D.end()` 把 `stream` 换成了尾音并设置
`ending = true`，但它从不调用 `play()`。更糟的是，`_on_finished()` 只检查
当前 stream 是否是 `sound_start`——而 `end()` 重新赋值之后它已经不是了。
所以尾音根本不会被听到；如果 `end()` 在前奏还在播放时被调用，循环也不会
接上：音频直接静默。

这正是那种能逃过人工 playtest 的 bug，因为没有任何东西崩溃，而静默很容易被
忽略。verifier 把它标为 "unverifiable"（这是一个运行时行为断言，静态索引证明
不了），所以它以警告而不是错误的级别落进报告——这是诚实的答案，也足以把它
摆到人面前。

## 架构

```text
┌─────────┐    ┌───────────┐    ┌─────────────┐
│  START  │───▶│ preflight │───▶│ build_index │
└─────────┘    │   check   │    └──────┬──────┘
               └─────┬─────┘           │
                     │ abort            ▼
                     ▼           ┌─────────────┐
                   ┌─────┐       │resolve_scope│
                   │ END │       └──────┬──────┘
                   └─────┘              │
                                        ▼
                               ┌────────────────┐
                               │  route_files   │  LinUCB bandit:
                               └───────┬────────┘  file → model tier
                                       ▼
                           ┌────────────────────────┐
                           │  [pipeline_check  │    │
                           │   code_review]    │◄───┘ (parallel)
                           └──────────┬───────────┘
                                      ▼
                           ┌────────────────────────┐
                           │        playtest        │  headless Godot boots the
                           └──────────┬─────────────┘  game, watches the bus
                                      ▼
                           ┌────────────────────────┐
                           │    runtime_verifier    │  dormant claims re-judged
                           └──────────┬─────────────┘  against observed signals
                                      ▼
                           ┌────────────────────────┐
                           │    cross_correlator    │
                           └──────────┬─────────────┘
                                      ▼
                           ┌────────────────────────┐
                           │     filter_refuted     │
                           └──────────┬─────────────┘
                                      ▼
                           ┌────────────────────────┐
                           │     findings_dedup     │
                           └──────────┬─────────────┘
                                      ▼
                           ┌────────────────────────┐
                           │      report_merger     │
                           └──────────┬─────────────┘
                                      ▼
                        ┌─────────────────────────┐
                        │  human_gate (interrupt) │  ◄── only with --auto-fix:
                        └──────────┬──────────────┘      pauses for human approval
                                   ▼
                        ┌────────────┐   ┌────────────┐   ┌─────┐
                        │  auto_fix  │──▶│ fix_verify │──▶│ END │
                        └────────────┘   └────────────┘   └─────┘
```

每个节点边界都会持久化到 SQLite checkpointer
（`.kimi/checkpoints/pipeline.sqlite`），所以崩溃的运行可以从断点继续，
而不是从头再来——见下文 `--resume`。一个诚实的提醒：
checkpoint 是按*节点*粒度的，所以如果在 `code_review`（那个很长的逐文件
循环）中途崩溃，该节点会从头重跑。

### 流水线节点

| 节点 | 作用 |
| ---- | ---- |
| `preflight_check` | 校验 API key、Godot 二进制、project.godot |
| `build_index` | 扫描项目，构建带 mtime 缓存的符号索引 |
| `resolve_scope` | Git diff 检测 + 依赖扩展 |
| `route_files` | LinUCB bandit 为每个文件分配模型档位（便宜↔高端） |
| `pipeline_check` | 静态分析（import、语法、资源） |
| `code_review` | LLM 驱动的评审，带逐文件重试 |
| `playtest` | 通过注入测试控制器做无头（headless）Godot playtest（不依赖 LLM） |
| `runtime_verifier` | 拿观察到的信号重新判定休眠中的运行时断言 |
| `cross_correlator` | 关联各项发现，检测 index_gap 模式 |
| `filter_refuted` | 移除被驳回的（final_severity="drop"）发现 |
| `findings_dedup` | 级联去重 |
| `report_merger` | 生成最终的 markdown + JSON 报告 |
| `human_gate` | 仅 `--auto-fix`：中断图执行，等待人工批准或拒绝修复计划 |
| `auto_fix` | 仅 `--auto-fix`：应用 LLM 对 error 级发现的修复（有 git stash + `godot --check-only` 保护） |
| `fix_verify` | 仅 `--auto-fix`：修复后重跑 playtest——如果某个修复弄坏了之前能通过的场景，会被标记为回归 |

## 快速开始 —— 完整命令链

```bash
# 1. Enter the agent directory
cd GameAgent/harvester-agent

# 2. Create and activate a virtual environment
py -3.11 -m venv .venv          # Windows  (use python3 on Linux/Mac)
.venv\Scripts\activate           # Windows cmd/PowerShell
# source .venv/bin/activate      # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
copy .env.example .env           # Windows  (cp on Linux/Mac)
# Edit .env: set LLM_PROVIDER and the matching API key

# 5. Run the pipeline
#    Windows with non-UTF-8 terminal (GBK/CP936) — set PYTHONUTF8=1
PYTHONUTF8=1 python game_agent_pipeline.py \
    --project "../godot-2d-space-game/project" \
    --no-display

# Reports are written to reports/<timestamp>/
#   final_report.md           — human-readable summary
#   deduplicated_findings.json — machine-readable findings
```

## 安装

### 前置条件

- Python 3.11+
- Godot 4.x（可执行文件路径在 `tools/godot_runner.py` 中配置）
- Claude（Anthropic）或 Kimi（Moonshot）的 API key

### 安装步骤

```bash
# Clone and enter directory
cd GameAgent/harvester-agent

# Create virtual environment
# Windows: if `python` points to the Microsoft Store stub, use `py -3.11` instead
py -3.11 -m venv .venv        # Windows (recommended)
# python3 -m venv .venv       # Linux/Mac

# Activate the virtual environment — required before all commands below
.venv\Scripts\activate        # Windows (cmd / PowerShell)
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env with your API keys
```

### 配置

编辑 `.env`：

```env
# LLM Provider: "claude" | "kimi"
LLM_PROVIDER=kimi

# Enable fallback to other provider on failure
LLM_FALLBACK=0

# API Keys (at least one required)
ANTHROPIC_API_KEY=your-anthropic-key
MOONSHOT_API_KEY=your-moonshot-key
```

## 用法

### 基本用法

> **Windows 注意：** 先激活 venv（`.venv\Scripts\activate`），再用 `python`。
> 在非 UTF-8 代码页（例如中文 Windows 的 GBK）下，流水线会以
> `errors="replace"` 重新配置 stdout/stderr——装饰性字符可能显示为 `?`，
> 但运行绝不会因为控制台编码而崩溃。`--no-display` 只是跳过视觉测试。

```bash
# Activate venv (required every new terminal session)
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # Linux/Mac

# Review the Harvester game project
python game_agent_pipeline.py --project "../godot-2d-space-game/project"

# Windows with non-UTF-8 terminal (GBK / CP936) — use --no-display
python game_agent_pipeline.py --project "../godot-2d-space-game/project" --no-display

# Custom output directory
python game_agent_pipeline.py --project "../godot-2d-space-game/project" --output "./my-reports"
```

### 崩溃恢复与人在环（human-in-the-loop）自动修复

每次运行都会边跑边写 checkpoint。如果有什么东西中途杀掉了流水线
（断电、不合时宜的 Ctrl-C、重试耗尽的 Godot 崩溃），你不必从头再来：

```bash
# Pick up from the last completed node
python game_agent_pipeline.py --project "../godot-2d-space-game/project" --resume
```

如果你希望流水线不止出报告，传 `--auto-fix`：报告就绪后，图会
**停下来等你**。看一眼 `reports/<timestamp>/final_report.md`，然后决定
——只有人说了 yes，修复才会发生：

```bash
python game_agent_pipeline.py --project "../godot-2d-space-game/project" --auto-fix
# ... pipeline runs, report is saved, then it pauses:
#     "Apply auto-fix to these error findings? [y/N]"

# Headless / CI: the pause prints a resume command instead of prompting —
python game_agent_pipeline.py --project "../godot-2d-space-game/project" --resume --approve
python game_agent_pipeline.py --project "../godot-2d-space-game/project" --resume --reject
```

获批的修复仍然要走独立 auto-fixer 的那套安全护栏：
任何写入之前先打一个带标签的 `git stash`，且每个重写后的文件必须通过
`godot --check-only`，否则恢复原文件。

重试是声明在图上的，不是手工拼的：`pipeline_check` 带一个
LangGraph `retry_policy`（3 次尝试，指数退避），用来应对 Godot 子进程的
临时性故障。`code_review` 刻意保留自己的*逐文件*重试——对整个节点重试
意味着只因为一次抖动的 API 调用就要重审所有文件。

### Exit Code（退出码）

`game_agent_pipeline.py` 以质量门（quality-gate）退出码退出（与 `review.py` 同一约定）：

| 退出码 | 含义 |
|------|---------|
| `0` | 健康度为绿或黄 |
| `1` | 健康度为**红**——存在已验证的错误 |
| `2` | 流水线失败（项目路径错误、preflight 中止） |

`.github/workflows/pr-gate.yml` 里的 CI 门依赖这个约定：它对每个
pull request 跑流水线，健康度为红则 job 失败。没有 LLM secrets 的 fork PR
会自动降级为纯静态模式，并且仍然按静态错误卡门。

### 独立评审脚本

```bash
# Traditional pipeline (without LangGraph)
python review.py --project "../godot-2d-space-game/project"

# Target specific subdirectory
python review.py --project "../godot-2d-space-game/project" --target "ships/enemies"

# Force index rebuild
python review.py --project "../godot-2d-space-game/project" --rebuild-index

# Include addons/libraries
python review.py --project "../godot-2d-space-game/project" --review-addons
```

### 查看流水线图

```bash
python game_agent_pipeline.py --print-graph
```

## Chat / 问答入口

`chat.py` 是一个产品风格的自然语言前门。它把每条消息分类成意图并分发
——评审流水线本身完全不动。

```bash
# REPL
python chat.py

# Single question
python chat.py --ask "MiningShip 的信号在哪声明"

# Headless demo: no LLM calls at all (rule intents + retrieval-only answers)
python chat.py --no-llm --ask "MiningShip 是什么"

# Show the classification before dispatching
python chat.py --show-intent --ask "审查这次提交的改动"
```

| 意图 | 示例 | 动作 |
|--------|---------|--------|
| `full_review` | "帮我审查一下整个项目" | 运行 `review.run_pipeline` |
| `diff_review` | "看看 diff 里有没有问题" | 子进程调用 `game_agent_pipeline.py`（git-diff 范围） |
| `symbol_lookup` | "MiningShip 是什么" | 纯索引查询，绝不调用 LLM |
| `code_qa` | "采矿船怎么收集矿石？" | 索引 RAG：top-k 检索 → LLM 作答，附 `file:line` 引用 |
| `unknown` | 其他任何东西 | 打印用法帮助 |

降级策略：没有 API key（或带 `--no-llm`）时，意图识别回退到关键词规则，
`code_qa` 则返回检索到的索引条目的诚实结构化摘要，而不是编造答案
（`llm_used=False`）。只有 `full_review` 需要可用的 key。

## 输出

报告保存在 `reports/<timestamp>/`：

```text
reports/20250328_161030/
├── final_report.md          # Human-readable summary
├── run_metadata.json        # Run configuration
├── findings.json            # All code review findings
├── pipeline_issues.json     # Static analysis issues
├── correlated_findings.json # Cross-correlated issues
├── deduplicated_findings.json
├── agent_status.json        # Per-node execution status
├── fix_results.json         # Auto-fix outcomes (only with --auto-fix)
├── playtest_results.json    # Automated playtest scenarios
├── observed_signals.json    # Events bus signals observed during playtest
├── runtime_coverage.json    # Scripts loaded in the live scene tree
├── routing_plan.json        # Per-file model tier assignments (bandit)
├── fix_verify_results.json  # Post-fix playtest comparison (only with --auto-fix)
└── ...
```

### 报告小节

- **Agent Status**：逐节点的执行结果
- **Pipeline Check**：import/语法/资源问题
- **Code Review**：LLM 发现，带严重级别与验证状态
- **Correlated Findings**：多来源的根因分析
- **Meta Findings**：索引缺口诊断

## 关键特性

### 1. 双 LLM 支持

自动检测并支持：

- **Claude**（Anthropic）：`claude-sonnet-4-6`
- **Kimi**（Moonshot）：`kimi-k2.7-code`（可用环境变量 `KIMI_TEXT_MODEL` 覆盖）

`LLM_FALLBACK=1` 时提供方之间自动 fallback。

### 2. Propose-Verify（提出—验证）模式

1. LLM 提出发现，附带断言
2. 通过 Godot CLI 自动验证
3. 验证通过的发现保留，被驳回的标记为 "drop"
4. 交叉关联器分析被驳回的模式

### 3. 智能上下文注入

构建项目索引，包括：

- 脚本及其 class_name
- 场景文件及其节点树
- 信号连接
- 跨文件依赖

上下文被注入 LLM prompt，以降低误报。

### 4. Git 集成

- 通过 `git diff` 检测变更文件
- 通过依赖图分析扩展范围
- 只评审受影响的文件（CI 跑得更快）

### 5. 静态兜底模式

API key 不可用时：

- 只跑语法检查
- 休眠 bug 以警告级别报告
- 在报告中标明降级模式

### 6. 元发现（Meta-Finding）检测

检测系统性的"断言被驳回"模式（≥3 个文件），以识别：

- 缺失的索引条目
- 常见误报模式
- 上下文注入缺口

### 7. 经验记忆（测试时学习，Test-Time Learning）

灵感来自 [Dynamic Cheatsheet](https://arxiv.org/abs/2504.07952)（EACL 2026），
但加了一道 DC 没有的硬性正确性闸门：只有 verifier 背书的结果才会入库。

- **写入**：每次运行后，被静态 verifier **驳回**的发现成为
  `false_positive` 记忆；**确认**的成为 `confirmed_pattern` 条目。
  无法验证的发现永远不入库——没有 ground truth，就没有记忆。
- **读取**：评审某个文件之前，检索相关的 FP 模式
  （TF-IDF 余弦 × 同文件加成 × 实体匹配加成 × 按衰减加权的命中数，
  硬性 800 token 预算），作为"已知误报"注入 prompt
- **生命周期**：条目按 `(assertion_type, entity)` 合并，累积命中数，
  每天按 0.95 衰减，低于有效命中阈值后归档
- **A/B**：`--no-memory` 关闭检索 + 记忆整理，用于基线运行
- 存储：`.kimi/memory/review_memory.jsonl`（已 gitignore）

效果是复利式的：反复出现的误报被逐步压制，
所以 precision（精确率）随每次运行提升，而不是每次重置。

### 8. 运行时验证（以 Playtest 作为 Ground Truth）

静态分析有个盲区：像"这个信号发射永远不可能执行"这样的断言，
靠读代码是判不了的。这类断言过去会堆成*休眠 bug*——听起来合理，
但没有证据。流水线现在通过真的把游戏跑起来补上了这个缺口。

- **Playtest 节点**：向项目注入一个 SceneTree 测试控制器，
  以无头方式启动游戏，并监听 `Events` 总线。每个被监听的信号
  都会计数；同时遍历活的场景树，看哪些脚本真的被加载了。
  不依赖 LLM，所以纯静态模式下也能跑。
- **运行时 verifier**：携带 `runtime_signal_not_emitted` 断言的发现，
  会拿 playtest 实际观察到的情况重新判定：
  - 信号发射了 → 该断言被**驳回**（附证据）
  - 触发上下文被执行到了但信号静默 → 休眠**升级为已确认错误**
  - 场景从未被执行到 → 原样保留，如实保留
- **上下文门控才是重点**：信号缺失并不能证明有 bug——
  也许 playtest 根本没触发它。只有当所有前置信号都发射了而目标信号
  保持静默时，缺席才成为证据。（来自演示游戏的真实数据：
  `mine_finished` 发射了 191 次，而 `mine_started` 一次都没发射——
  这是一个值得人看一眼的异常，纯静态分析永远浮不出它。）
- **覆盖率备注**：范围内但 playtest 从未加载的脚本，会以 info 级别
  备注出现（"未被 playtest 执行到"），而不是判成死代码。
- **修复—验证闭环**：带 `--auto-fix` 时，获批的修复之后会重跑
  playtest；弄坏了之前能通过场景的修复，会在报告中被标记为回归。

一个值得知道的坑：用 `godot --script` 启动的脚本，即使单例在运行时
就在场景树里，编译期也无法解析 autoload 全局（`Events`）——
所以控制器改走节点来连接。

### 9. 成本感知的模型路由（Contextual Bandit，上下文老虎机）

不是每个文件都配得上同一个模型。30 行的 UI 脚本和 600 行的状态机
是不同的活——所以 `route_files` 节点会在评审之前给范围内每个文件
分配一个模型档位，用的是 **disjoint LinUCB**（每个臂一个线性 bandit，
在一个 9 维结构特征向量上做 UCB 选择：大小、方法数、信号数、
preload 数、async、继承深度……）。

让这件事不需要标注预算就能成立的关键：**奖励信号是免费的**。
每个文件评审完后，verifier 已经把每个发现分类为确认或驳回——所以

```text
reward = n_confirmed − n_refuted − λ · (tokens / 1000) · relative_cost
```

是在线计算的，bandit 状态跨运行持久化
（`.kimi/router/linucb.json`）。干净文件只赚到（负的）成本项，
这会随时间把简单的活推向更便宜的档位；而发现经得起验证的文件
会奖励找到它们的更强模型。

- **臂（arm）** 根据可用的 API key 解析：DeepSeek V4 Flash / V4 Pro、
  Kimi highspeed / standard / K3，外加设了 `ANTHROPIC_API_KEY` 时的
  Claude Sonnet。Preflight 会 ping 每个配置了的提供方；key 失效的那个
  在本次运行中被排除出路由和升级阶梯（它已学到的 bandit 状态保留——
  修好 key 不该抹掉历史）。
- **升级（escalation）**：如果便宜档位的 API 调用失败，逐文件重试阶梯
  会爬到下一档，而不是重试同一档。认证错误（401）完全跳过退避——
  坏 key 等多久都好不了。
- **失败也回馈**：一个耗尽了阶梯的文件仍然会更新 bandit——给每个
  失败的臂一个负奖励。没有这一条，坏掉的高端臂会永远保持"冷"状态，
  持续吸引 UCB 流量（惨痛的教训：有一次运行把 62/62 个文件全路由到
  一个 key 已失效的臂上）。
- **A/B**：`--no-router` 把所有文件钉在默认模型上作为基线；
  `ROUTER_ARMS` 白名单档位；`ROUTER_LAMBDA` 调节成本敏感度。
- 学习是复利式的：每次运行的臂分布、每臂平均奖励、升级次数都落在
  `run_metadata.json` → `routing_plan.json`。

思想脉络：RouteLLM（arXiv 2406.18665）和 *Learning to Route LLMs from
Bandit Feedback*（arXiv 2510.07429）——这里把它用在了代码评审上，
由 verifier 替代人工偏好标注。

### 10. 自我改进的 Prompt（离线 Prompt 进化）

评审 prompt 不是手工调出来的玄学——它是针对人工标注的 golden set
（32 个文件上的 49 条发现，每条附评审者备注）进化出来的。
`agents/gepa.py` 实现了 GEPA（arXiv 2507.19457，Algorithm 1/2）的核心，
并适配到代码评审：

1. **按实例做回放式适应度评估**：候选 prompt 重新评审 golden set 覆盖的
   文件；发现与人工判定做匹配（文件 + 断言实体，或行号 ±3）。分数按
   **逐文件**追踪——这正是 Pareto 前沿（Pareto frontier）需要的实例级
   分数矩阵。
2. **基于执行轨迹的反思式变异**：变异模型读取父代模板及其*执行轨迹*
   ——每条产出的发现都标注了 verifier 的判定、复现的 FP 对应的人工
   驳回备注、以及漏掉的已确认 bug——然后重写模板。丢失
   `{filename}`/`{autoload_list}`/`{context_and_code}` 占位符的候选，
   在花掉任何一次评估调用之前就被丢弃。
3. **接收门**：变异体只有在同一 minibatch 上击败父代才能进入候选池。
   被拒的变异体绝不消耗验证轮次。
4. **Pareto 父代选择**（不是贪心选最优均值——论文表明贪心会坍缩到
   局部最优）：取各实例赢家的并集，移除被支配的候选，按获胜次数
   成比例采样父代。一个唯一能处理某种文件*类型*的 prompt，即使
   平均分平庸，也能把它的 DNA 留在池子里。
5. **成本纪律**：在便宜档位（deepseek-v4-flash）上做 minibatch 反思，
   按（候选， 文件）做回放缓存，再加上接收门，一起把 6 轮迭代的运行
   控制在 20 分钟左右。

```bash
python -m agents.gepa --iterations 6 --minibatch 8 --pareto-size 12
python -m agents.gepa --dry-run            # validate setup, no LLM calls
```

胜者只有在留出（held-out）的 `D_pareto` 划分上击败基线时，才会写入
`prompts/review_prompt_evolved.txt`（绝不自动覆盖生产 prompt）；
`eval/gepa_report.md` 记录每个候选的血统、被门拒掉的记录、以及
逐实例分数。相对论文的简化：System-Aware Merge（互补候选之间的
交叉）尚未实现。

已知局限：一个零发现的文件无法区分"真的干净"和"模型看走了眼"——
verifier 只评判被提出来的东西。这一点在 v1 中被接受，并且部分由
playtest 层兜底：不管是哪个模型审的文件，运行时可见的破坏都能被抓到。

## 项目结构

```text
harvester-agent/
├── game_agent_pipeline.py    # Main LangGraph pipeline
├── review.py                 # Standalone review script
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
├── agents/
│   ├── llm_client.py         # Unified LLM client (Claude/Kimi)
│   ├── code_review.py        # LLM code review agent
│   ├── pipeline_check.py     # Static analysis agent
│   ├── playtest.py           # Runtime testing agent
│   ├── auto_fix.py           # Automated fix suggestions
│   └── config_gen.py         # Configuration generator
├── tools/
│   ├── godot_runner.py       # Godot CLI wrapper
│   ├── project_parser.py     # Project.godot parser
│   ├── project_indexer.py    # Symbol index builder
│   ├── context_builder.py    # LLM context assembler
│   ├── assertion_verifier.py # Finding verification
│   ├── parser.py             # Output sanitizers
│   └── report_generator.py   # Report writers
├── godot_test_scripts/
│   └── test_controller.gd    # Injected runtime test harness
└── prompts/
    ├── review_prompt_v2.txt   # LLM review prompt template
    └── gdscript4_reference.md # Language reference
```

## 配置选项

### 环境变量

| 变量 | 说明 | 默认值 |
| -------- | ----------- | ------- |
| `LLM_PROVIDER` | 主 LLM：`claude` 或 `kimi` | 自动检测 |
| `LLM_FALLBACK` | 启用 fallback 提供方 | `0` |
| `ANTHROPIC_API_KEY` | Claude API key | - |
| `MOONSHOT_API_KEY` | Kimi API key | - |
| `DEEPSEEK_API_KEY` | DeepSeek API key（启用 v4-flash / v4-pro 路由臂） | - |
| `DEEPSEEK_TEXT_MODEL` | DeepSeek pro 档位模型 | `deepseek-v4-pro` |
| `DEEPSEEK_FLASH_MODEL` | DeepSeek flash 档位模型 | `deepseek-v4-flash` |
| `KIMI_TEXT_MODEL` | 用于代码评审的 Kimi 模型 | `kimi-k2.7-code` |
| `KIMI_VISION_MODEL` | 用于截图的 Kimi 模型 | `moonshot-v1-8k-vision-preview` |

### Godot 可执行文件

`tools/godot_runner.py` 按以下顺序解析二进制：

1. `GODOT_EXECUTABLE` 环境变量（CI 中使用）
2. `PATH` 上的 `godot`
3. `tools/godot_runner.py` 里硬编码的本地路径（改成匹配你的安装）

### 流水线选项

```bash
# Skip visual testing (headless CI)
--no-display

# Custom output directory
--output PATH

# Force index rebuild
--rebuild-index

# Include addon/library sources
--review-addons

# Anchor findings from previous run (A/B testing)
--anchor FINDINGS_JSON

# Disable experience memory (A/B baseline)
--no-memory

# Append human-gated auto-fix to the graph
--auto-fix

# Resume from the last checkpoint (after a crash or a human-gate pause)
--resume [--approve | --reject]

# Disable bandit model routing (A/B baseline)
--no-router
```

## 测试

```bash
# Run unit tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_llm_client.py

# Run with verbose output
python -m pytest tests/ -v
```

## 评估数据集（precision 与 recall）

一个数据集没法同时衡量两者。本项目刻意保留两个：

- **`eval/golden_set.jsonl` —— 衡量 precision（精确率）。** 49 条历史
  流水线发现，带人工判定（`tp`/`fp`/`partial`）和评审者备注。它回答
  "我们报出来的东西里有多少是真的？"它回答不了"存在的 bug 里我们找到
  多少？"——任何一次运行都没产出过的 bug 对它是不可见的。
  `python eval/run_eval.py` 打印基准结果。
- **`eval/mutant_set.jsonl` —— 衡量 recall（召回率）。** 由
  `eval/mutant_eval.py` 构建：复制游戏项目，然后机械地向每个文件注入
  一个已知 bug（重命名的信号发射、被死守卫挡住的发射、坏掉的 preload
  路径、未定义的类引用——每种都对应流水线声称能抓到的一类 bug）。
  检出与否是二值的 ground truth，所以 `recall = detected / injected`
  终于有了分母。游戏仓库绝不被触碰；变异体生活在 `.kimi/` 下一个
  已 gitignore 的副本里。

已知局限，直说：注入的 bug 比自然 bug 干净（真 bug 不会自报家门），
所以变异体 recall 是真实 recall 的上界。precision 仍以自然产生的
golden set 为准。

## 故障排查

### "Godot executable not found"

把 `tools/godot_runner.py` 中的 `GODOT_EXECUTABLE` 更新为你的 Godot 路径。

### "API key rejected"

检查 `.env` 文件，确认为你的 `LLM_PROVIDER` 设置了正确的 key。

### "Static-only mode"

流水线在没有 LLM 分析的情况下运行。在 `.env` 中设置 `ANTHROPIC_API_KEY` 或 `MOONSHOT_API_KEY`。

### 索引不更新

强制重建：

```bash
python review.py --project PATH --rebuild-index
```

## 许可证

MIT License - 详见 LICENSE 文件。

## 致谢

为 [Harvester Godot 4 太空采矿游戏](https://github.com/gdquest-demos/godot-2d-space-game)项目构建，使用了：

- [LangGraph](https://github.com/langchain-ai/langgraph) 做工作流编排
- [Anthropic Claude](https://www.anthropic.com/claude) 与 [Moonshot Kimi](https://www.moonshot.cn/) 做 LLM 分析
- Godot 4 CLI 做静态与运行时校验
