English | [简体中文](README.zh-CN.md)

# Godot Game Code Review Agent

An AI-powered code review and quality assurance pipeline for Godot 4 GDScript game projects. Built and test by running the [Harvester space mining game](https://github.com/gdquest-demos/godot-2d-space-game) and is adaptable to other Godot project.

## Overview

Harvester-Agent uses a multi-stage LangGraph pipeline to analyze GDScript code, combining:

- **Static Analysis**: Godot CLI validation (syntax, imports, resources)
- **LLM Code Review**: AI-powered review with cross-file context injection
- **Automated Verification**: Propose-verify pattern to reduce false positives
- **Cross-Correlation**: Detects root causes across pipeline and code findings

## A Real Bug It Caught

On a full-project run against the Harvester game (62 files, one pass), the pipeline
came back with a single warning — and it was a genuine bug that had been quietly
sitting in the audio code:

`LoopingAudioStreamPlayer2D.end()` swaps `stream` to the tail sound and sets
`ending = true`, but it never calls `play()`. Worse, `_on_finished()` only checks
whether the current stream is `sound_start` — which it no longer is after `end()`
reassigns it. So the tail sound is never heard at all, and if `end()` is called
while the intro is still playing, the loop never kicks in either: the audio just
goes silent.

It's exactly the kind of bug that survives manual playtesting, because nothing
crashes and silence is easy to overlook. The verifier marked it "unverifiable"
(it's a runtime-behavior claim, not something a static index can prove), so it
landed as a warning instead of an error — which is the honest answer, and still
enough to put it in front of a human.

## Architecture

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

Every node boundary is persisted to a SQLite checkpointer
(`.kimi/checkpoints/pipeline.sqlite`), so a crashed run can pick up where it
stopped instead of starting over — see `--resume` below. One honest caveat:
checkpoints are per *node*, so a crash in the middle of `code_review` (the long
per-file loop) re-runs that node from its start.

### Pipeline Nodes

| Node | Purpose |
| ---- | ------- |
| `preflight_check` | Validate API keys, Godot binary, project.godot |
| `build_index` | Scan project, build mtime-cached symbol index |
| `resolve_scope` | Git diff detection + dependency expansion |
| `route_files` | LinUCB bandit assigns each file a model tier (cheap↔premium) |
| `pipeline_check` | Static analysis (import, syntax, resources) |
| `code_review` | LLM-powered review with per-file retry |
| `playtest` | Headless Godot playtest via injected test controller (LLM-free) |
| `runtime_verifier` | Re-judge dormant runtime claims against observed signals |
| `cross_correlator` | Correlate findings, detect index_gap patterns |
| `filter_refuted` | Remove refuted (final_severity="drop") findings |
| `findings_dedup` | Cascading deduplication |
| `report_merger` | Generate final markdown + JSON reports |
| `human_gate` | `--auto-fix` only: interrupt the graph and wait for a human to approve or reject the fix plan |
| `auto_fix` | `--auto-fix` only: apply LLM fixes for error findings (git stash + `godot --check-only` protected) |
| `fix_verify` | `--auto-fix` only: re-run the playtest after fixing — a fix that breaks a previously passing scenario is flagged as a regression |

## Quick Start — Full Command Chain

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

## Installation

### Prerequisites

- Python 3.11+
- Godot 4.x (executable path configured in `tools/godot_runner.py`)
- API key for Claude (Anthropic) or Kimi (Moonshot)

### Setup

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

### Configuration

Edit `.env`:

```env
# LLM Provider: "claude" | "kimi"
LLM_PROVIDER=kimi

# Enable fallback to other provider on failure
LLM_FALLBACK=0

# API Keys (at least one required)
ANTHROPIC_API_KEY=your-anthropic-key
MOONSHOT_API_KEY=your-moonshot-key
```

## Usage

### Basic Usage

> **Windows note:** Activate the venv first (`.venv\Scripts\activate`), then use `python`.
> On non-UTF-8 code pages (e.g. GBK on Chinese Windows) the pipeline reconfigures
> stdout/stderr with `errors="replace"` — decorative glyphs may render as `?` but
> the run never crashes on console encoding. `--no-display` only skips visual testing.

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

### Crash Recovery and Human-in-the-Loop Auto-Fix

Every run writes checkpoints as it goes. If something kills the pipeline halfway
(power, Ctrl-C at the wrong moment, a Godot crash that outlived its retries),
you don't start over:

```bash
# Pick up from the last completed node
python game_agent_pipeline.py --project "../godot-2d-space-game/project" --resume
```

Pass `--auto-fix` when you want the pipeline to go one step further than
reporting: after the report is ready, the graph **stops and waits for you**.
Look at `reports/<timestamp>/final_report.md`, then decide — the fix only
happens if a human says yes:

```bash
python game_agent_pipeline.py --project "../godot-2d-space-game/project" --auto-fix
# ... pipeline runs, report is saved, then it pauses:
#     "Apply auto-fix to these error findings? [y/N]"

# Headless / CI: the pause prints a resume command instead of prompting —
python game_agent_pipeline.py --project "../godot-2d-space-game/project" --resume --approve
python game_agent_pipeline.py --project "../godot-2d-space-game/project" --resume --reject
```

Approved fixes still go through the safety rails from the standalone auto-fixer:
a labelled `git stash` before any write, and each rewritten file must pass
`godot --check-only` or the original is restored.

Retries are declared on the graph, not hand-rolled: `pipeline_check` carries a
LangGraph `retry_policy` (3 attempts, exponential backoff) for transient Godot
subprocess failures. `code_review` deliberately keeps its own *per-file* retry
— retrying the whole node would mean re-reviewing every file after one flaky
API call.

### Exit Codes

`game_agent_pipeline.py` exits with a quality-gate code (same convention as `review.py`):

| Code | Meaning |
|------|---------|
| `0` | Health is green or yellow |
| `1` | Health is **red** — verified errors present |
| `2` | Pipeline failure (bad project path, preflight abort) |

The CI gate in `.github/workflows/pr-gate.yml` relies on this: it runs the pipeline
on every pull request and fails the job when health is red. Fork PRs without LLM
secrets automatically degrade to static-only mode and still gate on static errors.

### Standalone Review Script

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

### View Pipeline Graph

```bash
python game_agent_pipeline.py --print-graph
```

## Chat / Q&A Entry

`chat.py` is a product-style natural-language front door. It classifies each
message into an intent and dispatches — the review pipeline itself is untouched.

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

| Intent | Example | Action |
|--------|---------|--------|
| `full_review` | "帮我审查一下整个项目" | runs `review.run_pipeline` |
| `diff_review` | "看看 diff 里有没有问题" | subprocess `game_agent_pipeline.py` (git-diff scope) |
| `symbol_lookup` | "MiningShip 是什么" | pure index lookup, never calls the LLM |
| `code_qa` | "采矿船怎么收集矿石？" | index RAG: top-k retrieval → LLM answer with `file:line` citations |
| `unknown` | anything else | prints usage help |

Degradation: without an API key (or with `--no-llm`) intents fall back to
keyword rules, and `code_qa` returns an honest structured summary of the
retrieved index entries instead of inventing an answer (`llm_used=False`).
Only `full_review` requires a working key.

## Output

Reports are saved to `reports/<timestamp>/`:

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

### Report Sections

- **Agent Status**: Per-node execution results
- **Pipeline Check**: Import/syntax/resource issues
- **Code Review**: LLM findings with severity, verification status
- **Correlated Findings**: Multi-source root cause analysis
- **Meta Findings**: Index gap diagnostics

## Key Features

### 1. Dual LLM Support

Auto-detects and supports:

- **Claude** (Anthropic): `claude-sonnet-4-6`
- **Kimi** (Moonshot): `kimi-k2.7-code` (env-overridable via `KIMI_TEXT_MODEL`)

Automatic fallback between providers when `LLM_FALLBACK=1`.

### 2. Propose-Verify Pattern

1. LLM proposes findings with assertions
2. Automated verification via Godot CLI
3. Verified findings kept, refuted marked as "drop"
4. Cross-correlator analyzes refuted patterns

### 3. Smart Context Injection

Builds project index with:

- Scripts and their class_names
- Scene files with node trees
- Signal connections
- Cross-file dependencies

Context is injected into LLM prompts to reduce false positives.

### 4. Git Integration

- Detects changed files via `git diff`
- Expands scope through dependency graph analysis
- Only reviews impacted files (faster CI runs)

### 5. Static Fallback Mode

When API keys are unavailable:

- Runs syntax-only checks
- Reports dormant bugs as warnings
- Indicates degraded mode in report

### 6. Meta-Finding Detection

Detects systematic refuted-assertion patterns (≥3 files) to identify:

- Missing index entries
- Common false positive patterns
- Context injection gaps

### 7. Experience Memory (Test-Time Learning)

Inspired by [Dynamic Cheatsheet](https://arxiv.org/abs/2504.07952) (EACL 2026),
with a hard correctness gate DC lacks: only verifier-backed outcomes are stored.

- **Write**: after each run, findings the static verifier **refuted** become
  `false_positive` memories; **confirmed** ones become `confirmed_pattern`
  entries. Unverifiable findings are never stored — no ground truth, no memory.
- **Read**: before reviewing a file, relevant FP patterns are retrieved
  (TF-IDF cosine × same-file boost × entity-match boost × decay-weighted hits,
  hard 800-token budget) and injected into the prompt as "known false positives"
- **Lifecycle**: entries merge by `(assertion_type, entity)`, accumulate hits,
  decay at 0.95/day, and are archived below an effective-hits threshold
- **A/B**: `--no-memory` disables retrieval + curation for baseline runs
- Storage: `.kimi/memory/review_memory.jsonl` (gitignored)

The effect compounds: repeated false positives are progressively suppressed,
so precision improves with every run instead of resetting each time.

### 8. Runtime Verification (Playtest Ground Truth)

Static analysis has a blind spot: claims like "this signal emission can never
execute" can't be settled by reading code. Those used to pile up as *dormant
bugs* — plausible, but unproven. The pipeline now closes that gap by actually
running the game.

- **Playtest node**: injects a SceneTree test controller into the project,
  boots the game headless, and watches the `Events` bus. Every watched signal
  is counted; the live scene tree is walked to see which scripts were really
  loaded. LLM-free, so it also runs in static-only mode.
- **Runtime verifier**: findings carrying the `runtime_signal_not_emitted`
  assertion are re-judged against what the playtest observed:
  - signal fired → the claim is **refuted** (evidence attached)
  - trigger context exercised but signal silent → dormant **upgraded to
    confirmed error**
  - scenario never exercised → left untouched, honestly
- **Context gating is the point**: a missing signal is not proof of a bug —
  maybe the playtest never triggered it. Only when every prerequisite signal
  fired while the target stayed silent does absence become evidence. (Real
  data from the demo game: `mine_finished` fired 191× while `mine_started`
  never fired once — an anomaly worth a human look that pure static analysis
  would never surface.)
- **Coverage notes**: in-scope scripts the playtest never loaded show up as
  info-level notes ("not exercised by playtest"), not dead-code verdicts.
- **Fix-verify loop**: with `--auto-fix`, approved fixes are followed by a
  playtest re-run; a fix that breaks a previously passing scenario is flagged
  as a regression in the report.

One wrinkle worth knowing: scripts launched with `godot --script` cannot
resolve autoload globals (`Events`) at compile time even though the singleton
is in the tree at runtime — the controller connects through the node instead.

### 9. Cost-Aware Model Routing (Contextual Bandit)

Not every file deserves the same model. A 30-line UI script and a 600-line
state machine are different jobs — so a `route_files` node assigns each
in-scope file a model tier before review, using **disjoint LinUCB** (one
linear bandit per arm, UCB selection over a 9-dim structural feature vector:
size, methods, signals, preloads, async, inheritance depth…).

The part that makes this work without a labelling budget: **the reward signal
is free**. After each file's review, the verifier has already classified every
finding as confirmed or refuted — so

```text
reward = n_confirmed − n_refuted − λ · (tokens / 1000) · relative_cost
```

is computed online, and the bandit persists across runs
(`.kimi/router/linucb.json`). Clean files earn only the (negative) cost term,
which pushes easy work toward cheaper tiers over time; files where findings
hold up under verification reward the stronger models that found them.

- **Arms** are resolved from available API keys: DeepSeek V4 Flash / V4 Pro,
  Kimi highspeed / standard / K3, plus Claude Sonnet when `ANTHROPIC_API_KEY`
  is set. Preflight pings every configured provider; one with a bad key is
  excluded from routing and the escalation ladder for that run (its learned
  bandit state is kept — a fixed key should not wipe history).
- **Escalation**: if a cheap tier's API call fails, the per-file retry ladder
  climbs to the next tier instead of retrying the same one. Auth errors (401)
  skip the backoff entirely — a bad key never heals by waiting.
- **Failures feed back**: a file that exhausts its ladder still updates the
  bandit — with a negative reward for every arm that failed. Without this, a
  broken premium arm stays "cold" forever and keeps attracting UCB traffic
  (learned the hard way: one run routed 62/62 files to a dead-key arm).
- **A/B**: `--no-router` pins all files to the default model as a baseline;
  `ROUTER_ARMS` whitelists tiers; `ROUTER_LAMBDA` tunes cost sensitivity.
- Learning compounds: each run's arm distribution, per-arm mean reward, and
  escalation count land in `run_metadata.json` → `routing_plan.json`.

Idea lineage: RouteLLM (arXiv 2406.18665) and *Learning to Route LLMs from
Bandit Feedback* (arXiv 2510.07429) — here applied to code review, where the
verifier replaces human preference labels.

### 10. Self-Improving Prompts (Offline Prompt Evolution)

The review prompt is not hand-tuned folklore — it evolves against the
human-labelled golden set (49 findings over 32 files, each with reviewer
notes). `agents/gepa.py` implements the core of GEPA (arXiv 2507.19457,
Algorithm 1/2), adapted to code review:

1. **Fitness by replay, per instance**: a candidate prompt re-reviews
   golden-covered files; findings are matched against human verdicts
   (file + assertion entity, or line ±3). Scores are tracked **per file** —
   the instance-level score matrix the Pareto frontier needs.
2. **Reflective mutation on traces**: the mutator model reads the parent's
   template plus its *execution traces* — each produced finding annotated
   with the verifier's verdict, the human-rejection notes for reproduced
   FPs, and the confirmed bugs it missed — then rewrites the template.
   Candidates that lose the `{filename}`/`{autoload_list}`/`{context_and_code}`
   placeholders are discarded before spending a single evaluation call.
3. **Acceptance gate**: a mutant enters the pool only if it beats its parent
   on the same minibatch. Rejected mutants never spend a validation pass.
4. **Pareto parent selection** (not greedy best-mean — the paper shows greedy
   collapses to local optima): per-instance winners are unioned, dominated
   candidates removed, and the parent is sampled proportional to win count.
   A prompt that is the only one that handles a certain file *type* keeps its
   DNA in the pool even with a mediocre average.
5. **Cost discipline**: minibatch reflection on a cheap tier
   (deepseek-v4-flash), per-(candidate, file) replay caching, and the
   acceptance gate together keep a 6-iteration run around 20 minutes.

```bash
python -m agents.gepa --iterations 6 --minibatch 8 --pareto-size 12
python -m agents.gepa --dry-run            # validate setup, no LLM calls
```

The winner is written to `prompts/review_prompt_evolved.txt` only if it beats
the baseline on the held-out `D_pareto` split (never overwrites the
production prompt automatically); `eval/gepa_report.md` records every
candidate's lineage, gate rejections, and per-instance scores.
Simplification vs. the paper: System-Aware Merge (crossover between
complementary candidates) is not yet implemented.

Known limitation: a file with zero findings can't distinguish "genuinely
clean" from "model looked away" — the verifier only judges what was proposed.
That's accepted in v1 and partly covered by the playtest layer catching
runtime-visible breakage regardless of which model reviewed the file.

## Project Structure

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

## Configuration Options

### Environment Variables

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `LLM_PROVIDER` | Primary LLM: `claude` or `kimi` | Auto-detect |
| `LLM_FALLBACK` | Enable fallback provider | `0` |
| `ANTHROPIC_API_KEY` | Claude API key | - |
| `MOONSHOT_API_KEY` | Kimi API key | - |
| `DEEPSEEK_API_KEY` | DeepSeek API key (enables v4-flash / v4-pro router arms) | - |
| `DEEPSEEK_TEXT_MODEL` | DeepSeek pro-tier model | `deepseek-v4-pro` |
| `DEEPSEEK_FLASH_MODEL` | DeepSeek flash-tier model | `deepseek-v4-flash` |
| `KIMI_TEXT_MODEL` | Kimi model for code review | `kimi-k2.7-code` |
| `KIMI_VISION_MODEL` | Kimi model for screenshots | `moonshot-v1-8k-vision-preview` |

### Godot Executable

`tools/godot_runner.py` resolves the binary in this order:

1. `GODOT_EXECUTABLE` environment variable (used in CI)
2. `godot` on `PATH`
3. Hardcoded local path in `tools/godot_runner.py` (edit it to match your install)

### Pipeline Options

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

## Testing

```bash
# Run unit tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_llm_client.py

# Run with verbose output
python -m pytest tests/ -v
```

## Evaluation Datasets (precision vs recall)

One dataset can't measure both. This project keeps two, on purpose:

- **`eval/golden_set.jsonl` — precision.** 49 past pipeline findings with
  human verdicts (`tp`/`fp`/`partial`) and reviewer notes. It answers "of the
  things we report, how many are real?" It cannot answer "of the bugs that
  exist, how many do we find?" — bugs no run ever produced are invisible to
  it. `python eval/run_eval.py` prints the benchmark.
- **`eval/mutant_set.jsonl` — recall.** Built by `eval/mutant_eval.py`: the
  game project is copied, and one known bug per file is mechanically
  injected (renamed signal emissions, dead-guarded emissions, broken preload
  paths, undefined class references — each maps to a bug class the pipeline
  claims to catch). Detection is binary ground truth, so
  `recall = detected / injected` finally has a denominator. The game repo is
  never touched; mutants live in a gitignored copy under `.kimi/`.

Known limitation, stated plainly: injected bugs are cleaner than natural
ones (real bugs don't announce their type), so mutant recall is an upper
bound of true recall. Precision stays with the natural golden set.

## Troubleshooting

### "Godot executable not found"

Update `GODOT_EXECUTABLE` in `tools/godot_runner.py` with your Godot path.

### "API key rejected"

Check your `.env` file and ensure the correct key is set for your `LLM_PROVIDER`.

### "Static-only mode"

Pipeline runs without LLM analysis. Set `ANTHROPIC_API_KEY` or `MOONSHOT_API_KEY` in `.env`.

### Index not updating

Force rebuild:

```bash
python review.py --project PATH --rebuild-index
```

## License

MIT License - See LICENSE file for details.

## Credits

Built for the [Harvester Godot 4 space mining game](https://github.com/gdquest-demos/godot-2d-space-game) project using:

- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- [Anthropic Claude](https://www.anthropic.com/claude) & [Moonshot Kimi](https://www.moonshot.cn/) for LLM analysis
- Godot 4 CLI for static and runtime validation
