# A Simple Godot Game Code Test Agent

An AI-powered code review and quality assurance pipeline for Godot 4 GDScript game projects. Built and test by running the "Harvester" space mining game and is adaptable to other Godot project.

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
                           ┌────────────────────────┐
                           │  [pipeline_check  │    │
                           │   code_review]    │◄───┘ (parallel)
                           └──────────┬───────────┘
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
                                    ┌─────┐
                                    │ END │
                                    └─────┘
```

### Pipeline Nodes

| Node | Purpose |
| ---- | ------- |
| `preflight_check` | Validate API keys, Godot binary, project.godot |
| `build_index` | Scan project, build mtime-cached symbol index |
| `resolve_scope` | Git diff detection + dependency expansion |
| `pipeline_check` | Static analysis (import, syntax, resources) |
| `code_review` | LLM-powered review with per-file retry |
| `cross_correlator` | Correlate findings, detect index_gap patterns |
| `filter_refuted` | Remove refuted (final_severity="drop") findings |
| `findings_dedup` | Cascading deduplication |
| `report_merger` | Generate final markdown + JSON reports |

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

Built for the Harvester Godot 4 space mining game project using:

- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- [Anthropic Claude](https://www.anthropic.com/claude) & [Moonshot Kimi](https://www.moonshot.cn/) for LLM analysis
- Godot 4 CLI for static and runtime validation
