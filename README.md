# EAG Benchmarks

> Benchmark harness for **Execution-Augmented Generation** — comparing answer-producing agent paradigms against the [BIRD](https://bird-bench.github.io/) dataset.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-teal.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)
[![uv](https://img.shields.io/badge/managed%20with-uv-blue.svg)](https://github.com/astral-sh/uv)

---

## What is EAG?

**Execution-Augmented Generation** is a paradigm where the LLM generates executable code/queries (SQL, Python) that run **locally** against real data — the LLM never sees raw data, only schemas and computed results.

This harness compares EAG against baseline paradigms on the text-to-SQL task:

| Paradigm | How it works | Status |
| -------- | ------------ | ------ |
| **ReAct** | Interleaved reasoning + tool execution → answer | Working |
| **PAL** | Program-Aided Language — Python constructs SQL → answer | Transitional |
| **PoT** | Program-of-Thoughts — declarative reasoning → answer | Transitional |
| **EAG** | Schema-only plan generation + local execution → answer | Stub |

**Evaluation compares answers, not SQL.** Each agent produces an answer to the natural language question. The evaluator compares it against the gold answer (computed by executing the gold SQL internally) using an LLM-based verdict checker.

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- A [LiteLLM proxy](https://docs.litellm.ai/docs/proxy) running with your model keys configured

### Setup

```bash
# 1. Install dependencies
uv sync

# 2. Download BIRD Mini-Dev dataset (~200 MB)
bash scripts/download_bird.sh

# 3. Configure API keys
cp .env.example .env
# Edit .env and add your LITELLM_API_KEY
```

### Run a Benchmark

```bash
# Evaluate ReAct agent on 10 samples with Groq
uv run python -m benchmarks.run --agent react --samples 10

# With a different model
uv run python -m benchmarks.run --agent react --model openrouter --samples 5

# With live step-by-step tracing
uv run python -m benchmarks.run --agent react --samples 10 --trace

# Skip LLM verdict checker (faster, only checks non-empty answers)
uv run python -m benchmarks.run --agent react --samples 10 --no-verdict
```

**CLI Options:**

| Flag | Choices | Default | Description |
| ---- | ------- | ------- | ----------- |
| `--agent` | `react`, `pal`, `pot`, `eag` | `react` | Agent paradigm to evaluate |
| `--model` | `groq`, `openrouter` | `groq` | LLM provider |
| `--dataset` | `bird` | `bird` | Dataset to evaluate on |
| `--samples` | Any integer | `10` | Number of samples from dev set |
| `--trace` | — | off | Print live step-by-step agent execution |
| `--delay` | Any float | `2.0` | Seconds to wait between questions |
| `--no-verdict` | — | off | Disable LLM verdict checker |

Results are saved to `results/` as timestamped JSON files.

## Architecture

```
eag-benchmarks/
├── agents/           # Agent implementations (each conforms to AgentABC)
│   ├── base.py       # Abstract base class — run(task) → {answer, raw_output, usage, latency_ms, error, steps}
│   ├── react.py      # ReAct agent (native LLM tool calling, iterative loop)
│   ├── pal.py        # PAL agent (transitional)
│   ├── pot.py        # PoT agent (transitional)
│   └── eag.py        # EAG stub (NotImplementedError)
├── tools/            # Agent tools for environment interaction
│   ├── __init__.py   # ToolABC base class, @register_tool decorator, registry
│   ├── execute_sql.py # ExecuteSQLTool — sandboxed SQLite execution
│   └── get_schema.py # GetSchemaTool — enriched schema with descriptions, PKs, FKs, row counts
├── eval/             # Answer-based evaluation pipeline
│   ├── gold.py       # Gold answer generation (executes gold SQL → normalized result)
│   ├── verdict.py    # VerdictChecker — LLM-based semantic comparison (match/wrong/mismatch/etc.)
│   ├── normalizer.py # Type coercion, NULL handling, shape detection
│   ├── metrics.py    # BenchmarkMetrics (verdicts, parse_failures, avg_confidence)
│   └── executor.py   # Low-level SQL execution utility
├── datasets/
│   └── bird/
│       └── loader.py # BIRD Mini-Dev loader — schema-only, no raw data exposure
├── llm/
│   ├── provider.py      # LLMInterface (abstract) — generate(), chat(), get_usage()
│   └── litellm_client.py # LiteLLMClient — unified adapter via LiteLLM proxy
├── benchmarks/
│   └── run.py        # CLI entry point
├── configs/          # Model and dataset YAML configs
├── scripts/
│   └── download_bird.sh  # Download + extract BIRD Mini-Dev
└── results/          # Run artifacts (gitignored)
```

### Evaluation Flow

```
Agent output (answer field) ─────────────────────────┐
                                                      ├→ VerdictChecker.check() → VerdictResult
Gold SQL → execute → normalize_result() → gold answer─┘
```

### Adding a New Agent

1. Create `agents/your_agent.py` inheriting from `AgentABC`
2. Implement `run(task)` returning `{answer, raw_output, usage, latency_ms, error, steps}`
3. Register it in `benchmarks/run.py` under `AGENTS`

### Adding a New Tool

1. Create `tools/your_tool.py` inheriting from `ToolABC`
2. Add `@register_tool` decorator, implement `name`, `description`, `run(params)`
3. Tools receive `db_id` as a parameter per call — do not store it at construction time

### Adding a New LLM Provider

1. Add an entry to `configs/models.yaml` with the model alias registered on your LiteLLM proxy
2. Add a lambda in `benchmarks/run.py` under `MODELS` mapping to `LiteLLMClient(config_name="...")`

## BIRD Mini-Dev

This project uses the [BIRD Mini-Dev](https://github.com/bird-bench/mini_dev) dataset — a curated subset of 500 text-to-SQL pairs across 11 databases.

Each example contains:
- **question** — natural language question about a database
- **evidence** — expert-annotated domain knowledge hints
- **SQL** — ground truth SQLite query
- **db_id** — which database to query
- **difficulty** — `simple` (30%), `moderate` (50%), `challenging` (20%)

## Development

```bash
uv run ruff check .          # lint
uv run ruff check --fix .    # lint with auto-fix
```

## License

[Apache License 2.0](LICENSE)
