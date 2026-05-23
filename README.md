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
| **ReAct** | Interleaved reasoning + tool execution → answer | Working (Phase 1) |
| **PAL** | Program-Aided Language — Python constructs SQL → answer | Transitional |
| **PoT** | Program-of-Thoughts — declarative reasoning → answer | Transitional |
| **EAG** | Schema-only plan generation + local execution → answer | Stub |

**Evaluation compares answers, not SQL.** Each agent produces an answer to the natural language question. The evaluator compares it against the gold answer (computed by executing the gold SQL internally) using tiered matching: exact → set equivalence → fuzzy numeric.

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- A [Groq](https://console.groq.com/) API key

### Setup

```bash
# 1. Install dependencies
uv sync

# 2. Download BIRD Mini-Dev dataset (~200 MB)
bash scripts/download_bird.sh

# 3. Configure API keys
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Run a Benchmark

```bash
# Evaluate ReAct agent on 10 samples with Groq
uv run python -m benchmarks.run --agent react --samples 10

# With a different model
uv run python -m benchmarks.run --agent react --model glm --samples 5
```

**CLI Options:**

| Flag | Choices | Default | Description |
| ---- | ------- | ------- | ----------- |
| `--agent` | `react`, `pal`, `pot`, `eag` | `react` | Agent paradigm to evaluate |
| `--model` | `groq`, `glm` | `groq` | LLM provider |
| `--dataset` | `bird` | `bird` | Dataset to evaluate on |
| `--samples` | Any integer | `10` | Number of samples from dev set |

Results are saved to `results/` as timestamped JSON files.

## Architecture

```
eag-benchmarks/
├── agents/           # Agent implementations (each conforms to AgentABC)
│   ├── base.py       # Abstract base class — run(task) → {answer, raw_output, usage, latency_ms, error}
│   ├── react.py      # ReAct agent (reasoning + tool execution)
│   ├── pal.py        # PAL agent (transitional)
│   ├── pot.py        # PoT agent (transitional)
│   └── eag.py        # EAG stub (NotImplementedError)
├── tools/            # Agent tools for environment interaction
│   ├── __init__.py   # ToolABC base class, @register_tool decorator, registry
│   └── execute_sql.py # ExecuteSQLTool — sandboxed SQLite execution
├── eval/             # Answer-based evaluation pipeline
│   ├── gold.py       # Gold answer generation (executes gold SQL → normalized result)
│   ├── answer_extractor.py # Parse agent output → structured answer
│   ├── comparators.py # Three-tier matching: exact → set → fuzzy
│   ├── normalizer.py # Type coercion, NULL handling, shape detection
│   ├── metrics.py    # BenchmarkMetrics (match_tiers, parse_failures, avg_confidence)
│   └── executor.py   # Low-level SQL execution utility
├── datasets/
│   └── bird/
│       └── loader.py # BIRD Mini-Dev loader — schema-only, no raw data exposure
├── llm/
│   ├── provider.py   # LLMInterface (abstract) — swap providers with zero code changes
│   ├── groq.py       # GroqClient (gpt-oss-120b)
│   └── glm.py        # GLMClient (glm-5.1)
├── benchmarks/
│   └── run.py        # CLI entry point
├── configs/          # Model and dataset YAML configs
├── scripts/
│   └── download_bird.sh  # Download + extract BIRD Mini-Dev
└── results/          # Run artifacts (gitignored)
```

### Evaluation Flow

```
Agent output → extract_answer() → canonical form ─┐
                                                    ├→ compare_answers() → (correct, confidence, tier)
Gold SQL → execute → normalize_result() ───────────┘
```

### Adding a New Agent

1. Create `agents/your_agent.py` inheriting from `AgentABC`
2. Implement `run(task)` returning `{answer, raw_output, usage, latency_ms, error}`
3. Register it in `benchmarks/run.py` under `AGENTS`

### Adding a New Tool

1. Create `tools/your_tool.py` inheriting from `ToolABC`
2. Add `@register_tool` decorator, implement `name`, `description`, `run(params)`
3. Tools receive `db_id` as a parameter per call — do not store it at construction time

### Adding a New LLM Provider

1. Create `llm/your_provider.py` inheriting from `LLMInterface`
2. Implement `generate()` and `get_usage()`
3. Register it in `benchmarks/run.py` under `MODELS`

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
