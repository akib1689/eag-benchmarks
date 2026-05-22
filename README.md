# EAG Benchmarks

> Benchmark harness for **Execution-Augmented Generation** — comparing text-to-SQL paradigms against the [BIRD](https://bird-bench.github.io/) dataset.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-teal.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)
[![uv](https://img.shields.io/badge/managed%20with-uv-blue.svg)](https://github.com/astral-sh/uv)

---

## What is EAG?

**Execution-Augmented Generation** is a paradigm where the LLM generates executable code/queries (SQL, Python) that run **locally** against real data — the LLM never sees raw data, only schemas and computed results.

This harness compares EAG against baseline paradigms on the text-to-SQL task:

| Paradigm | How it works | Status |
| -------- | ------------ | ------ |
| **ReAct** | Interleaved reasoning + SQL generation | Working |
| **PAL** | Program-Aided Language — Python constructs SQL | Stub |
| **PoT** | Program-of-Thoughts — declarative reasoning → SQL | Stub |
| **EAG** | Schema-only plan generation + local execution | Stub |

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
│   ├── base.py       # Abstract base class — run(task) → {sql, usage, latency_ms, error}
│   ├── react.py      # Full ReAct implementation
│   ├── pal.py        # PAL stub
│   ├── pot.py        # PoT stub
│   └── eag.py        # EAG stub (NotImplementedError)
├── datasets/
│   └── bird/
│       └── loader.py # BIRD Mini-Dev loader — schema-only, no raw data exposure
├── eval/
│   ├── executor.py   # Sandboxed SQLite execution + execution_accuracy()
│   └── metrics.py    # BenchmarkMetrics, Timer
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

### Adding a New Agent

1. Create `agents/your_agent.py` inheriting from `AgentABC`
2. Implement `run(task)` returning `{sql, usage, latency_ms, error}`
3. Register it in `benchmarks/run.py` under `AGENTS`

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
