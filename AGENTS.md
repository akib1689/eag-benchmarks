# AGENTS.md

## Project

Benchmark harness for Execution-Augmented Generation (EAG). Compares text-to-SQL paradigms (ReAct, PAL, PoT, EAG) against the BIRD Mini-Dev dataset. Apache 2.0 licensed.

## Commands

```bash
uv sync                              # install deps (use uv, not pip/poetry)
uv run python -m benchmarks.run      # run benchmark CLI
uv run ruff check .                  # lint
uv run ruff check --fix .            # lint with auto-fix
```

CLI flags: `--agent {react,pal,pot,eag}` `--model {groq,glm}` `--dataset {bird}` `--samples N`

## Setup before running

1. `cp .env.example .env` and set `GROQ_API_KEY` (required for groq model)
2. Download BIRD data: `bash scripts/download_bird.sh` — extracts into `data/bird/` (gitignored)

## Architecture

```
llm/provider.py    → LLMInterface (abstract) — swap providers with zero code changes
llm/groq.py        → GroqClient (primary, gpt-oss-120b)
llm/glm.py         → GLMClient (stub, glm-5.1)

datasets/bird/loader.py → load_mini_dev(), get_schema() — schema-only, no raw data exposure
eval/executor.py        → execute_sql() (sandboxed, read-only pragma), execution_accuracy()
eval/metrics.py         → BenchmarkMetrics, Timer

agents/base.py   → AgentABC abstract class, extract_sql() helper
agents/react.py  → Full implementation (reasoning → FINAL_SQL)
agents/pal.py    → Stub (prompt template only)
agents/pot.py    → Stub (prompt template only)
agents/eag.py    → Stub (raises NotImplementedError)
```

All agents conform to `AgentABC.run(task) → {sql, usage, latency_ms, error}`.

## Key conventions

- `benchmarks/run.py` loads `.env` via `python-dotenv` before importing project modules — E402 noqa comments on those imports are intentional
- Hatchling build system; packages are explicitly listed in `[tool.hatch.build.targets.wheel]`
- Ruff config is under `[tool.ruff.lint]` (not top-level `[tool.ruff]`) — `line-length = 100`, rules `E, F, I`
- `results/*.json` are gitignored but `results/.gitkeep` is tracked

## Gotchas

- BIRD Mini-Dev data must be downloaded separately (~200 MB); repo has no data
- `data/` is gitignored — never commit datasets
- The `eag` agent will raise `NotImplementedError` — do not use `--agent eag`
- `load_dotenv()` must run before any LLM client instantiation (they read env vars at init)
