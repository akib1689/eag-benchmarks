# AGENTS.md

## Project

Benchmark harness for Execution-Augmented Generation (EAG). Compares text-to-SQL paradigms (ReAct, PAL, PoT, EAG) against the BIRD Mini-Dev dataset. Apache 2.0 licensed.

EAG is a zero-data agent architecture: the LLM operates as a schema-constrained planner, never receiving raw data. See `docs/` for the theoretical model (planner/executor separation, dual-stage validation, threat model).

## Commands

```bash
uv sync                              # install deps (use uv, not pip/poetry)
uv run python -m benchmarks.run      # run benchmark CLI
uv run ruff check .                  # lint
uv run ruff check --fix .            # lint with auto-fix
```

CLI flags: `--agent {react,pal,pot,eag}` `--model {groq,glm}` `--dataset {bird}` `--samples N`

## Setup before running

1. `cp .env.example .env` and set API keys:
   - `GROQ_API_KEY` — required for `--model groq`
   - `GLM_API_KEY` — required for `--model glm` (optional `GLM_BASE_URL` override)
2. Download BIRD data: `bash scripts/download_bird.sh` — extracts into `data/bird/` (gitignored)

## Architecture

```
llm/provider.py    → LLMInterface (abstract) — generate(), get_usage()
llm/groq.py        → GroqClient (OpenAI-compatible, gpt-oss-120b)
llm/glm.py         → GLMClient (OpenAI-compatible, glm-5.1)

datasets/bird/loader.py → load_mini_dev(), get_schema() — schema-only, no raw data exposure
eval/executor.py        → execute_sql() (sandboxed, read-only pragma), execution_accuracy()
eval/metrics.py         → BenchmarkMetrics, Timer

agents/base.py   → AgentABC abstract class, extract_sql() helper
agents/react.py  → Full implementation (reasoning → FINAL_SQL)
agents/pal.py    → Working (Python-aided SQL construction → FINAL_SQL)
agents/pot.py    → Working (declarative program-of-thoughts → FINAL_SQL)
agents/eag.py    → Stub only (raises NotImplementedError)

configs/models.yaml    → Model provider configs (temperature, base_url, etc.)
configs/datasets.yaml  → Dataset split and path config
```

All agents conform to `AgentABC.run(task) → {sql, usage, latency_ms, error}`.
Both LLM clients use the `openai` library with different `base_url` and include 3-retry exponential backoff.

### Extending

- **New agent**: Create `agents/<name>.py` inheriting `AgentABC`, implement `run()` + `name`, register in `benchmarks/run.py` under `AGENTS`.
- **New LLM provider**: Create `llm/<name>.py` inheriting `LLMInterface`, implement `generate()` + `get_usage()`, register in `benchmarks/run.py` under `MODELS`.

### SQL extraction convention

Agents extract SQL from LLM output via `FINAL_SQL:` prefix. Fallback: `extract_sql()` in `base.py` handles ```sql fences.

## Key conventions

- `benchmarks/run.py` loads `.env` via `python-dotenv` before importing project modules — E402 noqa comments on those imports are intentional
- Hatchling build system; packages are explicitly listed in `[tool.hatch.build.targets.wheel]`
- Ruff config is under `[tool.ruff.lint]` (not top-level `[tool.ruff]`) — `line-length = 100`, rules `E, F, I`
- `results/*.json` are gitignored but `results/.gitkeep` is tracked
- No test suite exists yet — `pytest` is a dev dependency but no test files are present
- `BIRD_ROOT` is hardcoded as `Path("data/bird")` in both `eval/executor.py` and `datasets/bird/loader.py`
- `docs/` contains research/theory documentation (EAG architecture, validation, related work) — not implementation docs

## Gotchas

- BIRD Mini-Dev data must be downloaded separately (~200 MB); repo has no data
- `data/` is gitignored — never commit datasets
- The `eag` agent will raise `NotImplementedError` — do not use `--agent eag`
- `load_dotenv()` must run before any LLM client instantiation (they read env vars at init)
- Using `--model glm` without `GLM_API_KEY` set will raise `ValueError` at client init
