# AGENTS.md

## Project

Benchmark harness for Execution-Augmented Generation (EAG). Compares answer-producing agent paradigms (ReAct, PAL, PoT, EAG) against the BIRD Mini-Dev dataset. Apache 2.0 licensed.

**Core thesis**: EAG (metadata-only, zero data access) can answer BIRD questions as accurately as data-access agents. Evaluation compares **answers**, not SQL.

## Commands

```bash
uv sync                              # install deps (use uv, not pip/poetry)
uv run python -m benchmarks.run      # run benchmark CLI
uv run ruff check .                  # lint
uv run ruff check --fix .            # lint with auto-fix
```

CLI flags:
- `--agent {react,pal,pot,eag}` — agent paradigm (default: `react`)
- `--model {groq,openrouter}` — LLM provider (default: `groq`)
- `--dataset {bird}` — dataset (default: `bird`)
- `--samples N` — number of samples (default: `10`)
- `--trace` — print live step-by-step agent execution
- `--delay N` — seconds between questions (default: `2.0`)
- `--no-verdict` — disable LLM-based verdict checker (only checks non-empty)

## Setup before running

1. `cp .env.example .env` and set API keys:
   - `LITELLM_API_KEY` — virtual key for your LiteLLM proxy (required for all models)
   - `LITELLM_BASE_URL` — proxy URL (defaults to `http://localhost:4000`)
2. Download BIRD data: `bash scripts/download_bird.sh` — extracts into `data/bird/` (gitignored)

## Architecture

### Evaluation pipeline (answer-based, not SQL-based)

The evaluator never sees SQL. Gold answers come from executing gold SQL internally. Agent answers are compared against gold via an LLM-based verdict checker.

```
benchmarks/run.py
  → eval/gold.py           get_gold_answer(gold_sql, db_id) → normalized gold answer
  → eval/verdict.py        VerdictChecker.check() → VerdictResult (match/wrong_answer/mismatch/parse_error/unclear)
  → eval/normalizer.py     normalize_result(), canonicalize() — type coercion, NULL handling, shape detection
  → eval/metrics.py        BenchmarkMetrics (verdicts, parse_failures, avg_confidence)
```

The verdict checker (`eval/verdict.py`) uses a smaller LLM (`groq-verdict` config) with structured output (`VerdictResult` Pydantic model) to semantically compare agent answers against gold answers. It handles numeric tolerance, case-insensitive strings, set-equivalent lists, and type mismatches.

### Agent contract

All agents return `{answer, raw_output, usage, latency_ms, error, steps}`. The `answer` field is what gets evaluated. Agents produce answers — not SQL.

```
agents/base.py    → AgentABC abstract class, extract_sql() (internal utility only)
agents/react.py   → ReAct agent — native LLM tool calling with iterative loop
                    Terminates via finish tool call or plain text response
agents/pal.py     → PAL agent — generates Python that constructs SQL, extracts SQL as answer
agents/pot.py     → PoT agent — declarative reasoning program, extracts SQL as answer
agents/eag.py     → Stub only (raises NotImplementedError)
```

### Tools

Agents interact with the environment through the `tools/` directory. Tools receive `db_id` as a parameter per call.

```
tools/__init__.py      → ToolABC base class, @register_tool decorator, registry
tools/execute_sql.py   → ExecuteSQLTool — sandboxed SQLite execution, formatted text output
tools/get_schema.py    → GetSchemaTool — enriched schema with column descriptions, PKs, FKs,
                        distinct values, date format detection, row counts
```

To add a new tool: create `tools/<name>.py`, inherit `ToolABC`, implement `name`, `description`, `run(params)`, add `@register_tool`.

### LLM layer

```
llm/provider.py        → LLMInterface (abstract) — generate(), chat(), get_usage()
llm/litellm_client.py  → LiteLLMClient — unified adapter, reads configs/models.yaml
configs/models.yaml    → Per-model config (model alias, temperature, max_tokens, etc.)
```

All models are accessed through `LiteLLMClient` using the litellm SDK with the `litellm_proxy/` prefix. The client reads model configuration from `configs/models.yaml` by name. A locally-running LiteLLM proxy handles provider routing, key rotation, and rate-limit retries. The benchmark harness only needs a virtual key + proxy URL.

### Other modules

```
datasets/bird/loader.py → load_mini_dev(), get_schema() — schema-only, no raw data exposure
eval/executor.py        → execute_sql() — low-level utility, dual-path DB resolution
configs/models.yaml     → Model provider configs (groq, groq-verdict, openrouter)
configs/datasets.yaml   → Dataset split and path config
```

### Extending

- **New agent**: Create `agents/<name>.py` inheriting `AgentABC`, implement `run()` returning `{answer, raw_output, ...}` + `name`, register in `benchmarks/run.py` under `AGENTS`.
- **New LLM provider**: Add an entry to `configs/models.yaml`, then add a lambda in `benchmarks/run.py` under `MODELS` mapping to `LiteLLMClient(config_name="...")`.
- **New tool**: Create `tools/<name>.py`, inherit `ToolABC`, add `@register_tool`, implement `run(params)`.

## Key conventions

- `benchmarks/run.py` loads `.env` via `python-dotenv` before importing project modules — E402 noqa comments on those imports are intentional
- Hatchling build system; packages are explicitly listed in `[tool.hatch.build.targets.wheel]` — **must include `tools`** when adding new packages
- Ruff config is under `[tool.ruff.lint]` (not top-level `[tool.ruff]`) — `line-length = 100`, rules `E, F, I`
- `results/*.json` are gitignored but `results/.gitkeep` is tracked
- No test suite exists yet — `pytest` is a dev dependency but no test files are present
- `BIRD_ROOT = Path("data/bird")` is hardcoded in 5 files: `eval/executor.py`, `eval/gold.py`, `tools/execute_sql.py`, `tools/get_schema.py`, `datasets/bird/loader.py`
- `docs/` contains research/theory documentation (EAG architecture, validation, related work) — not implementation docs

## Gotchas

- BIRD Mini-Dev data must be downloaded separately (~200 MB); repo has no data
- Data lives at `data/bird/minidev/MINIDEV/` — most modules use dual-path resolution (try `minidev/MINIDEV/dev_databases/` first, fall back to `dev_databases/`). `datasets/bird/loader.py` also checks dual paths for the JSON file, but `get_schema()` only checks the fallback path and may fail if data layout changes.
- `data/` is gitignored — never commit datasets
- The `eag` agent will raise `NotImplementedError` — do not use `--agent eag`
- `load_dotenv()` must run before any LLM client instantiation (they read env vars at init)
- Using any `--model` without `LITELLM_API_KEY` set will raise `ValueError` at client init
- `ExecuteSQLTool` returns full result tables (up to 50 rows displayed + total count). Large result sets are truncated in display but the full row count is included.
- ReAct agent uses native LLM tool calling — the `finish` tool submits the answer, or the agent can respond with plain text. There is no `Finish[answer]` text pattern matching.
- PAL and PoT agents extract SQL as their answer (via `FINAL_SQL:` marker or `extract_sql()`). They do not execute SQL — the SQL text itself is the answer. This means they will generally score poorly until upgraded.
- The verdict checker uses the `groq-verdict` model config (smaller model, 512 max tokens). It requires the same `LITELLM_API_KEY`. Use `--no-verdict` to skip it.
