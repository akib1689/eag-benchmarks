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

CLI flags: `--agent {react,pal,pot,eag}` `--model {groq,glm}` `--dataset {bird}` `--samples N`

## Setup before running

1. `cp .env.example .env` and set API keys:
   - `GROQ_API_KEY` — required for `--model groq`
   - `GLM_API_KEY` — required for `--model glm` (optional `GLM_BASE_URL` override)
2. Download BIRD data: `bash scripts/download_bird.sh` — extracts into `data/bird/` (gitignored)

## Architecture

### Evaluation pipeline (answer-based, not SQL-based)

The evaluator never sees SQL. Gold answers come from executing gold SQL internally. Agent answers are extracted from raw output and compared against gold via tiered matching.

```
benchmarks/run.py
  → eval/gold.py           get_gold_answer(gold_sql, db_id) → normalized gold answer
  → eval/answer_extractor.py  extract_answer(raw_output) → parsed agent answer
  → eval/comparators.py    compare_answers(pred, gold) → (correct, confidence, tier)
                            tiers: exact (1.0) → set (0.95) → fuzzy (0.99, 1% tolerance)
  → eval/normalizer.py     type coercion, NULL handling, shape detection (scalar/row/column/table)
  → eval/metrics.py        BenchmarkMetrics (match_tiers, parse_failures, avg_confidence)
```

### Agent contract

All agents return `{answer, raw_output, usage, latency_ms, error}`. The `answer` field is what gets evaluated. Agents produce answers — not SQL. The `sql` field does not exist in the contract.

Internal SQL generation is fine (e.g., ReAct runs `Execute[sql]` as a tool call), but the final output must be an answer.

```
agents/base.py    → AgentABC abstract class, extract_sql() (internal utility only)
agents/react.py   → Reasoning agent (Phase 2: multi-step Thought/Action/Observation loop)
agents/pal.py     → PAL agent (transitional — produces SQL as answer pending Phase 2)
agents/pot.py     → PoT agent (transitional — produces SQL as answer pending Phase 2)
agents/eag.py     → Stub only (raises NotImplementedError)
```

### Tools

Agents interact with the environment through the `tools/` directory. Tools receive `db_id` as a parameter per call.

```
tools/__init__.py      → ToolABC base class, @register_tool decorator, registry
tools/execute_sql.py   → ExecuteSQLTool — sandboxed SQLite execution, formatted text output
```

To add a new tool: create `tools/<name>.py`, inherit `ToolABC`, implement `name`, `description`, `run(params)`, add `@register_tool`.

### LLM layer

```
llm/provider.py    → LLMInterface (abstract) — generate(), get_usage()
llm/groq.py        → GroqClient (OpenAI-compatible, gpt-oss-120b)
llm/glm.py         → GLMClient (OpenAI-compatible, glm-5.1)
```

Both clients use the `openai` library with different `base_url` and include 3-retry exponential backoff.

### Other modules

```
datasets/bird/loader.py → load_mini_dev(), get_schema() — schema-only, no raw data exposure
eval/executor.py        → execute_sql() — low-level utility, dual-path DB resolution
configs/models.yaml     → Model provider configs (temperature, base_url, etc.)
configs/datasets.yaml   → Dataset split and path config
```

### Extending

- **New agent**: Create `agents/<name>.py` inheriting `AgentABC`, implement `run()` returning `{answer, raw_output, ...}` + `name`, register in `benchmarks/run.py` under `AGENTS`.
- **New LLM provider**: Create `llm/<name>.py` inheriting `LLMInterface`, implement `generate()` + `get_usage()`, register in `benchmarks/run.py` under `MODELS`.
- **New tool**: Create `tools/<name>.py`, inherit `ToolABC`, add `@register_tool`, implement `run(params)`.

## Key conventions

- `benchmarks/run.py` loads `.env` via `python-dotenv` before importing project modules — E402 noqa comments on those imports are intentional
- Hatchling build system; packages are explicitly listed in `[tool.hatch.build.targets.wheel]` — **must include `tools`** when adding new packages
- Ruff config is under `[tool.ruff.lint]` (not top-level `[tool.ruff]`) — `line-length = 100`, rules `E, F, I`
- `results/*.json` are gitignored but `results/.gitkeep` is tracked
- No test suite exists yet — `pytest` is a dev dependency but no test files are present
- `BIRD_ROOT = Path("data/bird")` is hardcoded in 4 files: `eval/executor.py`, `eval/gold.py`, `tools/execute_sql.py`, `datasets/bird/loader.py`
- `docs/` contains research/theory documentation (EAG architecture, validation, related work) — not implementation docs

## Gotchas

- BIRD Mini-Dev data must be downloaded separately (~200 MB); repo has no data
- Data lives at `data/bird/minidev/MINIDEV/` — three modules (`tools/`, `eval/executor.py`, `eval/gold.py`) use dual-path resolution (try `minidev/MINIDEV/dev_databases/` first, fall back to `dev_databases/`). `datasets/bird/loader.py` only checks the fallback path and may fail if data layout changes.
- `data/` is gitignored — never commit datasets
- The `eag` agent will raise `NotImplementedError` — do not use `--agent eag`
- `load_dotenv()` must run before any LLM client instantiation (they read env vars at init)
- Using `--model glm` without `GLM_API_KEY` set will raise `ValueError` at client init
- `ExecuteSQLTool` returns full result tables (up to 50 rows displayed + total count). Large result sets are truncated in display but the full row count is included.
- Answer extractor tries strategies in order: `Finish[answer]` → JSON → numeric scalar → last-line → fallback. The `Finish[answer]` pattern is the intended ReAct output format.
