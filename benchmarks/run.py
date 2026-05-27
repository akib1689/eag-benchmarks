#!/usr/bin/env python3
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agents.eag import EAGAgent  # noqa: E402
from agents.pal import PALAgent  # noqa: E402
from agents.pot import POTAgent  # noqa: E402
from agents.react import ReActAgent  # noqa: E402
from datasets.bird.loader import load_mini_dev  # noqa: E402
from eval.gold import get_gold_answer  # noqa: E402
from eval.metrics import BenchmarkMetrics, Timer  # noqa: E402
from eval.precheck import precheck_match  # noqa: E402
from eval.verdict import VerdictChecker  # noqa: E402
from llm.direct_client import DirectLLMClient  # noqa: E402
from llm.litellm_client import LiteLLMClient  # noqa: E402

AGENTS = {
    "react": lambda llm: ReActAgent(llm),
    "pal": lambda llm: PALAgent(llm),
    "pot": lambda llm: POTAgent(llm),
    "eag": lambda llm: EAGAgent(llm),
}

PROXY_MODELS = {
    "groq": lambda: LiteLLMClient(config_name="groq"),
    "openrouter": lambda: LiteLLMClient(config_name="openrouter"),
}

DIRECT_MODELS = {
    "groq": lambda: DirectLLMClient(config_name="groq"),
    "openrouter": lambda: DirectLLMClient(config_name="openrouter"),
}

PROXY_VERDICT_CONFIG = "groq-verdict"
DIRECT_VERDICT_CONFIG = "groq-verdict"


def parse_args():
    parser = argparse.ArgumentParser(
        description="EAG Benchmarks — evaluate text-to-SQL agents on BIRD"
    )
    parser.add_argument(
        "--agent",
        choices=list(AGENTS.keys()),
        default="react",
        help="Agent paradigm to benchmark (default: react)",
    )
    parser.add_argument(
        "--dataset",
        choices=["bird"],
        default="bird",
        help="Dataset to evaluate on (default: bird)",
    )
    parser.add_argument(
        "--mode",
        choices=["proxy", "direct"],
        default="proxy",
        help="LLM mode: proxy or direct (default: proxy)",
    )
    parser.add_argument(
        "--model",
        choices=list(PROXY_MODELS.keys()),
        default="groq",
        help="LLM provider (default: groq)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples from the dev set (default: 10)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        default=False,
        help="Print live step-by-step agent execution (Thought/Action/Observation)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between questions (default: 2.0)",
    )
    parser.add_argument(
        "--no-verdict",
        action="store_true",
        default=False,
        help="Disable verdict checker (only checks if answer is non-empty)",
    )
    return parser.parse_args()


def _is_empty_answer(answer: object) -> bool:
    if answer is None:
        return True
    if isinstance(answer, str) and answer.strip() == "":
        return True
    return False


def run_benchmark(
    agent_name: str,
    model_name: str,
    samples: int,
    trace: bool = False,
    delay: float = 2.0,
    use_verdict: bool = True,
    mode: str = "proxy",
):
    print(f"\n{'='*60}")
    print("  EAG Benchmarks")
    print(f"  Agent: {agent_name} | Model: {model_name} | Samples: {samples} | Mode: {mode}")
    if trace:
        print("  Trace: ON")
    if use_verdict:
        print("  Verdict: ON (precheck + LLM)")
    print(f"{'='*60}\n")

    models = PROXY_MODELS if mode == "proxy" else DIRECT_MODELS
    llm = models[model_name]()
    agent = AGENTS[agent_name](llm)

    verdict_checker = None
    verdict_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    verdict_llm = None
    if use_verdict:
        if mode == "proxy":
            verdict_llm = LiteLLMClient(config_name=PROXY_VERDICT_CONFIG)
        else:
            verdict_llm = DirectLLMClient(config_name=DIRECT_VERDICT_CONFIG)
        verdict_checker = VerdictChecker(verdict_llm)

    if hasattr(agent, "trace"):
        agent.trace = trace

    dataset = load_mini_dev(limit=samples)
    metrics = BenchmarkMetrics()

    for i, item in enumerate(dataset):
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["SQL"]

        print(f"[{i+1}/{len(dataset)}] {db_id}: {question[:80]}...")

        with Timer() as timer:
            result = agent.run(item)

        agent_answer = result.get("answer", "")

        if result["error"]:
            print(f"  ERROR: {result['error'][:80]}")
            metrics.errors += 1
            metrics.total += 1
            metrics.verdicts["parse_error"] += 1
            metrics.per_item.append({
                "db_id": db_id,
                "question": question,
                "agent_answer": agent_answer,
                "gold_answer": None,
                "status": "parse_error",
                "error": result["error"],
                "latency_ms": round(timer.elapsed_ms, 2),
                "usage": result.get("usage", {}),
                "steps": result.get("steps", []),
            })
            print()
            if i < len(dataset) - 1:
                time.sleep(delay)
            continue

        gold = get_gold_answer(gold_sql, db_id)

        if gold is None:
            print(f"  GOLD_ERROR (db={db_id})")
            metrics.errors += 1
            metrics.total += 1
            metrics.verdicts["parse_error"] += 1
            metrics.per_item.append({
                "db_id": db_id,
                "question": question,
                "agent_answer": agent_answer,
                "gold_answer": None,
                "status": "parse_error",
                "error": "Gold SQL execution failed",
                "latency_ms": round(timer.elapsed_ms, 2),
                "usage": result.get("usage", {}),
                "steps": result.get("steps", []),
            })
            print()
            if i < len(dataset) - 1:
                time.sleep(delay)
            continue

        gold_answer = gold["answer"]

        if _is_empty_answer(agent_answer):
            status = "parse_error"
            verdict_confidence = 0.0
            verdict_source = "empty"
        else:
            status, verdict_confidence, verdict_source = _evaluate_answer(
                agent_answer, gold_answer, gold_sql, question,
                verdict_checker, verdict_llm, verdict_usage, use_verdict,
            )

        correct = status == "match"
        metrics.total += 1
        metrics.total_latency_ms += timer.elapsed_ms
        if correct:
            metrics.correct += 1
        metrics.record_verdict(status, verdict_confidence, verdict_source)
        if result.get("usage"):
            usage = result["usage"]
            metrics.prompt_tokens += usage.get("prompt_tokens", 0)
            metrics.completion_tokens += usage.get("completion_tokens", 0)
            metrics.total_tokens += usage.get("total_tokens", 0)

        tag = "CORRECT" if correct else status.upper()
        n_steps = len(result.get("steps", []))
        src_tag = f"[{verdict_source}]" if verdict_source else ""
        print(f"  {tag} {src_tag} | Answer: {agent_answer} | "
              f"Gold: {gold_answer} | Steps: {n_steps} | "
              f"{metrics.accuracy:.1%}")
        print()

        per_item = {
            "db_id": db_id,
            "question": question,
            "agent_answer": agent_answer,
            "gold_answer": gold_answer,
            "gold_sql": gold_sql,
            "status": status,
            "verdict_confidence": round(verdict_confidence, 4),
            "verdict_source": verdict_source,
            "correct": correct,
            "latency_ms": round(timer.elapsed_ms, 2),
            "usage": result.get("usage", {}),
            "steps": result.get("steps", []),
        }
        metrics.per_item.append(per_item)

        if i < len(dataset) - 1:
            time.sleep(delay)

    return metrics, verdict_usage


def _evaluate_answer(
    agent_answer, gold_answer, gold_sql, question,
    verdict_checker, verdict_llm, verdict_usage, use_verdict,
):
    """Run precheck first, then fall back to LLM verdict if undecided.

    Returns (status, confidence, source).
    """
    pc = precheck_match(agent_answer, gold_answer)
    if pc.status != "undecided":
        if pc.status == "match":
            return pc.status, pc.confidence, "precheck"
        return pc.status, pc.confidence, "precheck"

    if not use_verdict or verdict_checker is None:
        return "wrong_answer", 0.5, "precheck-undecided"

    vr = verdict_checker.check(
        question=question,
        agent_answer=agent_answer,
        gold_answer=gold_answer,
        gold_sql=gold_sql,
    )
    status = vr.verdict
    verdict_confidence = vr.confidence

    if verdict_confidence == 0.0 and status == "unclear":
        metrics_ref = verdict_usage
        metrics_ref["_verdict_failures"] = metrics_ref.get("_verdict_failures", 0) + 1

    v_usage = verdict_llm.get_usage()
    verdict_usage["prompt_tokens"] += v_usage.get("prompt_tokens", 0)
    verdict_usage["completion_tokens"] += v_usage.get("completion_tokens", 0)
    verdict_usage["total_tokens"] += v_usage.get("total_tokens", 0)

    return status, verdict_confidence, "llm"


def main():
    args = parse_args()

    metrics, verdict_usage = run_benchmark(
        args.agent, args.model, args.samples,
        trace=args.trace, delay=args.delay,
        use_verdict=not args.no_verdict,
        mode=args.mode,
    )

    print(f"\n{'='*60}")
    print("  Results")
    print(f"{'='*60}")
    print(f"  Answer Accuracy: {metrics.accuracy:.4f} ({metrics.correct}/{metrics.total})")
    print(f"  Avg Latency: {metrics.avg_latency_ms:.2f} ms")
    print(f"  Total Tokens: {metrics.total_tokens}")
    print(f"  Verdicts: {metrics.verdicts}")
    print(f"  Precheck matches: {metrics.precheck_matches}")
    print(f"  Verdict LLM matches: {metrics.verdict_llm_matches}")
    print(f"  Verdict LLM failures: {metrics.verdict_failures}")
    if not args.no_verdict:
        print(f"  Verdict Tokens: {verdict_usage['total_tokens']}")
    if metrics.errors:
        print(f"  Errors: {metrics.errors}")
    print()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"bird_{args.agent}_{args.model}_{timestamp}.json"

    output = {
        "config": {
            "agent": args.agent,
            "model": args.model,
            "mode": args.mode,
            "samples": args.samples,
            "timestamp": timestamp,
            "verdict_enabled": not args.no_verdict,
        },
        "summary": metrics.summary(),
        "results": metrics.per_item,
    }
    if not args.no_verdict:
        output["verdict_usage"] = verdict_usage

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
