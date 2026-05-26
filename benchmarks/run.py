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
from eval.verdict import VerdictChecker  # noqa: E402
from llm.litellm_client import LiteLLMClient  # noqa: E402

AGENTS = {
    "react": lambda llm: ReActAgent(llm),
    "pal": lambda llm: PALAgent(llm),
    "pot": lambda llm: POTAgent(llm),
    "eag": lambda llm: EAGAgent(llm),
}

MODELS = {
    "groq": lambda: LiteLLMClient(config_name="groq"),
    "glm": lambda: LiteLLMClient(config_name="glm"),
    "openrouter": lambda: LiteLLMClient(config_name="openrouter"),
}

VERDICT_CONFIG = "groq-verdict"


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
        "--model",
        choices=list(MODELS.keys()),
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
):
    print(f"\n{'='*60}")
    print("  EAG Benchmarks")
    print(f"  Agent: {agent_name} | Model: {model_name} | Samples: {samples}")
    if trace:
        print("  Trace: ON")
    if use_verdict:
        print("  Verdict: ON")
    print(f"{'='*60}\n")

    llm = MODELS[model_name]()
    agent = AGENTS[agent_name](llm)

    verdict_checker = None
    verdict_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    verdict_llm = None
    if use_verdict:
        verdict_llm = LiteLLMClient(config_name=VERDICT_CONFIG)
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
            metrics.verdicts["not_processed"] += 1
            metrics.per_item.append({
                "db_id": db_id,
                "question": question,
                "agent_answer": agent_answer,
                "gold_answer": None,
                "status": "not_processed",
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
            metrics.verdicts["not_processed"] += 1
            metrics.per_item.append({
                "db_id": db_id,
                "question": question,
                "agent_answer": agent_answer,
                "gold_answer": None,
                "status": "not_processed",
                "error": "Gold SQL execution failed",
                "latency_ms": round(timer.elapsed_ms, 2),
                "usage": result.get("usage", {}),
                "steps": result.get("steps", []),
            })
            print()
            if i < len(dataset) - 1:
                time.sleep(delay)
            continue

        if _is_empty_answer(agent_answer):
            status = "not_processed"
            verdict_confidence = 0.0
            metrics.not_processed += 1
        elif verdict_checker is not None:
            vr = verdict_checker.check(
                question=question,
                agent_answer=agent_answer,
                gold_answer=gold["answer"],
                gold_sql=gold_sql,
            )
            status = vr.verdict
            verdict_confidence = vr.confidence
            v_usage = verdict_llm.get_usage()
            verdict_usage["prompt_tokens"] += v_usage.get("prompt_tokens", 0)
            verdict_usage["completion_tokens"] += v_usage.get("completion_tokens", 0)
            verdict_usage["total_tokens"] += v_usage.get("total_tokens", 0)
        else:
            status = "match" if agent_answer == gold["answer"] else "wrong_answer"
            verdict_confidence = 1.0

        correct = status == "match"
        metrics.total += 1
        metrics.total_latency_ms += timer.elapsed_ms
        if correct:
            metrics.correct += 1
        metrics.record_verdict(status, verdict_confidence)
        if result.get("usage"):
            usage = result["usage"]
            metrics.prompt_tokens += usage.get("prompt_tokens", 0)
            metrics.completion_tokens += usage.get("completion_tokens", 0)
            metrics.total_tokens += usage.get("total_tokens", 0)

        tag = "CORRECT" if correct else status.upper()
        n_steps = len(result.get("steps", []))
        print(f"  {tag} | Answer: {agent_answer} | Gold: {gold['answer']} | "
              f"Steps: {n_steps} | {metrics.accuracy:.1%}")
        print()

        per_item = {
            "db_id": db_id,
            "question": question,
            "agent_answer": agent_answer,
            "gold_answer": gold["answer"],
            "gold_sql": gold_sql,
            "status": status,
            "verdict_confidence": round(verdict_confidence, 4),
            "correct": correct,
            "latency_ms": round(timer.elapsed_ms, 2),
            "usage": result.get("usage", {}),
            "steps": result.get("steps", []),
        }
        metrics.per_item.append(per_item)

        if i < len(dataset) - 1:
            time.sleep(delay)

    return metrics, verdict_usage


def main():
    args = parse_args()

    metrics, verdict_usage = run_benchmark(
        args.agent, args.model, args.samples,
        trace=args.trace, delay=args.delay,
        use_verdict=not args.no_verdict,
    )

    print(f"\n{'='*60}")
    print("  Results")
    print(f"{'='*60}")
    print(f"  Answer Accuracy: {metrics.accuracy:.4f} ({metrics.correct}/{metrics.total})")
    print(f"  Avg Latency: {metrics.avg_latency_ms:.2f} ms")
    print(f"  Total Tokens: {metrics.total_tokens}")
    print(f"  Verdicts: {metrics.verdicts}")
    if not args.no_verdict:
        print(f"  Verdict Tokens: {verdict_usage['total_tokens']}")
    if metrics.not_processed:
        print(f"  Not Processed: {metrics.not_processed}")
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
