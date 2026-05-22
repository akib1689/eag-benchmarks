#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agents.eag import EAGAgent  # noqa: E402
from agents.pal import PALAgent  # noqa: E402
from agents.pot import POTAgent  # noqa: E402
from agents.react import ReActAgent  # noqa: E402
from datasets.bird.loader import load_mini_dev  # noqa: E402
from eval.answer_extractor import extract_answer  # noqa: E402
from eval.comparators import compare_answers  # noqa: E402
from eval.gold import get_gold_answer  # noqa: E402
from eval.metrics import BenchmarkMetrics, Timer  # noqa: E402
from llm.glm import GLMClient  # noqa: E402
from llm.groq import GroqClient  # noqa: E402

AGENTS = {
    "react": lambda llm: ReActAgent(llm),
    "pal": lambda llm: PALAgent(llm),
    "pot": lambda llm: POTAgent(llm),
    "eag": lambda llm: EAGAgent(llm),
}

MODELS = {
    "groq": lambda: GroqClient(model="gpt-oss-120b"),
    "glm": lambda: GLMClient(model="glm-5.1"),
}


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
    return parser.parse_args()


def run_benchmark(agent_name: str, model_name: str, samples: int):
    print(f"\n{'='*60}")
    print("  EAG Benchmarks")
    print(f"  Agent: {agent_name} | Model: {model_name} | Samples: {samples}")
    print(f"{'='*60}\n")

    llm = MODELS[model_name]()
    agent = AGENTS[agent_name](llm)

    dataset = load_mini_dev(limit=samples)
    metrics = BenchmarkMetrics()

    for i, item in enumerate(dataset):
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["SQL"]

        print(f"[{i+1}/{len(dataset)}] {db_id}: {question[:80]}...", end=" ")

        with Timer() as timer:
            result = agent.run(item)

        if result["error"]:
            print(f"ERROR: {result['error'][:60]}")
            metrics.errors += 1
            metrics.total += 1
            metrics.per_item.append({
                "db_id": db_id,
                "question": question,
                "error": result["error"],
                "correct": False,
            })
            continue

        gold = get_gold_answer(gold_sql, db_id)
        extracted = extract_answer(result.get("answer", result.get("raw_output", "")))

        if gold is None:
            print(f"GOLD_ERROR (db={db_id})")
            metrics.errors += 1
            metrics.total += 1
            metrics.per_item.append({
                "db_id": db_id,
                "question": question,
                "error": "Gold SQL execution failed",
                "correct": False,
            })
            continue

        is_correct, confidence, tier = compare_answers(extracted["answer"], gold["answer"])

        metrics.total += 1
        metrics.total_latency_ms += timer.elapsed_ms
        if is_correct:
            metrics.correct += 1
        metrics.record_match(tier, confidence, extracted["success"])
        if result.get("usage"):
            usage = result["usage"]
            metrics.prompt_tokens += usage.get("prompt_tokens", 0)
            metrics.completion_tokens += usage.get("completion_tokens", 0)
            metrics.total_tokens += usage.get("total_tokens", 0)

        status = f"{'CORRECT' if is_correct else 'WRONG'} [{tier}]"
        print(f"{status} ({metrics.accuracy:.1%})")

        metrics.per_item.append({
            "db_id": db_id,
            "question": question,
            "agent_answer": extracted["answer"],
            "gold_answer": gold["answer"],
            "match_tier": tier,
            "confidence": round(confidence, 4),
            "parse_method": extracted["method"],
            "parse_success": extracted["success"],
            "correct": is_correct,
            "latency_ms": round(timer.elapsed_ms, 2),
            "usage": result.get("usage", {}),
        })

    return metrics


def main():
    args = parse_args()

    metrics = run_benchmark(args.agent, args.model, args.samples)

    print(f"\n{'='*60}")
    print("  Results")
    print(f"{'='*60}")
    print(f"  Answer Accuracy: {metrics.accuracy:.4f} ({metrics.correct}/{metrics.total})")
    print(f"  Avg Latency: {metrics.avg_latency_ms:.2f} ms")
    print(f"  Total Tokens: {metrics.total_tokens}")
    print(f"  Match Tiers: {metrics.match_tiers}")
    if metrics.parse_failures:
        print(f"  Parse Failures: {metrics.parse_failures}")
    if metrics.errors:
        print(f"  Errors: {metrics.errors}")
    print()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"bird_{args.agent}_{args.model}_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "agent": args.agent,
                "model": args.model,
                "samples": args.samples,
                "timestamp": timestamp,
            },
            "summary": metrics.summary(),
            "results": metrics.per_item,
        }, f, indent=2)

    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
