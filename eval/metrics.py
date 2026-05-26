import time
from dataclasses import dataclass, field


@dataclass
class BenchmarkMetrics:
    """Container for all metrics collected during a single benchmark run."""

    correct: int = 0
    total: int = 0
    total_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    errors: int = 0
    not_processed: int = 0
    per_item: list = field(default_factory=list)
    verdicts: dict = field(
        default_factory=lambda: {
            "match": 0,
            "wrong_answer": 0,
            "mismatch": 0,
            "parse_error": 0,
            "unclear": 0,
            "not_processed": 0,
        }
    )
    _confidences: list = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total if self.total > 0 else 0.0

    @property
    def avg_confidence(self) -> float:
        if not self._confidences:
            return 0.0
        return sum(self._confidences) / len(self._confidences)

    def record_verdict(self, verdict: str, confidence: float):
        if verdict in self.verdicts:
            self.verdicts[verdict] += 1
        self._confidences.append(confidence)

    def summary(self) -> dict:
        return {
            "answer_accuracy": round(self.accuracy, 4),
            "correct": self.correct,
            "total": self.total,
            "errors": self.errors,
            "not_processed": self.not_processed,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_prompt_tokens": self.prompt_tokens,
            "total_completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "verdicts": dict(self.verdicts),
            "avg_confidence": round(self.avg_confidence, 4),
        }


class Timer:
    """Context manager for measuring latency."""

    def __init__(self):
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
