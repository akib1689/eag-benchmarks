import json
import logging
import re
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict

from llm.provider import LLMInterface

logger = logging.getLogger(__name__)

MAX_ANSWER_DISPLAY = 1500

VERDICT_SYSTEM_PROMPT = """\
You are an answer validation assistant for a database question-answering \
benchmark. You are given a question, a gold SQL query, the gold answer \
(computed by executing that SQL), and an AI agent's answer. Your job is to \
determine if the agent's answer matches the gold answer.

## Context

The gold SQL was executed against a real database to produce the gold answer. \
This is ground truth. The agent also queried the same database but may have \
used different SQL, different rounding, or produced its answer in a different \
format. Your task is a semantic comparison — not exact string matching.

## Evaluation rules

1. **Numeric answers**: Compare values within 1% relative tolerance or 0.01 \
absolute tolerance, whichever is larger. Rounding differences at the last \
decimal place are acceptable.

2. **String answers**: Case-insensitive comparison. Whitespace differences are \
ignored. "Yes"/"yes" match. "04" and "4" match if they represent the same value.

3. **List answers**: Order does NOT matter unless the question implies ranking. \
Compare as sets — missing or extra elements make it "wrong_answer".

4. **Type mismatches**: If the agent returns {"CustomerID": 47273, \
"Consumption": 0.74} but the gold is 47273, the agent returned the whole row \
instead of the specific column the question asked for. This is "mismatch".

5. **Wrong computation**: If the agent computed the answer differently from the \
gold SQL and got different numbers, this is "wrong_answer" even if the approach \
seemed reasonable.

6. **Text descriptions vs values**: If the question asks for a number and the \
agent gives a text description (e.g., "SME has the biggest increase"), that is \
"mismatch" — the question requires a numeric answer.

7. **Formatted text vs structured lists**: If the agent returns values \
separated by newlines or commas and the gold is a list, compare the individual \
values. "foo\\nbar" matches ["foo", "bar"] if the values are equivalent.

## Verdict definitions

- "match": Agent answer is semantically equivalent to gold answer.
- "wrong_answer": Agent provided a valid, parseable answer but the values are \
different from gold.
- "mismatch": Agent answer is structurally incompatible — wrong type, returned \
full row instead of specific value, gave text when number was expected, etc.
- "parse_error": Agent answer is empty, null, or completely unparseable.
- "unclear": Genuinely ambiguous — cannot determine with reasonable confidence.

## Output

Respond with ONLY a JSON object (no markdown, no code fences, no tool calls). \
The JSON must have exactly this structure:

{"answer_extracted": <value>, "is_match": <bool>, "verdict": "<one of: match, \
wrong_answer, mismatch, parse_error, unclear>", "confidence": <0.0-1.0>, \
"explanation": "<brief reason>"}
"""

SIMPLIFIED_PROMPT = """\
Compare these two answers and respond with ONLY a JSON object \
(no markdown, no code fences, no tool calls):

{"answer_extracted": <value>, "is_match": <bool>, \
"verdict": "match|wrong_answer|mismatch|parse_error|unclear", \
"confidence": 0.0-1.0, "explanation": "reason"}

Rules: numeric 1%% tolerance, strings case-insensitive, \
lists order-independent.
"""


class VerdictResult(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    answer_extracted: Optional[Union[str, float, list]]
    is_match: bool
    verdict: Literal["match", "wrong_answer", "mismatch", "parse_error", "unclear"]
    confidence: float
    explanation: Optional[str] = None


def _truncate(value: object, max_len: int = MAX_ANSWER_DISPLAY) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n... [truncated, {len(text)} total chars]"


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON from model response, handling markdown fences and noise."""
    text = text.strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


class VerdictChecker:
    """Answer validator using a small LLM with manual JSON extraction.

    Sits in the evaluation pipeline after the deterministic precheck.
    Receives the agent's raw answer, the gold answer, the gold SQL,
    and the question. Returns a structured VerdictResult.

    Does NOT use response_format to avoid triggering tool-call
    hallucination in smaller Groq models. Instead, parses the JSON
    response manually with regex fallback.
    """

    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def check(
        self,
        question: str,
        agent_answer: object,
        gold_answer: object,
        gold_sql: str,
    ) -> VerdictResult:
        agent_display = _truncate(agent_answer)
        gold_display = _truncate(gold_answer)

        user_prompt = (
            f"Question: {question}\n\n"
            f"Gold SQL:\n{gold_sql}\n\n"
            f"Gold answer: {gold_display}\n\n"
            f"Agent answer: {agent_display}"
        )

        vr = self._try_call(VERDICT_SYSTEM_PROMPT, user_prompt)
        if vr is not None:
            return vr

        logger.warning(
            "Verdict model primary call failed, retrying with simplified prompt"
        )
        simple_prompt = (
            f"{SIMPLIFIED_PROMPT}\n\n"
            f"Question: {question}\n"
            f"Gold: {gold_display}\n"
            f"Agent: {agent_display}"
        )
        vr = self._try_call(
            "You are a JSON-only response bot. No tool calls, no markdown, just JSON.",
            simple_prompt,
        )
        if vr is not None:
            return vr

        return VerdictResult(
            answer_extracted=str(agent_answer)[:200],
            is_match=False,
            verdict="unclear",
            confidence=0.0,
            explanation="Verdict model call failed after retry",
        )

    def _try_call(self, system: str, user: str) -> Optional[VerdictResult]:
        try:
            raw = self.llm.generate(
                system_prompt=system,
                user_prompt=user,
                max_completion_tokens=1024,
            )
            parsed = _extract_json(raw)
            if parsed is None:
                logger.warning("Verdict model returned non-JSON: %s", raw[:300])
                return None
            return VerdictResult.model_validate(parsed)
        except Exception as exc:
            logger.warning("Verdict model exception: %s", exc)
            return None
