from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict

from llm.provider import LLMInterface

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
decimal place are acceptable. If the agent computed a ratio as 2002/30459 and \
the gold is 0.0657, check if they evaluate to the same number.

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

## Verdict definitions

- "match": Agent answer is semantically equivalent to gold answer.
- "wrong_answer": Agent provided a valid, parseable answer but the values are \
different from gold.
- "mismatch": Agent answer is structurally incompatible — wrong type, returned \
full row instead of specific value, gave text when number was expected, etc.
- "parse_error": Agent answer is empty, null, or completely unparseable.
- "unclear": Genuinely ambiguous — cannot determine with reasonable confidence.

## Output

Extract the core answer value from the agent's response into answer_extracted. \
Set is_match to true only for "match" verdict. Set confidence based on how \
certain you are (0.0 to 1.0). Provide a brief explanation of your reasoning.
"""


class VerdictResult(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    answer_extracted: Optional[Union[str, float, list]]
    is_match: bool
    verdict: Literal["match", "wrong_answer", "mismatch", "parse_error", "unclear"]
    confidence: float
    explanation: Optional[str] = None


class VerdictChecker:
    """Answer validator using a small LLM with strict response_format.

    Sits in the evaluation pipeline. Receives the agent's raw answer, the gold
    answer, the gold SQL, and the question. Returns a structured VerdictResult.
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
        user_prompt = (
            f"Question: {question}\n\n"
            f"Gold SQL:\n{gold_sql}\n\n"
            f"Gold answer: {gold_answer}\n\n"
            f"Agent answer: {agent_answer}"
        )

        try:
            raw = self.llm.generate(
                system_prompt=VERDICT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_completion_tokens=512,
                response_format=VerdictResult,
            )
            return VerdictResult.model_validate_json(raw)
        except Exception:
            return VerdictResult(
                answer_extracted=str(agent_answer),
                is_match=False,
                verdict="unclear",
                confidence=0.0,
                explanation="Verdict model call failed",
            )
