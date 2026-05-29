import json
import logging
import re
from typing import Any, List, Literal, Optional, Set, Union

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
Compare as SETS — if the agent has the same unique values as gold, extra \
duplicates in either side should be ignored. Missing unique values make it \
"wrong_answer"; extra duplicates do not.

4. **Scalar vs repeated-element list**: If the agent returns a single value \
(like "DC Comics") and the gold is a list where every element is the same \
value (["dc comics", "dc comics", "dc comics"]), this is a "match" — the \
agent gave a concise answer equivalent to the full list.

5. **Type mismatches**: If the agent returns {"CustomerID": 47273, \
"Consumption": 0.74} but the gold is 47273, the agent returned the whole row \
instead of the specific column the question asked for. This is "mismatch".

6. **Wrong computation**: If the agent computed the answer differently from the \
gold SQL and got different numbers, this is "wrong_answer" even if the approach \
seemed reasonable.

7. **Text descriptions vs values**: If the question asks for a number and the \
agent gives a text description (e.g., "SME has the biggest increase"), that is \
"mismatch" — the question requires a numeric answer.

8. **Formatted text vs structured lists**: If the agent returns values \
separated by newlines or commas and the gold is a list, compare the individual \
values. "foo\\nbar" matches ["foo", "bar"] if the values are equivalent.

9. **Name formats**: "Timo Glock" (full name) is equivalent to \
["timo", "glock"] (split first/last). Compare names by concatenating or \
splitting as needed.

10. **JSON objects vs tuples**: If the agent returns {"school": "Buchanan High", \
"score": 507} and the gold is ["buchanan high", 507], compare the VALUES — \
the labeled JSON is equivalent to the unlabeled tuple if the values match.

11. **Extra columns in table rows**: If the agent returns rows with additional \
columns beyond what gold has, compare only the shared prefix. \
[["Spielburg", 47.22, 14.76, 29]] matches [["spielburg", 47.22, 14.76]] \
because the first 3 columns match and the 4th is extra context.

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
lists order-independent, duplicates ignored, full names = split names, \
JSON objects = tuples if values match.
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


def _normalize_for_salvage(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, bool):
        return str(val).lower()
    if isinstance(val, (int, float)):
        if isinstance(val, float) and val == int(val) and not (val != val):
            return int(val)
        return round(val, 10) if isinstance(val, float) else val
    if isinstance(val, str):
        v = val.strip().lower()
        if v.endswith('%'):
            v = v[:-1].strip()
        for sentinel in ("null", "none", "n/a", "nan", ""):
            if v == sentinel:
                return None
        try:
            return int(v)
        except ValueError:
            pass
        try:
            fv = float(v)
            if fv == int(fv):
                return int(fv)
            return round(fv, 10)
        except ValueError:
            pass
        return v
    return val


def _extract_unique_scalars(val: Any) -> Set[Any]:
    if val is None:
        return set()
    if isinstance(val, (list, tuple)):
        result = set()
        for item in val:
            if isinstance(item, (list, tuple)):
                for sub in item:
                    n = _normalize_for_salvage(sub)
                    if n is not None:
                        result.add(n)
            else:
                n = _normalize_for_salvage(item)
                if n is not None:
                    result.add(n)
        return result
    n = _normalize_for_salvage(val)
    return {n} if n is not None else set()


def _extract_table_rows(val: Any) -> List[tuple]:
    if isinstance(val, (list, tuple)):
        rows = []
        for item in val:
            if isinstance(item, (list, tuple)):
                rows.append(tuple(_normalize_for_salvage(v) for v in item))
            else:
                rows.append((_normalize_for_salvage(item),))
        return rows
    return [(_normalize_for_salvage(val),)]


def _values_fuzzy_match(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return True
        if isinstance(a, int) and isinstance(b, int):
            return False
        return abs(a - b) <= max(abs(a), abs(b)) * 0.01
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    return str(a).lower() == str(b).lower()


def _verdict_hints(agent_answer: object, gold_answer: object) -> str:
    """Detect patterns the LLM tends to miss and return annotation hints."""
    hints = []

    agent_uniques = _extract_unique_scalars(agent_answer)
    gold_uniques = _extract_unique_scalars(gold_answer)

    if gold_uniques and isinstance(gold_answer, (list, tuple)):
        gold_items = list(_flatten(gold_answer))
        if len(gold_items) > 1:
            normed = [_normalize_for_salvage(g) for g in gold_items]
            non_none = [n for n in normed if n is not None]
            if non_none and len(set(str(v) for v in non_none)) == 1:
                agent_norm = _normalize_for_salvage(agent_answer)
                if agent_norm is not None:
                    all_val = non_none[0]
                    if _values_fuzzy_match(agent_norm, all_val):
                        hints.append(
                            f"HINT: All {len(non_none)} gold values are identical "
                            f"({all_val!r}), and the agent's scalar answer matches."
                        )

    if agent_uniques and gold_uniques:
        if agent_uniques == gold_uniques and len(agent_uniques) >= 2:
            agent_count = len(_flatten(agent_answer))
            gold_count = len(_flatten(gold_answer))
            if agent_count != gold_count:
                hints.append(
                    f"HINT: Agent and gold have the same {len(agent_uniques)} "
                    f"unique values, but different counts "
                    f"({agent_count} vs {gold_count}). Compare as sets."
                )

    agent_parsed = _safe_parse_agent(agent_answer)
    if isinstance(agent_parsed, list) and len(agent_parsed) > 0:
        if isinstance(agent_parsed[0], dict):
            hints.append(
                "HINT: Agent returns JSON objects with labeled fields. "
                "Compare values against gold's unlabeled tuples."
            )

    return "\n".join(hints)


def _flatten(val: Any) -> list:
    items = []
    if isinstance(val, (list, tuple)):
        for item in val:
            if isinstance(item, (list, tuple)):
                items.extend(_flatten(item))
            else:
                items.append(item)
    else:
        items.append(val)
    return items


def _salvage_check(agent_answer: object, gold_answer: object) -> Optional[VerdictResult]:
    """Post-LLM deterministic check for patterns the verdict model missed.

    Returns a VerdictResult(match) if a salvageable pattern is detected,
    otherwise None.
    """
    if agent_answer is None or gold_answer is None:
        return None

    agent_str = str(agent_answer).strip()
    if not agent_str:
        return None

    agent_parsed = _safe_parse_agent(agent_answer)

    for salvage_fn in (
        _salvage_scalar_vs_uniform_list,
        _salvage_set_equivalent,
        _salvage_name_split,
        _salvage_json_dict_vs_tuple,
        _salvage_table_prefix_match,
    ):
        try:
            result = salvage_fn(agent_parsed, gold_answer)
            if result is not None:
                return result
        except Exception:
            pass

    return None


def _safe_parse_agent(agent_answer: object) -> object:
    if not isinstance(agent_answer, str):
        return agent_answer
    try:
        from .answer_parser import parse_agent_answer
        return parse_agent_answer(agent_answer)
    except Exception:
        return agent_answer


def _salvage_scalar_vs_uniform_list(
    agent_answer: object, gold_answer: object
) -> Optional[VerdictResult]:
    if not isinstance(gold_answer, (list, tuple)) or len(gold_answer) < 2:
        return None
    if isinstance(agent_answer, (list, tuple)):
        return None

    gold_items = list(_flatten(gold_answer))
    if not gold_items:
        return None
    normed = [_normalize_for_salvage(g) for g in gold_items]
    non_none = [n for n in normed if n is not None]
    if not non_none:
        return None

    if len(set(str(v) for v in non_none)) != 1:
        return None

    all_val = non_none[0]
    agent_norm = _normalize_for_salvage(agent_answer)
    if agent_norm is not None and _values_fuzzy_match(agent_norm, all_val):
        return VerdictResult(
            answer_extracted=agent_norm,
            is_match=True,
            verdict="match",
            confidence=0.95,
            explanation=f"Agent scalar {agent_norm!r} matches all "
                        f"{len(non_none)} identical gold values",
        )
    return None


def _salvage_set_equivalent(
    agent_answer: object, gold_answer: object
) -> Optional[VerdictResult]:
    agent_uniques = _extract_unique_scalars(agent_answer)
    gold_uniques = _extract_unique_scalars(gold_answer)
    if not agent_uniques or not gold_uniques:
        return None
    if len(agent_uniques) < 2 and len(gold_uniques) < 2:
        return None
    if agent_uniques == gold_uniques:
        agent_flat = _flatten(agent_answer)
        gold_flat = _flatten(gold_answer)
        if len(agent_flat) != len(gold_flat):
            return VerdictResult(
                answer_extracted=str(agent_answer)[:200],
                is_match=True,
                verdict="match",
                confidence=0.93,
                explanation=f"Set-equivalent: same {len(agent_uniques)} unique "
                            f"values, different counts "
                            f"({len(agent_flat)} vs {len(gold_flat)})",
            )
    return None


def _salvage_name_split(
    agent_answer: object, gold_answer: object
) -> Optional[VerdictResult]:
    if not isinstance(gold_answer, (list, tuple)):
        return None
    if len(gold_answer) == 0:
        return None

    first = gold_answer[0]
    if not isinstance(first, (list, tuple)) or len(first) < 2:
        return None

    gold_str_vals = [
        v for v in first if isinstance(v, str) and len(v.strip()) > 0
    ]
    if len(gold_str_vals) < 2:
        return None

    if isinstance(agent_answer, (list, tuple)):
        agent_str_items = [
            v for v in agent_answer if isinstance(v, str) and " " in v.strip()
        ]
    elif isinstance(agent_answer, str) and "," in agent_answer:
        parts = [p.strip() for p in agent_answer.split(",")]
        agent_str_items = [p for p in parts if " " in p]
    else:
        agent_str_items = (
            [agent_answer] if isinstance(agent_answer, str) and " " in agent_answer
            else []
        )

    if not agent_str_items:
        return None

    gold_name_parts = set()
    for row in gold_answer:
        if isinstance(row, (list, tuple)):
            for v in row:
                if isinstance(v, str):
                    gold_name_parts.add(v.lower().strip())

    for name in agent_str_items:
        words = [w.lower().strip() for w in name.split() if w.strip()]
        if len(words) == len(gold_str_vals):
            if all(w in gold_name_parts for w in words):
                gold_as_sets = set()
                for row in gold_answer:
                    if isinstance(row, (list, tuple)):
                        gold_as_sets.add(
                            tuple(str(v).lower().strip() for v in row)
                        )
                agent_as_sets = set()
                for name_item in agent_str_items:
                    words = tuple(
                        w.lower().strip() for w in name_item.split() if w.strip()
                    )
                    agent_as_sets.add(words)

                if len(agent_as_sets) == len(gold_as_sets):
                    expanded_gold = set()
                    for parts in gold_as_sets:
                        expanded_gold.add(tuple(sorted(parts)))
                    expanded_agent = set()
                    for parts in agent_as_sets:
                        expanded_agent.add(tuple(sorted(parts)))

                    if expanded_agent == expanded_gold:
                        return VerdictResult(
                            answer_extracted=str(agent_answer)[:200],
                            is_match=True,
                            verdict="match",
                            confidence=0.93,
                            explanation="Agent full names match gold split "
                                        "first/last name format",
                        )
    return None


def _salvage_json_dict_vs_tuple(
    agent_answer: object, gold_answer: object
) -> Optional[VerdictResult]:
    agent_parsed = None
    if isinstance(agent_answer, str):
        try:
            agent_parsed = json.loads(agent_answer)
        except (json.JSONDecodeError, ValueError):
            return None
    elif isinstance(agent_answer, list) and agent_answer:
        if isinstance(agent_answer[0], dict):
            agent_parsed = agent_answer

    if agent_parsed is None:
        return None

    if isinstance(agent_parsed, dict):
        agent_parsed = [agent_parsed]

    if not isinstance(agent_parsed, list) or not agent_parsed:
        return None
    if not isinstance(agent_parsed[0], dict):
        return None

    if isinstance(gold_answer, (list, tuple)):
        if len(gold_answer) > 0 and not isinstance(gold_answer[0], (list, tuple)):
            if len(agent_parsed) == 1:
                agent_vals = list(agent_parsed[0].values())
                gold_vals = list(gold_answer)
                if len(agent_vals) == len(gold_vals):
                    all_match = True
                    for av, gv in zip(agent_vals, gold_vals):
                        if not _values_fuzzy_match(
                            _normalize_for_salvage(av),
                            _normalize_for_salvage(gv),
                        ):
                            all_match = False
                            break
                    if all_match:
                        return VerdictResult(
                            answer_extracted=str(agent_answer)[:200],
                            is_match=True,
                            verdict="match",
                            confidence=0.95,
                            explanation="Agent JSON dict values match gold flat list",
                        )
            return None

        gold_rows = []
        for item in gold_answer:
            if isinstance(item, (list, tuple)):
                gold_rows.append([_normalize_for_salvage(v) for v in item])
            else:
                gold_rows.append([_normalize_for_salvage(item)])

        if len(agent_parsed) != len(gold_rows):
            return None

        for agent_dict, gold_row in zip(agent_parsed, gold_rows):
            agent_vals = [_normalize_for_salvage(v) for v in agent_dict.values()]
            if len(agent_vals) != len(gold_row):
                return None
            for av, gv in zip(agent_vals, gold_row):
                if not _values_fuzzy_match(av, gv):
                    return None

        return VerdictResult(
            answer_extracted=str(agent_answer)[:200],
            is_match=True,
            verdict="match",
            confidence=0.95,
            explanation="Agent JSON dict values match gold tuple values",
        )

    return None


def _salvage_table_prefix_match(
    agent_answer: object, gold_answer: object
) -> Optional[VerdictResult]:
    agent_rows = _extract_table_rows(agent_answer)
    gold_rows = _extract_table_rows(gold_answer)

    if not agent_rows or not gold_rows:
        return None
    if len(agent_rows) != len(gold_rows):
        return None

    min_cols = min(len(ar) for ar in agent_rows)
    min_cols = min(min_cols, min(len(gr) for gr in gold_rows))
    if min_cols == 0:
        return None

    for ar, gr in zip(sorted(agent_rows), sorted(gold_rows)):
        prefix_len = min(len(ar), len(gr))
        if prefix_len < min_cols:
            prefix_len = min_cols
        for i in range(prefix_len):
            if not _values_fuzzy_match(ar[i], gr[i]):
                return None

    return VerdictResult(
        answer_extracted=str(agent_answer)[:200],
        is_match=True,
        verdict="match",
        confidence=0.93,
        explanation="Agent table rows match gold table rows (prefix columns)",
    )


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

        hints = _verdict_hints(agent_answer, gold_answer)

        user_prompt = (
            f"Question: {question}\n\n"
            f"Gold SQL:\n{gold_sql}\n\n"
            f"Gold answer: {gold_display}\n\n"
            f"Agent answer: {agent_display}"
        )

        if hints:
            user_prompt += f"\n\n{hints}"

        vr = self._try_call(VERDICT_SYSTEM_PROMPT, user_prompt)
        if vr is not None:
            if vr.verdict in ("wrong_answer", "mismatch"):
                salvage = _salvage_check(agent_answer, gold_answer)
                if salvage is not None:
                    logger.info(
                        "Salvage override: LLM said %s but salvage check found match: %s",
                        vr.verdict, salvage.explanation,
                    )
                    return salvage
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
            if vr.verdict in ("wrong_answer", "mismatch"):
                salvage = _salvage_check(agent_answer, gold_answer)
                if salvage is not None:
                    logger.info(
                        "Salvage override (retry): LLM said %s but salvage "
                        "check found match: %s",
                        vr.verdict, salvage.explanation,
                    )
                    return salvage
            return vr

        salvage = _salvage_check(agent_answer, gold_answer)
        if salvage is not None:
            return salvage

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
