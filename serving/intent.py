"""LLM-based session intent inference using Ollama.

Infers what a user is shopping for from a short session and predicts which
semantic ID (c0, c1) prefixes best describe their likely next purchases.

This is the primary cold-start path when session length < COLD_START_THRESHOLD
(see serving/api/routes.py). On timeout or any parse failure the caller falls
back to the prefix-sampling path in _prefix_recommend().

Environment variables
---------------------
  OLLAMA_BASE_URL  base URL for Ollama (default: http://localhost:11434)
  OLLAMA_MODEL     model name to use  (default: llama3.2)
"""

import json
import os
from typing import Any

import httpx
from pydantic import BaseModel

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
_TIMEOUT_SECONDS = 0.2  # 200ms hard limit


class IntentTimeoutError(Exception):
    pass


class PredictedPrefix(BaseModel):
    c0: int
    c1: int
    weight: float


class IntentResult(BaseModel):
    intent: str
    predicted_prefixes: list[PredictedPrefix]
    confidence: float


_SYSTEM_PROMPT = """\
You are a recommendation system assistant. Your task is to infer a user's \
shopping intent from a short browsing session and predict which semantic ID \
prefix combinations best describe what they are likely to look for next.

Semantic IDs are hierarchical discrete codes assigned to products:
  c0 (0-255): coarse category — items sharing c0 belong to the same broad product family
  c1 (0-255): subcategory — items sharing (c0, c1) belong to the same subcategory

The user message lists the session items with their actual (c0, c1) values from \
the live codebook. Use those values as your vocabulary when predicting next prefixes \
— predict values you observe in the session or nearby ones that are semantically \
consistent. Do not invent arbitrary numbers.

Return ONLY valid JSON with this exact shape, no other text:
{
  "intent": "<short natural language description of what the user is likely shopping for>",
  "predicted_prefixes": [
    {"c0": <int>, "c1": <int>, "weight": <float>}
  ],
  "confidence": <float between 0.0 and 1.0>
}

Rules:
  - predicted_prefixes must have 1-3 entries
  - weights must sum to approximately 1.0
  - confidence reflects how clearly the session signals a single intent\
"""


def _build_user_message(item_titles: list[str], item_codes: list[list[int]]) -> str:
    lines = ["Session items (title → semantic ID prefix):"]
    for title, codes in zip(item_titles, item_codes):
        c0 = codes[0] if len(codes) > 0 else "?"
        c1 = codes[1] if len(codes) > 1 else "?"
        lines.append(f'  - "{title}" → c0={c0}, c1={c1}')
    lines.append("\nPredict the user's intent and likely next semantic ID prefixes.")
    return "\n".join(lines)


async def infer_session_intent(
    item_titles: list[str],
    item_codes: list[list[int]],
) -> IntentResult:
    """Infer session intent via an Ollama LLM call.

    Args:
        item_titles: display titles of the session items
        item_codes:  semantic ID tuples per item, same order as item_titles

    Returns:
        IntentResult with intent string, predicted (c0, c1) prefixes, and confidence

    Raises:
        IntentTimeoutError: LLM call exceeded 200ms
        pydantic.ValidationError: LLM returned malformed JSON
        httpx.HTTPError: network or HTTP error other than timeout
    """
    payload: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(item_titles, item_codes)},
        ],
        "stream": False,
        "format": "json",
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
            )
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise IntentTimeoutError(
            f"Ollama call exceeded {_TIMEOUT_SECONDS * 1000:.0f}ms"
        ) from exc

    raw_content: str = response.json()["message"]["content"]
    return IntentResult.model_validate(json.loads(raw_content))
