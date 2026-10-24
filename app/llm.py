import time
from typing import Dict, List

import httpx

from .config import settings

SYSTEM_PROMPT = """You are a concise, helpful assistant.
Rules:
- Answer in plain text, at most 5 sentences unless the user asks for more.
- If you are not sure of a fact, say so instead of guessing.
- Use the earlier messages in this conversation to stay consistent.

Example:
User: What is an API?
Assistant: An API is a defined way for one program to ask another program for data or actions, usually over HTTP.
"""

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRY_STATUS = {429, 500, 502, 503, 504}
RETRIES_PER_MODEL = 2


class LLMError(Exception):
    pass


def _call_model(model: str, payload: dict) -> str:
    """One model with short exponential backoff (1s, 2s) on rate limits / temporary server errors."""
    last = "unknown error"
    for attempt in range(RETRIES_PER_MODEL + 1):
        try:
            resp = httpx.post(
                GEMINI_URL.format(model=model),
                headers={"x-goog-api-key": settings.gemini_api_key},
                json=payload,
                timeout=30,
            )
            if resp.status_code in RETRY_STATUS:
                last = f"HTTP {resp.status_code}"
            else:
                resp.raise_for_status()
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except httpx.TransportError as exc:
            last = str(exc)
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        except (KeyError, IndexError) as exc:
            raise LLMError("LLM returned an unexpected response") from exc
        if attempt < RETRIES_PER_MODEL:
            time.sleep(2 ** attempt)
    raise LLMError(f"{model} is busy ({last})")


def generate_reply(history: List[Dict[str, str]]) -> str:
    """history: [{"role": "user"|"assistant", "content": str}], oldest first.

    Tries the main model, then the fallback model if the main one stays busy.
    """
    if not settings.gemini_api_key:
        raise LLMError("GEMINI_API_KEY is not set")

    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
            for m in history
        ],
        "generationConfig": {"temperature": settings.temperature},
    }
    errors = []
    for model in dict.fromkeys([settings.gemini_model, settings.gemini_fallback_model]):
        try:
            return _call_model(model, payload)
        except LLMError as exc:
            if "busy" not in str(exc):
                raise
            errors.append(str(exc))
    raise LLMError("The AI service is busy right now. Please try again in a moment. (" + "; ".join(errors) + ")")
