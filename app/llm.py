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


class LLMError(Exception):
    pass


def generate_reply(history: List[Dict[str, str]]) -> str:
    """history: [{"role": "user"|"assistant", "content": str}], oldest first."""
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
    try:
        resp = httpx.post(
            GEMINI_URL.format(model=settings.gemini_model),
            headers={"x-goog-api-key": settings.gemini_api_key},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (httpx.HTTPError, KeyError, IndexError) as exc:
        raise LLMError(f"LLM request failed: {exc}") from exc
