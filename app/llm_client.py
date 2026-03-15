"""
Somnus LLM client.

Wraps the OpenAI Python SDK configured for the hackathon GPT-OSS endpoint.
The base_url can point to any OpenAI-compatible server (OpenAI, HuggingFace
TGI, vLLM, etc.) — switching providers is a one-line env var change.

Configuration (all from environment / .env):
  OPENAI_API_KEY   — required; set to "test" for local dev without real calls
  OPENAI_BASE_URL  — optional; set to hackathon endpoint to use GPT-OSS
  OPENAI_MODEL     — optional; defaults to gpt-4o-mini

Usage:
  from app.llm_client import chat_json, is_configured

  if is_configured():
      result = chat_json(system="...", user="...")  # raises on failure

Call sites (strategist.py, journal_reflection.py) always wrap chat_json in
try/except and fall back to deterministic logic — the client never crashes
the app.
"""

import json
import logging
import os

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

# Module-level singleton — created once on first call, None if not configured.
_client: OpenAI | None = None


def is_configured() -> bool:
    """True if OPENAI_API_KEY is present in the environment."""
    return bool(os.getenv("OPENAI_API_KEY"))


def get_client() -> OpenAI | None:
    """
    Return the shared OpenAI client, initialising it on first call.

    Returns None if OPENAI_API_KEY is not set so callers can short-circuit
    without triggering an exception.
    """
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    kwargs: dict = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url

    _client = OpenAI(**kwargs)
    return _client


def default_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def chat_json(
    system: str,
    user: str,
    temperature: float = 0.3,
) -> dict:
    """
    Send a chat completion request and return the parsed JSON response.

    Constraint compliance:
      - Uses Chat Completions API (not Assistants).
      - Passes base_url from OPENAI_BASE_URL so this works with any
        OpenAI-compatible endpoint (OpenAI, HuggingFace TGI, vLLM, etc.).
      - Requests response_format={"type": "json_object"} for structured output.
        If the endpoint rejects this parameter (older or non-compliant servers)
        we retry once without it — the prompt already instructs JSON-only output
        so the response is still parseable.

    Args:
        system:      System prompt — sets the model's role and output format.
        user:        User prompt — the actual input/context for this call.
        temperature: Sampling temperature.  0.3 is a good default for
                     structured JSON generation (low variance, still flexible).

    Returns:
        Parsed dict from the model's JSON response.

    Raises:
        RuntimeError:        If OPENAI_API_KEY is not configured.
        openai.APIError:     On network or model-side failures (not retried).
        json.JSONDecodeError: If the model returns non-JSON despite instructions.
    """
    client = get_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured — cannot make LLM call")

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    model = default_model()

    try:
        # Preferred path: request JSON mode explicitly.
        # This activates grammar-constrained decoding on supporting endpoints
        # (OpenAI, TGI ≥ 2.0, vLLM) which makes the output reliably parseable.
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
    except BadRequestError as exc:
        # Some endpoints (older TGI builds, custom servers) reject the
        # response_format parameter entirely.  Retry without it — the system
        # prompt already enforces JSON-only output so parsing still works.
        if "response_format" in str(exc).lower() or "json" in str(exc).lower():
            logger.debug(
                "Endpoint rejected response_format=json_object (%s); "
                "retrying without it — will rely on prompt-based JSON enforcement",
                exc,
            )
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
        else:
            raise  # Unrelated 400 error — let it propagate to the fallback handler

    raw = response.choices[0].message.content or ""
    return _parse_json(raw)


def _parse_json(raw: str) -> dict:
    """
    Parse a JSON string, stripping markdown code fences if present.

    Some models wrap their JSON in ```json ... ``` even when told not to.
    """
    text = raw.strip()

    # Strip opening fence (```json or ```)
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop the first line (fence) and the last line if it's a closing fence
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()

    return json.loads(text)
