"""Real LLM backend for the LLM-powered world model path.

Points at any OpenAI-compatible chat/completions endpoint — the HuggingFace
Inference Endpoint in .env, a local vLLM/Ollama server, or OpenAI itself.

The rule-based simulator (simulation/simulator.py) remains the baseline/fallback.
This client is a drop-in replacement for MockLLMClient anywhere the planner
expects a callable (prompt: str) -> str.

Configuration priority (highest → lowest)
------------------------------------------
1. Constructor arguments (api_key=, base_url=, model_name=)
2. OPENAI_* environment variables  ← our .env already sets these
3. SOMNUS_LLM_* environment variables  (secondary fallback)

The .env at the repo root already provides:
    OPENAI_API_KEY   = <key>
    OPENAI_BASE_URL  = https://<hf-endpoint>.aws.endpoints.huggingface.cloud/v1
    OPENAI_MODEL     = openai/gpt-oss-120b

python-dotenv is used to load .env automatically, so no manual export is needed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, APIError, APITimeoutError

# Load .env from the repo root (two levels up from this file) so callers
# do not have to remember to call load_dotenv() themselves.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _resolve(
    constructor_value: str | None,
    primary_env: str,
    fallback_env: str,
    label: str,
    required: bool = True,
) -> str | None:
    """Return the first non-empty value from: constructor → primary env → fallback env."""
    value = (
        constructor_value
        or os.getenv(primary_env)
        or os.getenv(fallback_env)
    )
    if required and not value:
        raise ValueError(
            f"{label} is required but was not found. "
            f"Set {primary_env} in your .env or pass it as a constructor argument."
        )
    return value


class OpenAICompatibleLLMClient:
    """Thin wrapper around the OpenAI SDK for any OpenAI-compatible endpoint.

    Implements both __call__ and generate so it is interchangeable with
    MockLLMClient anywhere the planner expects llm_client(prompt) -> str.

    Args:
        base_url:    Chat completions base URL (no trailing /chat/completions).
                     Defaults to OPENAI_BASE_URL → SOMNUS_LLM_BASE_URL.
        api_key:     Bearer token for the endpoint.
                     Defaults to OPENAI_API_KEY → SOMNUS_LLM_API_KEY.
        model_name:  Model identifier string sent in each request.
                     Defaults to OPENAI_MODEL → SOMNUS_LLM_MODEL.
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
                     0.7 is a reasonable default for structured-output tasks.
        timeout:     Per-request timeout in seconds.

    Example:
        # Config already in .env — no args needed:
        client = OpenAICompatibleLLMClient()

        # Or override individual fields:
        client = OpenAICompatibleLLMClient(
            base_url="http://localhost:8000/v1",
            model_name="mistral-7b-instruct",
        )

        best_action, score, bundle = choose_best_action_with_llm(
            state, user_profile=profile, horizon=5, llm_client=client
        )
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.7,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = _resolve(
            base_url, "OPENAI_BASE_URL", "SOMNUS_LLM_BASE_URL",
            label="base_url", required=False,
        )
        self.api_key = _resolve(
            api_key, "OPENAI_API_KEY", "SOMNUS_LLM_API_KEY",
            label="api_key", required=True,
        )
        self.model_name = _resolve(
            model_name, "OPENAI_MODEL", "SOMNUS_LLM_MODEL",
            label="model_name", required=True,
        )
        self.temperature = temperature
        self.timeout = timeout

        # Build the SDK client once; it is safe to reuse across calls.
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,  # None → uses api.openai.com
            timeout=self.timeout,
        )

    def generate(self, prompt: str) -> str:
        """Send a prompt to the endpoint and return the assistant's raw text.

        The prompt is sent as a single user message. System-role messages are
        omitted intentionally — the prompt itself contains all instructions.

        Args:
            prompt: The full prompt string built by build_world_model_prompt().

        Returns:
            Raw assistant message text (expected to be JSON).

        Raises:
            APITimeoutError: If the endpoint does not respond within self.timeout.
            APIError:        For 4xx/5xx responses from the endpoint.
            ValueError:      If the response shape is missing expected fields.
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
        except APITimeoutError as exc:
            raise APITimeoutError(
                f"Request to {self.base_url!r} timed out after {self.timeout}s. "
                "Check that the endpoint is reachable and increase timeout= if needed."
            ) from exc
        except APIError as exc:
            raise APIError(
                f"Endpoint returned an error: {exc}. "
                "Verify OPENAI_API_KEY and OPENAI_BASE_URL are correct.",
                response=exc.response,
                body=exc.body,
            ) from exc

        # Safely extract the first choice's message content
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise ValueError(
                f"Unexpected response shape — could not extract message content. "
                f"Raw response: {response}"
            ) from exc

        if not content:
            raise ValueError(
                "The model returned an empty response. "
                "Try raising max_tokens or check that the model is loaded correctly."
            )

        return content

    def __call__(self, prompt: str) -> str:
        """Allow the client to be passed directly as a callable llm_client."""
        return self.generate(prompt)
