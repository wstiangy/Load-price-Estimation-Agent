from __future__ import annotations

import json
import os
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime when optional dependency is missing.
    OpenAI = None


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_MAX_OUTPUT_TOKENS = 420


def llm_status() -> tuple[bool, str]:
    if OpenAI is None:
        return False, "The OpenAI Python SDK is not installed in this environment."
    if not os.environ.get("OPENAI_API_KEY"):
        return False, "Set OPENAI_API_KEY to enable the in-page research assistant."
    return True, f"Research assistant ready on {chat_model_name()}."


def chat_model_name() -> str:
    return os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)


def generate_chat_reply(
    user_message: str,
    page_context: dict[str, Any],
    previous_response_id: str | None = None,
) -> dict[str, str | None]:
    if OpenAI is None:
        raise RuntimeError("The OpenAI Python SDK is not installed in this environment.")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured for this deployment.")

    client = OpenAI(api_key=api_key)
    payload: dict[str, Any] = {
        "model": chat_model_name(),
        "instructions": _developer_instructions(),
        "input": [
            {
                "role": "user",
                "content": (
                    "Dashboard run context:\n"
                    f"{_format_context(page_context)}\n\n"
                    f"User question:\n{user_message.strip()}"
                ),
            }
        ],
        "max_output_tokens": int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)),
        "metadata": {"surface": "ieee14-dashboard"},
    }
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id

    response = client.responses.create(**payload)
    reply_text = (getattr(response, "output_text", None) or _extract_output_text(response)).strip()
    if not reply_text:
        reply_text = "I could not generate a text answer for that question."
    return {
        "reply": reply_text,
        "response_id": getattr(response, "id", None),
        "model": payload["model"],
    }


def _developer_instructions() -> str:
    return (
        "You are a research copilot embedded in an IEEE 14-bus electricity-price demo. "
        "Answer using the supplied run context first, then general power-systems reasoning. "
        "Be concise, analytical, and helpful. "
        "If the user asks for values that are not present in the context, say that the page would need "
        "to be rerun or extended rather than inventing numbers. "
        "Do not claim to have run a new OPF or forecast unless it is explicitly in the supplied context."
    )


def _format_context(page_context: dict[str, Any]) -> str:
    return json.dumps(page_context, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _extract_output_text(response: Any) -> str:
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)
