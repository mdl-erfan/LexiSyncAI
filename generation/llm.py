import json
import os

import ollama
from openai import OpenAI

MODEL = "llama3.2" # local Ollama model (default when no API key is set)

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence_sufficient": {"type": "boolean"},
        "answer": {"type": "string"},
    },
    "required": ["evidence_sufficient", "answer"],
}


class LLMUnavailableError(RuntimeError):
    """The hosted model could not be reached (rate limit, no credits, network...)."""


def cloud_enabled() -> bool:
    """Cloud mode is on when an API key is set (Streamlit secret or env variable)."""
    return bool(os.environ.get("LLM_API_KEY"))


def active_model() -> str:
    """Name of the model that answers: the cloud model, or local llama3.2."""
    if cloud_enabled():
        return os.environ.get("LLM_MODEL", "")
    return MODEL


def _cloud_client() -> OpenAI:
    """OpenAI-compatible client pointed at the provider (OpenRouter, Hugging Face...)."""
    return OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        timeout=60,
    )


def _parse_json(text: str | None) -> dict:
    """Parse the model's JSON reply; tolerate ```json fences. Raises ValueError if invalid."""
    if not text:
        raise ValueError("empty reply from the model")
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    result = json.loads(text)           # JSONDecodeError is a ValueError
    if not isinstance(result, dict):
        raise ValueError("reply is not a JSON object")
    return result


def _generate_local(messages: list[dict]) -> dict:
    """Local Ollama, exactly as before (used when no API key is set)."""
    response = ollama.chat(
        model=MODEL,
        messages=messages,
        format=ANSWER_SCHEMA,
        options={"temperature": 0, "num_predict": 400, "seed": 42},
        keep_alive="30m",
    )
    return json.loads(response["message"]["content"])


def _generate_cloud(messages: list[dict]) -> dict:
    """Hosted model through an OpenAI-compatible API (OpenRouter, Hugging Face...)."""
    request = {
        "model": active_model(),
        "messages": messages,
        "temperature": 0,
        "seed": 42,
        "max_tokens": 400,
    }
    if os.environ.get("LLM_JSON_MODE", "1") != "0":
        request["response_format"] = {"type": "json_object"}
    if os.environ.get("LLM_REASONING_OFF") == "1":
        request["extra_body"] = {"reasoning": {"enabled": False}}
    try:
        response = _cloud_client().chat.completions.create(**request)
    except Exception as exc:
        raise LLMUnavailableError(f"The hosted model is not available right now: {exc}") from exc
    return _parse_json(response.choices[0].message.content)


def generate(messages: list[dict]) -> dict:
    """Ask the LLM and return the parsed JSON: {'evidence_sufficient', 'answer'}."""
    if cloud_enabled():
        return _generate_cloud(messages)
    return _generate_local(messages)