import json

import ollama

MODEL = "llama3.2"

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence_sufficient": {"type": "boolean"},
        "answer": {"type": "string"},
    },
    "required": ["evidence_sufficient", "answer"],
}


def generate(messages: list[dict]) -> dict:
    """Ask the local LLM and return the parsed JSON: {'evidence_sufficient', 'answer'}."""
    response = ollama.chat(
        model=MODEL,
        messages=messages,
        format=ANSWER_SCHEMA,
        options={"temperature": 0, "num_predict": 400, "seed": 42},
        keep_alive="30m",
    )
    return json.loads(response["message"]["content"])