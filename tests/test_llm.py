import pytest

from generation import llm


class FakeClient:
    """Stands in for the OpenAI client: records the request, returns a prepared reply."""

    def __init__(self, content=None, error=None):
        self.content, self.error, self.kwargs = content, error, None
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        message = type("M", (), {"content": self.content})()
        return type("R", (), {"choices": [type("C", (), {"message": message})()]})()


MESSAGES = [{"role": "user", "content": "hi"}]


def use_cloud(monkeypatch, client):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_cloud_client", lambda: client)


def test_no_key_means_local_ollama(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert not llm.cloud_enabled()
    assert llm.active_model() == llm.MODEL


def test_key_switches_to_cloud_model(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "some/model")
    assert llm.cloud_enabled()
    assert llm.active_model() == "some/model"


def test_cloud_reply_is_parsed_and_request_is_deterministic(monkeypatch):
    client = FakeClient('{"evidence_sufficient": true, "answer": "30 days [1]"}')
    use_cloud(monkeypatch, client)
    assert llm.generate(MESSAGES) == {"evidence_sufficient": True, "answer": "30 days [1]"}
    assert client.kwargs["temperature"] == 0
    assert client.kwargs["seed"] == 42
    assert client.kwargs["response_format"] == {"type": "json_object"}


def test_json_mode_can_be_switched_off(monkeypatch):
    client = FakeClient('{"evidence_sufficient": false, "answer": "no"}')
    use_cloud(monkeypatch, client)
    monkeypatch.setenv("LLM_JSON_MODE", "0")
    llm.generate(MESSAGES)
    assert "response_format" not in client.kwargs


def test_code_fence_around_json_is_tolerated(monkeypatch):
    use_cloud(monkeypatch, FakeClient('```json\n{"evidence_sufficient": true, "answer": "x [1]"}\n```'))
    assert llm.generate(MESSAGES)["answer"] == "x [1]"


def test_invalid_json_raises_value_error_so_answer_question_refuses(monkeypatch):
    use_cloud(monkeypatch, FakeClient("sorry, I cannot do that"))
    with pytest.raises(ValueError):
        llm.generate(MESSAGES)


def test_service_error_is_not_a_value_error(monkeypatch):
    use_cloud(monkeypatch, FakeClient(error=RuntimeError("402 payment required")))
    with pytest.raises(llm.LLMUnavailableError) as info:
        llm.generate(MESSAGES)
    assert not isinstance(info.value, ValueError)
    assert "402" in str(info.value)
