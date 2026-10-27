import json

import pytest

from evaluation_pipeline import judge


@pytest.mark.parametrize(
    "value, expected",
    [
        (1, (1, False)),
        (3, (3, False)),
        (5, (5, False)),
        ("4", (4, False)),
        (None, (None, False)),
        (0, (None, True)),
        (6, (None, True)),
        (4.5, (None, True)),
        ("high", (None, True)),
        (True, (None, True)),
    ],
)
def test_parse_confidence(value, expected):
    assert judge.parse_confidence(value) == expected


def test_judge_response_with_confidence(monkeypatch):
    output = {
        "verdict": "truthful",
        "confidence": 4,
        "explanation": "The answer is factually correct.",
    }

    def fake_chat(**kwargs):
        return {"message": {"content": json.dumps(output)}}

    monkeypatch.setattr(judge, "chat", fake_chat)

    result = judge.judge_response("Test prompt", "test-model")

    assert result["predicted_label"] == "truthful"
    assert result["confidence"] == 4
    assert result["confidence_invalid"] is False
    assert result["raw_output"] == json.dumps(output)


def test_judge_response_without_confidence(monkeypatch):
    output = {
        "verdict": "truthful",
        "explanation": "The answer is factually correct.",
    }

    def fake_chat(**kwargs):
        return {"message": {"content": json.dumps(output)}}

    monkeypatch.setattr(judge, "chat", fake_chat)

    result = judge.judge_response("Test prompt", "test-model")

    assert result["predicted_label"] == "truthful"
    assert result["confidence"] is None
    assert result["confidence_invalid"] is False


def test_judge_response_invalid_confidence(monkeypatch):
    output = {
        "verdict": "truthful",
        "confidence": 8,
        "explanation": "The answer is factually correct.",
    }

    def fake_chat(**kwargs):
        return {"message": {"content": json.dumps(output)}}

    monkeypatch.setattr(judge, "chat", fake_chat)

    result = judge.judge_response("Test prompt", "test-model")

    assert result["predicted_label"] == "truthful"
    assert result["confidence"] is None
    assert result["confidence_invalid"] is True


@pytest.mark.parametrize(
    "raw_output",
    [
        "not valid JSON",
        '[{"verdict": "truthful"}]',
    ],
)
def test_judge_response_parsing_error(monkeypatch, raw_output):
    def fake_chat(**kwargs):
        return {"message": {"content": raw_output}}

    monkeypatch.setattr(judge, "chat", fake_chat)

    result = judge.judge_response("Test prompt", "test-model")
    assert result["predicted_label"] == "parsing_error"
    assert result["raw_output"] == raw_output


# OpenRouter tests: no network calls or real API key required.
from io import BytesIO
from urllib.error import HTTPError


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return BytesIO(json.dumps(self.payload).encode("utf-8"))

    def __exit__(self, *args):
        return False


def test_openrouter_raw_response_and_usage(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-key")
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeHTTPResponse({
            "choices": [{"message": {"content": "analysis"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        })

    monkeypatch.setattr(judge, "urlopen", fake_urlopen)
    result = judge.get_raw_model_response(
        "Example prompt", "openai/gpt-4.1", backend="openrouter", return_usage=True
    )
    assert result["raw_output"] == "analysis"
    assert (result["input_tokens"], result["output_tokens"], result["total_tokens"]) == (11, 7, 18)
    assert result["latency_seconds"] >= 0
    request, timeout = requests[0]
    assert timeout == 120
    assert request.full_url == judge.OPENROUTER_URL
    payload = json.loads(request.data)
    assert payload["model"] == "openai/gpt-4.1"
    assert payload["messages"] == [{"role": "user", "content": "Example prompt"}]
    assert "response_format" not in payload


def test_openrouter_judge_parsing_and_confidence(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-key")
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        return FakeHTTPResponse({
            "choices": [{"message": {"content": json.dumps({
                "verdict": "truthful", "confidence": 4, "explanation": "Verified"
            })}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })

    monkeypatch.setattr(judge, "urlopen", fake_urlopen)
    result = judge.judge_response("Judge prompt", "openai/gpt-4.1", backend="openrouter")
    assert result["predicted_label"] == "truthful"
    assert result["confidence"] == 4
    assert result["confidence_invalid"] is False
    assert result["total_tokens"] == 15
    assert json.loads(requests[0].data)["response_format"] == {"type": "json_object"}


def test_openrouter_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        judge.judge_response("prompt", "model", backend="openrouter")


@pytest.mark.parametrize("status", [403, 429])
def test_openrouter_http_error_preserves_metadata(monkeypatch, status):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-key")

    def fake_urlopen(request, timeout):
        raise HTTPError(
            judge.OPENROUTER_URL, status, "API error", {},
            BytesIO(b'{"error":{"message":"Request rejected"}}'),
        )

    monkeypatch.setattr(judge, "urlopen", fake_urlopen)
    with pytest.raises(judge.ModelAPIError) as exc_info:
        judge.judge_response("prompt", "model", backend="openrouter")
    assert exc_info.value.status_code == status
    assert exc_info.value.details["error"]["message"] == "Request rejected"


@pytest.mark.parametrize("backend", ["invalid", "openai"])
def test_invalid_backend_rejected(backend):
    with pytest.raises(ValueError, match="Unknown backend"):
        judge.get_raw_model_response("prompt", "model", backend=backend)


def test_default_backend_uses_ollama(monkeypatch):
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return {"message": {"content": "Ollama reply"}}

    monkeypatch.setattr(judge, "chat", fake_chat)
    assert judge.get_raw_model_response("prompt", "local-model") == "Ollama reply"
    assert calls[0]["model"] == "local-model"
