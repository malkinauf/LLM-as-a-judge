
import json
from io import BytesIO
from urllib.error import HTTPError

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

    assert result["predicted_label"] is None
    assert result["parsing_failed"] is True
    assert result["raw_output"] == raw_output


# OpenRouter tests: no network calls or real API key required.


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
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
        })

    monkeypatch.setattr(judge, "urlopen", fake_urlopen)

    result = judge.get_raw_model_response(
        "Example prompt",
        "openai/gpt-4.1-mini",
        backend="openrouter",
        return_usage=True,
    )

    assert result["raw_output"] == "analysis"
    assert (
        result["input_tokens"],
        result["output_tokens"],
        result["total_tokens"],
    ) == (11, 7, 18)
    assert result["latency_seconds"] >= 0

    request, timeout = requests[0]
    assert timeout == judge.OPENROUTER_TIMEOUT_SECONDS
    assert request.full_url == judge.OPENROUTER_URL

    payload = json.loads(request.data)
    assert payload["model"] == "openai/gpt-4.1-mini"
    assert payload["messages"] == [
        {"role": "user", "content": "Example prompt"}
    ]
    assert "response_format" not in payload


def test_openrouter_judge_parsing_and_confidence(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-key")
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        return FakeHTTPResponse({
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "verdict": "truthful",
                            "confidence": 4,
                            "explanation": "Verified",
                        })
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
            },
        })

    monkeypatch.setattr(judge, "urlopen", fake_urlopen)

    result = judge.judge_response(
        "Judge prompt",
        "openai/gpt-4.1-mini",
        backend="openrouter",
    )

    assert result["predicted_label"] == "truthful"
    assert result["confidence"] == 4
    assert result["confidence_invalid"] is False
    assert result["total_tokens"] == 15
    assert json.loads(requests[0].data)["response_format"] == {
        "type": "json_object"
    }


def test_openrouter_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        judge.judge_response(
            "prompt",
            "openai/gpt-4.1-mini",
            backend="openrouter",
        )



@pytest.mark.parametrize("status", [403, 429])
def test_openrouter_http_error_preserves_metadata(monkeypatch, status):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-key")

    attempts = []
    waits = []

    def fake_urlopen(request, timeout):
        attempts.append(1)
        raise HTTPError(
            judge.OPENROUTER_URL,
            status,
            "API error",
            {},
            BytesIO(b'{"error":{"message":"Request rejected"}}'),
        )

    monkeypatch.setattr(judge, "urlopen", fake_urlopen)
    monkeypatch.setattr(judge.time, "sleep", lambda seconds: waits.append(seconds))

    with pytest.raises(judge.ModelAPIError) as exc_info:
        judge.judge_response(
            "prompt",
            "openai/gpt-4.1-mini",
            backend="openrouter",
        )

    assert exc_info.value.status_code == status
    assert exc_info.value.details["error"]["message"] == "Request rejected"

    if status == 429:
        assert len(attempts) == 5
        assert waits == [10, 20, 40, 60]
    else:
        assert len(attempts) == 1
        assert waits == []


@pytest.mark.parametrize("backend", ["invalid", "openai"])
def test_invalid_backend_rejected(backend):
    with pytest.raises(ValueError, match="Unknown backend"):
        judge.get_raw_model_response(
            "prompt",
            "model",
            backend=backend,
        )


def test_default_backend_uses_ollama(monkeypatch):
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return {"message": {"content": "Ollama reply"}}

    monkeypatch.setattr(judge, "chat", fake_chat)

    assert (
        judge.get_raw_model_response("prompt", "local-model")
        == "Ollama reply"
    )
    assert calls[0]["model"] == "local-model"


# Verify that OpenRouter requests use settings from models.yaml.


@pytest.mark.parametrize(
    "model_id, expected_fields",
    [
        (
            "openai/gpt-4.1-mini",
            {},
        ),
        (
            "deepseek/deepseek-v4.1-flash-1017",
            {
                "provider": {
                    "only": ["relace/fp4"],
                    "allow_fallbacks": False,
                },
                "reasoning": {
                    "enabled": False,
                },
            },
        ),
        (
            "google/gemini-2.5-flash",
            {
                "provider": {
                    "only": ["google-ai-studio"],
                    "allow_fallbacks": False,
                },
                "reasoning": {
                    "enabled": False,
                },
            },
        ),
        (
            "openai/gpt-5.4-mini",
            {
                "provider": {
                    "only": ["openai"],
                    "allow_fallbacks": False,
                },
                "reasoning": {
                    "enabled": False,
                },
                "max_tokens": 1024,
            },
        ),
    ],
)
@pytest.mark.parametrize("json_mode", [False, True])
def test_openrouter_model_configuration(
    monkeypatch,
    model_id,
    expected_fields,
    json_mode,
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-key")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        return FakeHTTPResponse({
            "choices": [
                {
                    "message": {
                        "content": '{"verdict": "truthful"}'
                    }
                }
            ],
            "usage": {},
        })

    monkeypatch.setattr(judge, "urlopen", fake_urlopen)

    judge._openrouter_call(
        "Test prompt",
        model_id,
        json_mode=json_mode,
    )

    payload = captured["payload"]

    assert payload["model"] == model_id
    assert payload["temperature"] == 0
    assert payload["messages"] == [
        {"role": "user", "content": "Test prompt"}
    ]

    for field in ("provider", "reasoning", "max_tokens"):
        if field in expected_fields:
            assert payload[field] == expected_fields[field]
        else:
            assert field not in payload

    if json_mode:
        assert payload["response_format"] == {
            "type": "json_object"
        }
    else:
        assert "response_format" not in payload


def test_openrouter_retry_then_success(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-key")

    attempts = []
    waits = []

    def fake_urlopen(request, timeout):
        attempts.append(1)

        if len(attempts) == 1:
            raise HTTPError(
                judge.OPENROUTER_URL,
                429,
                "Too Many Requests",
                {},
                BytesIO(b'{"error":{"message":"Rate limit exceeded"}}'),
            )

        return FakeHTTPResponse({
            "choices": [
                {"message": {"content": '{"verdict": "truthful"}'}}
            ],
            "usage": {},
        })

    monkeypatch.setattr(judge, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        judge.time,
        "sleep",
        lambda seconds: waits.append(seconds),
    )

    result = judge.judge_response(
        "Test prompt",
        "openai/gpt-4.1-mini",
        backend="openrouter",
    )

    assert len(attempts) == 2
    assert waits == [10]
    assert result["predicted_label"] == "truthful"
    assert result["parsing_failed"] is False

