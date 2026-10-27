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
