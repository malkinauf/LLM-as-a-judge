
import pytest

from evaluation_pipeline import runner


EXAMPLE = {
    "id": "example-1",
    "question": "Is this statement true?",
    "model_response": "Yes.",
    "y_true": "truthful",
}


def run_experiment(method, confidence_mode=False):
    return runner.run_judge_experiment(
        dataset=[EXAMPLE],
        run_id="test-run",
        model="test-model",
        method=method,
        baseline_criterion="truthfulqa",
        baseline_criterion_detail="minimal",
        dataset_id="test-dataset",
        dynamic_prediction_variant="minimal",
        confidence_mode=confidence_mode,
    )


@pytest.mark.parametrize("confidence_mode", [False, True])
def test_baseline_confidence_modes(monkeypatch, confidence_mode):
    calls = []

    def fake_baseline_prompt(**kwargs):
        calls.append(kwargs)
        return "baseline prompt"

    def fake_judge_response(**kwargs):
        return {
            "predicted_label": "truthful",
            "explanation": "Correct answer.",
            "confidence": 4 if confidence_mode else None,
            "confidence_invalid": False,
            "raw_output": '{"verdict": "truthful"}',
        }

    monkeypatch.setattr(
        runner, "build_baseline_prompt", fake_baseline_prompt
    )
    monkeypatch.setattr(
        runner, "judge_response", fake_judge_response
    )

    result = run_experiment("baseline", confidence_mode)[0]

    assert calls[0]["confidence_mode"] is confidence_mode
    assert result["predicted_label"] == "truthful"
    assert result["first_level_confidence"] == (
        4 if confidence_mode else None
    )
    assert result["confidence_mode"] is confidence_mode
    assert result["first_level_confidence_invalid"] is False
    assert result["prediction_prompt"] is None


@pytest.mark.parametrize("confidence_mode", [False, True])
def test_dynamic_confidence_modes(monkeypatch, confidence_mode):
    prediction_calls = []
    dynamic_calls = []

    def fake_prediction_prompt(**kwargs):
        prediction_calls.append(kwargs)
        return "prediction prompt"

    def fake_raw_response(**kwargs):
        return "Preliminary analysis"

    def fake_dynamic_prompt(**kwargs):
        dynamic_calls.append(kwargs)
        return "dynamic prompt"

    def fake_judge_response(**kwargs):
        return {
            "predicted_label": "truthful",
            "explanation": "Correct answer.",
            "confidence": 5 if confidence_mode else None,
            "confidence_invalid": False,
            "raw_output": '{"verdict": "truthful"}',
        }

    monkeypatch.setattr(
        runner, "build_prediction_prompt", fake_prediction_prompt
    )
    monkeypatch.setattr(
        runner, "get_raw_model_response", fake_raw_response
    )
    monkeypatch.setattr(
        runner, "build_dynamic_prompt", fake_dynamic_prompt
    )
    monkeypatch.setattr(
        runner, "judge_response", fake_judge_response
    )

    result = run_experiment("dynamic", confidence_mode)[0]

    assert "confidence_mode" not in prediction_calls[0]
    assert dynamic_calls[0]["confidence_mode"] is confidence_mode
    assert dynamic_calls[0]["prediction_response"] == (
        "Preliminary analysis"
    )
    assert result["prediction_prompt"] == "prediction prompt"
    assert result["prediction_raw_output"] == "Preliminary analysis"
    assert result["first_level_confidence"] == (
        5 if confidence_mode else None
    )
    assert result["predicted_label"] == "truthful"
    assert result["dynamic_prediction_variant"] == "minimal"


def test_second_level_without_confidence(monkeypatch):
    calls = []

    def fake_baseline_prompt(**kwargs):
        assert kwargs["confidence_mode"] is False
        return "baseline prompt"

    def fake_second_level_prompt(**kwargs):
        return "second-level prompt"

    def fake_judge_response(**kwargs):
        calls.append(kwargs["prompt"])

        if kwargs["prompt"] == "baseline prompt":
            return {
                "predicted_label": "truthful",
                "explanation": "First assessment.",
                "confidence": None,
                "confidence_invalid": False,
                "raw_output": '{"verdict": "truthful"}',
            }

        return {
            "predicted_label": "correct",
            "explanation": "The first judge was correct.",
            "corrected_verdict": None,
            "corrected_explanation": None,
            "raw_output": '{"verdict": "correct"}',
        }

    monkeypatch.setattr(
        runner, "build_baseline_prompt", fake_baseline_prompt
    )
    monkeypatch.setattr(
        runner, "build_second_level_prompt", fake_second_level_prompt
    )
    monkeypatch.setattr(
        runner, "judge_response", fake_judge_response
    )

    result = run_experiment("second_level")[0]

    assert calls == ["baseline prompt", "second-level prompt"]
    assert result["second_level_verdict"] == "correct"
    assert result["predicted_label"] == "truthful"
    assert result["confidence_mode"] is False


def test_second_level_rejects_confidence():
    with pytest.raises(ValueError, match="Confidence Scale"):
        run_experiment("second_level", confidence_mode=True)


def test_invalid_confidence_preserves_verdict(monkeypatch):
    monkeypatch.setattr(
        runner,
        "build_baseline_prompt",
        lambda **kwargs: "baseline prompt",
    )
    monkeypatch.setattr(
        runner,
        "judge_response",
        lambda **kwargs: {
            "predicted_label": "truthful",
            "explanation": "Correct answer.",
            "confidence": None,
            "confidence_invalid": True,
            "raw_output": '{"verdict":"truthful","confidence":8}',
        },
    )

    result = run_experiment("baseline", confidence_mode=True)[0]

    assert result["predicted_label"] == "truthful"
    assert result["first_level_confidence"] is None
    assert result["first_level_confidence_invalid"] is True


def test_experiment_metadata(monkeypatch):
    monkeypatch.setattr(
        runner,
        "build_baseline_prompt",
        lambda **kwargs: "baseline prompt",
    )
    monkeypatch.setattr(
        runner,
        "judge_response",
        lambda **kwargs: {
            "predicted_label": "truthful",
            "explanation": "Correct answer.",
            "confidence": None,
            "confidence_invalid": False,
            "raw_output": '{"verdict":"truthful"}',
        },
    )

    result = run_experiment("baseline")[0]

    assert result["run_id"] == "test-run"
    assert result["dataset_id"] == "test-dataset"
    assert result["baseline_criterion"] == "truthfulqa"
    assert result["baseline_criterion_detail"] == "minimal"
    assert result["dynamic_prediction_variant"] is None
    assert result["confidence_mode"] is False


def test_first_level_parsing_error(monkeypatch):
    monkeypatch.setattr(
        runner,
        "build_baseline_prompt",
        lambda **kwargs: "baseline prompt",
    )
    monkeypatch.setattr(
        runner,
        "judge_response",
        lambda **kwargs: {
            "predicted_label": "parsing_error",
            "explanation": "Invalid JSON.",
            "confidence": None,
            "confidence_invalid": False,
            "raw_output": "invalid",
        },
    )

    result = run_experiment("baseline")[0]

    assert result["predicted_label"] == "parsing_error"
    assert result["first_raw_output"] == "invalid"
