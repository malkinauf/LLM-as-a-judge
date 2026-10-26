
import pytest

from evaluation_pipeline.prompts import builder
from evaluation_pipeline.prompts.templates import (
    CONFIDENCE_SCALE_TEMPLATE,
    OUTPUT_TEMPLATE,
)


# ============================================
# Test configuration
# ============================================

@pytest.fixture
def mock_criteria(monkeypatch):
    """Provide a controlled criterion configuration."""

    criteria = {
        "test_criterion": {
            "descriptions": {
                "basic": "Test evaluation criterion.",
                "detailed": "Detailed test evaluation criterion.",
            },
            "labels": {
                "positive": "positive",
                "negative": "negative",
            },
            "dynamic": {
                "prediction_instructions": {
                    "paper_based": "Analyze the response.",
                },
            },
        }
    }

    monkeypatch.setattr(builder, "CRITERIA", criteria)
    return criteria


@pytest.fixture
def prompt_data():
    return {
        "criterion": "test_criterion",
        "criterion_detail": "basic",
        "question": "Test question?",
        "model_response": "Test model response.",
    }


# ============================================
# Baseline body
# ============================================

@pytest.mark.parametrize("confidence_mode", [False, True])
def test_baseline_body_confidence(
    mock_criteria, prompt_data, confidence_mode
):
    body = builder.build_baseline_body(
        **prompt_data,
        confidence_mode=confidence_mode,
    )

    assert "Test evaluation criterion." in body
    assert '"positive"' in body
    assert '"negative"' in body
    assert "Test question?" in body
    assert "Test model response." in body

    assert ("### Confidence Scale" in body) == confidence_mode
    assert ('"confidence"' in body) == confidence_mode

    if confidence_mode:
        assert CONFIDENCE_SCALE_TEMPLATE in body
    else:
        assert CONFIDENCE_SCALE_TEMPLATE not in body

    assert "### Your Output" not in body


def test_baseline_body_detailed(mock_criteria, prompt_data):
    body = builder.build_baseline_body(
        **{**prompt_data, "criterion_detail": "detailed"}
    )

    assert "Detailed test evaluation criterion." in body


# ============================================
# Baseline prompt
# ============================================

@pytest.mark.parametrize("confidence_mode", [False, True])
def test_baseline_prompt_confidence(
    mock_criteria, prompt_data, confidence_mode
):
    prompt = builder.build_baseline_prompt(
        **prompt_data,
        confidence_mode=confidence_mode,
    )

    assert ("### Confidence Scale" in prompt) == confidence_mode
    assert ('"confidence"' in prompt) == confidence_mode
    assert '"verdict"' in prompt
    assert '"explanation"' in prompt
    assert prompt.endswith(OUTPUT_TEMPLATE)


def test_baseline_default_without_confidence(
    mock_criteria, prompt_data
):
    default_prompt = builder.build_baseline_prompt(**prompt_data)

    explicit_prompt = builder.build_baseline_prompt(
        **prompt_data,
        confidence_mode=False,
    )

    assert default_prompt == explicit_prompt


# ============================================
# Second-Level
# ============================================

def test_second_level_prompt(mock_criteria):
    prompt = builder.build_second_level_prompt(
        criterion="test_criterion",
        first_level_prompt="Original judge task.",
        first_level_response='{"verdict": "positive"}',
    )

    assert "Original judge task." in prompt
    assert '{"verdict": "positive"}' in prompt
    assert '"correct"' in prompt
    assert '"not_correct"' in prompt
    assert '"positive | negative"' in prompt


# ============================================
# Dynamic prediction
# ============================================

def test_prediction_prompt(mock_criteria):
    prompt = builder.build_prediction_prompt(
        criterion="test_criterion",
        prediction_variant="paper_based",
        question="Test question?",
        model_response="Test model response.",
    )

    assert "Analyze the response." in prompt
    assert "Test question?" in prompt
    assert "Test model response." in prompt


# ============================================
# Dynamic final prompt
# ============================================

@pytest.mark.parametrize("confidence_mode", [False, True])
def test_dynamic_prompt_confidence(
    mock_criteria, prompt_data, confidence_mode
):
    prompt = builder.build_dynamic_prompt(
        **prompt_data,
        prediction_response="Preliminary analysis.",
        confidence_mode=confidence_mode,
    )

    assert ("### Confidence Scale" in prompt) == confidence_mode
    assert ('"confidence"' in prompt) == confidence_mode

    assert "Test evaluation criterion." in prompt
    assert "Test question?" in prompt
    assert "Test model response." in prompt
    assert "### Additional Analysis" in prompt
    assert "Preliminary analysis." in prompt
    assert prompt.endswith(OUTPUT_TEMPLATE)


def test_dynamic_default_without_confidence(
    mock_criteria, prompt_data
):
    default_prompt = builder.build_dynamic_prompt(
        **prompt_data,
        prediction_response="Preliminary analysis.",
    )

    explicit_prompt = builder.build_dynamic_prompt(
        **prompt_data,
        prediction_response="Preliminary analysis.",
        confidence_mode=False,
    )

    assert default_prompt == explicit_prompt


@pytest.mark.parametrize("confidence_mode", [False, True])
def test_dynamic_reuses_baseline_body(
    mock_criteria, prompt_data, confidence_mode
):
    body = builder.build_baseline_body(
        **prompt_data,
        confidence_mode=confidence_mode,
    )

    dynamic_prompt = builder.build_dynamic_prompt(
        **prompt_data,
        prediction_response="Preliminary analysis.",
        confidence_mode=confidence_mode,
    )

    assert dynamic_prompt.startswith(body)


# ============================================
# Error handling
# ============================================

@pytest.mark.parametrize(
    "function_name, kwargs",
    [
        (
            "build_baseline_body",
            {
                "criterion_detail": "basic",
                "question": "Question",
                "model_response": "Response",
            },
        ),
        (
            "build_baseline_prompt",
            {
                "criterion_detail": "basic",
                "question": "Question",
                "model_response": "Response",
            },
        ),
        (
            "build_second_level_prompt",
            {
                "first_level_prompt": "Prompt",
                "first_level_response": "Response",
            },
        ),
        (
            "build_prediction_prompt",
            {
                "prediction_variant": "paper_based",
                "question": "Question",
                "model_response": "Response",
            },
        ),
        (
            "build_dynamic_prompt",
            {
                "criterion_detail": "basic",
                "question": "Question",
                "model_response": "Response",
                "prediction_response": "Analysis",
            },
        ),
    ],
)
def test_unknown_criterion(
    mock_criteria, function_name, kwargs
):
    function = getattr(builder, function_name)

    with pytest.raises(ValueError, match="Unknown criterion"):
        function(criterion="unknown", **kwargs)


def test_unknown_criterion_detail(mock_criteria, prompt_data):
    with pytest.raises(
        ValueError, match="Unknown criterion detail"
    ):
        builder.build_baseline_body(
            **{**prompt_data, "criterion_detail": "unknown"}
        )


def test_unknown_prediction_variant(mock_criteria):
    with pytest.raises(
        ValueError, match="Unknown prediction variant"
    ):
        builder.build_prediction_prompt(
            criterion="test_criterion",
            prediction_variant="unknown",
            question="Question",
            model_response="Response",
        )


def test_dynamic_unknown_criterion_detail(
    mock_criteria, prompt_data
):
    with pytest.raises(
        ValueError, match="Unknown criterion detail"
    ):
        builder.build_dynamic_prompt(
            **{**prompt_data, "criterion_detail": "unknown"},
            prediction_response="Analysis",
        )
