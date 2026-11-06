
"""Tests for OpenRouter model configuration validation."""

from copy import deepcopy

import pytest

from evaluation_pipeline.experiment_config.model_registry import (
    get_openrouter_models,
    get_openrouter_model_config,
    load_model_config,
    validate_model,
)


def test_load_model_config():
    """YAML configuration can be loaded."""
    config = load_model_config()

    assert isinstance(config, dict)
    assert "openrouter" in config


def test_openrouter_models():
    """All configured models pass validation."""
    models = get_openrouter_models()

    assert len(models) > 0
    assert len({model["id"] for model in models}) == len(models)

    for model in models:
        assert isinstance(model["id"], str)
        assert isinstance(model["label"], str)


def test_model_lookup():
    """A model can be retrieved by its ID."""
    model = get_openrouter_models()[0]

    result = get_openrouter_model_config(model["id"])

    assert result["id"] == model["id"]


def test_unknown_model():
    """Unknown model IDs are rejected."""
    with pytest.raises(ValueError, match="Unknown OpenRouter model"):
        get_openrouter_model_config("unknown/model")


@pytest.mark.parametrize(
    "changes",
    [
        {"id": ""},
        {"label": ""},
        {"label": 123},
        {"max_tokens": -1},
        {"max_tokens": "1024"},
        {"max_tokens": True},
        {"reasoning": {"enabled": "false"}},
        {"provider": {"only": [], "allow_fallbacks": False}},
        {"provider": {"only": ["openai"], "allow_fallbacks": "false"}},
        {"reasonning": {"enabled": False}},
    ],
)
def test_invalid_model(changes):
    """Invalid model settings are rejected."""
    config = load_model_config()
    rules = config["openrouter"]["allowed_fields"]

    model = {
        "id": "test/model",
        "label": "Test Model",
    }
    model.update(deepcopy(changes))

    with pytest.raises(ValueError):
        validate_model(model, 1, rules)
