
"""Load and validate OpenRouter model configurations."""

import yaml

from importlib.resources import files


def load_model_config() -> dict:
    """Load the model configuration from YAML."""
    config_path = files(
        "evaluation_pipeline.experiment_config"
    ).joinpath("models.yaml")

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except OSError as exc:
        raise ValueError(
            f"Cannot read models.yaml: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML syntax in models.yaml: {exc}"
        ) from exc

    if not isinstance(config, dict):
        raise ValueError(
            "models.yaml must contain a YAML mapping."
        )

    return config


def validate_allowed_fields(
    data: dict,
    allowed_fields: list,
    context: str,
) -> None:
    """Check for unknown configuration fields."""
    if (
        not isinstance(allowed_fields, list)
        or not allowed_fields
        or any(
            not isinstance(field, str) or not field.strip()
            for field in allowed_fields
        )
        or len(allowed_fields) != len(set(allowed_fields))
    ):
        raise ValueError(
            f"Invalid allowed_fields definition for '{context}'."
        )

    unknown = set(data) - set(allowed_fields)

    if unknown:
        raise ValueError(
            f"Unknown fields in '{context}': {sorted(unknown)}"
        )


def validate_model(
    model: dict,
    index: int,
    rules: dict,
) -> None:
    """Validate a single OpenRouter model."""
    if not isinstance(model, dict):
        raise ValueError(
            f"Model entry {index} must be a mapping."
        )

    model_id = model.get("id")
    label = model.get("label")

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(
            f"Model entry {index} must have a non-empty string ID."
        )

    if not isinstance(label, str) or not label.strip():
        raise ValueError(
            f"Model '{model_id}' must have a non-empty string label."
        )

    validate_allowed_fields(
        model, rules["model"], f"model '{model_id}'"
    )

    if "provider" in model:
        provider = model["provider"]

        if not isinstance(provider, dict):
            raise ValueError(
                f"Provider for '{model_id}' must be a mapping."
            )

        validate_allowed_fields(
            provider, rules["provider"], f"provider of '{model_id}'"
        )

        only = provider.get("only")

        if (
            not isinstance(only, list)
            or not only
            or any(
                not isinstance(item, str) or not item.strip()
                for item in only
            )
        ):
            raise ValueError(
                f"Provider.only for '{model_id}' must be "
                "a non-empty list of strings."
            )

        if not isinstance(provider.get("allow_fallbacks"), bool):
            raise ValueError(
                f"Provider.allow_fallbacks for '{model_id}' "
                "must be true or false."
            )

    if "reasoning" in model:
        reasoning = model["reasoning"]

        if not isinstance(reasoning, dict):
            raise ValueError(
                f"Reasoning for '{model_id}' must be a mapping."
            )

        validate_allowed_fields(
            reasoning, rules["reasoning"], f"reasoning of '{model_id}'"
        )

        if not isinstance(reasoning.get("enabled"), bool):
            raise ValueError(
                f"Reasoning.enabled for '{model_id}' "
                "must be true or false."
            )

    if "max_tokens" in model:
        max_tokens = model["max_tokens"]

        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise ValueError(
                f"Max_tokens for '{model_id}' "
                "must be a positive integer."
            )


def get_openrouter_models() -> list[dict]:
    """Return all validated OpenRouter models."""
    config = load_model_config()

    openrouter = config.get("openrouter")

    if not isinstance(openrouter, dict):
        raise ValueError(
            "Missing or invalid 'openrouter' section."
        )

    rules = openrouter.get("allowed_fields")

    if not isinstance(rules, dict):
        raise ValueError(
            "Missing or invalid 'openrouter.allowed_fields' section."
        )

    for section in ("model", "provider", "reasoning"):
        if section not in rules:
            raise ValueError(
                f"Missing allowed_fields section: '{section}'."
            )

    models = openrouter.get("models")

    if not isinstance(models, list) or not models:
        raise ValueError(
            "'openrouter.models' must be a non-empty list."
        )

    seen_ids = set()

    for index, model in enumerate(models, start=1):
        validate_model(model, index, rules)

        model_id = model["id"]

        if model_id in seen_ids:
            raise ValueError(
                f"Duplicate model ID: {model_id}"
            )

        seen_ids.add(model_id)

    return models


def get_openrouter_model_config(model_id: str) -> dict:
    """Return the configuration of one OpenRouter model."""
    for model in get_openrouter_models():
        if model["id"] == model_id:
            return model.copy()

    raise ValueError(
        f"Unknown OpenRouter model: {model_id}"
    )
