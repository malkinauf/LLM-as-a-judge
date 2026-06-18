from pathlib import Path

PROMPT_TYPE_CONFIG = {
    "baseline": {
        "template": "baseline",
        "fields": [
            "question",
            "model_response",
        ],
    },
    "second_level": {
        "template": "second_level",
        "fields": [
            "question",
            "model_response",
            "first_judge_verdict",
            "first_judge_explanation",
        ],
    },
    "prediction": {
        "template": "prediction",
        "fields": [
            "question",
            "model_response",
        ],
    },
    "dynamic": {
        "template": "dynamic",
        "fields": [
            "question",
            "model_response",
            "prediction_response",
        ],
    },
}


def load_prompt_template(prompt_file: str | Path) -> str:
    """
    Load a prompt template from a text file.

    Args:
        prompt_file:
            Path to the prompt template file.

    Returns:
        Template content as a string.

    Raises:
        FileNotFoundError:
            If the template file does not exist.
    """

    prompt_path = Path(prompt_file)

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}"
        )

    return prompt_path.read_text(encoding="utf-8")


def build_prompt(prompt_template: str, **kwargs) -> str:
    """
    Populate a prompt template with input values.

    Args:
        prompt_template:
            Template containing format placeholders.
        **kwargs:
            Values used to replace placeholders.

    Returns:
        Formatted prompt string.

    Raises:
        ValueError:
            If a required placeholder value is missing.
    """

    try:
        return prompt_template.format(**kwargs)

    except KeyError as e:
        missing_variable = e.args[0]
        raise ValueError(
            f"Missing variable for prompt template: '{missing_variable}'"
        ) from e


def validate_prompt_inputs(
    prompt_type: str,
    templates: dict[str, str],
    data: dict[str, object],
) -> dict[str, object]:
    """
    Validate inputs required for prompt construction.

    Ensures that the requested prompt type exists,
    the corresponding template is available,
    and all required input fields are present.

    Args:
        prompt_type:
            Evaluation strategy identifier.
        templates:
            Dictionary of available prompt templates.
        data:
            Input data used to populate the prompt.

    Returns:
        Configuration dictionary for the selected
        prompt type.

    Raises:
        ValueError:
            If validation fails.
    """

    if prompt_type not in PROMPT_TYPE_CONFIG:
        raise ValueError(
            f"Unknown prompt_type: '{prompt_type}'. "
            f"Expected one of: {list(PROMPT_TYPE_CONFIG.keys())}"
        )

    config = PROMPT_TYPE_CONFIG[prompt_type]
    template_name = config["template"]
    required_fields = config["fields"]

    if template_name not in templates:
        raise ValueError(
            f"Missing template '{template_name}' "
            f"for prompt type '{prompt_type}'"
        )

    missing_fields = [
        field for field in required_fields
        if field not in data
    ]

    if missing_fields:
        raise ValueError(
            f"Prompt type '{prompt_type}' requires fields "
            f"{required_fields}, but missing {missing_fields}."
        )

    return config


def build_experiment_prompt(
    prompt_type: str,
    templates: dict[str, str],
    data: dict[str, object],
) -> str:
    """
    Build a prompt for a specific evaluation strategy.

    The function validates the selected prompt type,
    checks that all required templates and input fields
    are available, and returns a formatted prompt ready
    to be sent to an LLM.

    Args:
        prompt_type:
            Evaluation strategy to use
            (e.g. baseline, second_level, prediction, dynamic).
        templates:
            Dictionary containing loaded prompt templates.
        data:
            Input values required to populate the selected
            prompt template.

    Returns:
        Fully formatted prompt string.

    Raises:
        ValueError:
            If the selected prompt type cannot be built
            due to missing templates or required fields.
    """

    config = validate_prompt_inputs(
        prompt_type=prompt_type,
        templates=templates,
        data=data,
    )

    template_name = config["template"]
    required_fields = config["fields"]

    prompt_variables = {
        field: data[field]
        for field in required_fields
    }

    return build_prompt(
        templates[template_name],
        **prompt_variables,
    )
