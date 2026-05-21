from pathlib import Path


def load_prompt_template(prompt_file: str) -> str:
    prompt_path = Path(prompt_file)

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}"
        )

    return prompt_path.read_text(encoding="utf-8")


def build_prompt(prompt_template: str, **kwargs) -> str:
    try:
        return prompt_template.format(**kwargs)

    except KeyError as e:
        raise ValueError(
            f"Missing variable for prompt: {e}"
        ) from e


def build_experiment_prompt(
    prompt_type: str,
    templates: dict,
    data: dict,
) -> str:
    required_templates = {
        "baseline": ["baseline"],
        "second_level": ["second_level"],
    }

    required_data_fields = {
        "baseline": [
            "question",
            "model_response",
        ],
        "second_level": [
            "question",
            "model_response",
            "first_judge_verdict",
            "first_judge_explanation",
        ],
    }

    if prompt_type not in required_templates:
        raise ValueError(
            f"Unknown prompt_type: {prompt_type}. "
            f"Expected one of: {list(required_templates.keys())}"
        )

    for template_name in required_templates[prompt_type]:
        if template_name not in templates:
            raise ValueError(
                f"Missing template '{template_name}' "
                f"for prompt type '{prompt_type}'"
            )

    for field_name in required_data_fields[prompt_type]:
        if field_name not in data:
            raise ValueError(
                f"Missing data field '{field_name}' "
                f"for prompt type '{prompt_type}'"
            )

    if prompt_type == "baseline":
        return build_prompt(
            templates["baseline"],
            question=data["question"],
            model_response=data["model_response"],
        )

    elif prompt_type == "second_level":
        return build_prompt(
            templates["second_level"],
            question=data["question"],
            model_response=data["model_response"],
            first_judge_verdict=data["first_judge_verdict"],
            first_judge_explanation=data["first_judge_explanation"],
        )
    