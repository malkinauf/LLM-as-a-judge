from pathlib import Path

def load_prompt_template(prompt_file: str) -> str:
    prompt_path = Path(prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")

def build_prompt(prompt_template: str, **kwargs) -> str:
    try:
        return prompt_template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing variable for prompt: {e}")

def build_experiment_prompt(
    prompt_type: str,
    templates: dict,
    data: dict
) -> str:
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
            #first_judge_raw_output=data["first_judge_raw_output"],
        )

    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")