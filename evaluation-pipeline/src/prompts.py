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


def build_experiment_prompt(prompt_type: str, templates: dict, data: dict) -> str:
    if prompt_type == "baseline":
        return build_prompt(
            templates["baseline"],
            question=data["question"],
            answer=data["answer"],
        )

    if prompt_type == "second_level":
        return build_prompt(
            templates["second_level"],
            judge_task=data["judge_task"],
            judge_output=data["judge_output"],
        )

    if prompt_type == "hint":
        return build_prompt(
            templates["hint"],
            question=data["question"],
            answer=data["answer"],
        )

    if prompt_type == "dynamic":
        return build_prompt(
            templates["dynamic"],
            question=data["question"],
            answer=data["answer"],
            hint=data["hint"],
        )

    raise ValueError(f"Unknown prompt type: {prompt_type}")
