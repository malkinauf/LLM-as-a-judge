from evaluation_pipeline.prompts.registry import CRITERIA

from evaluation_pipeline.prompts.templates import (
    BASELINE_TEMPLATE,
    SECOND_LEVEL_TEMPLATE,
    DYNAMIC_PREDICTION_TEMPLATE,
    DYNAMIC_TEMPLATE,
)


def build_baseline_prompt(
    criterion: str,
    criterion_detail: str,
    question: str,
    model_response: str,
) -> str:
    """
    Build a complete baseline judge prompt.

    Args:
        criterion: Evaluation criterion, e.g. "truthfulness" or "safety".
        criterion_detail: Detail level defined for the selected criterion.
        question: User question to evaluate.
        model_response: Model response to evaluate.

    Returns:
        Complete baseline judge prompt.

    Raises:
        ValueError: If criterion or criterion_detail is unsupported.
    """

    if criterion not in CRITERIA:
        raise ValueError(
            f"Unknown criterion: '{criterion}'. "
            f"Expected one of: {list(CRITERIA)}"
        )

    criterion_config = CRITERIA[criterion]
    descriptions = criterion_config["descriptions"]

    if criterion_detail not in descriptions:
        raise ValueError(
            f"Unknown criterion detail: '{criterion_detail}'. "
            f"Expected one of: {list(descriptions)}"
        )

    return BASELINE_TEMPLATE.format(
        criterion_description=descriptions[criterion_detail],
        decision_rules=criterion_config["decision_rules"],
        positive_label=criterion_config["labels"]["positive"],
        negative_label=criterion_config["labels"]["negative"],
        question=question,
        model_response=model_response,
    )


def build_second_level_prompt(
    criterion: str,
    first_level_prompt: str,
    first_level_response: str,
) -> str:
    """
    Build the fixed second-level judge prompt.

    The second-level judge receives the complete first-level
    judge task and the complete response produced by the
    first-level judge.

    Args:
        criterion: Criterion used by the first-level judge.
        first_level_prompt: Complete prompt given to the
            first-level judge.
        first_level_response: Complete response produced by the
            first-level judge.

    Returns:
        Complete second-level judge prompt.

    Raises:
        ValueError: If the criterion is unsupported.
    """

    if criterion not in CRITERIA:
        raise ValueError(
            f"Unknown criterion: '{criterion}'. "
            f"Expected one of: {list(CRITERIA)}"
        )

    criterion_config = CRITERIA[criterion]

    return SECOND_LEVEL_TEMPLATE.format(
        first_level_prompt=first_level_prompt,
        first_level_response=first_level_response,
        positive_label=criterion_config["labels"]["positive"],
        negative_label=criterion_config["labels"]["negative"],
    )


def build_prediction_prompt(
    criterion: str,
    prediction_variant: str,
    question: str,
    model_response: str,
) -> str:
    """
    Build the preliminary prediction prompt used by
    the dynamic prompting method.

    Args:
        criterion: Evaluation criterion, e.g. "truthfulness" or "safety".
        prediction_variant: Prediction instruction variant defined
            for the selected criterion.
        question: User question to evaluate.
        model_response: Model response to evaluate.

    Returns:
        Complete prediction prompt.

    Raises:
        ValueError: If the criterion or prediction variant is unsupported.
    """

    if criterion not in CRITERIA:
        raise ValueError(
            f"Unknown criterion: '{criterion}'. "
            f"Expected one of: {list(CRITERIA)}"
        )

    criterion_config = CRITERIA[criterion]

    dynamic_config = criterion_config.get("dynamic", {})
    instructions = dynamic_config.get(
        "prediction_instructions",
        {},
    )

    if prediction_variant not in instructions:
        raise ValueError(
            f"Unknown prediction variant: '{prediction_variant}'. "
            f"Expected one of: {list(instructions)}"
        )

    return DYNAMIC_PREDICTION_TEMPLATE.format(
        prediction_instruction=instructions[prediction_variant],
        question=question,
        model_response=model_response,
    )


def build_dynamic_prompt(
    baseline_prompt: str,
    prediction_response: str,
) -> str:
    """
    Build the final dynamic judge prompt.

    The dynamic prompt contains the complete baseline judge
    prompt and the preliminary prediction generated in the
    first step.

    Args:
        baseline_prompt: Complete baseline judge prompt.
        prediction_response: Preliminary analysis generated
            by the prediction step.

    Returns:
        Complete dynamic judge prompt.
    """

    return DYNAMIC_TEMPLATE.format(
        baseline_prompt=baseline_prompt,
        prediction_response=prediction_response,
    )
