from evaluation_pipeline.prompts.registry import CRITERIA

from evaluation_pipeline.prompts.templates import (
    BASELINE_BODY_TEMPLATE,
    OUTPUT_TEMPLATE,
    SECOND_LEVEL_TEMPLATE,
    DYNAMIC_PREDICTION_TEMPLATE,
    DYNAMIC_HINT_TEMPLATE,
)


def build_baseline_body(
    criterion: str,
    criterion_detail: str,
    question: str,
    model_response: str,
) -> str:
    """
    Build the shared baseline prompt body.

    Args:
        criterion: Evaluation criterion, e.g. "truthfulness" or "safety".
        criterion_detail: Detail level defined for the selected criterion.
        question: User question to evaluate.
        model_response: Model response to evaluate.

    Returns:
        Baseline prompt body without the final output section.

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

    return BASELINE_BODY_TEMPLATE.format(
        criterion_description=descriptions[criterion_detail],
        decision_rules=criterion_config["decision_rules"],
        positive_label=criterion_config["labels"]["positive"],
        negative_label=criterion_config["labels"]["negative"],
        question=question,
        model_response=model_response,
    )


def build_baseline_prompt(
    criterion: str,
    criterion_detail: str,
    question: str,
    model_response: str,
) -> str:
    """
    Build the complete baseline judge prompt.

    Args:
        criterion: Evaluation criterion, e.g. "truthfulness" or "safety".
        criterion_detail: Detail level defined for the selected criterion.
        question: User question to evaluate.
        model_response: Model response to evaluate.

    Returns:
        Complete baseline judge prompt.
    """

    baseline_body = build_baseline_body(
        criterion=criterion,
        criterion_detail=criterion_detail,
        question=question,
        model_response=model_response,
    )

    return (
        f"{baseline_body}\n\n"
        f"{OUTPUT_TEMPLATE}"
    )


def build_second_level_prompt(
    criterion: str,
    first_level_prompt: str,
    first_level_response: str,
) -> str:
    """
    Build the fixed second-level judge prompt.

    Args:
        criterion: Criterion used by the first-level judge.
        first_level_prompt: Complete prompt given to the first-level judge.
        first_level_response: Complete response produced by the first-level judge.

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
        criterion: Evaluation criterion.
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
    criterion: str,
    criterion_detail: str,
    question: str,
    model_response: str,
    prediction_response: str,
) -> str:
    """
    Build the final dynamic judge prompt.

    The dynamic prompt uses the same baseline prompt body,
    adds the preliminary analysis as a hint, and places the
    final output section at the end.

    Args:
        criterion: Evaluation criterion.
        criterion_detail: Detail level defined for the selected criterion.
        question: User question to evaluate.
        model_response: Model response to evaluate.
        prediction_response: Preliminary analysis generated
            by the prediction step.

    Returns:
        Complete dynamic judge prompt.
    """

    baseline_body = build_baseline_body(
        criterion=criterion,
        criterion_detail=criterion_detail,
        question=question,
        model_response=model_response,
    )

    dynamic_hint = DYNAMIC_HINT_TEMPLATE.format(
        prediction_response=prediction_response,
    )

    return (
        f"{baseline_body}\n\n"
        f"{dynamic_hint}\n\n"
        f"{OUTPUT_TEMPLATE}"
    )