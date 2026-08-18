from evaluation_pipeline.prompts.registry import (
    CRITERIA,
    SECOND_LEVEL_REVIEW,
)
from evaluation_pipeline.prompts.templates import (
    BASELINE_TEMPLATE,
    SECOND_LEVEL_STRUCTURED_TEMPLATE,
    SECOND_LEVEL_TEMPLATE,
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
    detail: str,
    input_mode: str,
    criterion: str,
    first_level_criterion_detail: str,
    question: str,
    model_response: str,
    first_level_prompt: str,
    first_level_response: str,
) -> str:
    """
    Build a complete second-level judge prompt.

    Args:
        detail: Detail level of the second-level review.
        input_mode: Input representation for the second-level judge:
            "full" or "structured".
        criterion: Criterion used by the first-level judge.
        first_level_criterion_detail: Detail level of the criterion
            used by the first-level judge.
        question: Original user question.
        model_response: Original model response being evaluated.
        first_level_prompt: Complete prompt used by the first-level judge.
        first_level_response: Complete response produced by the
            first-level judge.

    Returns:
        Complete second-level judge prompt.

    Raises:
        ValueError: If the detail level, input mode, criterion,
            or first-level criterion detail is unsupported.
    """

    review_descriptions = SECOND_LEVEL_REVIEW["descriptions"]

    if detail not in review_descriptions:
        raise ValueError(
            f"Unknown second-level detail: '{detail}'. "
            f"Expected one of: {list(review_descriptions)}"
        )

    if input_mode not in {"full", "structured"}:
        raise ValueError(
            f"Unknown second-level input mode: '{input_mode}'. "
            "Expected one of: ['full', 'structured']"
        )

    if criterion not in CRITERIA:
        raise ValueError(
            f"Unknown criterion: '{criterion}'. "
            f"Expected one of: {list(CRITERIA)}"
        )

    criterion_config = CRITERIA[criterion]
    criterion_descriptions = criterion_config["descriptions"]

    if first_level_criterion_detail not in criterion_descriptions:
        raise ValueError(
            f"Unknown criterion detail: "
            f"'{first_level_criterion_detail}'. "
            f"Expected one of: {list(criterion_descriptions)}"
        )

    review_description = review_descriptions[detail]
    positive_label = criterion_config["labels"]["positive"]
    negative_label = criterion_config["labels"]["negative"]

    if input_mode == "full":
        return SECOND_LEVEL_TEMPLATE.format(
            review_description=review_description,
            positive_label=positive_label,
            negative_label=negative_label,
            first_level_prompt=first_level_prompt,
            first_level_response=first_level_response,
        )

    return SECOND_LEVEL_STRUCTURED_TEMPLATE.format(
        review_description=review_description,
        criterion_description=criterion_descriptions[
            first_level_criterion_detail
        ],
        first_level_decision_rules=criterion_config["decision_rules"],
        question=question,
        model_response=model_response,
        first_level_response=first_level_response,
        positive_label=positive_label,
        negative_label=negative_label,
    )