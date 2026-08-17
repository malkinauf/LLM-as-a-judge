from evaluation_pipeline.prompts.registry import CRITERIA
from evaluation_pipeline.prompts.second_level import SECOND_LEVEL
from evaluation_pipeline.prompts.templates import BASELINE_TEMPLATE, SECOND_LEVEL_TEMPLATE


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
    first_level_prompt: str,
    first_level_response: str,
    positive_label: str,
    negative_label: str,
) -> str:
    """
    Build a complete second-level judge prompt.
    """

    if detail not in SECOND_LEVEL:
        raise ValueError(
            f"Unknown second-level detail: '{detail}'. "
            f"Expected one of: {list(SECOND_LEVEL)}"
        )

    config = SECOND_LEVEL[detail]

    return SECOND_LEVEL_TEMPLATE.format(
        instruction=config["instruction"],
        review_focus=config["review_focus"],
        first_level_prompt=first_level_prompt,
        first_level_response=first_level_response,
        first_level_positive_label=positive_label,
        first_level_negative_label=negative_label,
    )