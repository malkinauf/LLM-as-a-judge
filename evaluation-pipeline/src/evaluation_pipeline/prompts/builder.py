from evaluation_pipeline.prompts.registry import CRITERIA, SECOND_LEVEL_REVIEW
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
    criterion: str,
    first_level_prompt: str,
    first_level_response: str,
) -> str:
    """
    Build the second-level meta-judge prompt.

    Args:
        detail: Second-level review detail level.
        criterion: First-level evaluation criterion.
        first_level_prompt: Complete prompt given to the first-level judge.
        first_level_response: Complete response produced by the first-level judge.

    Returns:
        Fully formatted second-level judge prompt.

    Raises:
        ValueError: If the detail level or criterion is unknown.
    """

    if detail not in SECOND_LEVEL_REVIEW["descriptions"]:
        raise ValueError(
            f"Unknown second-level detail: '{detail}'."
        )

    if criterion not in CRITERIA:
        raise ValueError(
            f"Unknown criterion: '{criterion}'."
        )

    review_description = SECOND_LEVEL_REVIEW["descriptions"][detail]

    criterion_config = CRITERIA[criterion]

    positive_label = criterion_config["labels"]["positive"]
    negative_label = criterion_config["labels"]["negative"]

    return SECOND_LEVEL_TEMPLATE.format(
        review_description=review_description,
        positive_label=positive_label,
        negative_label=negative_label,
        first_level_prompt=first_level_prompt,
        first_level_response=first_level_response,
    )