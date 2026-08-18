import logging
from typing import Any

from tqdm import tqdm

from evaluation_pipeline.prompts.builder import build_baseline_prompt
from evaluation_pipeline.prompts_utils import build_experiment_prompt
from evaluation_pipeline.judge import get_raw_model_response, judge_response
from evaluation_pipeline.prompts.builder import (
    build_baseline_prompt,
    build_second_level_prompt,
)
from evaluation_pipeline.prompts.registry import CRITERIA

logger = logging.getLogger(__name__)


VALID_METHODS = (
    "baseline",
    "second_level",
    "dynamic",
)

VALID_JUDGE_LABELS = {
    "truthful",
    "not_truthful",
    "safe",
    "not_safe",
    "attack_failed",
    "attack_success",
    "correct",
    "not_correct",

}

VALID_SECOND_LEVEL_VERDICTS = {
    "correct",
    "not_correct",
}

PARSING_ERROR = "parsing_error"
RUNTIME_ERROR = "runtime_error"

NON_SECOND_LEVEL_METHODS = {"baseline", "dynamic"}


def build_base_result(
    example: dict[str, Any],
    run_id: str,
    model: str,
    method: str,
    dataset_file: str,
) -> dict[str, Any]:
    """
    Create the default result record for one evaluated example.

    The record contains common metadata and empty fields for
    baseline, second-level, and dynamic prompting outputs.

    Args:
        example: Dataset example being evaluated.
        run_id: Identifier of the current experiment run.
        model: Judge model name.
        method: Evaluation method.
        dataset_file: Source dataset file name.

    Returns:
        Initialized result dictionary.
    """

    return {
        "id": example["id"],
        "question": example["question"],
        "model_response": example["model_response"],
        "true_label": example["y_true"],

        "model": model,
        "method": method,
        "run_id": run_id,
        "dataset_file": dataset_file,

        "prediction_prompt": None,
        "prediction_raw_output": None,

        "first_prompt": None,
        "first_raw_output": None,
        "first_level_label": None,
        "first_level_explanation": None,

        "second_level_prompt": None,
        "second_level_raw_output": None,
        "second_level_verdict": None,
        "second_level_explanation": None,

        "predicted_label": None,
        "error": None,
    }


def run_first_level_judge(
    example: dict[str, Any],
    model: str,
    method: str,
    templates: dict[str, str],
    result: dict[str, Any],
    baseline_criterion: str,
    baseline_criterion_detail: str,
) -> None:
    """
    Run the first-level judge for one dataset example.

    Baseline and second-level methods use the generated baseline
    prompt. Dynamic prompting uses the prediction and dynamic
    prompt templates.

    Args:
        example: Dataset example being evaluated.
        model: Judge model name.
        method: Evaluation method.
        templates: Prompt templates required by non-baseline methods.
        result: Result dictionary updated in place.
        baseline_criterion: Criterion used for the baseline prompt.
        baseline_criterion_detail: Detail level of the baseline criterion.
    """

    if method == "dynamic":
        prediction_prompt = build_experiment_prompt(
            prompt_type="prediction",
            templates=templates,
            data=example,
        )

        prediction_response = get_raw_model_response(
            prompt=prediction_prompt,
            model=model,
        )

        first_prompt = build_experiment_prompt(
            prompt_type="dynamic",
            templates=templates,
            data={
                **example,
                "prediction_response": prediction_response,
            },
        )

        result["prediction_prompt"] = prediction_prompt
        result["prediction_raw_output"] = prediction_response

    else:
        first_prompt = build_baseline_prompt(
            criterion=baseline_criterion,
            criterion_detail=baseline_criterion_detail,
            question=example["question"],
            model_response=example["model_response"],
        )

    first_judge_result = judge_response(
        prompt=first_prompt,
        model=model,
    )

    result["first_prompt"] = first_prompt
    result["first_raw_output"] = first_judge_result.get("raw_output")
    result["first_level_label"] = first_judge_result.get(
        "predicted_label"
    )
    result["first_level_explanation"] = first_judge_result.get(
        "explanation"
    )


def apply_baseline_decision(
    result: dict[str, Any],
) -> None:
    """
    Set the final predicted label from the first-level judge output.

    Args:
        result: Result dictionary updated in place.
    """

    first_level_label = result["first_level_label"]

    if first_level_label in VALID_JUDGE_LABELS:
        result["predicted_label"] = first_level_label
    else:
        result["predicted_label"] = PARSING_ERROR


def apply_second_level_decision(
    model: str,
    result: dict[str, Any],
    second_level_detail: str,
    baseline_criterion: str,
) -> None:
    """
    Run the second-level judge and update the final prediction.

    The second-level judge reviews whether the first-level judge
    correctly applied the original evaluation criteria and decision
    rules. If the first-level judgment is correct, its verdict is kept.
    Otherwise, the corrected verdict is used as the final prediction.

    Args:
        model: Judge model name.
        result: Result dictionary updated in place.
        second_level_detail: Detail level of the second-level review.
        baseline_criterion: Criterion used by the first-level judge.
    """

    first_level_label = result["first_level_label"]

    if first_level_label not in VALID_JUDGE_LABELS:
        result["predicted_label"] = PARSING_ERROR
        return

    second_level_prompt = build_second_level_prompt(
        detail=second_level_detail,
        criterion=baseline_criterion,
        first_level_prompt=result["first_prompt"],
        first_level_response=result["first_raw_output"],
    )

    second_result = judge_response(
        prompt=second_level_prompt,
        model=model,
    )

    second_level_verdict = second_result.get("predicted_label")

    result["second_level_prompt"] = second_level_prompt
    result["second_level_raw_output"] = second_result.get("raw_output")
    result["second_level_verdict"] = second_level_verdict
    result["second_level_explanation"] = (
        second_result.get("corrected_explanation")
        or second_result.get("explanation")
    )

    if second_level_verdict not in VALID_SECOND_LEVEL_VERDICTS:
        result["predicted_label"] = PARSING_ERROR
        return

    if second_level_verdict == "correct":
        result["predicted_label"] = first_level_label
        return

    corrected_verdict = second_result.get("corrected_verdict")

    if corrected_verdict in VALID_JUDGE_LABELS:
        result["predicted_label"] = corrected_verdict
    else:
        result["predicted_label"] = PARSING_ERROR


def run_judge_experiment(
    dataset: list[dict[str, Any]],
    run_id: str,
    model: str,
    method: str,
    templates: dict[str, str],
    dataset_file: str,
    baseline_criterion: str,
    baseline_criterion_detail: str,
    second_level_detail: str,
) -> list[dict[str, Any]]:
    """
    Run a judge experiment over a prepared dataset.

    The selected evaluation method is applied to each dataset example.
    Baseline and second-level methods use the configured first-level
    criterion. The second-level method additionally reviews the
    first-level judgment using the selected review detail.

    Args:
        dataset: Prepared dataset examples to evaluate.
        run_id: Identifier of the current experiment run.
        model: Judge model name.
        method: Evaluation method to run.
        templates: Prompt templates required by methods that still use
            template files, such as dynamic prompting.
        dataset_file: Source dataset file name.
        baseline_criterion: Criterion used by the first-level judge.
        baseline_criterion_detail: Detail level of the first-level criterion.
        second_level_detail: Detail level of the second-level review.

    Returns:
        List of result dictionaries, one for each evaluated example.

    Raises:
        ValueError: If the selected evaluation method is unsupported.
    """

    if method not in VALID_METHODS:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Expected one of: {VALID_METHODS}"
        )

    results: list[dict[str, Any]] = []

    if not run_id:
        logger.info(
            "Experiment skipped because run_id is empty."
        )
        return results

    for example in tqdm(
        dataset,
        desc=f"Running {method} judge experiment",
    ):
        result = build_base_result(
            example=example,
            run_id=run_id,
            model=model,
            method=method,
            dataset_file=dataset_file,
        )

        try:
            run_first_level_judge(
                example=example,
                model=model,
                method=method,
                templates=templates,
                result=result,
                baseline_criterion=baseline_criterion,
                baseline_criterion_detail=baseline_criterion_detail,
            )

        except Exception as e:
            logger.exception(
                f"First-level judge failed for example "
                f"{example.get('id')}: {e}"
            )

            result["predicted_label"] = RUNTIME_ERROR
            result["error"] = str(e)
            results.append(result)
            continue

        if method in NON_SECOND_LEVEL_METHODS:
            apply_baseline_decision(result)
            results.append(result)
            continue

        if method == "second_level":
            try:
                apply_second_level_decision(
                    model=model,
                    result=result,
                    second_level_detail=second_level_detail,
                    baseline_criterion=baseline_criterion,
                )

            except Exception as e:
                logger.exception(
                    f"Second-level judge failed for example "
                    f"{example.get('id')}: {e}"
                )

                result["predicted_label"] = RUNTIME_ERROR
                result["error"] = str(e)

            results.append(result)

    logger.info(
        f"Finished. Collected {len(results)} results."
    )

    return results
