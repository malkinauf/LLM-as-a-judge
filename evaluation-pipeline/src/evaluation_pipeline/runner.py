import logging
from typing import Any

from tqdm import tqdm

from evaluation_pipeline.judge import judge_response
from evaluation_pipeline.prompts import build_experiment_prompt


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
}

VALID_SECOND_LEVEL_VERDICTS = {
    "correct",
    "not_correct",
}


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
) -> None:
    """
    Run the first-level judge for baseline, second-level, or dynamic methods.

    For baseline and second-level methods, the first-level judge uses
    the baseline prompt. For the dynamic method, the function first
    generates a prediction response and then injects it into the
    dynamic prompt.

    Args:
        example: Dataset example being evaluated.
        model: Judge model name.
        method: Evaluation method.
        templates: Loaded prompt templates.
        result: Result dictionary updated with prompts and outputs.

    Raises:
        Exception: If the judge call fails.
    """

    if method == "dynamic":
        prediction_prompt = build_experiment_prompt(
            prompt_type="prediction",
            templates=templates,
            data=example,
        )

        prediction_result = judge_response(
            prediction_prompt,
            model,
        )

        prediction_response = prediction_result.get("raw_output")

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
        first_prompt = build_experiment_prompt(
            prompt_type="baseline",
            templates=templates,
            data=example,
        )

    first_judge_result = judge_response(
        first_prompt,
        model,
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
        result["predicted_label"] = "parsing_error"


def apply_second_level_decision(
    example: dict[str, Any],
    model: str,
    templates: dict[str, str],
    result: dict[str, Any],
) -> None:
    """
    Run the second-level judge and update the final predicted label.

    The second-level judge checks whether the first-level verdict is
    correct. If it is not correct, the corrected verdict is used as
    the final prediction.

    Args:
        example: Dataset example being evaluated.
        model: Judge model name.
        templates: Loaded prompt templates.
        result: Result dictionary updated in place.
    """

    first_level_label = result["first_level_label"]

    if first_level_label not in VALID_JUDGE_LABELS:
        result["predicted_label"] = "parsing_error"
        return

    second_level_prompt = build_experiment_prompt(
        prompt_type="second_level",
        templates=templates,
        data={
            "question": example["question"],
            "model_response": example["model_response"],
            "first_judge_verdict": first_level_label,
            "first_judge_explanation": result[
                "first_level_explanation"
            ],
        },
    )

    second_result = judge_response(
        second_level_prompt,
        model,
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
        result["predicted_label"] = "parsing_error"
        return

    if second_level_verdict == "correct":
        result["predicted_label"] = first_level_label
        return

    corrected_verdict = second_result.get("corrected_verdict")

    if corrected_verdict in VALID_JUDGE_LABELS:
        result["predicted_label"] = corrected_verdict
    else:
        result["predicted_label"] = "parsing_error"


def run_judge_experiment(
    dataset: list[dict[str, Any]],
    run_id: str,
    model: str,
    method: str,
    templates: dict[str, str],
    dataset_file: str,
) -> list[dict[str, Any]]:
    """
    Run a judge experiment over a dataset.

    Supported methods are baseline, second_level, and dynamic.
    The dynamic method follows a two-step prompting scheme:
    prediction prompt first, then dynamic judge prompt with the
    prediction response inserted as auxiliary guidance.

    Args:
        dataset: List of prepared dataset examples.
        run_id: Identifier of the current experiment run.
        model: Judge model name.
        method: Evaluation method to run.
        templates: Loaded prompt templates.
        dataset_file: Source dataset file name.

    Returns:
        List of result dictionaries, one per evaluated example.

    Raises:
        ValueError: If the selected method is not supported.
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
            )

        except Exception as e:
            logger.exception(
                f"First-level judge failed for example "
                f"{example.get('id')}: {e}"
            )

            result["predicted_label"] = "runtime_error"
            result["error"] = str(e)
            results.append(result)
            continue

        if method in {"baseline", "dynamic"}:
            apply_baseline_decision(result)
            results.append(result)
            continue

        if method == "second_level":
            try:
                apply_second_level_decision(
                    example=example,
                    model=model,
                    templates=templates,
                    result=result,
                )

            except Exception as e:
                logger.exception(
                    f"Second-level judge failed for example "
                    f"{example.get('id')}: {e}"
                )

                result["predicted_label"] = "runtime_error"
                result["error"] = str(e)

            results.append(result)

    logger.info(
        f"Finished. Collected {len(results)} results."
    )

    return results
