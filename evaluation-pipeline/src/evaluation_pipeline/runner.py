import logging
from typing import Any

from tqdm import tqdm

from evaluation_pipeline.judge import (
    ModelAPIError,
    get_raw_model_response,
    judge_response,
)
from evaluation_pipeline.prompts.builder import (
    build_baseline_prompt,
    build_dynamic_prompt,
    build_prediction_prompt,
    build_second_level_prompt,
)
from evaluation_pipeline.prompts.registry import CRITERIA 

logger = logging.getLogger(__name__)


USAGE_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "latency_seconds")
USAGE_STAGES = ("prediction", "first_level", "second_level")
OPENROUTER_METADATA_FIELDS = (
    "openrouter_generation_id",
    "openrouter_response_model",
)

VALID_METHODS = (
    "baseline",
    "second_level",
    "dynamic",
)

VALID_SECOND_LEVEL_VERDICTS = {
    "correct",
    "not_correct",
}

PARSING_ERROR = "parsing_error"
RUNTIME_ERROR = "runtime_error"
API_ERROR = "api_error"
INVALID_LABEL = "invalid_label"


NON_SECOND_LEVEL_METHODS = {
    "baseline",
    "dynamic",
}


def record_usage(
    result: dict[str, Any],
    stage: str,
    response: dict[str, Any],
) -> None:
    """Record token usage, latency, and API metadata for a completed call.

    Update stage-specific fields and recompute aggregate usage across
    completed stages. Missing usage values remain unknown.

    Args:
        result: Example result dictionary to update in place.
        stage: Completed model-call stage.
        response: Model response containing usage and metadata fields.
    """
    for field in USAGE_FIELDS:
        result[f"{stage}_{field}"] = response.get(field)

    for field in OPENROUTER_METADATA_FIELDS:
        result[f"{stage}_{field}"] = response.get(field)

    result[f"{stage}_completed"] = True

    completed = [
        name for name in USAGE_STAGES
        if result[f"{name}_completed"]
    ]

    for field in USAGE_FIELDS:
        values = [
            result[f"{name}_{field}"]
            for name in completed
        ]
        result[field] = (
            sum(values)
            if all(value is not None for value in values)
            else None
        )

def record_error(
    result: dict[str, Any],
    error_type: str,
    message: str,
    stage: str,
    *,
    http_status: int | None = None,
) -> None:
    """Record an unsuccessful evaluation without assigning a prediction.

    Args:
        result: Example result dictionary to update in place.
        error_type: Category of the failure.
        message: Description of the failure.
        stage: Stage at which the failure occurred.
        http_status: HTTP response status, when available.
    """
    result["predicted_label"] = None
    result["error_type"] = error_type
    result["error_message"] = message
    result["error_stage"] = stage
    result["http_status"] = http_status

def record_exception(
    result: dict[str, Any], exc: Exception, stage: str
) -> None:
    """Record an API or pipeline exception in the example result.

    Args:
        result: Example result dictionary to update in place.
        exc: Exception raised during evaluation.
        stage: Stage at which the exception occurred.
    """
    is_api_error = isinstance(exc, ModelAPIError)
    record_error(
        result,
        API_ERROR if is_api_error else RUNTIME_ERROR,
        str(exc),
        stage,
        http_status=exc.status_code if is_api_error else None,
    )

def record_parsing_error(
    result: dict[str, Any], message: str, stage: str
) -> None:
    """Record an unusable judge output as a parsing error.

    Args:
        result: Example result dictionary to update in place.
        message: Description of the invalid output.
        stage: Stage that produced the invalid output.
    """
    record_error(result, PARSING_ERROR, message, stage)

def record_invalid_label(
    result: dict[str, Any], message: str, stage: str
) -> None:
    """Record a valid JSON response with an invalid verdict."""
    record_error(result, INVALID_LABEL, message, stage)

def allowed_labels(criterion: str) -> set[str]:
    """Return the valid classification labels for a criterion.

    Args:
        criterion: Evaluation criterion registered in CRITERIA.

    Returns:
        Set containing the positive and negative labels.
    """
    config = CRITERIA[criterion]["labels"]
    return {config["positive"], config["negative"]}


def build_base_result(
    example: dict[str, Any],
    run_id: str,
    model: str,
    method: str,
    dataset_id: str,
    evaluation_criterion: str,
    baseline_criterion_detail: str,
    dynamic_prediction_variant: str,
    confidence_mode: bool = False,
    backend: str = "ollama",
) -> dict[str, Any]:
    """Initialize the result record for one dataset example.

    The record contains experiment metadata, stage-specific outputs,
    usage measurements, and error fields for all evaluation methods.

    Args:
        example: Dataset example containing its ID, question, response,
            and ground-truth label.
        run_id: Identifier of the experiment run.
        model: Judge model identifier.
        method: Evaluation method.
        dataset_id: Dataset identifier.
        evaluation_criterion: Criterion used to assess the response.
        baseline_criterion_detail: Detail level of the baseline criterion.
        dynamic_prediction_variant: Prediction variant for dynamic prompting.
        confidence_mode: Whether the first-level prompt requests confidence.
        backend: Model backend used for evaluation.

    Returns:
        Initialized result dictionary with unset outputs and error fields.
    """
    return {
        "id": example["id"],
        "question": example["question"],
        "model_response": example["model_response"],
        "true_label": example["y_true"],

        "model": model,
        "backend": backend,
        "method": method,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "evaluation_criterion": evaluation_criterion,
        "baseline_criterion_detail": baseline_criterion_detail,
        "dynamic_prediction_variant": (dynamic_prediction_variant if method == "dynamic" else None),
        "confidence_mode": confidence_mode,

        # Per-stage and per-example model usage
        **{f"{stage}_{field}": None for stage in USAGE_STAGES for field in USAGE_FIELDS},
        **{f"{stage}_completed": False for stage in USAGE_STAGES},
        **{field: None for field in USAGE_FIELDS},
        **{f"{stage}_{field}": None for stage in USAGE_STAGES for field in OPENROUTER_METADATA_FIELDS},

        # Dynamic prediction step
        "prediction_prompt": None,
        "prediction_raw_output": None,

        # First-level judge
        "first_prompt": None,
        "first_raw_output": None,
        "first_level_label": None,
        "first_level_parsing_failed": False,
        "first_level_explanation": None,
        "first_level_confidence": None,
        "first_level_confidence_invalid": False,

        # Second-level judge
        "second_level_prompt": None,
        "second_level_raw_output": None,
        "second_level_verdict": None,
        "second_level_parsing_failed": False,
        "second_level_explanation": None,
        "second_level_corrected_verdict": None,
        "second_level_corrected_explanation": None,

        # Final result
        "predicted_label": None,
        "error_message": None,
        "error_stage": None,
        "current_stage": "first_level",
        "error_type": None,
        "http_status": None,
    }

def run_first_level_judge(
    example: dict[str, Any],
    model: str,
    method: str,
    result: dict[str, Any],
    evaluation_criterion: str,
    baseline_criterion_detail: str,
    dynamic_prediction_variant: str,
    confidence_mode: bool = False,
    backend: str = "ollama",
) -> None:
    """Execute the first-level evaluation for one example.

    Baseline and second-level methods use the baseline prompt. Dynamic
    prompting first generates a preliminary analysis, then includes it
    as a hint in the first-level judge prompt. Outputs and usage are
    stored in the provided result dictionary.

    Args:
        example: Dataset example to evaluate.
        model: Judge model identifier.
        method: Evaluation method.
        result: Example result dictionary to update in place.
        evaluation_criterion: Criterion used to assess the response.
        baseline_criterion_detail: Detail level of the baseline criterion.
        dynamic_prediction_variant: Prediction variant for dynamic prompting.
        confidence_mode: Whether the first-level prompt requests confidence.
        backend: Model backend used for evaluation.
    """
    if method == "dynamic":
        result["current_stage"] = "prediction"
        # Step 1:
        # Generate the preliminary analysis.
        prediction_prompt = build_prediction_prompt(
            criterion=evaluation_criterion,
            prediction_variant=dynamic_prediction_variant,
            question=example["question"],
            model_response=example["model_response"],
        )

        result["prediction_prompt"] = prediction_prompt

        prediction_call = get_raw_model_response(
            prompt=prediction_prompt,
            model=model,
            return_usage=True,
            backend=backend,
        )
        record_usage(result, "prediction", prediction_call)
        prediction_response = prediction_call["raw_output"]
        result["prediction_raw_output"] = prediction_response

        # Step 2:
        # Build the final dynamic judge prompt from the shared
        # baseline body + dynamic hint + output section.
        result["current_stage"] = "first_level"
        first_prompt = build_dynamic_prompt(
            criterion=evaluation_criterion,
            criterion_detail=baseline_criterion_detail,
            question=example["question"],
            model_response=example["model_response"],
            prediction_response=prediction_response,
            confidence_mode=confidence_mode,
        )

    else:
        # Baseline and second-level use the normal baseline prompt.
        first_prompt = build_baseline_prompt(
            criterion=evaluation_criterion,
            criterion_detail=baseline_criterion_detail,
            question=example["question"],
            model_response=example["model_response"],
            confidence_mode=confidence_mode,
        )

    # Run the first-level classification.
    result["first_prompt"] = first_prompt
    result["current_stage"] = "first_level"
    first_judge_result = judge_response(
        prompt=first_prompt,
        model=model,
        backend=backend,
    )

    record_usage(result, "first_level", first_judge_result)

    result["first_raw_output"] = first_judge_result.get(
        "raw_output"
    )

    result["first_level_parsing_failed"] = first_judge_result.get(
        "parsing_failed", False
    )

    label = first_judge_result.get("predicted_label")

    result["first_level_label"] = (
        label
        if label in allowed_labels(evaluation_criterion)
        else None
    )

    result["first_level_explanation"] = first_judge_result.get(
        "explanation"
    )

    result["first_level_confidence"] = first_judge_result.get(
        "confidence"
    )
    result["first_level_confidence_invalid"] = first_judge_result.get(
        "confidence_invalid", False
    )


def apply_baseline_decision(
    result: dict[str, Any],
) -> None:
    """Use the first-level verdict as the final prediction."""

    first_level_label = result["first_level_label"]

    if result.get("first_level_parsing_failed", False):
        record_parsing_error(
            result,
            f"Could not parse first-level output: {result['first_raw_output']!r}",
            "first_level",
        )

    elif first_level_label in allowed_labels(
        result["evaluation_criterion"]
    ):
        result["predicted_label"] = first_level_label

    else:
        record_invalid_label(
            result,
            f"Invalid first-level verdict: {result['first_raw_output']!r}",
            "first_level",
        )

def apply_second_level_decision(
    model: str,
    result: dict[str, Any],
    evaluation_criterion: str,
    backend: str = "ollama",
) -> None:
    """Review the first-level verdict and determine the final prediction.

    A valid first-level verdict is reviewed by a second model call.
    The original verdict is retained when the review says ``correct``;
    otherwise a valid corrected verdict is required.

    Malformed JSON responses are recorded as parsing errors, while
    missing or unsupported verdict labels are recorded as invalid labels.

    Args:
        model: Judge model identifier.
        result: Example result dictionary to update in place.
        evaluation_criterion: Criterion defining valid final labels.
        backend: Model backend used for evaluation.
    """
    first_level_label = result["first_level_label"]

    if result.get("first_level_parsing_failed", False):
        record_parsing_error(
            result,
            f"Could not parse first-level output: {result['first_raw_output']!r}",
            "first_level",
        )
        return

    if first_level_label not in allowed_labels(evaluation_criterion):
        record_invalid_label(
            result,
            f"Invalid first-level verdict: {result['first_raw_output']!r}",
            "first_level",
        )
        return

    result["current_stage"] = "second_level"
    second_level_prompt = build_second_level_prompt(
        criterion=evaluation_criterion,
        first_level_prompt=result["first_prompt"],
        first_level_response=result["first_raw_output"],
    )

    result["second_level_prompt"] = second_level_prompt

    second_result = judge_response(
        prompt=second_level_prompt,
        model=model,
        backend=backend,
    )

    record_usage(result, "second_level", second_result)
    result["second_level_parsing_failed"] = second_result.get(
        "parsing_failed", False
    )
    verdict = second_result.get("predicted_label")

    second_level_verdict = (
        verdict
        if verdict in VALID_SECOND_LEVEL_VERDICTS
        else None
    )

    result["second_level_raw_output"] = second_result.get(
        "raw_output"
    )
    result["second_level_verdict"] = second_level_verdict

    corrected_verdict = second_result.get("corrected_verdict")

    result["second_level_corrected_verdict"] = (
        corrected_verdict
        if corrected_verdict in allowed_labels(evaluation_criterion)
        else None
    )

    result["second_level_corrected_explanation"] = second_result.get(
        "corrected_explanation"
    )

    if result.get("second_level_parsing_failed", False):
        record_parsing_error(
            result,
            f"Could not parse second-level output: {result['second_level_raw_output']!r}",
            "second_level",
        )
        return

    if second_level_verdict not in VALID_SECOND_LEVEL_VERDICTS:
        record_invalid_label(
            result,
            f"Invalid second-level verdict: {verdict!r}",
            "second_level",
        )
        return

    if second_level_verdict == "correct":
        result["second_level_explanation"] = second_result.get(
            "explanation"
        )
        result["predicted_label"] = first_level_label
        return

    result["second_level_explanation"] = second_result.get(
        "corrected_explanation"
    )

    if corrected_verdict in allowed_labels(evaluation_criterion):
        result["predicted_label"] = corrected_verdict
    else:
        record_invalid_label(
            result,
            f"Invalid corrected verdict: {corrected_verdict!r}",
            "second_level",
        )

def run_judge_experiment(
    dataset: list[dict[str, Any]],
    run_id: str,
    model: str,
    method: str,
    evaluation_criterion: str,
    baseline_criterion_detail: str,
    dataset_id: str,
    dynamic_prediction_variant: str,
    confidence_mode: bool = False,
    backend: str = "ollama",
) -> list[dict[str, Any]]:
    """Evaluate a dataset with baseline, dynamic, or second-level prompting.

    Process examples individually, recording predictions, model usage,
    and parsing, API, or runtime errors. Dynamic prompting includes a
    preliminary prediction stage; second-level prompting reviews the
    first-level judgment with an additional model call.

    Args:
        dataset: Prepared examples to evaluate.
        run_id: Identifier of the experiment run.
        model: Judge model identifier.
        method: One of ``baseline``, ``dynamic``, or ``second_level``.
        evaluation_criterion: Criterion used to assess each response.
        baseline_criterion_detail: Detail level of the baseline criterion.
        dataset_id: Dataset identifier.
        dynamic_prediction_variant: Prediction variant for dynamic prompting.
        confidence_mode: Whether the first-level prompt requests confidence.
        backend: Model backend used for evaluation.

    Returns:
        Result dictionaries for processed examples, or an empty list
        when the run identifier is empty.

    Raises:
        ValueError: If the evaluation method is unsupported.
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
            dataset_id=dataset_id,
            evaluation_criterion=evaluation_criterion,
            baseline_criterion_detail=baseline_criterion_detail,
            dynamic_prediction_variant=dynamic_prediction_variant,
            confidence_mode=confidence_mode,
            backend=backend,
        )

        try:
            run_first_level_judge(
                example=example,
                model=model,
                method=method,
                result=result,
                evaluation_criterion=evaluation_criterion,
                baseline_criterion_detail=(
                    baseline_criterion_detail
                ),
                dynamic_prediction_variant=(
                    dynamic_prediction_variant
                ),
                confidence_mode=confidence_mode,
                backend=backend,
            )

        except Exception as e:
            logger.exception(
                f"First-level judge failed for example "
                f"{example.get('id')}: {e}"
            )
            record_exception(result, e, result["current_stage"])
            results.append(result)
            continue
        if method in NON_SECOND_LEVEL_METHODS:
            try:
                apply_baseline_decision(result)
            except Exception as e:
                logger.exception("Final decision failed for example %s: %s", example.get("id"), e)
                record_exception(result, e, "first_level")
            results.append(result)
            continue
        if method == "second_level":
            try:
                apply_second_level_decision(
                    model=model,
                    result=result,
                    evaluation_criterion=evaluation_criterion,
                    backend=backend,
                )
            except Exception as e:
                logger.exception(
                    f"Second-level judge failed for example "
                    f"{example.get('id')}: {e}"
                )

                record_exception(result, e, result["current_stage"])
            results.append(result)

    for result in results:
        result.pop("current_stage", None)

    logger.info(
        f"Finished. Collected {len(results)} results."
    )
    return results
