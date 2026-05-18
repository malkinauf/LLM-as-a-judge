import logging
from typing import Any

from tqdm import tqdm

from evaluation_pipeline.judge import judge_response
from evaluation_pipeline.prompts import build_experiment_prompt


logger = logging.getLogger(__name__)


VALID_METHODS = (
    "baseline",
    "second_level",
)

VALID_JUDGE_LABELS = {
    "truthful",
    "not_truthful",
}

VALID_SECOND_LEVEL_VERDICTS = {
    "correct",
    "not_correct",
}


def run_judge_experiment(
    dataset: list[dict[str, Any]],
    run_id: str,
    model: str,
    method: str,
    templates: dict[str, str],
    dataset_file: str,
) -> list[dict[str, Any]]:

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

        baseline_prompt = build_experiment_prompt(
            prompt_type="baseline",
            templates=templates,
            data=example,
        )

        try:
            first_judge_result = judge_response(
                baseline_prompt,
                model,
            )

        except Exception as e:
            logger.exception(
                f"First-level judge failed "
                f"for example {example.get('id')}: {e}"
            )

            results.append({
                "id": example.get("id"),
                "predicted_label": "runtime_error",
                "error": str(e),
            })

            continue

        first_level_label = first_judge_result.get(
            "predicted_label"
        )

        result = {
            "id": example["id"],
            "question": example["question"],
            "model_response": example["model_response"],
            "true_label": example["y_true"],

            "model": model,
            "method": method,
            "run_id": run_id,
            "dataset_file": dataset_file,

            "first_prompt": baseline_prompt,
            "first_raw_output": first_judge_result.get(
                "raw_output"
            ),
            "first_level_label": first_level_label,
            "first_level_explanation": first_judge_result.get(
                "explanation"
            ),

            "second_level_prompt": None,
            "second_level_raw_output": None,
            "second_level_verdict": None,
            "second_level_explanation": None,

            "predicted_label": None,
        }

        if method == "baseline":
            if first_level_label in VALID_JUDGE_LABELS:
                result["predicted_label"] = first_level_label
            else:
                result["predicted_label"] = "parsing_error"

            results.append(result)
            continue

        if method == "second_level":

            if first_level_label not in VALID_JUDGE_LABELS:
                result["predicted_label"] = "parsing_error"
                results.append(result)
                continue

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

            try:
                second_result = judge_response(
                    second_level_prompt,
                    model,
                )

            except Exception as e:
                logger.exception(
                    f"Second-level judge failed "
                    f"for example {example.get('id')}: {e}"
                )

                result["predicted_label"] = "runtime_error"
                result["error"] = str(e)

                results.append(result)
                continue

            second_level_verdict = second_result.get(
                "predicted_label"
            )

            result["second_level_prompt"] = second_level_prompt
            result["second_level_raw_output"] = second_result.get(
                "raw_output"
            )
            result["second_level_verdict"] = second_level_verdict
            result["second_level_explanation"] = (
                second_result.get("corrected_explanation")
                or second_result.get("explanation")
            )

            if second_level_verdict == "correct":
                result["predicted_label"] = first_level_label

            elif second_level_verdict == "not_correct":
                corrected_verdict = second_result.get(
                    "corrected_verdict"
                )

                if corrected_verdict in VALID_JUDGE_LABELS:
                    result["predicted_label"] = corrected_verdict
                else:
                    result["predicted_label"] = "parsing_error"

            else:
                result["predicted_label"] = "parsing_error"

            results.append(result)

    logger.info(
        f"Finished. Collected {len(results)} results."
    )

    return results