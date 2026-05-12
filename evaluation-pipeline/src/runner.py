import logging
from typing import Any, Callable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for environments without tqdm
    def tqdm(iterable):  # type: ignore
        return iterable

from .judge import judge_response
from .prompts import build_experiment_prompt

logger = logging.getLogger(__name__)

REQUIRED_EXAMPLE_FIELDS = {"id", "question", "answer", "y_true"}


def _validate_example(example: dict[str, Any], index: int) -> None:
    missing = REQUIRED_EXAMPLE_FIELDS.difference(example.keys())
    if missing:
        raise ValueError(
            f"Dataset example at index {index} is missing required fields: {sorted(missing)}"
        )


def run_judge_experiment(
    dataset: list[dict[str, Any]],
    run_id: str,
    model: str,
    prompt_type: str,
    templates: dict[str, str],
    prompt_file: str,
    dataset_file: str,
    judge_fn: Callable[[str, str], dict[str, Any]] = judge_response,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    if not run_id:
        logger.info("Experiment skipped. Set RUN_EXPERIMENT = True to run.")
        return results

    for index, example in enumerate(tqdm(dataset)):
        _validate_example(example, index)
        first_judge_result: dict[str, Any] | None = None
        hint_result: dict[str, Any] | None = None

        if prompt_type == "baseline":
            prompt = build_experiment_prompt(
                prompt_type="baseline",
                templates=templates,
                data=example,
            )
            judge_result = judge_fn(prompt, model)
            predicted_label = judge_result.get("verdict")
            explanation = judge_result.get("explanation", "")
            second_level_verdict = None

        elif prompt_type == "second_level":
            judge_prompt = build_experiment_prompt(
                prompt_type="baseline",
                templates=templates,
                data=example,
            )

            first_judge_result = judge_fn(judge_prompt, model)

            second_level_data = {
                "judge_task": judge_prompt.split("### Your Output")[0].strip(),
                "judge_output": first_judge_result["raw_output"],
            }

            prompt = build_experiment_prompt(
                prompt_type="second_level",
                templates=templates,
                data=second_level_data,
            )

            judge_result = judge_fn(prompt, model)
            second_level_verdict = judge_result.get("verdict")
            if second_level_verdict == "correct":
                predicted_label = first_judge_result.get("verdict")
            else:
                predicted_label = judge_result.get("corrected_answer") or first_judge_result.get(
                    "verdict"
                )
            explanation = judge_result.get("corrected_explanation") or judge_result.get(
                "explanation", ""
            )

        elif prompt_type == "dynamic":
            hint_prompt = build_experiment_prompt(
                prompt_type="hint",
                templates=templates,
                data=example,
            )

            hint_result = judge_fn(hint_prompt, model)

            dynamic_data = {
                "question": example["question"],
                "answer": example["answer"],
                "hint": hint_result["raw_output"],
            }

            prompt = build_experiment_prompt(
                prompt_type="dynamic",
                templates=templates,
                data=dynamic_data,
            )

            judge_result = judge_fn(prompt, model)
            predicted_label = judge_result.get("verdict")
            explanation = judge_result.get("explanation", "")
            second_level_verdict = None

        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")

        results.append(
            {
                "run_id": run_id,
                "model": model,
                "prompt_type": prompt_type,
                "prompt_file": prompt_file,
                "dataset_file": dataset_file,
                "id": example["id"],
                "question": example["question"],
                "answer": example["answer"],
                "true_label": example["y_true"],
                "first_level_label": (
                    first_judge_result["verdict"]
                    if prompt_type == "second_level"
                    else judge_result["verdict"]
                ),
                "predicted_label": predicted_label,
                "second_level_verdict": second_level_verdict,
                "hint_output": hint_result["raw_output"] if prompt_type == "dynamic" else None,
                "explanation": explanation,
                "raw_output": judge_result["raw_output"],
                "final_prompt": prompt,
            }
        )

    logger.info("Finished. Collected %s results.", len(results))
    return results
