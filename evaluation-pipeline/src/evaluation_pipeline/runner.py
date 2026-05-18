import logging

from tqdm import tqdm
from evaluation_pipeline.judge import judge_response
from evaluation_pipeline.prompts import build_experiment_prompt


logger = logging.getLogger(__name__)


def run_judge_experiment(
    dataset,
    run_id,
    model,
    method,
    templates,
    dataset_file
):
    results = []
    
# ToDo: need refactoring of the code

    if not run_id:
        print("Experiment skipped. Set RUN_EXPERIMENT = True to run.")
        return results

    for example in tqdm(dataset):
        baseline_prompt = build_experiment_prompt(
            prompt_type="baseline",
            templates=templates,
            data=example
        )

        first_judge_result = judge_response(baseline_prompt, model)

        result = {
            "id": example["id"],
            "question": example["question"],
            "model_response": example["model_response"],
            "true_label": example["y_true"],

            "model": model,
            "method": method,
            "run_id": run_id,

            "dataset_file": dataset_file,

            # ToDo "answer" to "label" in judge.py ?
            "first_prompt": baseline_prompt,
            "first_raw_output": first_judge_result.get("raw_output"),
            "first_level_label": first_judge_result.get("predicted_label"),
            "first_level_explanation": first_judge_result.get("explanation"),
        }

        if method == "baseline":
            result["predicted_label"] = result["first_level_label"]
            result["second_level_prompt"] = None
            result["second_level_raw_output"] = None
            result["second_level_verdict"] = None
            result["second_level_explanation"] = None

        elif method == "second_level":
            second_level_prompt = build_experiment_prompt(
                prompt_type="second_level",
                templates=templates,
                data={
                    "question": example["question"],
                    "model_response": example["model_response"],
                    "first_judge_verdict": result["first_level_label"],
                    "first_judge_explanation": result["first_level_explanation"],
                },
            )

            second_result = judge_response(second_level_prompt, model)

            result["second_level_prompt"] = second_level_prompt
            result["second_level_raw_output"] = second_result.get("raw_output")
            result["second_level_verdict"] = second_result.get(
                "predicted_label")
            result["second_level_explanation"] = second_result.get(
                "corrected_explanation")

            if result["second_level_verdict"] == "correct":
                result["predicted_label"] = result["first_level_label"]
            elif result["second_level_verdict"] == "not_correct":
                result["predicted_label"] = second_result.get("corrected_verdict")

            else:
                result["predicted_label"] = "parsing_error"

        else:
            raise ValueError(f"Unknown method: {method}")

        results.append(result)

    logger.info(f"Finished. Collected {len(results)} results.")
    return results
