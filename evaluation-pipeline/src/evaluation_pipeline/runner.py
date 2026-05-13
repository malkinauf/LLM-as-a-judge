from tqdm import tqdm
from .judge import judge_response
from .prompts import build_experiment_prompt
import logging

logger = logging.getLogger(__name__)

def run_judge_experiment(
    dataset,
    run_id,
    model,
    prompt_type,
    templates,
    prompt_file,
    dataset_file
):
    results = []
# ToDo: need refactoring of the code
    if not run_id:
        print("Experiment skipped. Set RUN_EXPERIMENT = True to run.")
        return results

    for example in tqdm(dataset):

        if prompt_type == "baseline":
            prompt = build_experiment_prompt(
                prompt_type="baseline",
                templates=templates,
                data=example
            )
            judge_result = judge_response(prompt, model)

        elif prompt_type == "second_level":
            judge_prompt = build_experiment_prompt(
                prompt_type="baseline",
                templates=templates,
                data=example
            )

            first_judge_result = judge_response(judge_prompt, model)

            second_level_data = {
                "judge_task": judge_prompt.split("### Your Output")[0].strip(),
                "judge_output": first_judge_result["raw_output"]
            }

            prompt = build_experiment_prompt(
                prompt_type="second_level",
                templates=templates,
                data=second_level_data
            )

            judge_result = judge_response(prompt, model)

        elif prompt_type == "dynamic":
            hint_prompt = build_experiment_prompt(
                prompt_type="hint",
                templates=templates,
                data=example
            )

            hint_result = judge_response(hint_prompt, model)

            dynamic_data = {
                "question": example["question"],
                "answer": example["answer"],
                "hint": hint_result["raw_output"]
            }

            prompt = build_experiment_prompt(
                prompt_type="dynamic",
                templates=templates,
                data=dynamic_data
            )

            judge_result = judge_response(prompt, model)

        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")

        results.append({
            "run_id": run_id,
            "model": model,
            "prompt_type": prompt_type,
            "prompt_file": prompt_file,
            "dataset_file": dataset_file,

            "id": example["id"],
            "question": example["question"],
            "answer": example["answer"],

            "true_label": example["y_true"],

            # prediction of the first-level judge
            "first_level_label": (
                first_judge_result["predicted_label"]
                if prompt_type == "second_level"
                else judge_result["predicted_label"]
            ),

            # final prediction used for metrics
            "predicted_label": (
                first_judge_result["predicted_label"]
                if prompt_type == "second_level" and judge_result["predicted_label"] == "correct"
                else (
                    judge_result.get("corrected_answer")
                    if prompt_type == "second_level"
                    else judge_result["predicted_label"]
                )
            ),

            # whether the second-level judge thinks the first judge was correct
            "second_level_verdict": (
                judge_result["predicted_label"]
                if prompt_type == "second_level"
                else None
            ),
            "hint_output": (
                hint_result["raw_output"]
                if prompt_type == "dynamic"
                else None
            ),


            "explanation": judge_result.get("corrected_explanation")
            if prompt_type == "second_level"
            else judge_result.get("explanation"),

            "raw_output": judge_result["raw_output"],
            "final_prompt": prompt
        })

    logger.info(f"Finished. Collected {len(results)} results.")
    return results
