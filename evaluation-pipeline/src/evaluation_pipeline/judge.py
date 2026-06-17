import json
import logging
from typing import Any

from ollama import chat


logger = logging.getLogger(__name__)


def get_raw_model_response(prompt: str, model: str) -> str:
    """
    Send a prompt to the model and return the raw text response.

    This function is used for prompts that do not necessarily
    return a judge verdict, such as the prediction step in
    dynamic prompting.

    Args:
        prompt: Fully formatted prompt sent to the model.
        model: Ollama model name.

    Returns:
        Raw model response as a string.
    """

    response = chat(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": 0,
        },
    )

    return response["message"]["content"]


def get_json_model_response(prompt: str, model: str) -> str:
    """
    Send a judge prompt to the model and return a structured JSON response.

    This function uses Ollama's JSON formatting mode to reduce
    parsing errors for prompts that are expected to return a
    structured judge verdict.

    Args:
        prompt: Fully formatted judge prompt sent to the model.
        model: Ollama model name.

    Returns:
        Raw JSON-formatted model response as a string.
    """

    response = chat(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        format="json",
        options={
            "temperature": 0,
        },
    )

    return response["message"]["content"]


def normalize_judge_output(
    parsed_output: dict[str, Any],
    raw_output: str,
) -> dict[str, Any]:
    """
    Convert parsed model JSON into the normalized judge result format.

    Args:
        parsed_output: Parsed JSON object returned by the model.
        raw_output: Original raw model response.

    Returns:
        Normalized judge result dictionary.
    """

    return {
        "predicted_label": parsed_output.get("verdict"),
        "explanation": parsed_output.get("explanation"),
        "corrected_verdict": parsed_output.get("corrected_verdict"),
        "corrected_explanation": parsed_output.get(
            "corrected_explanation"
        ),
        "raw_output": raw_output,
    }


def judge_response(prompt: str, model: str) -> dict[str, Any]:
    """
    Send a judge prompt to the model and parse its JSON response.

    This function is used for prompts that are expected to return
    a structured JSON verdict, such as baseline, second-level,
    and final dynamic judge prompts.

    Args:
        prompt: Fully formatted judge prompt sent to the model.
        model: Ollama model name.

    Returns:
        Normalized judge result dictionary. If JSON parsing fails,
        the returned dictionary contains predicted_label set to
        parsing_error.
    """

    raw_output = get_json_model_response(
        prompt=prompt,
        model=model,
    )

    try:
        parsed_output = json.loads(raw_output)
        return normalize_judge_output(
            parsed_output=parsed_output,
            raw_output=raw_output,
        )

    except json.JSONDecodeError as e:
        logger.warning(
            f"Failed to parse model output as JSON: {e}"
        )
        logger.warning(
            f"RAW OUTPUT REPR: {repr(raw_output)}"
        )
        logger.warning(
            f"RAW OUTPUT END: {repr(raw_output[-300:])}"
        )

        return {
            "predicted_label": "parsing_error",
            "explanation": "Could not parse model output as JSON.",
            "corrected_verdict": None,
            "corrected_explanation": None,
            "raw_output": raw_output,
        }