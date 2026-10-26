import json
import logging
from ollama import chat
from typing import Any

logger = logging.getLogger(__name__)

def parse_confidence(value: Any) -> tuple[int | None, bool]:
    """
    Normalize confidence to an integer from 1 to 5.

    Returns:
        A tuple containing the normalized confidence and
        whether the supplied value was invalid.
    """

    if value is None:
        return None, False

    if isinstance(value, bool):
        return None, True

    if isinstance(value, int):
        if 1 <= value <= 5:
            return value, False
        return None, True

    if isinstance(value, str):
        if value in {"1", "2", "3", "4", "5"}:
            return int(value), False

    return None, True

def get_raw_model_response(prompt: str, model: str) -> str:
    """
    Send a prompt to the judge model and return its raw response text.
    """

    response = chat(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        options={
            "temperature": 0,
        }
    )

    return response["message"]["content"]


def judge_response(prompt: str, model: str) -> dict[str, Any]:
    """
    Send a prepared prompt to the judge model and parse its JSON response.
    """

    response = chat(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        format="json",
        options={
            "temperature": 0,
        }
    )

    raw_output = response["message"]["content"]

    try:
        parsed_output = json.loads(raw_output)

        if not isinstance(parsed_output, dict):
            raise ValueError("Expected a JSON object.")

        confidence, confidence_invalid = parse_confidence(
            parsed_output.get("confidence")
        )

        return {
            "predicted_label": parsed_output.get("verdict"),
            "explanation": parsed_output.get("explanation"),
            "confidence": confidence,
            "confidence_invalid": confidence_invalid,
            "corrected_verdict": parsed_output.get("corrected_verdict"),
            "corrected_explanation": parsed_output.get("corrected_explanation"),
            "raw_output": raw_output,
        }

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse model output: {e}")
        logger.warning(f"RAW OUTPUT REPR: {repr(raw_output)}")
        logger.warning(f"RAW OUTPUT END: {repr(raw_output[-300:])}")

        return {
            "predicted_label": "parsing_error",
            "explanation": "Could not parse model output as a JSON object.",
            "confidence": None,
            "confidence_invalid": False,
            "corrected_verdict": None,
            "corrected_explanation": None,
            "raw_output": raw_output,
        }