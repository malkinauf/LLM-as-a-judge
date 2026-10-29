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

def extract_usage(response: Any) -> dict[str, Any]:
    """Extract Ollama usage; missing counts remain unknown rather than zero."""
    input_tokens = response.get("prompt_eval_count")
    output_tokens = response.get("eval_count")
    duration = response.get("total_duration")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None else None
        ),
        "latency_seconds": duration / 1e9 if duration is not None else None,
    }

def get_raw_model_response(
    prompt: str, model: str, *, return_usage: bool = False
) -> str | dict[str, Any]:
    """Return text by default; optionally return text and usage together."""
    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    raw_output = response["message"]["content"]
    if return_usage:
        return {"raw_output": raw_output, **extract_usage(response)}
    return raw_output


def judge_response(prompt: str, model: str) -> dict[str, Any]:
    """Send a prepared prompt to the judge model and parse its JSON response."""

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
    usage = extract_usage(response)

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
            **usage,
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
            **usage,
        }