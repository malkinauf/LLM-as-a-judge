import json
import logging
from ollama import chat
from typing import Any

logger = logging.getLogger(__name__)


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

        return {
            "predicted_label": parsed_output.get("verdict"),
            "explanation": parsed_output.get("explanation"),
            "corrected_verdict": parsed_output.get("corrected_verdict"),
            "corrected_explanation": parsed_output.get("corrected_explanation"),
            "raw_output": raw_output

        }

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse model output as JSON: {e}")
        logger.warning(f"RAW OUTPUT REPR: {repr(raw_output)}")
        logger.warning(f"RAW OUTPUT END: {repr(raw_output[-300:])}")
        return {
            "predicted_label": "parsing_error",
            "explanation": "Could not parse model output as JSON.",
            "raw_output": raw_output
        }
