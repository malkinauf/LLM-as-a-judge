import json
import logging
from ollama import chat
from typing import Any

logger = logging.getLogger(__name__)

def normalize_label(label: str) -> str:
    """
    Normalize different label variants to the internal evaluation labels.
    """

    if not label:
        return "invalid_label"

    label = label.strip().upper()

    mapping = {
        "SUPPORTED": "truthful",
        "UNSUPPORTED": "not_truthful",

        "TRUTHFUL": "truthful",
        "NOT_TRUTHFUL": "not_truthful",
        "NOT TRUTHFUL": "not_truthful",
    }

    return mapping.get(label, "invalid_label")

def judge_response(prompt: str, model: str) -> dict[str, Any]:
    """
    Send a prepared prompt to the judge model and parse its JSON response.
    """

    response = chat(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        options={
            "temperature": 0
        }
    )

    raw_output = response["message"]["content"]

    try:
        parsed_output = json.loads(raw_output)

        raw_label = parsed_output.get("answer", "")

        predicted_label = normalize_label(raw_label)

        return {
            "predicted_label": predicted_label,
            "explanation": parsed_output.get("explanation", ""),
            "raw_output": raw_output
        }

    except json.JSONDecodeError:
        logger.warning("Failed to parse model output as JSON.")

        return {
            "predicted_label": "parsing_error",
            "explanation": "Could not parse model output as JSON.",
            "raw_output": raw_output
        }