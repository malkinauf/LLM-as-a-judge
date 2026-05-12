import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _normalize_verdict(value: Any) -> str:
    if value is None:
        return "invalid_label"
    verdict = str(value).strip()
    if verdict.startswith("[") and verdict.endswith("]"):
        verdict = verdict[1:-1].strip()
    return verdict or "invalid_label"


def _default_chat(model: str, prompt: str) -> dict[str, Any]:
    try:
        from ollama import chat  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "ollama is required for live judge calls. Install ollama or inject chat_fn in tests."
        ) from exc

    return chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )


def judge_response(
    prompt: str,
    model: str,
    chat_fn: Callable[[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    response = (chat_fn or _default_chat)(model, prompt)
    raw_output = response.get("message", {}).get("content", "")

    try:
        parsed_output = json.loads(raw_output)
    except json.JSONDecodeError:
        logger.warning("Failed to parse model output as JSON.")
        return {
            "verdict": "parsing_error",
            "explanation": "Could not parse model output as JSON.",
            "corrected_answer": None,
            "corrected_explanation": None,
            "raw_output": raw_output,
        }

    verdict = _normalize_verdict(
        parsed_output.get("verdict")
        or parsed_output.get("answer")
        or parsed_output.get("predicted_label")
    )

    return {
        "verdict": verdict,
        "explanation": parsed_output.get("explanation", ""),
        "corrected_answer": parsed_output.get("corrected_answer"),
        "corrected_explanation": parsed_output.get("corrected_explanation"),
        "raw_output": raw_output,
    }
