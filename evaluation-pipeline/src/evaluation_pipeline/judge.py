import json
import logging
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ollama import chat
from typing import Any

logger = logging.getLogger(__name__)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
VALID_BACKENDS = {"ollama", "openrouter"}

class ModelAPIError(RuntimeError):
    """An API request failed; status and provider details are preserved."""

    def __init__(self, message: str, *, status_code: int | None = None, details: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details

def parse_confidence(value: Any) -> tuple[int | None, bool]:
    """
    Normalize confidence to an integer from 1 to 5."""

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

def _openrouter_call(prompt: str, model: str, *, json_mode: bool) -> dict[str, Any]:
    """Return the same raw_output and usage keys as Ollama calls."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    request = Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urlopen(request, timeout=120) as response:
            data = json.load(response)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            details = json.loads(body)
        except ValueError:
            details = {"message": body[:1000]}
        raise ModelAPIError(
            f"OpenRouter API HTTP {exc.code}", status_code=exc.code, details=details
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise ModelAPIError("OpenRouter connection failed") from exc

    latency = time.perf_counter() - start
    try:
        raw_output = data["choices"][0]["message"]["content"]
        if not isinstance(raw_output, str):
            raise TypeError("Expected textual content")
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelAPIError("Unexpected OpenRouter response format", details=data) from exc

    usage = data.get("usage") or {}
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return {
        "raw_output": raw_output,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency_seconds": latency,
        "openrouter_generation_id": data.get("id"),
        "openrouter_response_model": data.get("model"),
        "openrouter_provider": data.get("provider"),
    }

def get_raw_model_response(
    prompt: str,
    model: str,
    *,
    return_usage: bool = False,
    backend: str = "ollama",
) -> str | dict[str, Any]:
    """Return model text, optionally including usage metrics."""

    if backend not in VALID_BACKENDS:
        raise ValueError(f"Unknown backend: {backend}")

    if backend == "openrouter":
        result = _openrouter_call(
            prompt,
            model,
            json_mode=False,
        )
    else:
        response = chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        result = {
            "raw_output": response["message"]["content"],
            **extract_usage(response),
        }

    return result if return_usage else result["raw_output"]


def judge_response(prompt: str, model: str, *, backend: str = "ollama") -> dict[str, Any]:
    """Send a prepared prompt to the judge model and parse its JSON response."""
    if backend not in VALID_BACKENDS:
        raise ValueError(f"Unknown backend: {backend}")
    if backend == "openrouter":
        result = _openrouter_call(prompt, model, json_mode=True)
    else:
        response = chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0},
        )
        result = {"raw_output": response["message"]["content"], **extract_usage(response)}

    raw_output = result["raw_output"]
    usage = {key: result[key] for key in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "latency_seconds",
    )}

    openrouter_metadata = {
        "openrouter_generation_id": result.get("openrouter_generation_id"),
        "openrouter_response_model": result.get("openrouter_response_model"),
        "openrouter_provider": result.get("openrouter_provider"),
    }
    try:
        parsed_output = json.loads(raw_output)
        if not isinstance(parsed_output, dict):
            raise ValueError("Expected a JSON object.")
        confidence, confidence_invalid = parse_confidence(parsed_output.get("confidence"))
        return {
            "predicted_label": parsed_output.get("verdict"),
            "explanation": parsed_output.get("explanation"),
            "confidence": confidence,
            "confidence_invalid": confidence_invalid,
            "corrected_verdict": parsed_output.get("corrected_verdict"),
            "corrected_explanation": parsed_output.get("corrected_explanation"),
            "raw_output": raw_output,
            **usage,
            **openrouter_metadata,
        }
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse model output: %s", exc)
        logger.warning("RAW OUTPUT REPR: %r", raw_output)
        logger.warning("RAW OUTPUT END: %r", raw_output[-300:])
        return {
            "predicted_label": "parsing_error",
            "explanation": "Could not parse model output as a JSON object.",
            "confidence": None,
            "confidence_invalid": False,
            "corrected_verdict": None,
            "corrected_explanation": None,
            "raw_output": raw_output,
            **usage,
            **openrouter_metadata,
        }