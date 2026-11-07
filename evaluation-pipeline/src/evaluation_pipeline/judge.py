import json
import logging
import os
import time
from typing import Any

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ollama import chat

from evaluation_pipeline.experiment_config.model_registry import get_openrouter_model_config

logger = logging.getLogger(__name__)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_TIMEOUT_SECONDS = 120 
VALID_BACKENDS = {"ollama", "openrouter"}

class ModelAPIError(RuntimeError):
    """An API request failed; status and provider details are preserved."""

    def __init__(self, message: str, *, status_code: int | None = None, details: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details

def parse_confidence(value: Any) -> tuple[int | None, bool]:
    """Normalize confidence to an integer from 1 to 5.

    Return the normalized value and a flag indicating invalid input.
    Missing confidence is allowed and is not considered invalid.
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

def _openrouter_call(prompt: str, model: str, *, json_mode: bool) -> dict[str, Any]:
    """Call OpenRouter using model settings from the YAML configuration.

    Return model output, token usage, latency, and OpenRouter metadata.
    Raise ModelAPIError for HTTP, connection, or response format errors.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    model_config = get_openrouter_model_config(model)

    for field in ("provider", "reasoning", "max_tokens"):
        if field in model_config:
            payload[field] = model_config[field]

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

    # Retry only when OpenRouter returns HTTP 429.
    for attempt in range(5):
        try:
            with urlopen(
                request, timeout=OPENROUTER_TIMEOUT_SECONDS
            ) as response:
                data = json.load(response)
            break

        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")

            try:
                details = json.loads(body)
            except ValueError:
                details = {"message": body[:1000]}

            error_info = (
                details.get("error", {})
                if isinstance(details, dict)
                else {}
            )

            error_message = (
                error_info.get("message", "")
                if isinstance(error_info, dict)
                else str(error_info)
            )

            if exc.code == 429 and attempt < 4:
                wait = min(10 * (2 ** attempt), 60)

                retry_after = exc.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = max(wait, float(retry_after))
                    except ValueError:
                        pass

                metadata = (
                    error_info.get("metadata", {})
                    if isinstance(error_info, dict)
                    else {}
                )

                headers = (
                    metadata.get("headers", {})
                    if isinstance(metadata, dict)
                    else {}
                )

                reset_ms = headers.get("X-RateLimit-Reset")

                if reset_ms:
                    try:
                        wait = max(
                            wait,
                            float(reset_ms) / 1000 - time.time() + 1,
                        )
                    except (ValueError, TypeError):
                        pass

                logger.warning(
                    "OpenRouter HTTP 429: model=%s, retry %s/4 in %.1fs",
                    model,
                    attempt + 1,
                    wait,
                )

                time.sleep(wait)
                continue

            logger.error(
                "OpenRouter API error: model=%s, HTTP=%s, message=%s",
                model,
                exc.code,
                error_message,
            )

            raise ModelAPIError(
                f"OpenRouter API HTTP {exc.code}",
                status_code=exc.code,
                details=details,
            ) from exc

        except (URLError, TimeoutError) as exc:
            raise ModelAPIError(
                "OpenRouter connection failed"
            ) from exc

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
    }

def get_raw_model_response(
    prompt: str,
    model: str,
    *,
    return_usage: bool = False,
    backend: str = "ollama",
) -> str | dict[str, Any]:
    """Return raw model text from Ollama or OpenRouter.

    Optionally include token usage, latency, and available metadata.
    """
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
    """Evaluate a prepared prompt and parse the judge's JSON response.

    Return the predicted label, explanation, confidence information,
    usage metrics, and available OpenRouter metadata.
    Invalid JSON is reported as a parsing error.
    """ 
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
    }
    try:
        parsed_output = json.loads(raw_output)
        if not isinstance(parsed_output, dict):
            raise ValueError("Expected a JSON object.")
        confidence, confidence_invalid = parse_confidence(parsed_output.get("confidence"))
        return {
            "predicted_label": parsed_output.get("verdict"),
            "parsing_failed": False,
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
            "predicted_label": None,
            "parsing_failed": True,
            "explanation": "Could not parse model output as a JSON object.",
            "confidence": None,
            "confidence_invalid": False,
            "corrected_verdict": None,
            "corrected_explanation": None,
            "raw_output": raw_output,
            **usage,
            **openrouter_metadata,
        }